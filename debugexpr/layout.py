"""Caller-supplied finite type layouts for structured access.

A layout describes exactly what the caller knows about a type: field
offsets, array lengths, element types and alignment.  Nothing is
inferred from C compiler rules; every number comes from the caller.

Layout inventory
----------------
IntLayout(int_type, align=None)     scalar integer; align defaults to size
PtrLayout(target_region, align=8)   8-byte pointer into a named region
StructLayout(fields, align=1, union=False, size=None)
ArrayLayout(element, length, align=None)   length may be 0 (empty array)

Layout *definition* errors (negative offsets, overlapping fields in a
non-union struct, bad alignment, ...) raise ``ValueError`` at
construction time, mirroring ``IntType``.  Evaluation-time failures
(unknown field name, out-of-bounds subscript, misaligned address) raise
``EvalError`` subclasses carrying a locatable ``path``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import IntType


def _check_align(align: int) -> int:
    if type(align) is not int or isinstance(align, bool) or align < 1:
        raise ValueError(f"align must be a positive int, got {align!r}")
    return align


class Layout:
    """Base class of every type layout."""

    __slots__ = ()

    @property
    def size(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def align(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class IntLayout(Layout):
    """Layout of a scalar integer of an explicit IntType."""

    int_type: IntType
    align: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.int_type, IntType):
            raise ValueError(f"int_type must be an IntType, got {self.int_type!r}")
        object.__setattr__(self, "align", _check_align(self.align if self.align is not None else self.int_type.size))

    @property
    def size(self) -> int:
        return self.int_type.size


@dataclass(frozen=True)
class PtrLayout(Layout):
    """Layout of an 8-byte pointer stored in memory.

    The stored value is a little-endian u64 byte offset into
    ``target_region``.  The target region is part of the layout because
    the caller's type information is what ties the pointer to its
    address space.
    """

    target_region: str
    align: int = 8

    def __post_init__(self) -> None:
        if type(self.target_region) is not str or not self.target_region:
            raise ValueError(f"target_region must be a non-empty str, got {self.target_region!r}")
        _check_align(self.align)

    @property
    def size(self) -> int:
        return 8


@dataclass(frozen=True)
class StructField:
    """One named field: explicit byte offset plus its layout."""

    name: str
    offset: int
    layout: Layout

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError(f"field name must be a non-empty str, got {self.name!r}")
        if type(self.offset) is not int or isinstance(self.offset, bool) or self.offset < 0:
            raise ValueError(f"field {self.name!r}: offset must be a non-negative int, got {self.offset!r}")
        if not isinstance(self.layout, Layout):
            raise ValueError(f"field {self.name!r}: layout must be a Layout, got {self.layout!r}")


@dataclass(frozen=True)
class StructLayout(Layout):
    """Layout of a struct (or explicit union) with caller-given offsets.

    A plain struct (``union=False``) rejects overlapping fields.  A
    union (``union=True``) explicitly permits them: the same bytes are
    then interpretable through more than one field.
    """

    fields: tuple[StructField, ...]
    align: int = 1
    union: bool = False
    size: int | None = None

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        object.__setattr__(self, "fields", fields)
        _check_align(self.align)
        if type(self.union) is not bool:
            raise ValueError(f"union must be a bool, got {self.union!r}")
        seen: set[str] = set()
        for field in fields:
            if not isinstance(field, StructField):
                raise ValueError(f"fields must be StructField instances, got {field!r}")
            if field.name in seen:
                raise ValueError(f"duplicate field name {field.name!r}")
            seen.add(field.name)
        if not self.union:
            intervals = sorted((f.offset, f.offset + f.layout.size, f.name) for f in fields)
            for (prev_start, prev_end, prev_name), (start, end, name) in zip(intervals, intervals[1:]):
                if start < prev_end:
                    raise ValueError(
                        f"fields {prev_name!r} [{prev_start}, {prev_end}) and {name!r} [{start}, {end}) "
                        f"overlap; pass union=True to allow overlapping fields explicitly"
                    )
        content_end = max((f.offset + f.layout.size for f in fields), default=0)
        if self.size is None:
            size = -(-content_end // self.align) * self.align  # round up to align
        else:
            size = self.size
            if type(size) is not int or isinstance(size, bool) or size < content_end:
                raise ValueError(
                    f"explicit size must be an int >= the end of the last field ({content_end}), got {size!r}"
                )
        object.__setattr__(self, "size", size)

    def field(self, name: str) -> StructField | None:
        for field in self.fields:
            if field.name == name:
                return field
        return None


@dataclass(frozen=True)
class ArrayLayout(Layout):
    """Layout of a fixed-length array; ``length == 0`` is an empty array."""

    element: Layout
    length: int
    align: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.element, Layout):
            raise ValueError(f"element must be a Layout, got {self.element!r}")
        if type(self.length) is not int or isinstance(self.length, bool) or self.length < 0:
            raise ValueError(f"length must be a non-negative int, got {self.length!r}")
        object.__setattr__(
            self, "align", _check_align(self.align if self.align is not None else self.element.align)
        )

    @property
    def size(self) -> int:
        return self.element.size * self.length
