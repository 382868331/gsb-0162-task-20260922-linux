"""Caller-supplied, finite type layouts for structured field/array access.

The layouts here are *explicit*: the caller states every field offset,
array length, element type and alignment.  Nothing is derived from C
compiler rules (no padding is invented, no natural alignment is assumed),
so a layout describes exactly the bytes the caller means.

Layout objects
--------------
FieldInfo(name, offset, type, align)   one struct/union member
StructLayout(name, size, fields, union=False)
ArrayLayout(element_type, length)

``type`` of a field / ``element_type`` of an array may be an ``IntType``,
``POINTER``, another ``StructLayout`` or another ``ArrayLayout``
(nested arrays included).

Validation (at construction, ``ValueError`` — same style as ``IntType``):

* offsets are non-negative ints (bool rejected), fields must fit the
  declared struct ``size``;
* a plain (non-union) struct rejects overlapping field byte ranges;
  ``union=True`` is the explicit opt-in that permits overlap;
* array ``length`` may be 0 (empty array: every index is out of bounds);
* ``align`` is recorded caller metadata only — it is *not* enforced,
  so deliberately unaligned fields are expressible and readable.

Pointers in memory
------------------
A pointer stored in memory is encoded as ``POINTER_SIZE = 9`` bytes:
1 byte region tag (index into the ``region_table`` given to
``evaluate``) followed by an 8-byte little-endian unsigned offset.
``encode_pointer`` / ``decode_pointer`` are the only (de)serializers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import IntType, Pointer, PointerType

POINTER_SIZE = 9
"""Byte size of an in-memory pointer: 1-byte region tag + 8-byte LE offset."""


def _require_int_attr(value: object, what: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise ValueError(f"{what} must be an int (bool and float are rejected), got {value!r}")
    return value


def layout_size(layout_type) -> int:
    """Byte size of a layout/leaf type.  POINTER takes POINTER_SIZE bytes."""
    if isinstance(layout_type, IntType):
        return layout_type.size
    if isinstance(layout_type, PointerType):
        return POINTER_SIZE
    if isinstance(layout_type, StructLayout):
        return layout_type.size
    if isinstance(layout_type, ArrayLayout):
        return layout_type.length * layout_size(layout_type.element_type)
    raise ValueError(f"not a layout type: {layout_type!r}")


@dataclass(frozen=True)
class FieldInfo:
    """One named member of a struct/union layout."""

    name: str
    offset: int
    type: object  # IntType | PointerType | StructLayout | ArrayLayout
    align: int = 1

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError(f"field name must be a non-empty str, got {self.name!r}")
        _require_int_attr(self.offset, f"field {self.name!r} offset")
        if self.offset < 0:
            raise ValueError(f"field {self.name!r} offset must be >= 0, got {self.offset}")
        _require_int_attr(self.align, f"field {self.name!r} align")
        if self.align < 1:
            raise ValueError(f"field {self.name!r} align must be >= 1, got {self.align}")
        layout_size(self.type)  # validates the field type is a known layout


@dataclass(frozen=True)
class StructLayout:
    """An explicit struct (or union) layout: caller-given size and fields.

    A plain struct rejects overlapping field byte ranges; pass
    ``union=True`` to explicitly interpret overlaps as a union.
    """

    name: str
    size: int
    fields: tuple
    union: bool = False

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError(f"struct name must be a non-empty str, got {self.name!r}")
        _require_int_attr(self.size, f"struct {self.name!r} size")
        if self.size < 0:
            raise ValueError(f"struct {self.name!r} size must be >= 0, got {self.size}")
        if type(self.union) is not bool:
            raise ValueError(f"struct {self.name!r} union flag must be a bool, got {self.union!r}")
        fields = tuple(self.fields)
        object.__setattr__(self, "fields", fields)
        seen = set()
        ranges = []
        for field in fields:
            if not isinstance(field, FieldInfo):
                raise ValueError(f"struct {self.name!r}: fields must be FieldInfo, got {field!r}")
            if field.name in seen:
                raise ValueError(f"struct {self.name!r}: duplicate field name {field.name!r}")
            seen.add(field.name)
            end = field.offset + layout_size(field.type)
            if end > self.size:
                raise ValueError(
                    f"struct {self.name!r}: field {field.name!r} ends at {end}, "
                    f"beyond the declared size {self.size}"
                )
            ranges.append((field.offset, end, field.name))
        if not self.union:
            for i, (lo_a, hi_a, name_a) in enumerate(ranges):
                for lo_b, hi_b, name_b in ranges[i + 1:]:
                    if lo_a < hi_b and lo_b < hi_a:
                        raise ValueError(
                            f"struct {self.name!r}: fields {name_a!r} and {name_b!r} overlap "
                            f"([{lo_a}, {hi_a}) vs [{lo_b}, {hi_b})); "
                            f"pass union=True to allow overlapping fields"
                        )

    def field(self, name: str):
        for field in self.fields:
            if field.name == name:
                return field
        return None


@dataclass(frozen=True)
class ArrayLayout:
    """A fixed-length array layout; ``length`` 0 is a valid empty array."""

    element_type: object  # IntType | PointerType | StructLayout | ArrayLayout
    length: int

    def __post_init__(self) -> None:
        _require_int_attr(self.length, "array length")
        if self.length < 0:
            raise ValueError(f"array length must be >= 0, got {self.length}")
        layout_size(self.element_type)  # validates the element type

    @property
    def size(self) -> int:
        return self.length * layout_size(self.element_type)


def encode_pointer(ptr: Pointer, region_table: tuple) -> bytes:
    """Encode a pointer as 1-byte region tag + 8-byte LE unsigned offset."""
    tag = region_table.index(ptr.region_id)
    if not 0 <= ptr.offset < (1 << 64):
        raise ValueError(f"pointer offset {ptr.offset} does not fit u64")
    return bytes([tag]) + ptr.offset.to_bytes(8, "little")


def decode_pointer(raw: bytes, region_table: tuple) -> Pointer:
    """Decode a 9-byte in-memory pointer.

    Raises ``ValueError`` when the region tag does not index
    ``region_table``; the evaluator re-raises this as a ``PointerError``
    carrying the AST path.
    """
    if len(raw) != POINTER_SIZE:
        raise ValueError(f"encoded pointer must be {POINTER_SIZE} bytes, got {len(raw)}")
    tag = raw[0]
    if tag >= len(region_table):
        raise ValueError(
            f"region tag {tag} does not index the region table ({len(region_table)} regions)"
        )
    offset = int.from_bytes(raw[1:], "little")
    return Pointer(region_table[tag], offset)
