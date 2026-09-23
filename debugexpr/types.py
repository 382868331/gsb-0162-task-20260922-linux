"""Value types for the debug-expression evaluator.

Integers are always explicitly typed: signed or unsigned, 8/16/32/64 bit.
The canonical Python representation of an integer value is:

* signed type   -> Python int in [-(2**(bits-1)), 2**(bits-1) - 1]
* unsigned type -> Python int in [0, 2**bits - 1]

Pointers are (region_id, offset) pairs; ``offset`` is a byte offset that
may reach the one-past-end position of its region but never beyond.

When a register holds up to ``MAX_CANDIDATES`` possible integer values,
evaluation works over sets of possibilities.  The possible values of one
sub-expression are represented by :class:`CandidateSet` (1..8 distinct
members, integers or :class:`Pointer`).  A plain ``Value`` means a single
determined possibility and is interchangeable with a one-element set.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InputError

ALLOWED_BITS = (8, 16, 32, 64)
MAX_CANDIDATES = 8


@dataclass(frozen=True)
class IntType:
    """An explicitly sized, explicitly signed integer type."""

    bits: int
    signed: bool

    def __post_init__(self) -> None:
        if type(self.bits) is not int or self.bits not in ALLOWED_BITS:
            raise ValueError(f"bits must be one of {ALLOWED_BITS}, got {self.bits!r}")
        if type(self.signed) is not bool:
            raise ValueError(f"signed must be a bool, got {self.signed!r}")

    @property
    def size(self) -> int:
        """Size in bytes."""
        return self.bits // 8

    @property
    def min(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    def contains(self, value: int) -> bool:
        return self.min <= value <= self.max

    def wrap(self, value: int) -> int:
        """Reduce ``value`` modulo 2**bits and reinterpret per signedness."""
        raw = value & ((1 << self.bits) - 1)
        if self.signed and raw >= (1 << (self.bits - 1)):
            raw -= 1 << self.bits
        return raw

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{'s' if self.signed else 'u'}{self.bits}"


@dataclass(frozen=True)
class PointerType:
    """The type of (region_id, offset) pointers.  A single instance suffices."""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return "ptr"


POINTER = PointerType()

# Convenient prebuilt integer types.
U8, U16, U32, U64 = (IntType(b, False) for b in ALLOWED_BITS)
S8, S16, S32, S64 = (IntType(b, True) for b in ALLOWED_BITS)


@dataclass(frozen=True)
class Pointer:
    """A pointer value: region identity plus byte offset within the region."""

    region_id: str
    offset: int


@dataclass(frozen=True)
class Value:
    """A typed evaluation result.

    ``data`` is a canonical Python int for IntType, a Pointer for POINTER.
    """

    type: IntType | PointerType
    data: int | Pointer


def _member_sort_key(member: int | Pointer) -> tuple:
    if isinstance(member, Pointer):
        return (1, member.region_id, member.offset)
    return (0, member)


class CandidateSet:
    """A finite, deduplicated set of possible runtime values.

    Members are canonical Python ints (for an integer result) or
    :class:`Pointer` values (for a pointer result); a single set never
    mixes the two kinds.  There are between 1 and ``MAX_CANDIDATES``
    distinct members.  Members are exposed as a deterministically ordered
    tuple so that results and read evidence are reproducible.

    A :class:`CandidateSet` does not itself carry a type; all members of
    one set share the type recorded alongside it (see ``MaybeValue`` in
    the evaluator).
    """

    __slots__ = ("_members",)

    def __init__(self, members) -> None:
        if not isinstance(members, (tuple, list, frozenset, set)):
            raise InputError(
                f"candidate set must be built from a tuple/list/set, got {type(members).__name__}"
            )
        unique: set[int | Pointer] = set()
        saw_int = saw_ptr = False
        for member in members:
            if type(member) is bool or not isinstance(member, (int, Pointer)):
                raise InputError(
                    "candidate members must be ints or Pointers (bool and float rejected), "
                    f"got {member!r}"
                )
            if isinstance(member, Pointer):
                saw_ptr = True
            else:
                saw_int = True
            if saw_int and saw_ptr:
                raise InputError("a candidate set cannot mix integers and pointers")
            unique.add(member)
        if not unique:
            raise InputError("candidate set must contain at least one value")
        if len(unique) > MAX_CANDIDATES:
            raise InputError(
                f"candidate set has {len(unique)} values, exceeding the limit of {MAX_CANDIDATES}"
            )
        self._members: tuple[int | Pointer, ...] = tuple(sorted(unique, key=_member_sort_key))

    @classmethod
    def singleton(cls, member: int | Pointer) -> "CandidateSet":
        obj = cls.__new__(cls)
        obj._members = (member,)
        return obj

    @property
    def members(self) -> tuple[int | Pointer, ...]:
        return self._members

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self):
        return iter(self._members)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CandidateSet) and self._members == other._members

    def __hash__(self) -> int:
        return hash(self._members)

    def __repr__(self) -> str:
        return f"CandidateSet({self._members!r})"
