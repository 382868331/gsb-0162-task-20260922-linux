"""Value types for the debug-expression evaluator.

Integers are always explicitly typed: signed or unsigned, 8/16/32/64 bit.
The canonical Python representation of an integer value is:

* signed type   -> Python int in [-(2**(bits-1)), 2**(bits-1) - 1]
* unsigned type -> Python int in [0, 2**bits - 1]

Pointers are (region_id, offset) pairs; ``offset`` is a byte offset that
may reach the one-past-end position of its region but never beyond.
"""

from __future__ import annotations

from dataclasses import dataclass

ALLOWED_BITS = (8, 16, 32, 64)


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
