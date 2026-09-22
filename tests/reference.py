"""Independent integer-mask reference for the evaluator's arithmetic.

This module deliberately does NOT import anything from ``debugexpr``.
It re-derives the expected results directly from the contract using plain
Python integer masking, so the tests compare two independent
implementations of the same semantics.

Canonical value convention (same as the contract):
  signed   -> Python int in [-(2**(bits-1)), 2**(bits-1) - 1]
  unsigned -> Python int in [0, 2**bits - 1]
"""

from __future__ import annotations


def mask_of(bits: int) -> int:
    return (1 << bits) - 1


def wrap(value: int, bits: int, signed: bool) -> int:
    """Reduce modulo 2**bits and reinterpret per signedness."""
    raw = value & mask_of(bits)
    if signed and raw >= (1 << (bits - 1)):
        raw -= 1 << bits
    return raw


def trunc_divmod(a: int, b: int) -> tuple[int, int]:
    """C-style truncating division and matching remainder."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q, a - q * b


def ref_binop(op: str, a: int, b: int, bits: int, signed: bool) -> int:
    """Reference binary arithmetic on canonical values.

    Shift amounts must already satisfy 0 <= b < bits; div/mod require
    b != 0.  Everything is computed on masked raw values and wrapped.
    """
    m = mask_of(bits)
    ua, ub = a & m, b & m
    if op == "add":
        raw = (ua + ub) & m
    elif op == "sub":
        raw = (ua - ub) & m
    elif op == "mul":
        raw = (ua * ub) & m
    elif op == "shl":
        raw = (ua << ub) & m
    elif op == "shr":
        # arithmetic shift for signed (sign-extend from the raw sign bit),
        # logical shift for unsigned
        raw = (wrap(ua, bits, True) >> ub) & m if signed else (ua >> ub) & m
    elif op in ("div", "mod"):
        sa = wrap(ua, bits, True) if signed else ua
        sb = wrap(ub, bits, True) if signed else ub
        q, r = trunc_divmod(sa, sb)
        raw = (q if op == "div" else r) & m
    elif op == "bitand":
        raw = ua & ub
    elif op == "bitor":
        raw = ua | ub
    elif op == "bitxor":
        raw = ua ^ ub
    else:
        raise AssertionError(f"unknown op {op!r}")
    return wrap(raw, bits, signed)


def ref_cmp(op: str, a: int, b: int) -> int:
    """Reference comparison on canonical values -> 0/1."""
    return int(
        {
            "eq": a == b,
            "ne": a != b,
            "lt": a < b,
            "le": a <= b,
            "gt": a > b,
            "ge": a >= b,
        }[op]
    )


def ref_cast(value: int, src_bits: int, src_signed: bool, dst_bits: int, dst_signed: bool) -> int:
    """Reference cast: value-preserving when representable, otherwise
    modular truncation to the destination width (C-style conversion)."""
    return wrap(value, dst_bits, dst_signed)
