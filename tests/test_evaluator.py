"""Tests for the debug-expression evaluator.

Arithmetic expectations are cross-checked against ``tests/reference.py``,
an independent plain-integer-mask implementation that does not import the
core under test.
"""

from __future__ import annotations

import random
import unittest

from debugexpr import (
    MAX_NODES,
    And,
    BinOp,
    Cast,
    Cmp,
    Const,
    DivisionByZeroError,
    EvalError,
    Field,
    InputError,
    Load,
    MemorySnapshot,
    NodeBudgetError,
    Not,
    Or,
    Pointer,
    PointerError,
    PtrAdd,
    PtrConst,
    PtrSub,
    RangeError,
    Reg,
    ShiftError,
    TypeMismatchError,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    evaluate,
)
from reference import ref_binop, ref_cast, ref_cmp

EMPTY = MemorySnapshot({})
STACK = MemorySnapshot({"stack": bytes(range(64)), "heap": bytes(32)})

ALL_TYPES = [
    (U8, 8, False), (U16, 16, False), (U32, 32, False), (U64, 64, False),
    (S8, 8, True), (S16, 16, True), (S32, 32, True), (S64, 64, True),
]


def ev(ast, registers=None, memory=EMPTY):
    return evaluate(ast, registers or {}, memory).value


class TestArithmeticModular(unittest.TestCase):
    def test_unsigned_wrap(self):
        self.assertEqual(ev(BinOp("add", Const(200, U8), Const(100, U8))).data, 44)
        self.assertEqual(ev(BinOp("sub", Const(0, U8), Const(1, U8))).data, 255)
        self.assertEqual(ev(BinOp("mul", Const(16, U8), Const(16, U8))).data, 0)

    def test_signed_wrap(self):
        self.assertEqual(ev(BinOp("add", Const(127, S8), Const(1, S8))).data, -128)
        self.assertEqual(ev(BinOp("sub", Const(-128, S8), Const(1, S8))).data, 127)

    def test_type_mismatch_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(BinOp("add", Const(1, U8), Const(1, U16)))
        with self.assertRaises(TypeMismatchError):
            ev(BinOp("add", Const(1, U8), Const(1, S8)))

    def test_division_truncates_toward_zero(self):
        self.assertEqual(ev(BinOp("div", Const(-7, S8), Const(2, S8))).data, -3)
        self.assertEqual(ev(BinOp("mod", Const(-7, S8), Const(2, S8))).data, -1)
        self.assertEqual(ev(BinOp("div", Const(7, S8), Const(-2, S8))).data, -3)

    def test_division_by_zero(self):
        with self.assertRaises(DivisionByZeroError):
            ev(BinOp("div", Const(1, U8), Const(0, U8)))
        with self.assertRaises(DivisionByZeroError):
            ev(BinOp("mod", Const(1, U8), Const(0, U8)))

    def test_signed_division_overflow_wraps(self):
        # INT_MIN / -1 wraps per the modular rule, like every other result.
        self.assertEqual(ev(BinOp("div", Const(-128, S8), Const(-1, S8))).data, -128)


class TestShifts(unittest.TestCase):
    def test_signed_right_shift_is_arithmetic(self):
        # s8: -16 >> 2 == -4 (sign bits shifted in)
        self.assertEqual(ev(BinOp("shr", Const(-16, S8), Const(2, S8))).data, -4)
        self.assertEqual(ev(BinOp("shr", Const(-1, S8), Const(7, S8))).data, -1)
        # u8: 0xF0 >> 2 == 0x3C (zeros shifted in)
        self.assertEqual(ev(BinOp("shr", Const(0xF0, U8), Const(2, U8))).data, 0x3C)

    def test_left_shift_truncates(self):
        self.assertEqual(ev(BinOp("shl", Const(0xFF, U8), Const(4, U8))).data, 0xF0)
        self.assertEqual(ev(BinOp("shl", Const(1, S8), Const(7, S8))).data, -128)

    def test_shift_amount_must_be_below_width(self):
        with self.assertRaises(ShiftError):
            ev(BinOp("shr", Const(1, U8), Const(8, U8)))
        with self.assertRaises(ShiftError):
            ev(BinOp("shl", Const(1, S64), Const(64, S64)))
        with self.assertRaises(ShiftError):
            ev(BinOp("shl", Const(1, S8), Const(-1, S8)))


class TestComparisons(unittest.TestCase):
    def test_signedness_decides(self):
        # 0xFF as s8 is -1 (< 0); as u8 it is 255 (> 0).
        self.assertEqual(ev(Cmp("lt", Const(-1, S8), Const(0, S8))).data, 1)
        self.assertEqual(ev(Cmp("lt", Const(0xFF, U8), Const(0, U8))).data, 0)
        self.assertEqual(ev(Cmp("gt", Const(0xFF, U8), Const(0, U8))).data, 1)

    def test_result_is_u8(self):
        self.assertIs(ev(Cmp("eq", Const(1, U8), Const(1, U8))).type, U8)

    def test_mixed_types_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(Cmp("eq", Const(1, U8), Const(1, U16)))


class TestCast(unittest.TestCase):
    def test_cast_reinterprets(self):
        self.assertEqual(ev(Cast(Const(0xFF, U8), S8)).data, -1)
        self.assertEqual(ev(Cast(Const(-1, S16), U16)).data, 0xFFFF)
        self.assertEqual(ev(Cast(Const(0x1234, U16), U8)).data, 0x34)
        self.assertEqual(ev(Cast(Const(0x80, U8), S16)).data, 0x80)  # zero-extend then reinterpret

    def test_cast_pointer_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(Cast(PtrConst("stack", 0), U64), memory=STACK)


class TestShortCircuit(unittest.TestCase):
    def test_and_short_circuits_out_of_bounds_load(self):
        # The right side would dereference one-past-end; it must never run.
        bad = Load(PtrConst("stack", 64), U8)  # one-past-end of a 64-byte region
        result = evaluate(And(Const(0, U8), bad), {}, STACK)
        self.assertEqual(result.value.data, 0)
        self.assertEqual(result.reads, ())  # no memory was touched

    def test_or_short_circuits_out_of_bounds_load(self):
        bad = Load(PtrConst("stack", 64), U8)
        result = evaluate(Or(Const(1, U8), bad), {}, STACK)
        self.assertEqual(result.value.data, 1)
        self.assertEqual(result.reads, ())

    def test_non_short_circuited_side_is_evaluated(self):
        bad = Load(PtrConst("stack", 64), U8)
        with self.assertRaises(PointerError):
            evaluate(And(Const(1, U8), bad), {}, STACK)
        with self.assertRaises(PointerError):
            evaluate(Or(Const(0, U8), bad), {}, STACK)

    def test_logic_values(self):
        self.assertEqual(ev(And(Const(3, U8), Const(5, U8))).data, 1)
        self.assertEqual(ev(Or(Const(0, U8), Const(0, U8))).data, 0)
        self.assertEqual(ev(Not(Const(0, U8))).data, 1)
        self.assertEqual(ev(Not(Const(9, U8))).data, 0)


class TestPointers(unittest.TestCase):
    def test_ptr_add_within_region_including_one_past_end(self):
        value = ev(PtrAdd(PtrConst("stack", 60), Const(4, S8)), memory=STACK)
        self.assertEqual(value.data, Pointer("stack", 64))  # one-past-end is a valid pointer

    def test_ptr_add_beyond_one_past_end_rejected(self):
        with self.assertRaises(PointerError):
            ev(PtrAdd(PtrConst("stack", 60), Const(5, U8)), memory=STACK)
        with self.assertRaises(PointerError):
            ev(PtrAdd(PtrConst("stack", 0), Const(-1, S8)), memory=STACK)

    def test_deref_one_past_end_rejected(self):
        with self.assertRaises(PointerError):
            ev(Load(PtrConst("stack", 64), U8), memory=STACK)
        # a 4-byte load starting at the last valid byte also runs past the end
        with self.assertRaises(PointerError):
            ev(Load(PtrConst("stack", 60), U64), memory=STACK)

    def test_misaligned_deref_rejected(self):
        with self.assertRaises(PointerError):
            ev(Load(PtrConst("stack", 2), U32), memory=STACK)
        with self.assertRaises(PointerError):
            ev(Field(PtrConst("stack", 0), 3, U16), memory=STACK)

    def test_cross_region_subtraction_rejected(self):
        with self.assertRaises(PointerError):
            ev(PtrSub(PtrConst("stack", 8), PtrConst("heap", 8)), memory=STACK)

    def test_same_region_subtraction(self):
        value = ev(PtrSub(PtrConst("stack", 40), PtrConst("stack", 8)), memory=STACK)
        self.assertEqual((value.type, value.data), (S64, 32))

    def test_cross_region_ordering_rejected_but_equality_allowed(self):
        with self.assertRaises(PointerError):
            ev(Cmp("lt", PtrConst("stack", 0), PtrConst("heap", 0)), memory=STACK)
        self.assertEqual(
            ev(Cmp("ne", PtrConst("stack", 0), PtrConst("heap", 0)), memory=STACK).data, 1
        )

    def test_load_only_reads_snapshot_regions(self):
        with self.assertRaises(PointerError):
            ev(Load(PtrConst("nonexistent", 0), U8), memory=STACK)

    def test_load_is_little_endian_and_records_reads(self):
        # stack[i] == i, so u32 at offset 8 is 0x0B0A0908 little-endian.
        result = evaluate(Load(PtrConst("stack", 8), U32), {}, STACK)
        self.assertEqual(result.value.data, 0x0B0A0908)
        self.assertEqual(len(result.reads), 1)
        read = result.reads[0]
        self.assertEqual((read.region_id, read.offset, read.length), ("stack", 8, 4))

    def test_field_reads_at_offset(self):
        result = evaluate(Field(PtrConst("stack", 8), 8, U16), {}, STACK)
        self.assertEqual(result.value.data, 0x1110)
        self.assertEqual([(r.region_id, r.offset, r.length) for r in result.reads], [("stack", 16, 2)])


class TestRegisters(unittest.TestCase):
    def test_register_read(self):
        self.assertEqual(ev(Reg("rax", U64), {"rax": 42}).data, 42)

    def test_unknown_register(self):
        with self.assertRaises(InputError):
            ev(Reg("nope", U64))

    def test_register_out_of_declared_range(self):
        with self.assertRaises(RangeError):
            ev(Reg("al", U8), {"al": 256})


class TestInputValidation(unittest.TestCase):
    def test_bool_rejected_as_integer(self):
        with self.assertRaises(InputError):
            ev(Const(True, U8))
        with self.assertRaises(InputError):
            ev(Reg("r", U8), {"r": False})
        with self.assertRaises(InputError):
            ev(Field(PtrConst("stack", 0), True, U8), memory=STACK)

    def test_float_and_nan_infinity_rejected(self):
        for bad in (1.5, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InputError, msg=f"value {bad!r}"):
                ev(Const(bad, U8))
            with self.assertRaises(InputError, msg=f"register {bad!r}"):
                ev(Reg("r", U8), {"r": bad})

    def test_literal_out_of_range(self):
        with self.assertRaises(RangeError):
            ev(Const(256, U8))
        with self.assertRaises(RangeError):
            ev(Const(-129, S8))

    def test_bad_top_level_arguments(self):
        with self.assertRaises(InputError):
            evaluate("not an ast", {}, EMPTY)
        with self.assertRaises(InputError):
            evaluate(Const(1, U8), ["not", "a", "mapping"], EMPTY)
        with self.assertRaises(InputError):
            evaluate(Const(1, U8), {}, {"stack": b"x"})

    def test_error_path_locates_node(self):
        try:
            ev(BinOp("add", Const(1, U8), BinOp("shl", Const(1, U8), Const(8, U8))))
        except EvalError as err:
            # the path points at the offending shift-amount operand
            self.assertEqual(err.path, ("right", "right"))
            self.assertIn("right", str(err))
        else:  # pragma: no cover
            self.fail("expected ShiftError")


class TestNodeBudget(unittest.TestCase):
    def _chain(self, n):
        node = Const(1, U8)
        for _ in range(n - 1):
            node = Not(node)
        return node

    def test_up_to_500_nodes_accepted(self):
        # 1 Const + 499 Not nodes = 500 nodes; 499 negations of 1 give 0.
        self.assertEqual(ev(self._chain(MAX_NODES)).data, 0)

    def test_501_nodes_rejected(self):
        with self.assertRaises(NodeBudgetError):
            ev(self._chain(MAX_NODES + 1))


class TestAgainstIndependentReference(unittest.TestCase):
    """Randomized cross-check of arithmetic/comparison/cast against the
    independent integer-mask reference in tests/reference.py."""

    def test_binops(self):
        rng = random.Random(0x0162)
        ops = ["add", "sub", "mul", "div", "mod", "shl", "shr", "bitand", "bitor", "bitxor"]
        for itype, bits, signed in ALL_TYPES:
            for _ in range(150):
                op = rng.choice(ops)
                a = rng.randint(itype.min, itype.max)
                b = rng.randint(itype.min, itype.max)
                if op in ("div", "mod") and b == 0:
                    b = 1
                if op in ("shl", "shr"):
                    b = rng.randrange(bits)
                got = ev(BinOp(op, Const(a, itype), Const(b, itype)))
                want = ref_binop(op, a, b, bits, signed)
                self.assertEqual(got.data, want, f"{op}({a}, {b}) on {itype}")
                self.assertIs(got.type, itype)

    def test_cmps(self):
        rng = random.Random(0x0163)
        for itype, _bits, _signed in ALL_TYPES:
            for _ in range(50):
                op = rng.choice(["eq", "ne", "lt", "le", "gt", "ge"])
                a = rng.randint(itype.min, itype.max)
                b = rng.randint(itype.min, itype.max)
                got = ev(Cmp(op, Const(a, itype), Const(b, itype)))
                self.assertEqual(got.data, ref_cmp(op, a, b), f"{op}({a}, {b}) on {itype}")

    def test_casts(self):
        rng = random.Random(0x0164)
        for src, src_bits, src_signed in ALL_TYPES:
            for dst, dst_bits, dst_signed in ALL_TYPES:
                value = rng.randint(src.min, src.max)
                got = ev(Cast(Const(value, src), dst))
                want = ref_cast(value, src_bits, src_signed, dst_bits, dst_signed)
                self.assertEqual(got.data, want, f"cast {value} from {src} to {dst}")
                self.assertIs(got.type, dst)


if __name__ == "__main__":
    unittest.main()
