"""测试：固定宽度整数的位宽、模截断、符号性比较与移位语义。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from dbgx import (
    DivisionByZeroError,
    IntType,
    ShiftError,
    TypeMismatchError,
    Value,
    evaluate,
)


def i(v: int, w: int = 32, s: bool = False):
    return ("int", v, w, s)


def ev(node, regs=None, mem=None):
    return evaluate(node, regs or {}, mem or {})


class TestWrapping(unittest.TestCase):
    def test_add_wraps_each_width(self):
        for w in (8, 16, 32, 64):
            top = 1 << w
            r = ev(("add", i(top - 1, w), i(1, w))).value
            self.assertEqual(r.vtype, IntType(w, False))
            self.assertEqual(r.as_int(), 0)
            r2 = ev(("sub", i(0, w, True), i(1, w, True))).value
            self.assertEqual(r2.vtype, IntType(w, True))
            self.assertEqual(r2.as_signed(), -1)

    def test_mul_wraps_64(self):
        r = ev(("mul", i(0xFFFFFFFF, 64), i(0xFFFFFFFF, 64))).value
        # (2^32-1)^2 mod 2^64
        self.assertEqual(r.as_int(), ((1 << 32) - 1) ** 2 % (1 << 64))

    def test_negative_literal_wraps(self):
        r = ev(("add", i(-1, 8, True), i(1, 8, True))).value
        self.assertEqual(r.as_signed(), 0)
        r = ev(i(-128 - 1, 8, True)).value  # -129 -> 127
        self.assertEqual(r.as_signed(), 127)

    def test_bitops(self):
        r = ev(("xor", i(0xFF, 8), i(0x0F, 8))).value
        self.assertEqual(r.as_int(), 0xF0)
        r = ev(("and", i(0xFF, 8), i(0x0F, 8))).value
        self.assertEqual(r.as_int(), 0x0F)
        r = ev(("or", i(0xF0, 8), i(0x0F, 8))).value
        self.assertEqual(r.as_int(), 0xFF)

    def test_result_type_preserved(self):
        r = ev(("add", i(1, 16, True), i(2, 16, True))).value
        self.assertEqual(r.vtype, IntType(16, True))


class TestMixedTypes(unittest.TestCase):
    def test_width_mismatch_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(("add", i(1, 16), i(1, 32)))

    def test_sign_mismatch_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(("add", i(1, 32, False), i(1, 32, True)))

    def test_compare_sign_mismatch_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(("lt", i(0xFFFFFFFF, 32, False), i(-1, 32, True)))

    def test_int_ptr_arith_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(("add", i(1, 32), ("ptr", "r", 0, 8)))


class TestComparisons(unittest.TestCase):
    def test_signed_vs_unsigned_ordering(self):
        # 同样的位模式 0xFFFF：u16 下为 65535，i16 下为 -1
        unsigned = ev(("lt", i(0xFFFF, 16, False), i(0, 16, False))).value
        signed = ev(("lt", i(0xFFFF, 16, True), i(0, 16, True))).value
        self.assertEqual(unsigned.as_int(), 0)  # 65535 < 0 为假
        self.assertEqual(signed.as_int(), 1)  # -1 < 0 为真

    def test_eq_ne_identity(self):
        self.assertEqual(
            ev(("eq", i(0xFFFF, 16, True), i(-1, 16, True))).value.as_int(), 1
        )
        self.assertEqual(
            ev(("ne", i(1, 8), i(2, 8))).value.as_int(), 1
        )
        self.assertEqual(
            ev(("ge", i(5, 32, True), i(5, 32, True))).value.as_int(), 1
        )

    def test_comparison_yields_u8(self):
        r = ev(("gt", i(2, 8), i(1, 8))).value
        self.assertEqual(r.vtype, IntType(8, False))
        self.assertEqual(r.as_int(), 1)


class TestShift(unittest.TestCase):
    def test_shl_lsr_basic(self):
        self.assertEqual(ev(("shl", i(1, 8), i(3, 8))).value.as_int(), 8)
        self.assertEqual(ev(("lsr", i(0x80, 8), i(7, 8))).value.as_int(), 1)

    def test_arithmetic_right_shift_signed(self):
        # -16 (0xF0 i8) asr 2 => -4 (0xFC)，高位补符号位
        r = ev(("asr", i(-16, 8, True), i(2, 8, True))).value
        self.assertEqual(r.as_signed(), -4)
        self.assertEqual(r.as_int(), 0xFC)
        # -1 asr 7 仍是 -1
        r = ev(("asr", i(-1, 32, True), i(7, 32, True))).value
        self.assertEqual(r.as_signed(), -1)
        self.assertEqual(r.as_int(), 0xFFFFFFFF)
        # 正数 asr 与 lsr 一致
        r = ev(("asr", i(0x40, 8, True), i(2, 8, True))).value
        self.assertEqual(r.as_signed(), 0x10)
        # INT32_MIN asr 31 => -1
        r = ev(("asr", i(1 << 31, 32, True), i(31, 32, True))).value
        self.assertEqual(r.as_signed(), -1)

    def test_asr_on_unsigned_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(("asr", i(0xF0, 8, False), i(1, 8, False)))

    def test_shift_amount_must_be_less_than_width(self):
        for amt in (8, 9, 255):
            with self.assertRaises(ShiftError):
                ev(("shl", i(1, 8), i(amt, 8)))
        with self.assertRaises(ShiftError):
            ev(("lsr", i(1, 16), i(16, 16)))
        with self.assertRaises(ShiftError):
            ev(("asr", i(1, 32, True), i(32, 32, True)))
        with self.assertRaises(ShiftError):
            ev(("shl", i(1, 64), i(64, 64)))

    def test_shift_amount_zero_valid(self):
        self.assertEqual(ev(("shl", i(0xAB, 8), i(0, 8))).value.as_int(), 0xAB)

    def test_shift_signed_minus_one_amount_rejected(self):
        # 有符号 -1 的机器表示为 255，必然不小于位宽
        with self.assertRaises(ShiftError):
            ev(("shl", i(1, 8, True), i(-1, 8, True)))

    def test_shift_requires_same_type(self):
        with self.assertRaises(TypeMismatchError):
            ev(("shl", i(1, 8), i(1, 16)))


class TestDivision(unittest.TestCase):
    def test_unsigned_div_rem(self):
        self.assertEqual(ev(("udiv", i(10, 8), i(3, 8))).value.as_int(), 3)
        self.assertEqual(ev(("urem", i(10, 8), i(3, 8))).value.as_int(), 1)

    def test_signed_div_truncates_toward_zero(self):
        # -7 / 2 = -3（向零截断），余数 -1
        self.assertEqual(
            ev(("sdiv", i(-7, 8, True), i(2, 8, True))).value.as_signed(), -3
        )
        self.assertEqual(
            ev(("srem", i(-7, 8, True), i(2, 8, True))).value.as_signed(), -1
        )
        self.assertEqual(
            ev(("sdiv", i(7, 8, True), i(-2, 8, True))).value.as_signed(), -3
        )

    def test_int_min_div_minus_one_wraps(self):
        r = ev(("sdiv", i(-(1 << 31), 32, True), i(-1, 32, True))).value
        self.assertEqual(r.as_signed(), -(1 << 31))
        self.assertEqual(
            ev(("srem", i(-(1 << 31), 32, True), i(-1, 32, True))).value.as_signed(),
            0,
        )

    def test_divide_by_zero(self):
        with self.assertRaises(DivisionByZeroError):
            ev(("udiv", i(1, 8), i(0, 8)))
        with self.assertRaises(DivisionByZeroError):
            ev(("srem", i(-1, 8, True), i(0, 8, True)))


class TestRegisters(unittest.TestCase):
    def test_register_value_used(self):
        regs = {"x": Value(IntType(32, False), 0xDEADBEEF)}
        r = ev(
            ("add", ("reg", "x"), i(0x21524111, 32)),
            regs=regs,
        ).value
        self.assertEqual(r.as_int(), 0)

    def test_wrong_register_value_type_rejected(self):
        with self.assertRaises(Exception):
            evaluate(("reg", "x"), {"x": 123}, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
