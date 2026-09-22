"""测试：cast 位宽/符号转换与 field 位域提取。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from dbgx import IntType, TypeMismatchError, evaluate


def i(v, w=32, s=False):
    return ("int", v, w, s)


def ev(node, regs=None, mem=None):
    return evaluate(node, regs or {}, mem or {})


class TestCast(unittest.TestCase):
    def test_narrow_truncates(self):
        r = ev(("cast", i(0x1234, 32), 8, False)).value
        self.assertEqual(r.vtype, IntType(8, False))
        self.assertEqual(r.as_int(), 0x34)

    def test_widen_zero_extends_even_when_target_signed(self):
        # u8 0xFF 扩到 i32：位模式零扩展 => 255（非 -1）
        r = ev(("cast", i(0xFF, 8, False), 32, True)).value
        self.assertEqual(r.as_int(), 255)
        self.assertEqual(r.as_signed(), 255)

    def test_reinterpret_sign_same_pattern(self):
        u = ev(i(0xFFFFFFFF, 32, False)).value
        r = ev(("cast", u := ("int", 0xFFFFFFFF, 32, False), 32, True)).value
        self.assertEqual(r.as_int(), 0xFFFFFFFF)
        self.assertEqual(r.as_signed(), -1)

    def test_signed_narrow(self):
        # i32 -1 -> i8: 取低 8 位，i8 下仍为 -1
        r = ev(("cast", i(-1, 32, True), 8, True)).value
        self.assertEqual(r.as_signed(), -1)
        # i32 255 -> i8: 0xFF => -1
        r = ev(("cast", i(255, 32, True), 8, True)).value
        self.assertEqual(r.as_signed(), -1)


class TestField(unittest.TestCase):
    def test_extract_middle(self):
        # u32 = 0x000000B0：lsb=4 的 8 位字段为 (0xB0 >> 4) = 0x0B
        val = 0b0000_0000_0000_0000_0000_0000_1011_0000
        r = ev(("field", i(val, 32), 4, 8, False)).value
        self.assertEqual(r.vtype, IntType(8, False))
        self.assertEqual(r.as_int(), 0x0B)

    def test_extract_signed_field_sign_extends(self):
        # 源 u32，位域 [8,8) = 0x80 => 有符号 i8 解释为 -128
        val = 0x80 << 8
        r = ev(("field", i(val, 32), 8, 8, True)).value
        self.assertEqual(r.vtype, IntType(8, True))
        self.assertEqual(r.as_signed(), -128)

    def test_field_32(self):
        r = ev(("field", i(0xDEADBEEF, 32), 0, 32, False)).value
        self.assertEqual(r.as_int(), 0xDEADBEEF)

    def test_field_16_signed(self):
        # lsb=16 字段 0xFFFF，i16 为 -1
        r = ev(("field", i(0xFFFF0000, 32), 16, 16, True)).value
        self.assertEqual(r.as_signed(), -1)

    def test_field_64(self):
        r = ev(("field", i(1 << 63, 64), 0, 64, True)).value
        self.assertEqual(r.as_signed(), -(1 << 63))

    def test_field_out_of_range_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(("field", i(0xFF, 8), 4, 8, False))  # lsb+width 越界
        with self.assertRaises(TypeMismatchError):
            ev(("field", i(0xFF, 16), 8, 16, False))

    def test_cast_then_field_chain(self):
        # 从 32 位寄存器取最高字节，按有符号 i8 解释
        node = ("field", ("cast", ("reg", "x"), 32, False), 24, 8, True)
        from dbgx import Value

        r = evaluate(node, {"x": Value(IntType(32, False), 0x80000000)}, {}).value
        self.assertEqual(r.as_signed(), -128)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
