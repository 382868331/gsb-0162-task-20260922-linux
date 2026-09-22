"""测试：指针/区域地址空间语义、对齐、load 读取证据、短路求值。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from dbgx import (
    AddressError,
    IntType,
    PtrType,
    ReadRange,
    Region,
    TypeMismatchError,
    Value,
    evaluate,
)


def i(v, w=32, s=False):
    return ("int", v, w, s)


def make_mem():
    # 16 字节小端数据，base=0x1000
    data = bytes(
        [
            0x78, 0x56, 0x34, 0x12,  # +0x1000 u32 = 0x12345678
            0xFF, 0xFF, 0x00, 0x00,  # +0x1004 u16 = 0xFFFF, u8 ...
            0x01, 0x02, 0x03, 0x04,
            0x10, 0x20, 0x30, 0x40,
        ]
    )
    return {"ram": Region("ram", data, base=0x1000)}


class TestLoad(unittest.TestCase):
    def test_aligned_load_and_read_evidence(self):
        res = evaluate(("load", ("ptr", "ram", 0x1000, 32)), {}, make_mem())
        self.assertEqual(res.value.vtype, IntType(32, False))
        self.assertEqual(res.value.as_int(), 0x12345678)
        self.assertEqual(
            res.reads, (ReadRange("ram", 0x1000, 0x1004),)
        )

    def test_load_u8_always_aligned(self):
        res = evaluate(("load", ("ptr", "ram", 0x1005, 8)), {}, make_mem())
        self.assertEqual(res.value.as_int(), 0xFF)
        self.assertEqual(res.reads[0].nbytes, 1)

    def test_load_u16(self):
        res = evaluate(("load", ("ptr", "ram", 0x1004, 16)), {}, make_mem())
        self.assertEqual(res.value.as_int(), 0xFFFF)

    def test_misaligned_load_rejected(self):
        with self.assertRaisesRegex(AddressError, "未对齐"):
            evaluate(("load", ("ptr", "ram", 0x1001, 16)), {}, make_mem())
        with self.assertRaisesRegex(AddressError, "未对齐"):
            evaluate(("load", ("ptr", "ram", 0x1002, 32)), {}, make_mem())
        # 对齐错误优先于边界检查，且不得留下读取记录（异常即丢弃整个结果）
        try:
            evaluate(("load", ("ptr", "ram", 0x100F, 32)), {}, make_mem())
            self.fail("应当抛出地址错误")
        except AddressError:
            pass

    def test_load_one_past_end_pointer_rejected(self):
        # 区域 16 字节，尾后地址 = 0x1010；指针算术允许它，load 拒绝它
        res = evaluate(
            ("padd", ("ptr", "ram", 0x1000, 8), i(16, 32, True)),
            {},
            make_mem(),
        )
        self.assertEqual(res.value.vtype, PtrType("ram", 8))
        self.assertEqual(res.value.as_ptr(), ("ram", 0x1010))
        with self.assertRaisesRegex(AddressError, r"尾后|越界"):
            evaluate(
                ("load", ("padd", ("ptr", "ram", 0x1000, 8), i(16, 32, True))),
                {},
                make_mem(),
            )

    def test_pointer_beyond_one_past_end_rejected(self):
        with self.assertRaises(AddressError):
            evaluate(
                ("padd", ("ptr", "ram", 0x1000, 8), i(17, 32, True)),
                {},
                make_mem(),
            )

    def test_last_valid_load(self):
        # 最后一个合法 u32：0x100C..0x1010
        res = evaluate(("load", ("ptr", "ram", 0x100C, 32)), {}, make_mem())
        self.assertEqual(res.value.as_int(), 0x40302010)
        # 最后一个合法 u8：0x100F
        res = evaluate(("load", ("ptr", "ram", 0x100F, 8)), {}, make_mem())
        self.assertEqual(res.value.as_int(), 0x40)

    def test_psub_backwards(self):
        # 0x100C - 12 => 0x1000，再 load 得 0x12345678
        node = (
            "load",
            ("psub", ("ptr", "ram", 0x100C, 32), i(12, 32, True)),
        )
        res = evaluate(node, {}, make_mem())
        self.assertEqual(res.value.as_int(), 0x12345678)
        self.assertEqual(res.reads, (ReadRange("ram", 0x1000, 0x1004),))

    def test_psub_negative_address_rejected(self):
        with self.assertRaises(AddressError):
            evaluate(
                ("psub", ("ptr", "ram", 0x1002, 8), i(3, 32, True)),
                {},
                make_mem(),
            )

    def test_load_unknown_region_rejected(self):
        with self.assertRaises(AddressError):
            evaluate(("load", ("ptr", "rom", 0, 8)), {}, make_mem())

    def test_padd_unknown_region_rejected(self):
        with self.assertRaises(AddressError):
            evaluate(
                ("padd", ("ptr", "rom", 0, 8), i(1, 32, True)), {}, make_mem()
            )

    def test_load_only_uses_passed_snapshot(self):
        mem = make_mem()
        # 修改外部对象不影响已构造快照
        buf = bytearray(16)
        snap = Region("ram", buf, base=0)
        buf[0] = 0x99
        res = evaluate(("load", ("ptr", "ram", 0, 8)), {}, {"ram": snap})
        self.assertEqual(res.value.as_int(), 0)


class TestPointerDiff(unittest.TestCase):
    def test_same_region_diff(self):
        node = (
            "pdiff",
            ("ptr", "ram", 0x1008, 8),
            ("ptr", "ram", 0x1000, 8),
        )
        res = evaluate(node, {}, make_mem())
        self.assertEqual(res.value.vtype, IntType(64, True))
        self.assertEqual(res.value.as_signed(), 8)

    def test_diff_reversed_negative(self):
        node = (
            "pdiff",
            ("ptr", "ram", 0x1000, 8),
            ("ptr", "ram", 0x1008, 8),
        )
        res = evaluate(node, {}, make_mem())
        self.assertEqual(res.value.as_signed(), -8)

    def test_cross_region_diff_rejected(self):
        mem = dict(make_mem())
        mem["rom"] = Region("rom", b"\x00" * 8, base=0x2000)
        node = (
            "pdiff",
            ("ptr", "ram", 0x1000, 8),
            ("ptr", "rom", 0x2000, 8),
        )
        with self.assertRaisesRegex(AddressError, "跨区域"):
            evaluate(node, {}, mem)

    def test_pdiff_different_pointee_width_still_bytes(self):
        node = (
            "pdiff",
            ("ptr", "ram", 0x1004, 32),
            ("ptr", "ram", 0x1000, 8),
        )
        res = evaluate(node, {}, make_mem())
        # 字节差，不按元素宽度缩放
        self.assertEqual(res.value.as_signed(), 4)

    def test_pdiff_one_past_end_allowed(self):
        node = (
            "pdiff",
            ("padd", ("ptr", "ram", 0x1000, 8), i(16, 32, True)),
            ("ptr", "ram", 0x1000, 8),
        )
        res = evaluate(node, {}, make_mem())
        self.assertEqual(res.value.as_signed(), 16)


class TestPointerTypes(unittest.TestCase):
    def test_padd_requires_pointer_left(self):
        with self.assertRaises(TypeMismatchError):
            evaluate(("padd", i(1, 32), i(1, 32, True)), {}, make_mem())

    def test_padd_requires_integer_right(self):
        with self.assertRaises(TypeMismatchError):
            evaluate(
                ("padd", ("ptr", "ram", 0, 8), ("ptr", "ram", 0, 8)),
                {},
                make_mem(),
            )

    def test_pointer_type_preserved(self):
        res = evaluate(
            ("padd", ("ptr", "ram", 0x1000, 16), i(2, 32, True)), {}, make_mem()
        )
        self.assertEqual(res.value.vtype, PtrType("ram", 16))


class TestShortCircuit(unittest.TestCase):
    def test_band_skips_oob_load_when_false(self):
        # 左侧为假（u8 0），右侧是尾后越界 load —— 不得求值，读取证据为空
        node = (
            "band",
            ("eq", i(1, 8), i(2, 8)),
            ("load", ("ptr", "ram", 0x1010, 8)),  # 尾后，若求值必报错
        )
        res = evaluate(node, {}, make_mem())
        self.assertEqual(res.value.as_int(), 0)
        self.assertEqual(res.reads, ())

    def test_bor_skips_oob_load_when_true(self):
        node = (
            "bor",
            ("eq", i(1, 8), i(1, 8)),
            ("load", ("ptr", "ram", 0x9999, 8)),  # 越界
        )
        res = evaluate(node, {}, make_mem())
        self.assertEqual(res.value.as_int(), 1)
        self.assertEqual(res.reads, ())

    def test_band_evaluates_right_when_true_and_fails_oob(self):
        node = (
            "band",
            ("eq", i(1, 8), i(1, 8)),
            ("load", ("ptr", "ram", 0x1010, 8)),  # 尾后
        )
        with self.assertRaises(AddressError):
            evaluate(node, {}, make_mem())

    def test_band_evaluates_right_when_true_and_reads(self):
        node = (
            "band",
            ("eq", i(1, 8), i(1, 8)),
            ("load", ("ptr", "ram", 0x1004, 8)),
        )
        res = evaluate(node, {}, make_mem())
        self.assertEqual(res.value.as_int(), 1)
        self.assertEqual(res.reads, (ReadRange("ram", 0x1004, 0x1005),))

    def test_short_circuit_nested(self):
        # (false AND (OOB)) OR (true AND load@0x1000)
        node = (
            "bor",
            ("band", ("eq", i(0, 8), i(1, 8)), ("load", ("ptr", "ram", 0x1010, 32))),
            ("band", ("eq", i(1, 8), i(1, 8)), ("load", ("ptr", "ram", 0x1000, 8))),
        )
        res = evaluate(node, {}, make_mem())
        self.assertEqual(res.value.as_int(), 1)
        self.assertEqual(res.reads, (ReadRange("ram", 0x1000, 0x1001),))

    def test_short_circuit_result_is_u8(self):
        res = evaluate(("band", i(0, 16), i(7, 16)), {}, {}).value
        self.assertEqual(res.vtype, IntType(8, False))



if __name__ == "__main__":  # pragma: no cover
    unittest.main()
