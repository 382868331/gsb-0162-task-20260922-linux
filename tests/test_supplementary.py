"""补充测试：往返/自洽性质（不作为期望值的唯一来源）。

按验收要求，期望值主要由独立参考 tests/reference.py 差分给出；本文件只做
结构性自洽检查：
* 500 节点左/右旋链均可求值（递归深度保险）；
* cast 往返：窄值加宽后回到原宽度位模式不变；
* field 重组：用 field 拆出的字节按小端重组等于原机器表示；
* load 与原始字节往返：load 出的 u32 的 4 个字节等于快照对应字节；
* 失败求值不产生部分变更：registers/memory 对象内容与读取前一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from dbgx import AddressError, IntType, Region, Value, evaluate


def i(v, w=32, s=False):
    return ("int", v, w, s)


class TestSupplementary(unittest.TestCase):
    def test_500_node_left_skew_chain(self):
        node = i(0, 8)
        for _ in range(249):
            node = ("add", node, i(0, 8))
        node = ("lnot", node)  # 500
        self.assertEqual(evaluate(node, {}, {}).node_count, 500)

    def test_500_node_right_skew_chain(self):
        # 右旋：add(0, add(0, ... add(0, 0)))，共 250 个 add = 501 超限；
        # 249 个 add + 一个 lnot 根 = 500
        node = i(0, 8)
        for _ in range(249):
            node = ("add", i(0, 8), node)
        node = ("lnot", node)
        res = evaluate(node, {}, {})
        self.assertEqual(res.node_count, 500)
        self.assertEqual(res.value.as_int(), 1)

    def test_cast_roundtrip_preserves_pattern(self):
        for w1 in (8, 16):
            for s1 in (False, True):
                v = Value.int_(w1, s1, -7 if s1 else 200)
                regs = {"x": v}
                node = (
                    "cast",
                    ("cast", ("reg", "x"), 64, False),
                    w1,
                    s1,
                )
                r = evaluate(node, regs, {}).value
                self.assertEqual(r, v)

    def test_field_bytes_reassemble(self):
        # u32 拆成 4 个 u8 字段后按小端重组应等于原值
        raw = 0xDEADBEEF
        regs = {"x": Value(IntType(32, False), raw)}
        parts = [
            ("field", ("reg", "x"), 8 * k, 8, False) for k in range(4)
        ]
        rebuilt = parts[0]
        for k, p in enumerate(parts[1:], start=1):
            rebuilt = (
                "or",
                ("shl", ("cast", p, 32, False), i(8 * k, 32)),
                ("cast", rebuilt, 32, False),
            )
        r = evaluate(rebuilt, regs, {}).value
        self.assertEqual(r.as_int(), raw)

    def test_load_bytes_roundtrip(self):
        data = bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
        mem = {"ram": Region("ram", data, base=0x100)}
        word = evaluate(("load", ("ptr", "ram", 0x100, 32)), {}, mem).value
        for k in range(4):
            b = evaluate(
                ("load", ("ptr", "ram", 0x100 + k, 8)), {}, mem
            ).value
            self.assertEqual((word.as_int() >> (8 * k)) & 0xFF, data[k])
            self.assertEqual(b.as_int(), data[k])

    def test_failed_evaluation_leaves_inputs_untouched(self):
        data = bytearray(range(16))
        region = Region("ram", data, base=0)
        mem = {"ram": region}
        regs = {"x": Value(IntType(32, True), 5)}
        before = bytes(region.data)
        with self.assertRaises(AddressError):
            evaluate(
                ("load", ("ptr", "ram", 0x10, 8)), regs, mem
            )  # 尾后
        # 外部可变缓冲与快照均未被修改
        self.assertEqual(bytes(region.data), before)
        self.assertEqual(bytes(data), bytes(range(16)))
        self.assertEqual(regs["x"].raw, 5)
        # 快照隔离：构造后外部改数据不影响求值
        data[0] = 0xFF
        r = evaluate(("load", ("ptr", "ram", 0, 8)), regs, mem).value
        self.assertEqual(r.as_int(), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
