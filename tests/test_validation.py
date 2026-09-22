"""测试：静态校验、输入拒绝（NaN/Infinity、bool）、节点上限、错误定位路径。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from dbgx import (
    ASTError,
    DbgxError,
    InputError,
    IntType,
    Region,
    UndefinedRegisterError,
    Value,
    evaluate,
)


class TestLiteralValidation(unittest.TestCase):
    def test_bool_literal_rejected(self):
        with self.assertRaisesRegex(ASTError, "bool"):
            evaluate(("int", True, 8, False), {}, {})
        with self.assertRaisesRegex(ASTError, "bool"):
            evaluate(("int", False, 32, True), {}, {})

    def test_float_nan_inf_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ASTError, "NaN/Infinity|float"):
                evaluate(("int", bad, 32, False), {}, {})

    def test_regular_float_rejected(self):
        with self.assertRaisesRegex(ASTError, "int"):
            evaluate(("int", 1.5, 32, False), {}, {})

    def test_string_literal_rejected(self):
        with self.assertRaises(ASTError):
            evaluate(("int", "1", 32, False), {}, {})

    def test_bad_width_signed_rejected(self):
        with self.assertRaises(ASTError):
            evaluate(("int", 1, 7, False), {}, {})
        with self.assertRaises(ASTError):
            evaluate(("int", 1, 32, 0), {}, {})  # signed 非 bool

    def test_unknown_op(self):
        with self.assertRaisesRegex(ASTError, "未知算子"):
            evaluate(("wat",), {}, {})

    def test_non_tuple_node(self):
        with self.assertRaises(ASTError):
            evaluate(42, {}, {})

    def test_negative_ptr_offset_literal_rejected(self):
        with self.assertRaises(ASTError):
            evaluate(("ptr", "ram", -1, 8), {}, {})

    def test_field_bad_params(self):
        with self.assertRaises(ASTError):
            evaluate(("field", ("int", 1, 32, False), -1, 8, False), {}, {})
        with self.assertRaises(ASTError):
            evaluate(("field", ("int", 1, 32, False), 0, 12, False), {}, {})


class TestInputValidation(unittest.TestCase):
    def test_registers_must_be_dict(self):
        with self.assertRaises(InputError):
            evaluate(("int", 1, 8, False), [], {})

    def test_register_value_must_be_value(self):
        with self.assertRaises(InputError):
            evaluate(("reg", "x"), {"x": 3}, {})

    def test_register_key_must_be_str(self):
        with self.assertRaises(InputError):
            evaluate(("reg", "x"), {1: Value(IntType(8, False), 0)}, {})

    def test_memory_must_be_region_dict(self):
        with self.assertRaises(InputError):
            evaluate(("load", ("ptr", "r", 0, 8)), {}, {"r": b"x"})
        with self.assertRaises(InputError):
            evaluate(("load", ("ptr", "r", 0, 8)), {}, [])

    def test_memory_key_mismatch(self):
        with self.assertRaises(InputError):
            evaluate(
                ("load", ("ptr", "a", 0, 8)),
                {},
                {"b": Region("a", b"\x00")},
            )

    def test_region_base_rejects_bool_negative(self):
        with self.assertRaises(InputError):
            Region("r", b"", base=True)
        with self.assertRaises(InputError):
            Region("r", b"", base=-1)

    def test_value_raw_rejects_bool_and_negative(self):
        with self.assertRaises(InputError):
            Value(IntType(8, False), True)
        with self.assertRaises(InputError):
            Value(IntType(8, False), -1)
        with self.assertRaises(InputError):
            Value(IntType(8, False), 256)

    def test_bool_register_raw_rejected(self):
        with self.assertRaises(InputError):
            Value(IntType(8, False), True)

    def test_undefined_register_is_runtime_error(self):
        with self.assertRaises(UndefinedRegisterError):
            evaluate(("reg", "missing"), {}, {})


class TestNodeBudget(unittest.TestCase):
    @staticmethod
    def _chain(adds: int):
        # 每次 add 新增 1 个算子节点和 1 个叶子，共 1 + 2*adds 个节点
        node = ("int", 0, 8, False)
        for _ in range(adds):
            node = ("add", node, ("int", 0, 8, False))
        return node

    def test_500_nodes_accepted(self):
        # 249 个 add = 499 节点，外层包一个一元 lnot 恰好 500
        node = ("lnot", self._chain(249))
        res = evaluate(node, {}, {})
        self.assertEqual(res.node_count, 500)
        self.assertEqual(res.value.as_int(), 1)

    def test_501_nodes_rejected_before_evaluation(self):
        # 250 个 add = 501 节点
        with self.assertRaisesRegex(ASTError, "500"):
            evaluate(self._chain(250), {}, {})

    def test_budget_is_structural_regardless_of_short_circuit(self):
        # 即使右子树会被短路跳过，超限结构仍在静态校验阶段被拒
        over = ("int", 0, 8, False)
        for _ in range(500):  # 501 个节点
            over = ("add", over, ("int", 0, 8, False))
        node = ("band", ("int", 0, 8, False), over)
        with self.assertRaises(ASTError):
            evaluate(node, {}, {})


class TestErrorLocation(unittest.TestCase):
    def test_undefined_register_path(self):
        # root add(reg x, ...) -> 错误路径 (1,)
        node = ("add", ("reg", "x"), ("int", 1, 8, False))
        try:
            evaluate(node, {}, {})
        except UndefinedRegisterError as e:
            self.assertEqual(e.path, (1,))
        else:  # pragma: no cover
            self.fail("应当报错")

    def test_nested_div_zero_path(self):
        # bor(false, add(1, udiv(2, 0))) -> 路径 (2, 2, 2)
        node = (
            "bor",
            ("int", 0, 8, False),
            ("add", ("int", 1, 8, False), ("udiv", ("int", 2, 8, False),
                                            ("int", 0, 8, False))),
        )
        from dbgx import DivisionByZeroError

        try:
            evaluate(node, {}, {})
        except DivisionByZeroError as e:
            self.assertEqual(e.path, (2, 2))
        else:  # pragma: no cover
            self.fail("应当报错")

    def test_root_error_path_empty(self):
        try:
            evaluate(42, {}, {})
        except ASTError as e:
            self.assertEqual(e.path, ())

    def test_all_errors_are_dbgx_errors(self):
        bad_cases = [
            (("add", ("int", 1, 8, False), ("int", 1, 16, False)), {}, {}),
            (("shl", ("int", 1, 8, False), ("int", 8, 8, False)), {}, {}),
            (("load", ("ptr", "nope", 0, 8)), {}, {}),
            (("reg", "z"), {}, {}),
        ]
        for a, r, m in bad_cases:
            with self.assertRaises(DbgxError):
                evaluate(a, r, m)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
