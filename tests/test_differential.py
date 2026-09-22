"""差分测试：被测 dbgx 与独立掩码参考 tests.reference 交叉核对。

参考实现**不导入 dbgx**（见 tests/reference.py 模块文档），期望值由其独立
计算；本文件只负责用确定性随机生成 AST/寄存器/内存，对同一输入分别求值并
比对值与读取证据。错误情形按错误类别比对。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import unittest

from dbgx import (
    AddressError,
    DbgxError,
    DivisionByZeroError,
    IntType,
    PtrType,
    ReadRange,
    Region,
    ShiftError,
    TypeMismatchError,
    UndefinedRegisterError,
    Value,
    evaluate,
)

import reference as ref

WIDTHS = (8, 16, 32, 64)
_INT_OPS = (
    "add", "sub", "mul",
    "and", "or", "xor",
    "udiv", "sdiv", "urem", "srem",
    "shl", "lsr",
)
_CMPS = ("eq", "ne", "lt", "le", "gt", "ge")

# dbgx 错误 -> 参考 tag
_ERROR_TAG = {
    TypeMismatchError: "type",
    ShiftError: "shift",
    DivisionByZeroError: "div0",
    AddressError: "addr",
    UndefinedRegisterError: "undef",
}


class Fixture:
    """一次差分用例的完整确定性环境。"""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        # 两个区域：ram 24 字节 base=0，io 16 字节 base=0x4000
        self.ram = bytes(self.rng.randrange(256) for _ in range(24))
        self.io = bytes(self.rng.randrange(256) for _ in range(16))
        self.mem_dbgx = {
            "ram": Region("ram", self.ram, base=0),
            "io": Region("io", self.io, base=0x4000),
        }
        self.mem_ref = {"ram": (0, self.ram), "io": (0x4000, self.io)}
        # 寄存器：每个位宽/符号组合一个，外加两个指针寄存器
        self.regs_dbgx: dict[str, Value] = {}
        self.regs_ref: dict[str, object] = {}
        for w in WIDTHS:
            for signed in (False, True):
                name = f"r{w}{'s' if signed else 'u'}"
                val = self.rng.randrange(1 << w)
                self.regs_dbgx[name] = Value(IntType(w, signed), val)
                self.regs_ref[name] = ("i", w, signed, val)
        for name, rid, base, pw in (
            ("p_ram", "ram", 0, 16),
            ("p_io", "io", 0x4000, 32),
        ):
            off = base + self.rng.randrange(0, 8) * (pw // 8)
            self.regs_dbgx[name] = Value(PtrType(rid, pw), (rid, off))
            self.regs_ref[name] = ("p", rid, off, pw)


def _gen_int_expr(rng: random.Random, depth: int) -> tuple:
    if depth <= 0 or rng.random() < 0.45:
        w = rng.choice(WIDTHS)
        signed = rng.random() < 0.5
        return ("int", rng.randrange(-(1 << (w - 1)), 1 << w), w, signed)
    kind = rng.randrange(6)
    if kind < 3:
        op = rng.choice(_INT_OPS)
        w = rng.choice(WIDTHS)
        signed = op in ("sdiv", "srem", "asr") or rng.random() < 0.5
        l = _gen_typed_int(rng, depth - 1, w, signed)
        r = _gen_typed_int(rng, depth - 1, w, signed)
        return (op, l, r)
    if kind == 3:
        op = rng.choice(_CMPS)
        w = rng.choice(WIDTHS)
        signed = rng.random() < 0.5
        return (
            op,
            _gen_typed_int(rng, depth - 1, w, signed),
            _gen_typed_int(rng, depth - 1, w, signed),
        )
    if kind == 4:
        # cast / field
        if rng.random() < 0.5:
            w2 = rng.choice(WIDTHS)
            return ("cast", _gen_int_expr(rng, depth - 1), w2, rng.random() < 0.5)
        srcw = rng.choice((16, 32, 64))
        fw = rng.choice([w for w in WIDTHS if w <= srcw])
        lsb = rng.choice([x for x in (0, 8, 16, 24, 32, 40, 48, 56) if x + fw <= srcw])
        return ("field", _gen_typed_int(rng, depth - 1, srcw, rng.random() < 0.5),
                lsb, fw, rng.random() < 0.5)
    # asr
    w = rng.choice(WIDTHS)
    return (
        "asr",
        _gen_typed_int(rng, depth - 1, w, True),
        _gen_typed_int(rng, depth - 1, w, True),
    )


def _gen_typed_int(rng: random.Random, depth: int, w: int, signed: bool) -> tuple:
    if depth <= 0 or rng.random() < 0.6:
        # 寄存器或字面量
        if rng.random() < 0.4:
            return ("reg", f"r{w}{'s' if signed else 'u'}")
        return ("int", rng.randrange(-(1 << (w - 1)), 1 << w), w, signed)
    e = _gen_int_expr(rng, depth)
    # 用 cast 收敛到目标类型
    return ("cast", e, w, signed)


def _gen_ptr_expr(fx: Fixture, depth: int) -> tuple:
    rng = fx.rng
    base_node = rng.choice(
        [("reg", "p_ram"), ("reg", "p_io"),
         ("ptr", "ram", rng.randrange(0, 24) & ~1, 16),
         ("ptr", "io", 0x4000 + (rng.randrange(0, 16) & ~3), 32)]
    )
    if depth <= 0 or rng.random() < 0.5:
        return base_node
    # 小步长 ±0..6 字节，制造尾后/越界/对齐场景
    step = rng.randrange(-6, 7)
    op = rng.choice(("padd", "psub"))
    return (op, base_node, ("int", step, 32, True))


def _gen_mixed(fx: Fixture, depth: int) -> tuple:
    """生成可能含 load、短路、指针相减的混合表达式。"""
    rng = fx.rng
    if depth <= 0:
        return _gen_int_expr(rng, 0)
    kind = rng.randrange(8)
    if kind < 3:
        return _gen_int_expr(rng, depth)
    if kind in (3, 4):
        p = _gen_ptr_expr(fx, depth - 1)
        return ("load", p)
    if kind == 5:
        return (
            "pdiff",
            _gen_ptr_expr(fx, depth - 1),
            _gen_ptr_expr(fx, depth - 1),
        )
    if kind == 6:
        # 短路：左为比较，右可能是 load（制造短路越界读取）
        return (
            rng.choice(("band", "bor")),
            _gen_int_expr(rng, min(depth - 1, 2)),
            _gen_mixed(fx, depth - 1),
        )
    # 指针算术后 load
    return ("load", _gen_ptr_expr(fx, depth - 1))


def _ref_value_to_dbgx(rv) -> Value:
    if rv[0] == "i":
        return Value(IntType(rv[1], rv[2]), rv[3])
    return Value(PtrType(rv[1], rv[3]), (rv[1], rv[2]))


def _error_tag(exc: DbgxError) -> str | None:
    for cls, tag in _ERROR_TAG.items():
        if isinstance(exc, cls):
            return tag
    return None


class TestDifferential(unittest.TestCase):
    def test_pure_integer_expressions(self):
        n = 0
        for seed in range(400):
            fx = Fixture(seed)
            ast = _gen_int_expr(fx.rng, 4)
            self._compare(ast, fx, n)
            n += 1
        self.assertGreater(n, 0)

    def test_mixed_memory_expressions(self):
        n = 0
        for seed in range(1000, 1400):
            fx = Fixture(seed)
            ast = _gen_mixed(fx, 4)
            self._compare(ast, fx, n)
            n += 1
        self.assertGreater(n, 0)

    def test_node_count_agrees_on_valid(self):
        fx = Fixture(999)
        ast = _gen_mixed(fx, 3)
        res = evaluate(ast, fx.regs_dbgx, fx.mem_dbgx)
        self.assertLessEqual(res.node_count, 500)

    def _compare(self, ast, fx: Fixture, idx: int) -> None:
        try:
            refv, ref_reads = ref.ref_eval(ast, fx.regs_ref, fx.mem_ref)
        except ref._RefError as e:
            with self.assertRaises(DbgxError, msg=f"seed 用例 {idx}: {ast}"):
                evaluate(ast, fx.regs_dbgx, fx.mem_dbgx)
            try:
                evaluate(ast, fx.regs_dbgx, fx.mem_dbgx)
            except DbgxError as got:
                self.assertEqual(
                    _error_tag(got),
                    e.tag,
                    f"用例 {idx} 错误类别不一致：参考={e.tag} "
                    f"被测={type(got).__name__}: {got}\nAST={ast}",
                )
            return

        res = evaluate(ast, fx.regs_dbgx, fx.mem_dbgx)
        expect = _ref_value_to_dbgx(refv)
        self.assertEqual(
            res.value.vtype,
            expect.vtype,
            f"用例 {idx} 类型不一致\nAST={ast}",
        )
        self.assertEqual(
            res.value.raw,
            expect.raw,
            f"用例 {idx} 值不一致：期望 {expect} 实际 {res.value}\nAST={ast}",
        )
        self.assertEqual(
            tuple(res.reads),
            tuple(ReadRange(r[0], r[1], r[2]) for r in ref_reads),
            f"用例 {idx} 读取证据不一致\nAST={ast}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
