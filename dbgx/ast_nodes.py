"""AST 节点格式、静态校验与节点预算。

AST 节点统一为 Python tuple，首元素为算子名字符串，其余元素为静态参数或
子节点。节点树中任何非 tuple 的叶子只允许出现在字面量参数位置；整数字面量
必须是 ``int`` 且**拒绝 bool**；没有浮点算子，因此 ``float``（含 NaN、
Infinity）一律被静态校验拒绝。

节点一览（详见 README）::

    ("int",   value, width:int, signed:bool)      整数字面量
    ("reg",   name:str)                            读寄存器（值须为 Value）
    ("add"|"sub"|"mul"|"udiv"|"sdiv"|"urem"|"srem"
       |"and"|"or"|"xor"|"shl"|"lsr"|"asr", l, r) 二元运算
    ("eq"|"ne"|"lt"|"le"|"gt"|"ge", l, r)          比较
    ("lnot", x)                                    逻辑非（整数 != 0）
    ("band", l, r) / ("bor", l, r)                 短路与/或
    ("cast", x, width, signed)                     位宽/符号转换
    ("ptr",  region_id:str, offset:int, width:int) 指针常量
    ("padd", p, i) / ("psub", p, i)                指针 ± 整数
    ("pdiff", p1, p2)                              同区域指针相减
    ("load", p)                                    对齐解引用读内存
    ("field", x, lsb:int, width:int, signed:bool)  位域提取

节点数预算固定为 500：整棵树的 tuple 节点总数不得超过 500，
超过时静态校验直接失败，不执行任何求值。
"""

from __future__ import annotations

import math

from .errors import ASTError

MAX_NODES = 500

_BINARY_OPS = frozenset(
    {
        "add", "sub", "mul",
        "udiv", "sdiv", "urem", "srem",
        "and", "or", "xor",
        "shl", "lsr", "asr",
        "eq", "ne", "lt", "le", "gt", "ge",
    }
)
_SHORT_CIRCUIT_OPS = frozenset({"band", "bor"})
_UNARY_OPS = frozenset({"lnot"})
_WIDTHS = frozenset({8, 16, 32, 64})


def _is_real_tuple(node: object) -> bool:
    return isinstance(node, tuple) and len(node) > 0 and isinstance(node[0], str)


def _check_width(w: object, where: str) -> None:
    if isinstance(w, bool) or not isinstance(w, int) or w not in _WIDTHS:
        raise ASTError(f"{where} 位宽必须是 8/16/32/64 之一，得到 {w!r}")


def _check_signed(s: object, where: str) -> None:
    if not isinstance(s, bool):
        raise ASTError(f"{where} signed 标志必须是 bool，得到 {s!r}")


def _check_literal_int(v: object, where: str) -> None:
    """整数字面量入参：必须 int、拒绝 bool、拒绝 float/NaN/Infinity。"""
    if isinstance(v, float):
        if math.isfinite(v):
            raise ASTError(f"{where} 必须是 int，得到 float: {v!r}")
        raise ASTError(f"{where} 拒绝 NaN/Infinity，得到 {v!r}")
    if isinstance(v, bool) or not isinstance(v, int):
        kind = "bool" if isinstance(v, bool) else type(v).__name__
        raise ASTError(f"{where} 必须是 int 且不接受 bool，得到 {kind}: {v!r}")


def _check_str(v: object, where: str) -> None:
    if not isinstance(v, str) or not v:
        raise ASTError(f"{where} 必须是非空字符串，得到 {v!r}")


def validate(node: object) -> int:
    """对整棵 AST 做静态校验，返回节点总数。

    纯结构检查不触碰 registers/memory；任何非法结构抛
    :class:`~dbgx.errors.ASTError`。
    """
    count = 0

    def walk(n: object, path: tuple[int, ...]) -> None:
        nonlocal count
        count += 1
        if count > MAX_NODES:
            raise ASTError(
                f"AST 节点数超过上限 {MAX_NODES}（第 {count} 个节点位于 "
                f"{'/'.join(map(str, path)) or '根'}）",
                (),
            )
        if not _is_real_tuple(n):
            raise ASTError(
                f"节点必须是以算子名开头的非空 tuple，得到 {n!r}", path
            )
        op = n[0]

        if op == "int":
            if len(n) != 4:
                raise ASTError(f"(int) 需要 3 个参数 (value,width,signed)", path)
            _check_literal_int(n[1], "(int) value")
            _check_width(n[2], "(int)")
            _check_signed(n[3], "(int)")

        elif op == "reg":
            if len(n) != 2:
                raise ASTError("(reg) 需要 1 个参数 (name)", path)
            _check_str(n[1], "(reg) name")

        elif op in _BINARY_OPS:
            if len(n) != 3:
                raise ASTError(f"({op}) 需要 2 个子节点", path)
            walk(n[1], path + (1,))
            walk(n[2], path + (2,))

        elif op in _SHORT_CIRCUIT_OPS:
            if len(n) != 3:
                raise ASTError(f"({op}) 需要 2 个子节点", path)
            walk(n[1], path + (1,))
            walk(n[2], path + (2,))

        elif op in _UNARY_OPS:
            if len(n) != 2:
                raise ASTError(f"({op}) 需要 1 个子节点", path)
            walk(n[1], path + (1,))

        elif op == "cast":
            if len(n) != 4:
                raise ASTError("(cast) 需要 3 个参数 (child,width,signed)", path)
            walk(n[1], path + (1,))
            _check_width(n[2], "(cast)")
            _check_signed(n[3], "(cast)")

        elif op == "ptr":
            if len(n) != 4:
                raise ASTError(
                    "(ptr) 需要 3 个参数 (region_id,offset,pointee_width)", path
                )
            _check_str(n[1], "(ptr) region_id")
            _check_literal_int(n[2], "(ptr) offset")
            if n[2] < 0:
                raise ASTError(f"(ptr) offset 必须非负，得到 {n[2]}", path)
            _check_width(n[3], "(ptr) pointee")

        elif op in ("padd", "psub"):
            if len(n) != 3:
                raise ASTError(f"({op}) 需要 2 个子节点", path)
            walk(n[1], path + (1,))
            walk(n[2], path + (2,))

        elif op == "pdiff":
            if len(n) != 3:
                raise ASTError("(pdiff) 需要 2 个子节点", path)
            walk(n[1], path + (1,))
            walk(n[2], path + (2,))

        elif op == "load":
            if len(n) != 2:
                raise ASTError("(load) 需要 1 个子节点", path)
            walk(n[1], path + (1,))

        elif op == "field":
            if len(n) != 5:
                raise ASTError(
                    "(field) 需要 4 个参数 (child,lsb,width,signed)", path
                )
            walk(n[1], path + (1,))
            _check_literal_int(n[2], "(field) lsb")
            if n[2] < 0:
                raise ASTError(f"(field) lsb 必须非负，得到 {n[2]}", path)
            _check_width(n[3], "(field) 提取宽度")
            _check_signed(n[4], "(field)")

        else:
            raise ASTError(f"未知算子 {op!r}", path)

    walk(node, ())
    return count
