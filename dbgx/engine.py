"""核心求值器：位宽语义 + 地址空间语义。

入口 :func:`evaluate`，返回 :class:`EvalResult`（带类型值 + 实际读取区间
证据 + 静态节点数）。求值对 registers/memory 只读：区域在构造
:class:`~dbgx.types.Region` 时已拷贝为不可变 bytes，失败的求值不会留下
任何部分变更。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from . import ast_nodes

# 500 节点上限下最深可能有 500 层递归；默认 1000 的解释器栈余量不足，
# 显式抬高，保证恰好 500 节点的左/右旋链都能求值。
sys.setrecursionlimit(max(sys.getrecursionlimit(), 4000))
from .errors import (
    AddressError,
    ASTError,
    DivisionByZeroError,
    InputError,
    ShiftError,
    TypeMismatchError,
    UndefinedRegisterError,
)
from .types import IntType, PtrType, ReadRange, Region, Value

_ENDIAN = "little"


@dataclass(frozen=True)
class EvalResult:
    """求值结果。

    :param value: 带类型的结果值。
    :param reads: 求值过程中**实际执行**的 load 读取区间（按执行顺序）。
        被短路跳过的 load 不会出现。
    :param node_count: AST 静态节点总数（<= 500）。
    """

    value: Value
    reads: tuple[ReadRange, ...]
    node_count: int


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------

def _validate_inputs(registers: object, memory: object) -> tuple[dict, dict]:
    if not isinstance(registers, dict):
        raise InputError("registers 必须是 dict[str, Value]")
    regs: dict[str, Value] = {}
    for k, v in registers.items():
        if not isinstance(k, str) or not k:
            raise InputError(f"registers 的键必须是非空 str，得到 {k!r}")
        if not isinstance(v, Value):
            raise InputError(
                f"寄存器 {k!r} 的值必须是 dbgx.Value，得到 {type(v).__name__}"
            )
        regs[k] = v

    if not isinstance(memory, dict):
        raise InputError("memory 必须是 dict[str, Region]")
    mem: dict[str, Region] = {}
    for k, v in memory.items():
        if not isinstance(k, str) or not k:
            raise InputError(f"memory 的键必须是非空 str，得到 {k!r}")
        if not isinstance(v, Region):
            raise InputError(
                f"区域 {k!r} 必须是 dbgx.Region，得到 {type(v).__name__}"
            )
        if v.region_id != k:
            raise InputError(
                f"memory 键 {k!r} 与 Region.region_id={v.region_id!r} 不一致"
            )
        mem[k] = v
    return regs, mem


# ---------------------------------------------------------------------------
# 求值
# ---------------------------------------------------------------------------

class _Engine:
    def __init__(self, regs: dict[str, Value], mem: dict[str, Region]) -> None:
        self._regs = regs
        self._mem = mem
        self._reads: list[ReadRange] = []

    def run(self, node: tuple) -> Value:
        return self._eval(node, ())

    # -- 递归框架 -------------------------------------------------------
    # 约定：所有 DbgxError 均携带从根算起的绝对路径（tuple 子节点下标序列），
    # 子节点错误直接向上传播，不在中间层重复加前缀。
    def _eval(self, node: tuple, path: tuple[int, ...]) -> Value:
        op = node[0]

        if op == "int":
            _, v, width, signed = node
            return Value.int_(width, signed, v)

        if op == "reg":
            name = node[1]
            if name not in self._regs:
                raise UndefinedRegisterError(f"寄存器 {name!r} 未提供", path)
            return self._regs[name]

        if op == "lnot":
            x = self._eval(node[1], path + (1,))
            if not x.is_int:
                raise TypeMismatchError(
                    f"lnot 要求整数操作数，得到 {x.vtype}", path
                )
            return Value(IntType(8, False), 1 if x.as_int() == 0 else 0)

        if op in ("band", "bor"):
            return self._short_circuit(node, path)

        if op == "cast":
            x = self._eval(node[1], path + (1,))
            _, _, width, signed = node
            if not x.is_int:
                raise TypeMismatchError(
                    f"cast 只支持整数，得到 {x.vtype}", path
                )
            # 位模式保留：变窄按模截断，变宽零扩展，同宽换符号性不变位模式
            return Value.int_(width, signed, x.as_int())

        if op == "field":
            return self._field(node, path)

        if op == "ptr":
            _, region_id, offset, width = node
            return Value(PtrType(region_id, width), (region_id, offset))

        if op in ("padd", "psub"):
            return self._pointer_add(node, path)

        if op == "pdiff":
            return self._pointer_diff(node, path)

        if op == "load":
            return self._load(node, path)

        if op in ("eq", "ne", "lt", "le", "gt", "ge"):
            return self._compare(node, path)

        # 其余为二元整数算术
        return self._binary_arith(node, path)

    # -- 算子实现 -------------------------------------------------------
    def _int_pair(
        self, node: tuple, path: tuple[int, ...]
    ) -> tuple[IntType, int, int]:
        l = self._eval(node[1], path + (1,))
        r = self._eval(node[2], path + (2,))
        op = node[0]
        if not (l.is_int and r.is_int):
            raise TypeMismatchError(
                f"({op}) 要求两个整数操作数，得到 {l.vtype} 与 {r.vtype}", path
            )
        lt, rt = l.vtype, r.vtype
        assert isinstance(lt, IntType) and isinstance(rt, IntType)
        if lt != rt:
            raise TypeMismatchError(
                f"({op}) 要求两侧同类型（同位宽同符号性），"
                f"得到 {lt} 与 {rt}",
                path,
            )
        return lt, l.as_int(), r.as_int()

    def _binary_arith(self, node: tuple, path: tuple[int, ...]) -> Value:
        op = node[0]
        t, a, b = self._int_pair(node, path)
        mask = t.mask

        if op in ("add", "sub", "mul", "and", "or", "xor"):
            if op == "add":
                r = a + b
            elif op == "sub":
                r = a - b
            elif op == "mul":
                r = a * b
            elif op == "and":
                r = a & b
            elif op == "or":
                r = a | b
            else:
                r = a ^ b
            return Value(t, r & mask)

        if op in ("udiv", "urem"):
            if b == 0:
                raise DivisionByZeroError(f"({op}) 除数为零（{t}）", path)
            return Value(t, (a // b) if op == "udiv" else (a % b))

        if op in ("sdiv", "srem"):
            sa, sb = t.signed_value(a), t.signed_value(b)
            if sb == 0:
                raise DivisionByZeroError(f"({op}) 除数为零（{t}）", path)
            if op == "sdiv":
                # 向零截断；INT_MIN / -1 按模回绕为 INT_MIN
                q = abs(sa) // abs(sb)
                if (sa < 0) != (sb < 0):
                    q = -q
                return Value(t, q & mask)
            # srem：C 截断余数，符号随被除数；INT_MIN % -1 == 0
            q = abs(sa) // abs(sb)
            if (sa < 0) != (sb < 0):
                q = -q
            return Value(t, (sa - q * sb) & mask)

        if op in ("shl", "lsr", "asr"):
            return self._shift(op, t, a, b, path)

        raise ASTError(f"内部错误：未实现的算子 {op!r}", path)  # pragma: no cover

    @staticmethod
    def _shift(
        op: str, t: IntType, a: int, amt_raw: int, path: tuple[int, ...]
    ) -> Value:
        # 移位量按机器表示（无符号视角）读取；有符号 -1 的表示为 2**w-1，
        # 必然不小于位宽。
        if amt_raw >= t.width:
            raise ShiftError(
                f"({op}) 移位量 {amt_raw} 不小于位宽 {t.width}（合法区间 "
                f"0..{t.width - 1}）",
                path,
            )
        if op == "shl":
            return Value(t, (a << amt_raw) & t.mask)
        if op == "lsr":
            return Value(t, a >> amt_raw)
        # asr：仅允许有符号类型
        if not t.signed:
            raise TypeMismatchError(
                f"(asr) 算术右移要求有符号整数类型，得到无符号 {t}", path
            )
        signed = t.signed_value(a)
        return Value(t, (signed >> amt_raw) & t.mask)

    def _compare(self, node: tuple, path: tuple[int, ...]) -> Value:
        op = node[0]
        l = self._eval(node[1], path + (1,))
        r = self._eval(node[2], path + (2,))
        if not (l.is_int and r.is_int):
            raise TypeMismatchError(
                f"({op}) 只能比较整数，得到 {l.vtype} 与 {r.vtype}", path
            )
        lt, rt = l.vtype, r.vtype
        assert isinstance(lt, IntType) and isinstance(rt, IntType)
        if lt != rt:
            raise TypeMismatchError(
                f"({op}) 要求两侧同类型才能比较，得到 {lt} 与 {rt}", path
            )
        if op in ("eq", "ne"):
            cond = l.as_int() == r.as_int()
            if op == "ne":
                cond = not cond
        else:
            lv = l.as_signed() if lt.signed else l.as_int()
            rv = r.as_signed() if rt.signed else r.as_int()
            cond = {
                "lt": lv < rv,
                "le": lv <= rv,
                "gt": lv > rv,
                "ge": lv >= rv,
            }[op]
        return Value(IntType(8, False), 1 if cond else 0)

    def _short_circuit(self, node: tuple, path: tuple[int, ...]) -> Value:
        op = node[0]
        l = self._eval(node[1], path + (1,))
        if not l.is_int:
            raise TypeMismatchError(
                f"({op}) 操作数必须是整数，得到 {l.vtype}", path
            )
        l_true = l.as_int() != 0
        if op == "band":
            if not l_true:
                return Value(IntType(8, False), 0)  # 右子树不求值、不读内存
            r = self._eval(node[2], path + (2,))
            if not r.is_int:
                raise TypeMismatchError(
                    f"({op}) 操作数必须是整数，得到 {r.vtype}", path
                )
            return Value(IntType(8, False), 1 if r.as_int() != 0 else 0)
        # bor
        if l_true:
            return Value(IntType(8, False), 1)  # 右子树不求值、不读内存
        r = self._eval(node[2], path + (2,))
        if not r.is_int:
            raise TypeMismatchError(
                f"({op}) 操作数必须是整数，得到 {r.vtype}", path
            )
        return Value(IntType(8, False), 1 if r.as_int() != 0 else 0)

    def _field(self, node: tuple, path: tuple[int, ...]) -> Value:
        x = self._eval(node[1], path + (1,))
        _, _, lsb, width, signed = node
        if not x.is_int:
            raise TypeMismatchError(
                f"(field) 只支持整数源，得到 {x.vtype}", path
            )
        xt = x.vtype
        assert isinstance(xt, IntType)
        if lsb + width > xt.width:
            raise TypeMismatchError(
                f"(field) 位域 [lsb={lsb}, 宽度={width}] 超出源类型 {xt} "
                f"（最大 lsb={xt.width - width}）",
                path,
            )
        fmask = (1 << width) - 1
        raw = (x.as_int() >> lsb) & fmask
        # 装箱时按目标宽度保留位模式；signed 只影响后续解释（as_signed）
        return Value(IntType(width, signed), raw)

    def _ptr_region(
        self, rid: str, op: str, path: tuple[int, ...]
    ) -> Region:
        region = self._mem.get(rid)
        if region is None:
            raise AddressError(
                f"({op}) 引用了未传入快照的区域 {rid!r}，load/指针算术只能"
                f"使用 memory 中提供的区域",
                path,
            )
        return region

    def _pointer_add(self, node: tuple, path: tuple[int, ...]) -> Value:
        op = node[0]
        p = self._eval(node[1], path + (1,))
        i = self._eval(node[2], path + (2,))
        if not p.is_ptr:
            raise TypeMismatchError(
                f"({op}) 左操作数必须是指针，得到 {p.vtype}", path
            )
        if not i.is_int:
            raise TypeMismatchError(
                f"({op}) 右操作数必须是整数，得到 {i.vtype}", path
            )
        pt = p.vtype
        assert isinstance(pt, PtrType)
        rid, offset = p.as_ptr()
        region = self._ptr_region(rid, op, path)
        delta = i.as_signed()
        new_offset = offset + delta if op == "padd" else offset - delta
        # 含尾后位置 end；越界（含负地址）拒绝
        if not (region.base <= new_offset <= region.end):
            raise AddressError(
                f"({op}) 结果地址 {new_offset} 离开区域 {rid!r} 范围 "
                f"[{region.base}, {region.end}]（尾后位置 {region.end} "
                f"合法但不可再越过）",
                path,
            )
        return Value(pt, (rid, new_offset))

    def _pointer_diff(self, node: tuple, path: tuple[int, ...]) -> Value:
        p1 = self._eval(node[1], path + (1,))
        p2 = self._eval(node[2], path + (2,))
        if not (p1.is_ptr and p2.is_ptr):
            raise TypeMismatchError(
                f"(pdiff) 要求两个指针，得到 {p1.vtype} 与 {p2.vtype}", path
            )
        r1, o1 = p1.as_ptr()
        r2, o2 = p2.as_ptr()
        if r1 != r2:
            raise AddressError(
                f"(pdiff) 拒绝跨区域指针相减：{r1!r} 与 {r2!r}（不同地址"
                f"空间的偏移没有定义良好的差值）",
                path,
            )
        region = self._ptr_region(r1, "pdiff", path)
        for off in (o1, o2):
            if not (region.base <= off <= region.end):
                raise AddressError(
                    f"(pdiff) 指针偏移 {off} 不在区域 {r1!r} 范围 "
                    f"[{region.base}, {region.end}]（含尾后位置）内",
                    path,
                )
        # 字节差，有符号 i64；合法指针偏移均在区域 [base, end] 内
        return Value.int_(64, True, o1 - o2)

    def _load(self, node: tuple, path: tuple[int, ...]) -> Value:
        p = self._eval(node[1], path + (1,))
        if not p.is_ptr:
            raise TypeMismatchError(
                f"(load) 操作数必须是指针，得到 {p.vtype}", path
            )
        pt = p.vtype
        assert isinstance(pt, PtrType)
        rid, offset = p.as_ptr()
        region = self._mem.get(rid)
        if region is None:
            raise AddressError(
                f"(load) 区域 {rid!r} 不在传入的内存快照中", path
            )
        nbytes = pt.pointee_nbytes
        # 对齐：按自然访问宽度（offset 必须是 nbytes 的整数倍）
        if offset % nbytes != 0:
            raise AddressError(
                f"(load) 未对齐访问：区域 {rid!r} 偏移 {offset} 读取 "
                f"{pt.pointee_width} 位（要求 {nbytes} 字节对齐）",
                path,
            )
        # 边界（含尾后拒绝）：把 Region 的地址错误补上节点路径
        try:
            rr = region.check_read(offset, nbytes)
        except AddressError as e:
            raise AddressError(e.message, path) from None
        idx0 = offset - region.base
        raw = int.from_bytes(region.data[idx0 : idx0 + nbytes], _ENDIAN)
        self._reads.append(rr)
        return Value(IntType(pt.pointee_width, False), raw)

    @property
    def reads(self) -> tuple[ReadRange, ...]:
        return tuple(self._reads)


def evaluate(
    ast: object,
    registers: object,
    memory: object,
) -> EvalResult:
    """对调试表达式 AST 求值。

    :param ast: 符合 :mod:`dbgx.ast_nodes` 契约的 tuple 树（<= 500 节点）。
    :param registers: ``{寄存器名: Value}``，只读。
    :param memory: ``{region_id: Region}`` 区域快照，load 只能读这些区域。
    :returns: :class:`EvalResult`。
    :raises DbgxError: 各种子类，消息带节点路径，可定位。
    """
    regs, mem = _validate_inputs(registers, memory)
    node_count = ast_nodes.validate(ast)
    engine = _Engine(regs, mem)
    value = engine.run(ast)  # type: ignore[arg-type]
    return EvalResult(value=value, reads=engine.reads, node_count=node_count)
