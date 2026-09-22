"""The expression evaluator: ``evaluate(ast, registers, memory)``.

Semantics (fixed contract, see README):

* Integers are explicitly typed (signed/unsigned, 8/16/32/64 bit).
* Binary arithmetic requires *identical* operand types and truncates the
  result modulo 2**bits (wrap-around, no saturation, no silent widening).
* ``div``/``mod`` truncate toward zero (C semantics); division by zero is
  an error.  Signed overflow such as INT_MIN / -1 wraps per the modular
  rule, like every other arithmetic result.
* Shifts require 0 <= amount < bit width; ``shr`` is arithmetic for
  signed types and logical for unsigned types.
* Comparisons honor signedness and require identical integer types.
  Pointer ``eq``/``ne`` work across regions; ordering comparisons and
  ``PtrSub`` require a single shared region.
* ``and``/``or`` short-circuit: the right operand is not evaluated when
  the left decides the result (so its side conditions, e.g. loads, never
  trigger).
* Loads are little-endian, must be naturally aligned, and may only touch
  regions of the snapshot passed to ``evaluate``.  Dereferencing the
  one-past-end position is rejected.
* The AST is limited to ``MAX_NODES`` distinct nodes.

Every contract violation raises an ``EvalError`` subclass carrying a
``path`` that locates the offending node.  Nothing is silently corrected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import (
    DivisionByZeroError,
    EvalError,
    InputError,
    NodeBudgetError,
    PointerError,
    RangeError,
    ShiftError,
    TypeMismatchError,
)
from .memory import MemorySnapshot
from .nodes import And, BinOp, Cast, Cmp, Const, Field, Load, Node, Not, Or, PtrAdd, PtrConst, PtrSub, Reg
from .types import POINTER, IntType, Pointer, PointerType, S64, U8, Value

MAX_NODES = 500

_INT_BINOPS = frozenset({"add", "sub", "mul", "div", "mod", "shl", "shr", "bitand", "bitor", "bitxor"})
_SHIFT_OPS = frozenset({"shl", "shr"})
_CMP_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


@dataclass(frozen=True)
class ReadInterval:
    """One memory region interval actually read during evaluation."""

    region_id: str
    offset: int
    length: int


@dataclass(frozen=True)
class EvalResult:
    """Outcome of a successful evaluation.

    ``value`` is the typed result; ``reads`` lists every memory interval
    actually read, in evaluation order (evaluation evidence).
    """

    value: Value
    reads: tuple[ReadInterval, ...]


def _require_int(value: object, what: str, path: tuple[str, ...]) -> int:
    """Reject bool and every non-int (floats incl. NaN/Infinity included)."""
    if type(value) is bool or not isinstance(value, int):
        raise InputError(f"{what} must be an int (bool and float are rejected), got {value!r}", path)
    return value


def _trunc_divmod(a: int, b: int) -> tuple[int, int]:
    """Truncating (toward zero) division and the matching remainder."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q, a - q * b


class _Ctx:
    __slots__ = ("registers", "memory", "reads")

    def __init__(self, registers: Mapping[str, int], memory: MemorySnapshot) -> None:
        self.registers = registers
        self.memory = memory
        self.reads: list[ReadInterval] = []


def _count_nodes(root: Node) -> int:
    """Count distinct nodes (a shared subtree counts once), iteratively."""
    seen: set[int] = set()
    stack = [root]
    count = 0
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        count += 1
        for child in _children(node):
            stack.append(child)
    return count


def _children(node: Node) -> tuple[Node, ...]:
    if isinstance(node, (BinOp, Cmp, PtrSub, And, Or)):
        return (node.left, node.right)
    if isinstance(node, PtrAdd):
        return (node.ptr, node.index)
    if isinstance(node, (Cast, Not)):
        return (node.node,)
    if isinstance(node, (Load, Field)):
        return (node.ptr,)
    return ()


def _check_int_operands(op: str, left: Value, right: Value, path: tuple[str, ...]) -> IntType:
    if not isinstance(left.type, IntType) or not isinstance(right.type, IntType):
        raise TypeMismatchError(
            f"{op}: both operands must be integers, got {left.type} and {right.type}", path
        )
    if left.type != right.type:
        raise TypeMismatchError(
            f"{op}: operand types must be identical, got {left.type} and {right.type}", path
        )
    return left.type


def _eval_binop(node: BinOp, left: Value, right: Value, path: tuple[str, ...]) -> Value:
    if node.op not in _INT_BINOPS:
        raise InputError(f"unknown binary operator {node.op!r}", path)
    itype = _check_int_operands(node.op, left, right, path)
    a, b = left.data, right.data
    assert isinstance(a, int) and isinstance(b, int)

    if node.op in _SHIFT_OPS:
        if b < 0 or b >= itype.bits:
            raise ShiftError(
                f"{node.op}: shift amount {b} out of range [0, {itype.bits})", path + ("right",)
            )
        raw = (a << b) if node.op == "shl" else (a >> b)
        # Python's >> on the canonical signed value is an arithmetic shift;
        # on the non-negative canonical unsigned value it is a logical shift.
    elif node.op == "add":
        raw = a + b
    elif node.op == "sub":
        raw = a - b
    elif node.op == "mul":
        raw = a * b
    elif node.op in ("div", "mod"):
        if b == 0:
            raise DivisionByZeroError(f"{node.op}: division by zero", path + ("right",))
        q, r = _trunc_divmod(a, b)
        raw = q if node.op == "div" else r
    elif node.op == "bitand":
        raw = a & b
    elif node.op == "bitor":
        raw = a | b
    else:  # bitxor
        raw = a ^ b
    return Value(itype, itype.wrap(raw))


def _eval_cmp(node: Cmp, left: Value, right: Value, path: tuple[str, ...]) -> Value:
    if node.op not in _CMP_OPS:
        raise InputError(f"unknown comparison operator {node.op!r}", path)
    if isinstance(left.type, PointerType) and isinstance(right.type, PointerType):
        lp, rp = left.data, right.data
        assert isinstance(lp, Pointer) and isinstance(rp, Pointer)
        if node.op in ("eq", "ne"):
            result = (lp.region_id == rp.region_id and lp.offset == rp.offset)
            return Value(U8, int(result if node.op == "eq" else not result))
        if lp.region_id != rp.region_id:
            raise PointerError(
                f"{node.op}: ordering comparison across regions "
                f"{lp.region_id!r} and {rp.region_id!r}",
                path,
            )
        a, b = lp.offset, rp.offset
    else:
        _check_int_operands(node.op, left, right, path)
        a, b = left.data, right.data
        assert isinstance(a, int) and isinstance(b, int)
    result = {
        "eq": a == b,
        "ne": a != b,
        "lt": a < b,
        "le": a <= b,
        "gt": a > b,
        "ge": a >= b,
    }[node.op]
    return Value(U8, int(result))


def _check_pointer_bounds(memory: MemorySnapshot, ptr: Pointer, path: tuple[str, ...]) -> int:
    """Return region size; enforce 0 <= offset <= size (one-past-end ok)."""
    size = memory.region_size(ptr.region_id)  # raises PointerError if unknown
    if ptr.offset < 0 or ptr.offset > size:
        raise PointerError(
            f"pointer offset {ptr.offset} is outside region {ptr.region_id!r} "
            f"(size {size}; one-past-end {size} is the last valid position)",
            path,
        )
    return size


def _do_load(ctx: _Ctx, ptr: Pointer, itype: IntType, path: tuple[str, ...]) -> Value:
    region_size = _check_pointer_bounds(ctx.memory, ptr, path)
    size = itype.size
    if ptr.offset % size != 0:
        raise PointerError(
            f"misaligned {itype} load at offset {ptr.offset} in region {ptr.region_id!r} "
            f"(requires {size}-byte alignment)",
            path,
        )
    if ptr.offset + size > region_size:
        raise PointerError(
            f"{itype} load at offset {ptr.offset} in region {ptr.region_id!r} "
            f"needs {size} bytes but the region ends at {region_size} "
            f"(one-past-end dereference is rejected)",
            path,
        )
    raw = ctx.memory.read(ptr.region_id, ptr.offset, size)
    ctx.reads.append(ReadInterval(ptr.region_id, ptr.offset, size))
    return Value(itype, int.from_bytes(raw, "little", signed=itype.signed))


def _truth(value: Value, op: str, path: tuple[str, ...]) -> bool:
    if not isinstance(value.type, IntType):
        raise TypeMismatchError(f"{op}: operand must be an integer, got {value.type}", path)
    assert isinstance(value.data, int)
    return value.data != 0


def _eval(node: Node, ctx: _Ctx, path: tuple[str, ...]) -> Value:
    if isinstance(node, Const):
        if not isinstance(node.type, IntType):
            raise InputError(f"Const type must be an IntType, got {node.type!r}", path)
        value = _require_int(node.value, "Const value", path)
        if not node.type.contains(value):
            raise RangeError(
                f"literal {value} does not fit {node.type} "
                f"[{node.type.min}, {node.type.max}]",
                path,
            )
        return Value(node.type, value)

    if isinstance(node, PtrConst):
        if type(node.region_id) is not str or not node.region_id:
            raise InputError(f"PtrConst region_id must be a non-empty str, got {node.region_id!r}", path)
        offset = _require_int(node.offset, "PtrConst offset", path)
        ptr = Pointer(node.region_id, offset)
        _check_pointer_bounds(ctx.memory, ptr, path)
        return Value(POINTER, ptr)

    if isinstance(node, Reg):
        if type(node.name) is not str or not node.name:
            raise InputError(f"Reg name must be a non-empty str, got {node.name!r}", path)
        if not isinstance(node.type, IntType):
            raise InputError(f"Reg type must be an IntType, got {node.type!r}", path)
        if node.name not in ctx.registers:
            raise InputError(f"unknown register {node.name!r}", path)
        value = ctx.registers[node.name]
        if not node.type.contains(value):
            raise RangeError(
                f"register {node.name!r} value {value} does not fit {node.type} "
                f"[{node.type.min}, {node.type.max}]",
                path,
            )
        return Value(node.type, value)

    if isinstance(node, BinOp):
        left = _eval(node.left, ctx, path + ("left",))
        right = _eval(node.right, ctx, path + ("right",))
        return _eval_binop(node, left, right, path)

    if isinstance(node, Cmp):
        left = _eval(node.left, ctx, path + ("left",))
        right = _eval(node.right, ctx, path + ("right",))
        return _eval_cmp(node, left, right, path)

    if isinstance(node, Cast):
        if not isinstance(node.target, IntType):
            raise InputError(f"Cast target must be an IntType, got {node.target!r}", path)
        value = _eval(node.node, ctx, path + ("node",))
        if not isinstance(value.type, IntType):
            raise TypeMismatchError(f"cast: source must be an integer, got {value.type}", path + ("node",))
        assert isinstance(value.data, int)
        return Value(node.target, node.target.wrap(value.data))

    if isinstance(node, PtrAdd):
        base = _eval(node.ptr, ctx, path + ("ptr",))
        index = _eval(node.index, ctx, path + ("index",))
        if not isinstance(base.type, PointerType):
            raise TypeMismatchError(f"ptr-add: base must be a pointer, got {base.type}", path + ("ptr",))
        if not isinstance(index.type, IntType):
            raise TypeMismatchError(f"ptr-add: index must be an integer, got {index.type}", path + ("index",))
        assert isinstance(base.data, Pointer) and isinstance(index.data, int)
        ptr = Pointer(base.data.region_id, base.data.offset + index.data)
        _check_pointer_bounds(ctx.memory, ptr, path)
        return Value(POINTER, ptr)

    if isinstance(node, PtrSub):
        left = _eval(node.left, ctx, path + ("left",))
        right = _eval(node.right, ctx, path + ("right",))
        if not isinstance(left.type, PointerType) or not isinstance(right.type, PointerType):
            raise TypeMismatchError(
                f"ptr-sub: both operands must be pointers, got {left.type} and {right.type}", path
            )
        lp, rp = left.data, right.data
        assert isinstance(lp, Pointer) and isinstance(rp, Pointer)
        if lp.region_id != rp.region_id:
            raise PointerError(
                f"ptr-sub: cross-region subtraction {lp.region_id!r} - {rp.region_id!r} is rejected",
                path,
            )
        return Value(S64, S64.wrap(lp.offset - rp.offset))

    if isinstance(node, Load):
        if not isinstance(node.type, IntType):
            raise InputError(f"Load type must be an IntType, got {node.type!r}", path)
        ptr_value = _eval(node.ptr, ctx, path + ("ptr",))
        if not isinstance(ptr_value.type, PointerType):
            raise TypeMismatchError(f"load: operand must be a pointer, got {ptr_value.type}", path + ("ptr",))
        assert isinstance(ptr_value.data, Pointer)
        return _do_load(ctx, ptr_value.data, node.type, path)

    if isinstance(node, Field):
        if not isinstance(node.type, IntType):
            raise InputError(f"Field type must be an IntType, got {node.type!r}", path)
        offset = _require_int(node.offset, "Field offset", path)
        base = _eval(node.ptr, ctx, path + ("ptr",))
        if not isinstance(base.type, PointerType):
            raise TypeMismatchError(f"field: base must be a pointer, got {base.type}", path + ("ptr",))
        assert isinstance(base.data, Pointer)
        ptr = Pointer(base.data.region_id, base.data.offset + offset)
        return _do_load(ctx, ptr, node.type, path)

    if isinstance(node, And):
        left = _eval(node.left, ctx, path + ("left",))
        if not _truth(left, "and", path + ("left",)):
            return Value(U8, 0)  # short-circuit: right side never evaluated
        right = _eval(node.right, ctx, path + ("right",))
        return Value(U8, int(_truth(right, "and", path + ("right",))))

    if isinstance(node, Or):
        left = _eval(node.left, ctx, path + ("left",))
        if _truth(left, "or", path + ("left",)):
            return Value(U8, 1)  # short-circuit: right side never evaluated
        right = _eval(node.right, ctx, path + ("right",))
        return Value(U8, int(_truth(right, "or", path + ("right",))))

    if isinstance(node, Not):
        value = _eval(node.node, ctx, path + ("node",))
        return Value(U8, int(not _truth(value, "not", path + ("node",))))

    raise InputError(f"not an AST node: {node!r}", path)


def evaluate(ast: Node, registers: Mapping[str, int], memory: MemorySnapshot) -> EvalResult:
    """Evaluate ``ast`` against ``registers`` and the ``memory`` snapshot.

    Args:
        ast: an AST built from the node classes in ``debugexpr.nodes``
            (at most ``MAX_NODES`` distinct nodes).
        registers: mapping of register name -> Python int.  bool and float
            values (including NaN/Infinity) are rejected.
        memory: a ``MemorySnapshot``; loads may only touch its regions.

    Returns:
        ``EvalResult`` with the typed value and the list of memory
        intervals actually read.

    Raises:
        EvalError: (subclass) any contract violation, with a locatable path.
    """
    if not isinstance(ast, Node):
        raise InputError(f"ast must be a debugexpr Node, got {ast!r}")
    if not isinstance(registers, Mapping):
        raise InputError(f"registers must be a mapping of str -> int, got {type(registers).__name__}")
    clean_registers: dict[str, int] = {}
    for name, value in registers.items():
        if type(name) is not str or not name:
            raise InputError(f"register name must be a non-empty str, got {name!r}")
        clean_registers[name] = _require_int(value, f"register {name!r} value", ())
    if not isinstance(memory, MemorySnapshot):
        raise InputError(f"memory must be a MemorySnapshot, got {type(memory).__name__}")

    count = _count_nodes(ast)
    if count > MAX_NODES:
        raise NodeBudgetError(f"AST has {count} distinct nodes, exceeding the limit of {MAX_NODES}")

    ctx = _Ctx(clean_registers, memory)
    value = _eval(ast, ctx, ())
    return EvalResult(value=value, reads=tuple(ctx.reads))
