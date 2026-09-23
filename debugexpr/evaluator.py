"""The expression evaluator: ``evaluate(ast, registers, memory)``.

Semantics (fixed contract, see README):

* Integers are explicitly typed (signed/unsigned, 8/16/32/64 bit).
* Binary arithmetic requires *identical* operand types and truncates the
  result modulo 2**bits (wrap-around, no saturation, no silent widening).
* ``div``/``mod`` truncate toward zero (C semantics); division by zero is
  an error for the assignments that reach it.  Signed overflow such as
  ``INT_MIN / -1`` wraps per the modular rule.
* Shifts require 0 <= amount < bit width; ``shr`` is arithmetic for
  signed types and logical for unsigned types.
* Comparisons honor signedness and require identical integer types.
  Pointer ``eq``/``ne`` work across regions; ordering comparisons and
  ``PtrSub`` require a single shared region.
* ``and``/``or`` short-circuit and ``If`` selects a branch: for each
  concrete register assignment only the taken side is evaluated, so an
  untaken branch performs no reads and cannot raise.  When the condition
  is *unknown* (0 for some assignments, nonzero for others) both
  branches are explored, each only for the assignments that reach it.
* Registers may map to up to ``MAX_REGISTER_CANDIDATES`` candidate ints.
  The expression is then evaluated exactly over the feasible
  assignments: ``results`` is the set of distinct feasible values and
  ``reads`` is the union of intervals that some feasible assignment
  actually touches.  An error reached by some (but not all) assignments
  yields ``status == "possible_error"`` with the located errors in
  ``errors``; an error reached by every assignment is raised.
* The concrete assignment space is bounded by
  ``MAX_CANDIDATE_COMBINATIONS``.  Beyond it the result is
  ``status == "unknown"`` (or ``CandidateBudgetError`` when the caller
  passes ``on_budget_overrun="raise"``); static input legality is still
  checked and illegal input still raises.
* Loads are little-endian, must be naturally aligned, and may only touch
  regions of the snapshot passed to ``evaluate``.  Dereferencing the
  one-past-end position is rejected.
* The AST is limited to ``MAX_NODES`` distinct nodes.

Every contract violation raises an ``EvalError`` subclass carrying a
``path`` that locates the offending node.  Nothing is silently corrected.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass

from .errors import (
    CandidateBudgetError,
    DivisionByZeroError,
    EvalError,
    InputError,
    NodeBudgetError,
    PointerError,
    PossibleError,
    RangeError,
    ShiftError,
    TypeMismatchError,
)
from .memory import MemorySnapshot
from .nodes import And, BinOp, Cast, Cmp, Const, Field, If, Load, Node, Not, Or, PtrAdd, PtrConst, PtrSub, Reg
from .types import POINTER, IntType, Pointer, PointerType, S64, U8, Value

MAX_NODES = 500
MAX_REGISTER_CANDIDATES = 8
MAX_CANDIDATE_COMBINATIONS = 64

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
    """Outcome of evaluating an expression.

    Attributes:
        status: one of
            ``"exact"``          -- ``results`` lists every feasible value;
            ``"possible_error"`` -- some assignments succeed (``results``),
                                    others fail (``errors``);
            ``"unknown"``        -- the feasible-assignment space exceeded
                                    ``MAX_CANDIDATE_COMBINATIONS``; no values
                                    or reads are claimed (both are empty).
        results: distinct feasible typed values, deterministically ordered.
        errors: located errors reached by some feasible assignment (each an
            ``EvalError`` with its ``path``); empty for ``"exact"``.
        reads: memory intervals actually read by some feasible assignment,
            de-duplicated in a deterministic (evidence) order.
    """

    status: str
    results: tuple[Value, ...] = ()
    errors: tuple[EvalError, ...] = ()
    reads: tuple[ReadInterval, ...] = ()

    @property
    def value(self) -> Value:
        """Legacy accessor for expressions with a uniquely determined value."""
        if self.status == "unknown" or len(self.results) != 1:
            raise RuntimeError(
                f"result is not uniquely determined (status={self.status}, "
                f"{len(self.results)} feasible values); use .results"
            )
        return self.results[0]


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
    if isinstance(node, If):
        return (node.cond, node.then, node.else_)
    if isinstance(node, PtrAdd):
        return (node.ptr, node.index)
    if isinstance(node, (Cast, Not)):
        return (node.node,)
    if isinstance(node, (Load, Field)):
        return (node.ptr,)
    return ()


def _referenced_registers(root: Node) -> frozenset[str]:
    names: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, Reg):
            names.add(node.name)
        stack.extend(_children(node))
    return frozenset(names)


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

    if isinstance(node, If):
        cond = _eval(node.cond, ctx, path + ("cond",))
        # Only the selected branch is evaluated for this assignment; the
        # other branch's reads and errors cannot occur here.
        if _truth(cond, "if", path + ("cond",)):
            return _eval(node.then, ctx, path + ("then",))
        return _eval(node.else_, ctx, path + ("else_",))

    raise InputError(f"not an AST node: {node!r}", path)


# ---------------------------------------------------------------------------
# Candidate aggregation
# ---------------------------------------------------------------------------


def _value_key(value: Value):
    data = value.data
    if isinstance(data, Pointer):
        return (1, data.region_id, data.offset)
    return (0, str(value.type), data)


def _dedup_errors(errors: list[EvalError]) -> tuple[EvalError, ...]:
    seen: set[tuple] = set()
    out: list[EvalError] = []
    for err in errors:
        key = (type(err), err.path, err.message)
        if key not in seen:
            seen.add(key)
            out.append(err)
    return tuple(out)


def _dedup_reads(reads: list[ReadInterval]) -> tuple[ReadInterval, ...]:
    seen: set[tuple] = set()
    out: list[ReadInterval] = []
    for read in reads:
        key = (read.region_id, read.offset, read.length)
        if key not in seen:
            seen.add(key)
            out.append(read)
    return tuple(out)


def _normalize_registers(
    registers: Mapping[str, object],
) -> dict[str, tuple[int, ...]]:
    """Validate registers and return per-register candidate tuples.

    A plain ``int`` means one candidate; a tuple/list of ints gives up to
    ``MAX_REGISTER_CANDIDATES`` candidates (duplicates are removed,
    first-occurrence order kept).
    """
    if not isinstance(registers, Mapping):
        raise InputError(
            f"registers must be a mapping of str -> int (or int tuple/list), "
            f"got {type(registers).__name__}"
        )
    clean: dict[str, tuple[int, ...]] = {}
    for name, value in registers.items():
        if type(name) is not str or not name:
            raise InputError(f"register name must be a non-empty str, got {name!r}")
        if type(value) is tuple or type(value) is list:
            candidates: list[int] = []
            seen: set[int] = set()
            for item in value:
                item = _require_int(item, f"register {name!r} candidate", ())
                if item not in seen:
                    seen.add(item)
                    candidates.append(item)
            if not candidates:
                raise InputError(f"register {name!r} candidate list must not be empty", ())
            if len(candidates) > MAX_REGISTER_CANDIDATES:
                raise InputError(
                    f"register {name!r} has {len(candidates)} distinct candidates; "
                    f"at most {MAX_REGISTER_CANDIDATES} are allowed",
                    (),
                )
            clean[name] = tuple(candidates)
        else:
            clean[name] = (_require_int(value, f"register {name!r} value", ()),)
    return clean


def _run_worlds(
    ast: Node,
    candidates: Mapping[str, tuple[int, ...]],
    memory: MemorySnapshot,
) -> tuple[str, tuple[Value, ...], tuple[EvalError, ...], tuple[ReadInterval, ...]]:
    """Exact evaluation: one scalar run per concrete register assignment.

    Only registers referenced by the AST participate in the space.
    """
    referenced = _referenced_registers(ast)
    names = sorted(name for name in candidates if name in referenced)
    pools = [candidates[name] for name in names]

    values: dict[object, Value] = {}
    errors: list[EvalError] = []
    reads: list[ReadInterval] = []
    for combo in itertools.product(*pools):
        assignment = dict(zip(names, combo))
        ctx = _Ctx(assignment, memory)
        try:
            result = _eval(ast, ctx, ())
        except EvalError as err:
            errors.append(err)
        else:
            values[_value_key(result)] = result
        reads.extend(ctx.reads)

    if not values:
        # Every feasible assignment fails.  A uniform failure is a definite
        # error; mixed failures are reported together as a possible error.
        unique = _dedup_errors(errors)
        if len(unique) == 1:
            raise unique[0]
        raise PossibleError(unique)

    ordered_results = tuple(values[k] for k in sorted(values))
    unique_errors = _dedup_errors(errors)
    status = "possible_error" if unique_errors else "exact"
    return status, ordered_results, unique_errors, _dedup_reads(reads)


# ---------------------------------------------------------------------------
# Structural validation used when the assignment space is over budget.
# ---------------------------------------------------------------------------
#
# Only assignment-independent legality is enforced here: node field kinds,
# operator names, IntType annotations, int-ness of literal integers,
# register/region existence and literal pointer bounds (PtrConst is a
# literal, so an out-of-region one is illegal input regardless of the
# assignment).  Everything that depends on the concrete register values
# (range, division by zero, dynamic pointer bounds, alignment, types
# implied by a selected branch) is conservatively left unclaimed: the
# result is "unknown" by contract.


def _static_validate(
    node: Node,
    memory: MemorySnapshot,
    candidates: Mapping[str, tuple[int, ...]],
    path: tuple[str, ...],
) -> None:
    if isinstance(node, Const):
        if not isinstance(node.type, IntType):
            raise InputError(f"Const type must be an IntType, got {node.type!r}", path)
        value = _require_int(node.value, "Const value", path)
        if not node.type.contains(value):
            raise RangeError(
                f"literal {value} does not fit {node.type} [{node.type.min}, {node.type.max}]", path
            )
        return

    if isinstance(node, PtrConst):
        if type(node.region_id) is not str or not node.region_id:
            raise InputError(f"PtrConst region_id must be a non-empty str, got {node.region_id!r}", path)
        offset = _require_int(node.offset, "PtrConst offset", path)
        _check_pointer_bounds(memory, Pointer(node.region_id, offset), path)
        return

    if isinstance(node, Reg):
        if type(node.name) is not str or not node.name:
            raise InputError(f"Reg name must be a non-empty str, got {node.name!r}", path)
        if not isinstance(node.type, IntType):
            raise InputError(f"Reg type must be an IntType, got {node.type!r}", path)
        if node.name not in candidates:
            raise InputError(f"unknown register {node.name!r}", path)
        # Whether a candidate fits this Reg node's declared type is
        # assignment-dependent (a per-world RangeError); it cannot be
        # decided in the over-budget path, so it is left to "unknown".
        return

    if isinstance(node, BinOp):
        if node.op not in _INT_BINOPS:
            raise InputError(f"unknown binary operator {node.op!r}", path)
        _static_validate(node.left, memory, candidates, path + ("left",))
        _static_validate(node.right, memory, candidates, path + ("right",))
        return

    if isinstance(node, Cmp):
        if node.op not in _CMP_OPS:
            raise InputError(f"unknown comparison operator {node.op!r}", path)
        _static_validate(node.left, memory, candidates, path + ("left",))
        _static_validate(node.right, memory, candidates, path + ("right",))
        return

    if isinstance(node, Cast):
        if not isinstance(node.target, IntType):
            raise InputError(f"Cast target must be an IntType, got {node.target!r}", path)
        _static_validate(node.node, memory, candidates, path + ("node",))
        return

    if isinstance(node, PtrAdd):
        _static_validate(node.ptr, memory, candidates, path + ("ptr",))
        _static_validate(node.index, memory, candidates, path + ("index",))
        return

    if isinstance(node, PtrSub):
        _static_validate(node.left, memory, candidates, path + ("left",))
        _static_validate(node.right, memory, candidates, path + ("right",))
        return

    if isinstance(node, Load):
        if not isinstance(node.type, IntType):
            raise InputError(f"Load type must be an IntType, got {node.type!r}", path)
        _static_validate(node.ptr, memory, candidates, path + ("ptr",))
        return

    if isinstance(node, Field):
        if not isinstance(node.type, IntType):
            raise InputError(f"Field type must be an IntType, got {node.type!r}", path)
        _require_int(node.offset, "Field offset", path)
        _static_validate(node.ptr, memory, candidates, path + ("ptr",))
        return

    if isinstance(node, (And, Or)):
        _static_validate(node.left, memory, candidates, path + ("left",))
        # Constant left values statically decide the short-circuit, so the
        # unevaluated right side is skipped (same laziness as scalar runs).
        is_and = isinstance(node, And)
        if isinstance(node.left, Const):
            left_true = node.left.value != 0
            right_runs = (not left_true) if is_and else left_true
            if right_runs:
                _static_validate(node.right, memory, candidates, path + ("right",))
        else:
            _static_validate(node.right, memory, candidates, path + ("right",))
        return

    if isinstance(node, Not):
        _static_validate(node.node, memory, candidates, path + ("node",))
        return

    if isinstance(node, If):
        _static_validate(node.cond, memory, candidates, path + ("cond",))
        # Mirror the scalar engine's laziness for constant conditions;
        # for non-constant conditions both branches may be reached, so
        # both are checked (a malformed reachable branch is illegal input).
        if isinstance(node.cond, Const):
            if node.cond.value != 0:
                _static_validate(node.then, memory, candidates, path + ("then",))
            else:
                _static_validate(node.else_, memory, candidates, path + ("else_",))
        else:
            _static_validate(node.then, memory, candidates, path + ("then",))
            _static_validate(node.else_, memory, candidates, path + ("else_",))
        return

    raise InputError(f"not an AST node: {node!r}", path)


def evaluate(
    ast: Node,
    registers: Mapping[str, object],
    memory: MemorySnapshot,
    *,
    on_budget_overrun: str = "unknown",
) -> EvalResult:
    """Evaluate ``ast`` against ``registers`` and the ``memory`` snapshot.

    Args:
        ast: an AST built from the node classes in ``debugexpr.nodes``
            (at most ``MAX_NODES`` distinct nodes).
        registers: mapping of register name -> Python int, or -> tuple/list
            of up to ``MAX_REGISTER_CANDIDATES`` distinct ints.  bool and
            float values (including NaN/Infinity) are rejected.
        memory: a ``MemorySnapshot``; loads may only touch its regions.
        on_budget_overrun: ``"unknown"`` (default) returns
            ``EvalResult(status="unknown")`` when the feasible assignment
            space exceeds ``MAX_CANDIDATE_COMBINATIONS``; ``"raise"``
            raises ``CandidateBudgetError`` instead.

    Returns:
        ``EvalResult`` with ``status`` / ``results`` / ``errors`` /
        ``reads``.

    Raises:
        EvalError: a definite (all-assignment or assignment-independent)
            contract violation, with a locatable path.  Failures reached
            by only some assignments are collected in ``result.errors``
            (``status="possible_error"``); when no assignment succeeds,
            the uniform error is raised (or :class:`PossibleError` for
            non-uniform failures).
    """
    if not isinstance(ast, Node):
        raise InputError(f"ast must be a debugexpr Node, got {ast!r}")
    if on_budget_overrun not in ("unknown", "raise"):
        raise InputError(
            f"on_budget_overrun must be 'unknown' or 'raise', got {on_budget_overrun!r}"
        )
    if not isinstance(memory, MemorySnapshot):
        raise InputError(f"memory must be a MemorySnapshot, got {type(memory).__name__}")
    candidates = _normalize_registers(registers)

    count = _count_nodes(ast)
    if count > MAX_NODES:
        raise NodeBudgetError(f"AST has {count} distinct nodes, exceeding the limit of {MAX_NODES}")

    referenced = _referenced_registers(ast)
    missing = referenced - candidates.keys()
    if missing:
        raise InputError(f"unknown register {sorted(missing)[0]!r}")
    total = 1
    for name in referenced:
        total *= len(candidates[name])

    if total <= MAX_CANDIDATE_COMBINATIONS:
        status, results, errors, reads = _run_worlds(ast, candidates, memory)
    elif on_budget_overrun == "raise":
        raise CandidateBudgetError(
            f"register assignment space has {total} combinations, exceeding the budget "
            f"of {MAX_CANDIDATE_COMBINATIONS}"
        )
    else:
        # Definite, assignment-independent legality is still enforced;
        # dynamic outcomes are not claimed.
        _static_validate(ast, memory, candidates, ())
        status, results, errors, reads = "unknown", (), (), ()

    return EvalResult(status=status, results=results, errors=errors, reads=reads)
