"""The expression evaluator: ``evaluate(ast, registers, memory)``.

Scalar semantics (fixed contract, see README):

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
* ``and``/``or`` short-circuit: the right operand is not evaluated on the
  pairs where the left already decides the result.
* Loads are little-endian, must be naturally aligned, and may only touch
  regions of the snapshot passed to ``evaluate``.  Dereferencing the
  one-past-end position is rejected.
* The AST is limited to ``MAX_NODES`` distinct nodes.

Candidate semantics (this iteration):

* A register may hold up to ``MAX_CANDIDATES`` (8) possible integers.
* Every sub-expression then evaluates to a :class:`CandidateSet` of
  possible values (1..8 distinct members).  Binary nodes combine the
  operand sets as a Cartesian product; occurrences of the *same*
  register are combined independently (no equality correlation).
* ``Select`` only reads the branch that can actually run: a determined
  condition evaluates exactly one branch; a condition that may be either
  0 or 1 evaluates both and unions their values.
* Every enumerated combination is charged against ``max_combinations``.
  Exceeding the budget (or producing more than 8 distinct possible
  values) raises :class:`CandidateBudgetError` -- the explicit "unknown"
  signal, never raised while every register holds a single value.

Every illegal-input violation raises an ``EvalError`` subclass carrying a
``path`` that locates the offending node.  Nothing is silently corrected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import (
    CandidateBudgetError,
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
from .nodes import And, BinOp, Cast, Cmp, Const, Field, Load, Node, Not, Or, PtrAdd, PtrConst, PtrSub, Reg, Select
from .types import (
    MAX_CANDIDATES,
    POINTER,
    CandidateSet,
    IntType,
    Pointer,
    PointerType,
    S64,
    U8,
    Value,
)

MAX_NODES = 500
MAX_COMBINATIONS = 4096

_INT_BINOPS = frozenset({"add", "sub", "mul", "div", "mod", "shl", "shr", "bitand", "bitor", "bitxor"})
_SHIFT_OPS = frozenset({"shl", "shr"})
_CMP_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_REGISTER_CANDIDATE_KINDS = (tuple, list, frozenset, set)


@dataclass(frozen=True)
class ReadInterval:
    """One memory region interval actually read during evaluation."""

    region_id: str
    offset: int
    length: int


@dataclass(frozen=True)
class MaybeValue:
    """The possible values of one sub-expression: one shared type plus a
    deduplicated, deterministically ordered set of candidate members."""

    type: IntType | PointerType
    candidates: CandidateSet


@dataclass(frozen=True)
class EvalResult:
    """Outcome of a successful evaluation.

    ``type`` is the shared type of every possible result.  ``candidates``
    are the possible values (1..8).  ``reads`` lists every memory interval
    that may actually be read, in evaluation order (evaluation evidence).

    ``value`` is the uniquely determined :class:`Value`, or ``None`` when
    more than one candidate is possible (use ``values`` then).
    """

    type: IntType | PointerType
    candidates: CandidateSet
    reads: tuple[ReadInterval, ...]

    @property
    def is_determined(self) -> bool:
        return len(self.candidates) == 1

    @property
    def value(self) -> Value | None:
        if len(self.candidates) != 1:
            return None
        return Value(self.type, self.candidates.members[0])

    @property
    def values(self) -> tuple[int | Pointer, ...]:
        """The possible result data, in a deterministic order."""
        return self.candidates.members


def _mv_single(itype: IntType | PointerType, data: int | Pointer) -> MaybeValue:
    return MaybeValue(itype, CandidateSet.singleton(data))


def _mv_collect(
    itype: IntType | PointerType, members: set[int | Pointer], path: tuple[str, ...]
) -> MaybeValue:
    """Build a candidate result from computed members.

    More than ``MAX_CANDIDATES`` distinct outcomes is a representation
    limit: raise the explicit "unknown" error rather than dropping values.
    """
    if len(members) > MAX_CANDIDATES:
        raise CandidateBudgetError(
            f"expression has {len(members)} distinct possible values, exceeding the "
            f"limit of {MAX_CANDIDATES}; result is unknown",
            path,
        )
    return MaybeValue(itype, CandidateSet(members))


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
    __slots__ = ("registers", "memory", "reads", "combos", "combo_budget", "multi")

    def __init__(
        self,
        registers: Mapping[str, tuple[int, ...]],
        memory: MemorySnapshot,
        combo_budget: int,
    ) -> None:
        self.registers = registers
        self.memory = memory
        self.reads: list[ReadInterval] = []
        self.combos = 0
        self.combo_budget = combo_budget
        # Becomes True as soon as a register with several candidates is
        # read.  While it stays False every intermediate set is a
        # singleton, so enumeration is linear and can never go unknown.
        self.multi = False


def _charge(ctx: _Ctx, amount: int, path: tuple[str, ...]) -> None:
    """Account for ``amount`` enumerated candidate combinations.

    Charging only starts once a multi-candidate register is involved, so
    purely scalar evaluations (even over maximally shared DAGs) can never
    be reported unknown.
    """
    if not ctx.multi:
        return
    ctx.combos += amount
    if ctx.combos > ctx.combo_budget:
        raise CandidateBudgetError(
            f"candidate expansion enumerated {ctx.combos} combinations, exceeding the "
            f"budget of {ctx.combo_budget}; result is unknown (pass a larger max_combinations "
            f"to enumerate more)",
            path,
        )


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
    if isinstance(node, Select):
        return (node.cond, node.then, node.else_)
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


def _do_load(ctx: _Ctx, ptr: Pointer, itype: IntType, path: tuple[str, ...]) -> int:
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
    return int.from_bytes(raw, "little", signed=itype.signed)


def _require_integer(mv: MaybeValue, what: str, path: tuple[str, ...]) -> IntType:
    if not isinstance(mv.type, IntType):
        raise TypeMismatchError(f"{what}: operand must be an integer, got {mv.type}", path)
    return mv.type


def _require_pointer(mv: MaybeValue, what: str, path: tuple[str, ...]) -> None:
    if not isinstance(mv.type, PointerType):
        raise TypeMismatchError(f"{what}: operand must be a pointer, got {mv.type}", path)


def _eval_isolated(
    node: Node, ctx: _Ctx, path: tuple[str, ...]
) -> tuple[MaybeValue, tuple[ReadInterval, ...]]:
    """Evaluate one branch with a private read list, then restore the
    caller's list, so a conditional's branch evidence can be merged in a
    defined order (and discarded entirely if the branch is not taken)."""
    branch_reads: list[ReadInterval] = []
    saved = ctx.reads
    ctx.reads = branch_reads
    try:
        mv = _eval(node, ctx, path)
    finally:
        ctx.reads = saved
    return mv, tuple(branch_reads)


def _eval(node: Node, ctx: _Ctx, path: tuple[str, ...]) -> MaybeValue:
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
        return _mv_single(node.type, value)

    if isinstance(node, PtrConst):
        if type(node.region_id) is not str or not node.region_id:
            raise InputError(f"PtrConst region_id must be a non-empty str, got {node.region_id!r}", path)
        offset = _require_int(node.offset, "PtrConst offset", path)
        ptr = Pointer(node.region_id, offset)
        _check_pointer_bounds(ctx.memory, ptr, path)
        return _mv_single(POINTER, ptr)

    if isinstance(node, Reg):
        if type(node.name) is not str or not node.name:
            raise InputError(f"Reg name must be a non-empty str, got {node.name!r}", path)
        if not isinstance(node.type, IntType):
            raise InputError(f"Reg type must be an IntType, got {node.type!r}", path)
        if node.name not in ctx.registers:
            raise InputError(f"unknown register {node.name!r}", path)
        values = ctx.registers[node.name]
        for value in values:
            if not node.type.contains(value):
                raise RangeError(
                    f"register {node.name!r} value {value} does not fit {node.type} "
                    f"[{node.type.min}, {node.type.max}]",
                    path,
                )
        if len(values) > 1:
            ctx.multi = True
        _charge(ctx, len(values), path)
        candidates = CandidateSet(values) if len(values) > 1 else CandidateSet.singleton(values[0])
        return MaybeValue(node.type, candidates)

    if isinstance(node, BinOp):
        if node.op not in _INT_BINOPS:
            raise InputError(f"unknown binary operator {node.op!r}", path)
        left = _eval(node.left, ctx, path + ("left",))
        right = _eval(node.right, ctx, path + ("right",))
        # Type-check once up front (yields the same error the per-pair
        # scalar rule would) before spending the combination budget.
        itype = _check_int_operands(
            node.op, Value(left.type, 0), Value(right.type, 0), path
        )
        _charge(ctx, len(left.candidates) * len(right.candidates), path)
        outcomes: set[int] = set()
        for a in left.candidates:
            for b in right.candidates:
                outcomes.add(_eval_binop(node, Value(itype, a), Value(itype, b), path).data)
        return _mv_collect(itype, outcomes, path)

    if isinstance(node, Cmp):
        if node.op not in _CMP_OPS:
            raise InputError(f"unknown comparison operator {node.op!r}", path)
        left = _eval(node.left, ctx, path + ("left",))
        right = _eval(node.right, ctx, path + ("right",))
        _charge(ctx, len(left.candidates) * len(right.candidates), path)
        outcomes: set[int] = set()
        for a in left.candidates:
            for b in right.candidates:
                outcomes.add(_eval_cmp(node, Value(left.type, a), Value(right.type, b), path).data)
        return _mv_collect(U8, outcomes, path)

    if isinstance(node, Cast):
        if not isinstance(node.target, IntType):
            raise InputError(f"Cast target must be an IntType, got {node.target!r}", path)
        value = _eval(node.node, ctx, path + ("node",))
        _require_integer(value, "cast", path + ("node",))
        outcomes = {node.target.wrap(d) for d in value.candidates}
        return _mv_collect(node.target, outcomes, path)

    if isinstance(node, PtrAdd):
        base = _eval(node.ptr, ctx, path + ("ptr",))
        index = _eval(node.index, ctx, path + ("index",))
        _require_pointer(base, "ptr-add", path + ("ptr",))
        _require_integer(index, "ptr-add", path + ("index",))
        _charge(ctx, len(base.candidates) * len(index.candidates), path)
        outcomes: set[Pointer] = set()
        for p in base.candidates:
            assert isinstance(p, Pointer)
            for i in index.candidates:
                assert isinstance(i, int)
                ptr = Pointer(p.region_id, p.offset + i)
                _check_pointer_bounds(ctx.memory, ptr, path)
                outcomes.add(ptr)
        return _mv_collect(POINTER, outcomes, path)

    if isinstance(node, PtrSub):
        left = _eval(node.left, ctx, path + ("left",))
        right = _eval(node.right, ctx, path + ("right",))
        _require_pointer(left, "ptr-sub", path + ("left",))
        _require_pointer(right, "ptr-sub", path + ("right",))
        _charge(ctx, len(left.candidates) * len(right.candidates), path)
        outcomes: set[int] = set()
        for lp in left.candidates:
            for rp in right.candidates:
                assert isinstance(lp, Pointer) and isinstance(rp, Pointer)
                if lp.region_id != rp.region_id:
                    raise PointerError(
                        f"ptr-sub: cross-region subtraction {lp.region_id!r} - "
                        f"{rp.region_id!r} is rejected",
                        path,
                    )
                outcomes.add(S64.wrap(lp.offset - rp.offset))
        return _mv_collect(S64, outcomes, path)

    if isinstance(node, Load):
        if not isinstance(node.type, IntType):
            raise InputError(f"Load type must be an IntType, got {node.type!r}", path)
        ptr_value = _eval(node.ptr, ctx, path + ("ptr",))
        _require_pointer(ptr_value, "load", path + ("ptr",))
        _charge(ctx, len(ptr_value.candidates), path)
        outcomes: set[int] = set()
        for ptr in ptr_value.candidates:
            assert isinstance(ptr, Pointer)
            outcomes.add(_do_load(ctx, ptr, node.type, path))
        return _mv_collect(node.type, outcomes, path)

    if isinstance(node, Field):
        if not isinstance(node.type, IntType):
            raise InputError(f"Field type must be an IntType, got {node.type!r}", path)
        offset = _require_int(node.offset, "Field offset", path)
        base = _eval(node.ptr, ctx, path + ("ptr",))
        _require_pointer(base, "field", path + ("ptr",))
        _charge(ctx, len(base.candidates), path)
        outcomes: set[int] = set()
        for base_ptr in base.candidates:
            assert isinstance(base_ptr, Pointer)
            ptr = Pointer(base_ptr.region_id, base_ptr.offset + offset)
            outcomes.add(_do_load(ctx, ptr, node.type, path))
        return _mv_collect(node.type, outcomes, path)

    if isinstance(node, And):
        left = _eval(node.left, ctx, path + ("left",))
        _require_integer(left, "and", path + ("left",))
        left_zero = any(d == 0 for d in left.candidates)
        if left_zero and all(d == 0 for d in left.candidates):
            return _mv_single(U8, 0)  # short-circuit on every path: right never runs
        # At least one path has a non-zero left: the right side can run.
        right = _eval(node.right, ctx, path + ("right",))
        _require_integer(right, "and", path + ("right",))
        outcomes = {int(d != 0) for d in right.candidates}
        if left_zero:
            outcomes.add(0)  # paths where the left was 0 (right not evaluated there)
        return _mv_collect(U8, outcomes, path)

    if isinstance(node, Or):
        left = _eval(node.left, ctx, path + ("left",))
        _require_integer(left, "or", path + ("left",))
        left_nonzero = any(d != 0 for d in left.candidates)
        if left_nonzero and all(d != 0 for d in left.candidates):
            return _mv_single(U8, 1)  # short-circuit on every path: right never runs
        right = _eval(node.right, ctx, path + ("right",))
        _require_integer(right, "or", path + ("right",))
        outcomes = {int(d != 0) for d in right.candidates}
        if left_nonzero:
            outcomes.add(1)
        return _mv_collect(U8, outcomes, path)

    if isinstance(node, Not):
        value = _eval(node.node, ctx, path + ("node",))
        _require_integer(value, "not", path + ("node",))
        outcomes = {int(d == 0) for d in value.candidates}
        return _mv_collect(U8, outcomes, path)

    if isinstance(node, Select):
        cond = _eval(node.cond, ctx, path + ("cond",))
        _require_integer(cond, "select condition", path + ("cond",))
        cond_values = set(cond.candidates)
        for possible in cond_values:
            if possible not in (0, 1):
                raise RangeError(
                    f"select condition must be 0 or 1, got possible value {possible}",
                    path + ("cond",),
                )
        if cond_values == {1}:
            return _eval(node.then, ctx, path + ("then",))  # else_ never touched
        if cond_values == {0}:
            return _eval(node.else_, ctx, path + ("else_",))  # then never touched
        # Condition may be either: both branches can actually execute.
        then_mv, then_reads = _eval_isolated(node.then, ctx, path + ("then",))
        else_mv, else_reads = _eval_isolated(node.else_, ctx, path + ("else_",))
        if then_mv.type != else_mv.type:
            raise TypeMismatchError(
                f"select: branch types must be identical, got {then_mv.type} and {else_mv.type}",
                path,
            )
        ctx.reads.extend(then_reads)
        ctx.reads.extend(else_reads)
        members = set(then_mv.candidates) | set(else_mv.candidates)
        return _mv_collect(then_mv.type, members, path)

    raise InputError(f"not an AST node: {node!r}", path)


def _normalize_registers(registers: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    """Validate and normalize register inputs into sorted, deduplicated
    tuples of candidate ints (a plain int becomes a one-element tuple)."""
    clean: dict[str, tuple[int, ...]] = {}
    for name, value in registers.items():
        if type(name) is not str or not name:
            raise InputError(f"register name must be a non-empty str, got {name!r}")
        if type(value) is bool or not isinstance(value, (int,) + _REGISTER_CANDIDATE_KINDS):
            raise InputError(
                f"register {name!r} value must be an int or a tuple/list of ints "
                f"(up to {MAX_CANDIDATES}); bool and float are rejected, got {value!r}"
            )
        if isinstance(value, int):
            members = [value]
        else:
            members = list(value)
            if not members:
                raise InputError(f"register {name!r} candidate list must not be empty")
            if len(members) > MAX_CANDIDATES:
                raise InputError(
                    f"register {name!r} has {len(members)} candidate values, exceeding the "
                    f"limit of {MAX_CANDIDATES}"
                )
        for member in members:
            _require_int(member, f"register {name!r} candidate", ())
        unique = set(members)
        clean[name] = tuple(sorted(unique))
    return clean


def evaluate(
    ast: Node,
    registers: Mapping[str, int | tuple[int, ...] | list[int]],
    memory: MemorySnapshot,
    *,
    max_combinations: int = MAX_COMBINATIONS,
) -> EvalResult:
    """Evaluate ``ast`` against ``registers`` and the ``memory`` snapshot.

    Args:
        ast: an AST built from the node classes in ``debugexpr.nodes``
            (at most ``MAX_NODES`` distinct nodes).
        registers: mapping of register name to either a Python int or a
            tuple/list of 1..``MAX_CANDIDATES`` possible ints.  bool and
            float values (including NaN/Infinity) are rejected.
        memory: a ``MemorySnapshot``; loads may only touch its regions.
        max_combinations: budget for enumerated candidate combinations;
            exceeding it raises ``CandidateBudgetError`` (unknown).

    Returns:
        ``EvalResult`` with the shared result type, the set of possible
        values, and the memory intervals that may actually be read.

    Raises:
        EvalError: (subclass) any contract violation, with a locatable
            path.  ``CandidateBudgetError`` means "unknown", not an
            illegal input.
    """
    if not isinstance(ast, Node):
        raise InputError(f"ast must be a debugexpr Node, got {ast!r}")
    if not isinstance(registers, Mapping):
        raise InputError(f"registers must be a mapping of str -> int, got {type(registers).__name__}")
    if not isinstance(memory, MemorySnapshot):
        raise InputError(f"memory must be a MemorySnapshot, got {type(memory).__name__}")
    if type(max_combinations) is bool or not isinstance(max_combinations, int) or max_combinations < 1:
        raise InputError(f"max_combinations must be a positive int, got {max_combinations!r}")

    clean_registers = _normalize_registers(registers)

    count = _count_nodes(ast)
    if count > MAX_NODES:
        raise NodeBudgetError(f"AST has {count} distinct nodes, exceeding the limit of {MAX_NODES}")

    ctx = _Ctx(clean_registers, memory, max_combinations)
    maybe = _eval(ast, ctx, ())
    return EvalResult(type=maybe.type, candidates=maybe.candidates, reads=tuple(ctx.reads))
