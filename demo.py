"""Deterministic offline demo of the debug-expression evaluator.

Run:  python demo.py

Scenarios (all real computations, no precomputed answers or sleeps):

1. normal candidate-set evaluation: a register with several candidate
   values yields the exact set of feasible results plus the union of the
   memory intervals actually read;
2. conditional branch with an unknown condition: both branches are
   explored, but only the memory a branch can actually reach is read;
3. short-circuit and a *possible* division by zero: the error is
   reported for the assignments that reach it, not as a definite failure;
4. a definite rejection (one-past-end dereference) with its located path;
5. the candidate-combination budget: over-budget input becomes
   ``unknown`` (or raises CandidateBudgetError on request).
"""

from debugexpr import (
    And,
    BinOp,
    CandidateBudgetError,
    Const,
    EvalError,
    EvalResult,
    If,
    Load,
    MemorySnapshot,
    PtrAdd,
    PtrConst,
    Reg,
    S32,
    U8,
    evaluate,
)


def build_snapshot() -> MemorySnapshot:
    """Deterministic 64-byte region: m[i] == i (little-endian fields)."""
    return MemorySnapshot({"m": bytes(range(64))})


def show(title: str) -> None:
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


def dump(result: EvalResult) -> None:
    print(f"status : {result.status}")
    print("values :", [v.data for v in result.results])
    if result.errors:
        print("errors :")
        for err in result.errors:
            print(f"  - {type(err).__name__} at {err.path}: {err.message}")
    if result.reads:
        print("reads  :", [(r.region_id, r.offset, r.length) for r in result.reads])


def main() -> None:
    memory = build_snapshot()

    # 1) Candidate register: offset in {0, 8, 16} -> three u32 loads.
    show("1. normal: candidate register -> exact feasible result set")
    expr = Load(PtrAdd(PtrConst("m", 0), Reg("off", U8)), S32)
    result = evaluate(expr, {"off": (0, 8, 16)}, memory)
    print("expr   : load<s32>(m + off),  off in {0, 8, 16}")
    dump(result)

    # 2) Unknown condition: both branches explored, only feasible reads.
    show("2. conditional: unknown condition reads only reachable memory")
    branch = If(
        Reg("c", U8),
        Load(PtrConst("m", 0), U8),     # taken when c != 0
        Load(PtrConst("m", 8), U8),     # taken when c == 0
    )
    result = evaluate(branch, {"c": (0, 1)}, memory)
    print("expr   : if c then load(m,0) else load(m,8),  c in {0, 1}")
    dump(result)

    # Same construct, but the else branch dereferences one-past-end.
    show("2b. unknown condition, one branch can fail -> possible_error")
    branch_bad = If(
        Reg("c", U8),
        Load(PtrConst("m", 0), U8),
        Load(PtrConst("m", 64), U8),    # one-past-end: only fails when c == 0
    )
    result = evaluate(branch_bad, {"c": (0, 1)}, memory)
    print("expr   : if c then load(m,0) else load(m,64),  c in {0, 1}")
    dump(result)

    # 3) Short-circuit suppresses a possible division by zero; then a
    #    non-short-circuited expression reports the possible error.
    show("3. short-circuit vs. possible division by zero")
    risky_div = BinOp("div", Const(6, U8), Reg("z", U8))
    guarded = And(Reg("gate", U8), risky_div)
    result = evaluate(guarded, {"gate": (0, 1), "z": (0, 2, 3)}, memory)
    print("expr   : gate and (6 / z),  gate in {0,1}, z in {0,2,3}")
    dump(result)
    print("note   : gate=0 never divides (short-circuit); 6/2=3, 6/3=6 -> u8 truth 1")

    # 4) Definite rejection: the error occurs for every assignment.
    show("4. definite error: one-past-end dereference always fails")
    bad = Load(PtrConst("m", 64), U8)
    try:
        evaluate(bad, {"irrelevant": (1, 2, 3)}, memory)
    except EvalError as err:
        print(f"raised : {type(err).__name__}: {err.message}")
        print(f"path   : {err.path}")

    # 5) Budget: 8 * 8 * 2 = 128 > 64 worlds -> unknown; "raise" mode.
    show("5. candidate budget: 128 > 64 combinations")
    over = BinOp(
        "add",
        BinOp("add", Reg("a", U8), Reg("b", U8)),
        Reg("c", U8),
    )
    big_regs = {"a": tuple(range(8)), "b": tuple(range(8)), "c": (0, 1)}
    result = evaluate(over, big_regs, memory)
    print("expr   : a + b + c,  |a|=8 |b|=8 |c|=2 (128 assignments)")
    dump(result)
    try:
        evaluate(over, big_regs, memory, on_budget_overrun="raise")
    except CandidateBudgetError as err:
        print(f"raise  : {type(err).__name__}: {err.message}")
    # Structurally illegal input is still rejected when over budget.
    try:
        evaluate(Const("not-an-int", U8), big_regs, memory)
    except EvalError as err:
        print(f"illegal: {type(err).__name__} still raised despite over-budget input")


if __name__ == "__main__":
    main()
