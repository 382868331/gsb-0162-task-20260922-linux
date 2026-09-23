"""Deterministic offline demo of the debug-expression evaluator.

Sections, all computed for real (no sleep, no pre-recorded answers):

1. scalar expression: typed value plus memory-read evidence (old API);
2. candidate registers: the expression evaluates to a set of possibilities;
3. Select with an unknown condition: only memory that can actually be
   read is touched, and both branches' reads appear as evidence;
4. boundary: exactly 8 candidates succeed, 9 distinct outcomes become the
   explicit "unknown" (CandidateBudgetError);
5. a genuine rejection: a candidate combination that divides by zero.

Run:  python demo.py
"""

from debugexpr import (
    BinOp,
    CandidateBudgetError,
    Cast,
    Const,
    DivisionByZeroError,
    Field,
    Load,
    MemorySnapshot,
    PtrAdd,
    PtrConst,
    Reg,
    Select,
    S16,
    S32,
    U8,
    evaluate,
)


def build_snapshot() -> MemorySnapshot:
    """Deterministic 64-byte 'stack' region with two known fields and
    a small byte table at offset 0 (stack[i] == i)."""
    data = bytearray(range(64))
    data[16:20] = (0x00DEAD00).to_bytes(4, "little")  # s32 field at offset 16
    data[32:34] = (-1234).to_bytes(2, "little", signed=True)  # s16 field at offset 32
    return MemorySnapshot({"stack": bytes(data)})


def section(title: str) -> None:
    print()
    print(f"== {title} ==")


def main() -> None:
    memory = build_snapshot()

    # 1. Scalar expression -- the original entry point, unchanged.
    section("1. scalar evaluation (single-value registers)")
    registers = {"index": 16}
    expr = BinOp(
        "add",
        Load(PtrAdd(PtrConst("stack", 0), Reg("index", U8)), S32),
        Cast(Field(PtrConst("stack", 0), 32, S16), S32),
    )
    result = evaluate(expr, registers, memory)
    print("expr       : load<s32>(stack + index) + cast(field<s16>(stack, 32), s32)")
    print(f"registers  : {registers}")
    print(f"value      : {result.value.data} (type {result.value.type})")
    for read in result.reads:
        chunk = memory.read(read.region_id, read.offset, read.length)
        print(f"read       : {read.region_id!r} +{read.offset} x{read.length} bytes={chunk.hex()}")

    # 2. Candidate registers -> a set of possible values.
    section("2. candidate registers -> possible result set")
    # index in {0,4,8}: loading u32 at those aligned offsets
    expr2 = Load(PtrAdd(PtrConst("stack", 0), Reg("index", U8)), S32)
    r2 = evaluate(expr2, {"index": (0, 4, 8)}, memory)
    print("expr       : load<s32>(stack + index), index in {0,4,8}")
    print(f"results    : {tuple(hex(v & 0xFFFFFFFF) for v in r2.values)}")
    print("read evidence (only offsets that can actually be accessed):")
    for read in r2.reads:
        print(f"             {read.region_id!r} +{read.offset} x{read.length}")

    # 3. Select with an unknown condition: both branches can execute.
    section("3. Select under an unknown {0,1} condition")
    # flag=1 reads the s32 field at 16 (0x00DEAD00); flag=0 reads s32 at 8
    # (0x0B0A0908).  Both branches have the same type, per the contract.
    expr3 = Select(
        Reg("flag", U8),
        Load(PtrConst("stack", 16), S32),
        Load(PtrConst("stack", 8), S32),
    )
    r3 = evaluate(expr3, {"flag": (0, 1)}, memory)
    print("expr       : flag ? load<s32>(stack+16) : load<s32>(stack+8)")
    print(f"results    : {tuple(hex(v & 0xFFFFFFFF) for v in r3.values)}")
    print("read evidence (taken branch first, other branch second):")
    for read in r3.reads:
        print(f"             {read.region_id!r} +{read.offset} x{read.length}")
    # Determined condition: the other branch's one-past-end load never runs.
    bad_else = Load(PtrConst("stack", 64), U8)
    r3b = evaluate(Select(Const(1, U8), Load(PtrConst("stack", 0), U8), bad_else), {}, memory)
    print(f"flag fixed 1: else branch dereferences one-past-end but is never read")
    print(f"             result={r3b.values}, reads={[(r.offset, r.length) for r in r3b.reads]}")

    # 4. Boundary: 8 distinct outcomes OK; 9 distinct outcomes -> unknown.
    section("4. candidate boundary: 8 ok, 9 distinct outcomes -> unknown")
    r4 = evaluate(
        BinOp("mul", Reg("a", U8), Const(3, U8)), {"a": tuple(range(8))}, memory
    )
    print(f"a*3 for a in 0..7 -> {r4.values}  ({len(r4.candidates)} candidates)")
    try:
        evaluate(
            BinOp("add", Reg("a", U8), Reg("b", U8)),
            {"a": tuple(range(8)), "b": tuple(range(8))},
            memory,
        )
    except CandidateBudgetError as err:
        print("a+b for a,b in 0..7 (15 distinct outcomes):")
        print(f"  {type(err).__name__}: {err.message}")
    # Combination-budget variant of unknown.
    try:
        evaluate(
            BinOp("add", Reg("a", U8), Reg("b", U8)),
            {"a": (0, 1), "b": (0, 1)},
            memory,
            max_combinations=4,
        )
    except CandidateBudgetError as err:
        print("same with max_combinations=4:")
        print(f"  {type(err).__name__}: {err.message}")

    # 5. A genuine illegal-input/execution rejection among candidates.
    section("5. genuine rejection: a reachable combination divides by zero")
    try:
        # divisor may be 0: one real execution path hits division by zero
        evaluate(
            BinOp("div", Const(10, U8), Reg("d", U8)), {"d": (2, 0, 1)}, memory
        )
    except DivisionByZeroError as err:
        print("10 / d with d in {2,0,1}:")
        print(f"  {type(err).__name__}: {err}")
        print(f"  located at {err.path!r}")
    # And the determined-safe counterpart computes normally.
    r5 = evaluate(BinOp("div", Const(10, U8), Reg("d", U8)), {"d": (2, 5)}, memory)
    print(f"10 / d with d in {{2,5}} -> {r5.values}")


if __name__ == "__main__":
    main()
