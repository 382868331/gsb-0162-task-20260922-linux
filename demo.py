"""Deterministic offline demo of the debug-expression evaluator.

Builds a fixed memory snapshot and registers, evaluates one real
expression, prints the typed value together with the memory intervals
actually read (evaluation evidence), then triggers one genuine rejection
(a one-past-end dereference) and prints the located error.  A second
section demonstrates a one-shot edit transaction: typed stores with
read-your-writes, the access trace, a rollback on failure, and a commit
that leaves the source snapshot untouched.

Run:  python demo.py
"""

from debugexpr import (
    BinOp,
    Cast,
    Const,
    EvalError,
    Field,
    Load,
    MemorySnapshot,
    PtrAdd,
    PtrConst,
    Reg,
    S16,
    S32,
    Store,
    U8,
    U32,
    evaluate,
    run_transaction,
)


def build_snapshot() -> MemorySnapshot:
    """Deterministic 64-byte 'stack' region with two known fields."""
    data = bytearray(64)
    data[16:20] = (0x00DEAD00).to_bytes(4, "little")  # s32 field at offset 16
    data[32:34] = (-1234).to_bytes(2, "little", signed=True)  # s16 field at offset 32
    return MemorySnapshot({"stack": bytes(data)})


def main() -> None:
    memory = build_snapshot()
    registers = {"index": 16}  # byte offset of the first field

    # expr: load<s32>(stack + reg.index) + cast(field<s16>(stack, 32), s32)
    expr = BinOp(
        "add",
        Load(PtrAdd(PtrConst("stack", 0), Reg("index", U8)), S32),
        Cast(Field(PtrConst("stack", 0), 32, S16), S32),
    )

    result = evaluate(expr, registers, memory)
    print("== normal evaluation ==")
    print("expression : load<s32>(stack + reg.index) + cast(field<s16>(stack, 32), s32)")
    print(f"registers  : {registers}")
    print(f"value      : {result.value.data} (type {result.value.type})")
    print("read evidence (memory intervals actually read):")
    for read in result.reads:
        chunk = memory.read(read.region_id, read.offset, read.length)
        print(
            f"  region={read.region_id!r} offset={read.offset} length={read.length}"
            f" bytes={chunk.hex()}"
        )

    print()
    print("== triggered rejection ==")
    # One-past-end of the 64-byte region: a valid pointer, an invalid load.
    bad = Load(PtrConst("stack", 64), U8)
    try:
        evaluate(bad, registers, memory)
    except EvalError as err:
        print(f"rejected   : {type(err).__name__}: {err}")
        print(f"error path : {err.path!r} (locates the failing AST node)")

    print()
    print("== edit transaction: rollback on failure ==")
    # store u32 at stack+16, read its low byte back into stack+0, then a
    # one-past-end store fails and rolls the whole transaction back.
    txn = run_transaction(
        [
            Store(PtrConst("stack", 16), U32, Const(0x11223344, U32)),
            Store(PtrConst("stack", 0), U8, Load(PtrConst("stack", 16), U8)),
            Store(PtrConst("stack", 64), U8, Const(0x99, U8)),  # one-past-end
        ],
        registers,
        memory,
    )
    print(f"ok         : {txn.ok}")
    print(f"error      : {type(txn.error).__name__}: {txn.error}")
    print("access trace attempted before the failure:")
    for rec in txn.trace:
        print(f"  {rec.kind:5s} region={rec.region_id!r} offset={rec.offset} length={rec.length}")
    print(f"source snapshot still intact: {memory.read('stack', 0, 4).hex() == '00000000'}")

    print()
    print("== edit transaction: commit ==")
    txn = run_transaction(
        [
            Store(PtrConst("stack", 16), U32, Const(0x11223344, U32)),
            Store(PtrConst("stack", 0), U8, Load(PtrConst("stack", 16), U8)),
        ],
        registers,
        memory,
    )
    print(f"ok         : {txn.ok}")
    committed = txn.commit()
    print(f"committed  : stack[0:4]  = {committed.read('stack', 0, 4).hex()} (read-your-writes byte 0x44)")
    print(f"             stack[16:20]= {committed.read('stack', 16, 4).hex()}")
    print(f"source snapshot unchanged : stack[0:4]={memory.read('stack', 0, 4).hex()}")


if __name__ == "__main__":
    main()
