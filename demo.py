"""Deterministic offline demo of the debug-expression evaluator.

Builds a fixed memory snapshot and registers, evaluates one real
expression, prints the typed value together with the memory intervals
actually read (evaluation evidence), then triggers one genuine rejection
(a one-past-end dereference) and prints the located error.

The second half demonstrates the one-shot edit transaction: typed stores
are staged on top of the snapshot, expressions read the pending values,
and the caller commits; then a failing transaction rolls back and shows
its access trace while the caller's snapshot stays untouched.

Run:  python demo.py
"""

from debugexpr import (
    BinOp,
    Cast,
    EvalError,
    Field,
    Load,
    MemorySnapshot,
    Pointer,
    PtrAdd,
    PtrConst,
    Reg,
    Store,
    S16,
    S32,
    U8,
    begin_edit,
    evaluate,
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
    print("== edit transaction: preview, then commit ==")
    txn = begin_edit(memory)
    txn.store(Store(Pointer("stack", 16), S32, -7))  # overwrite the s32 field
    txn.store(Store(Pointer("stack", 17), U8, 0x01))  # overlapping u8 on top
    preview = txn.evaluate(Load(PtrConst("stack", 16), S32))
    print(f"staged     : store<s32>(stack+16, -7); store<u8>(stack+17, 0x01)")
    print(f"preview    : load<s32>(stack+16) = {preview.value.data} (pending write visible)")
    print("access trace so far:")
    for rec in txn.trace:
        print(f"  {rec.kind:5s} region={rec.region_id!r} offset={rec.offset} length={rec.length}")
    committed = txn.commit()
    print(f"committed  : new snapshot bytes[16:20] = {committed.read('stack', 16, 4).hex()}")
    print(f"original   : caller snapshot bytes[16:20] = {memory.read('stack', 16, 4).hex()} (untouched)")

    print()
    print("== edit transaction: write, read, fail -> rollback ==")
    txn2 = begin_edit(memory)
    txn2.store(Store(Pointer("stack", 32), S16, 999))
    seen = txn2.evaluate(Load(PtrConst("stack", 32), S16))
    print(f"staged     : store<s16>(stack+32, 999); preview read back {seen.value.data}")
    try:
        txn2.store(Store(Pointer("stack", 64), U8, 1))  # one-past-end: rejected
    except EvalError as err:
        print(f"rejected   : {type(err).__name__}: {err}")
    print(f"state      : {txn2.state} (whole transaction rolled back)")
    print("access trace of what was attempted:")
    for rec in txn2.trace:
        print(f"  {rec.kind:5s} region={rec.region_id!r} offset={rec.offset} length={rec.length}")
    try:
        txn2.commit()
    except EvalError as err:
        print(f"commit     : {type(err).__name__}: {err}")
    after = evaluate(Load(PtrConst("stack", 32), S16), {}, memory)
    print(f"original   : load<s16>(stack+32) on caller snapshot = {after.value.data} (no leak)")


if __name__ == "__main__":
    main()
