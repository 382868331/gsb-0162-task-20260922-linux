"""Deterministic offline demo of the debug-expression evaluator.

Builds a fixed memory snapshot and registers, evaluates one real
expression, prints the typed value together with the memory intervals
actually read (evaluation evidence), then triggers one genuine rejection
(a one-past-end dereference) and prints the located error.

Run:  python demo.py
"""

from debugexpr import (
    BinOp,
    Cast,
    EvalError,
    Field,
    Load,
    MemorySnapshot,
    PtrAdd,
    PtrConst,
    Reg,
    S16,
    S32,
    U8,
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


if __name__ == "__main__":
    main()
