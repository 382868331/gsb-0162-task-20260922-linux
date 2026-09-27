"""Deterministic offline demo of the debug-expression evaluator.

Builds a fixed sparse memory snapshot (multiple mapped segments with
per-segment read permission and an unmapped hole), evaluates one real
expression that spans two adjacent readable segments, prints the typed
value together with the precise per-segment read evidence, then triggers
two genuine, distinguishable rejections:

* touching an unmapped hole ........... UnmappedAddressError;
* touching a mapped non-readable page . ReadPermissionError.

Run:  python demo.py
"""

from debugexpr import (
    And,
    BinOp,
    Cast,
    Const,
    Field,
    Load,
    MemorySegment,
    MemorySnapshot,
    PtrConst,
    ReadPermissionError,
    U16,
    U8,
    U32,
    UnmappedAddressError,
    evaluate,
)


def build_snapshot() -> MemorySnapshot:
    """Sparse region ``io`` (mapped extent 32):

    [0,6)   readable: six deterministic bytes;
    [6,8)   readable: two deterministic bytes (adjacent, no gap);
    [8,16)  unmapped hole;
    [16,24) readable: deterministic field bytes;
    [24,32) mapped but non-readable (a guard page).
    """
    head = bytes([0x10, 0x20, 0x30, 0x40, 0xEE, 0xFF])
    neck = bytes([0x01, 0x02])
    field = bytearray(8)
    field[0:4] = (0x00DEAD00).to_bytes(4, "little")  # s32 at 16
    field[4:6] = (1234).to_bytes(2, "little")  # u16 at 20
    return MemorySnapshot(
        {
            "io": [
                MemorySegment(0, head, readable=True),
                MemorySegment(6, neck, readable=True),
                MemorySegment(16, bytes(field), readable=True),
                MemorySegment(24, b"guard!!!"[:8], readable=False),
            ]
        }
    )


def main() -> None:
    memory = build_snapshot()
    registers = {"index": 16}

    # Aligned u32 load at offset 4 spans adjacent readable segments
    # [4,6) and [6,8); the u16 field lives entirely inside [16,24).
    expr = BinOp(
        "add",
        Load(PtrConst("io", 4), U32),
        Cast(Field(PtrConst("io", 0), 20, U16), U32),
    )

    result = evaluate(expr, registers, memory)
    print("== normal evaluation ==")
    print("expression : load<u32>(io + 4) + cast(field<u16>(io, 20), u32)")
    print("registers  :", registers)
    print("value      :", result.value.data, f"(type {result.value.type})")
    print("read evidence (one record per backing segment, address order):")
    for read in result.reads:
        chunk = memory.read(read.region_id, read.offset, read.length)
        print(
            f"  region={read.region_id!r} offset={read.offset} length={read.length}"
            f" bytes={chunk.hex()}"
        )

    print()
    print("== unmapped hole rejected ==")
    try:  # aligned u32 at offset 8 sits inside the unmapped hole [8,16)
        evaluate(Load(PtrConst("io", 8), U32), registers, memory)
    except UnmappedAddressError as err:
        print(f"rejected   : {type(err).__name__}: {err}")
        print(f"error path : {err.path!r}")

    print()
    print("== non-readable mapping rejected ==")
    try:  # u8 at offset 24 touches the mapped but non-readable guard page
        evaluate(Load(PtrConst("io", 24), U8), registers, memory)
    except ReadPermissionError as err:
        print(f"rejected   : {type(err).__name__}: {err}")
        print(f"error path : {err.path!r}")

    print()
    print("== short-circuited branch never reads ==")
    # The hole load is behind a short-circuited branch: it never runs,
    # so no unmapped-address error is raised and no read is recorded.
    skipped = evaluate(And(Const(0, U8), Load(PtrConst("io", 8), U8)), registers, memory)
    print(f"value      : {skipped.value.data} reads={skipped.reads}")


if __name__ == "__main__":
    main()
