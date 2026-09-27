"""Deterministic offline demo of the debug-expression evaluator.

Builds a region out of multiple mapped segments and shows:

* one aligned load crossing two adjacent readable segments, with exact
  per-segment read evidence;
* the two distinct rejections: an unmapped hole vs mapped-but-unreadable;
* a short-circuited branch whose bad load never runs.

Run:  python demo.py
"""

from debugexpr import (
    And,
    Const,
    EvalError,
    Load,
    MemorySegment,
    MemorySnapshot,
   PtrConst,
   U16,
   U32,
   U8,
   evaluate,
)


def build_snapshot():
    # readable [0,8), unreadable [8,12), unmapped hole [12,16),
    # readable [16,20).
    return MemorySnapshot({
        "dev": [
            MemorySegment(0, bytes(range(0x10, 0x18)), readable=True),
            MemorySegment(8, b"\x00\x00\x00\x00", readable=False),
            MemorySegment(16, bytes(range(0x20, 0x24)), readable=True),
        ],
    })


def show(title, snapshot, offset, itype):
    print(f"-- {title} --")
    try:
        result = evaluate(Load(PtrConst("dev", offset), itype), {}, snapshot)
        width = itype.size * 2
        print(f"value    : 0x{result.value.data:0{width}x} ({result.value.type})")
        for read in result.reads:
            chunk = snapshot.read(read.region_id, read.offset, read.length)
            print(
                f"read     : region={read.region_id!r} offset={read.offset} "
                f"length={read.length} bytes={chunk.hex()}"
            )
    except EvalError as err:
        print(f"rejected : {type(err).__name__}: {err.message}")
        print(f"path     : {err.path!r}")
    print()


def main():
    gapped = build_snapshot()

    print("== normal: u32 load crosses two adjacent readable segments ==")
    flat = MemorySnapshot({"dev": [
        MemorySegment(0, b"\x01\x02"),
        MemorySegment(2, b"\x03\x04"),
    ]})
    show("u32 at 0 across [0,2) + [2,4)", flat, 0, U32)

    print("== rejections: hole vs unreadable are distinguished ==")
    show("u32 at 8: fully mapped but segment is unreadable", gapped, 8, U32)
    show("u16 at 14: crosses the unmapped hole [12,16)", gapped, 14, U16)
    show("u32 at 16: entirely readable segment", gapped, 16, U32)

    print("== short circuit: the unexecuted branch triggers no fault ==")
    bad_hole = Load(PtrConst("dev", 14), U16)
    result = evaluate(And(Const(0, U8), bad_hole), {}, gapped)
    print(f"and(0, hole-load) = {result.value.data}; reads={tuple(result.reads)}")


if __name__ == "__main__":
    main()
