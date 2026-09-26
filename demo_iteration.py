"""Deterministic offline demo of layout-driven field/index evaluation.

Builds a fixed two-region snapshot with a caller-supplied struct/array
layout, evaluates real expressions (nested arrays, an unaligned field,
a pointer chain crossing regions), prints the typed values together with
the exact memory intervals read — proving unrelated fields are never
touched — then triggers two genuine rejections (out-of-bounds index,
bad cross-region pointer tag) with their located paths.

Run:  python demo_iteration.py     (finishes in well under 90 seconds)
"""

from debugexpr import (
    POINTER,
    ArrayLayout,
    Const,
    EvalError,
    FieldInfo,
    FieldOf,
    IndexOf,
    MemorySnapshot,
    PtrConst,
    StructLayout,
    S32,
    U8,
    U32,
    U16,
    evaluate,
)

REGION_TABLE = ("stack", "heap")

# Caller-supplied layout (explicit offsets, no C compiler rules):
# struct Packet { u8 kind @0; s32 delta @1 (unaligned); u16 samples[3] @5 }
SAMPLES = ArrayLayout(U16, 3)
PACKET = StructLayout("Packet", 11, (
    FieldInfo("kind", 0, U8),
    FieldInfo("delta", 1, S32),
    FieldInfo("samples", 5, SAMPLES),
))
# struct Link { u32 id @0; ptr next @4 }  (ptr = 1-byte region tag + u64 offset)
LINK = StructLayout("Link", 13, (
    FieldInfo("id", 0, U32),
    FieldInfo("next", 4, POINTER),
))


def build_memory() -> MemorySnapshot:
    stack = bytearray(64)
    base = 10  # Packet lives at stack offset 10
    stack[base] = 3                                    # kind
    stack[base + 1:base + 5] = (-70000).to_bytes(4, "little", signed=True)  # delta @11 (unaligned)
    for i, sample in enumerate((441, 442, 443)):
        stack[base + 5 + 2 * i:base + 7 + 2 * i] = sample.to_bytes(2, "little")
    heap = bytearray(32)
    heap[0:4] = (1001).to_bytes(4, "little")          # Link.id
    heap[4] = 0                                       # next -> region tag 0 ("stack")
    heap[5:13] = (10).to_bytes(8, "little")           # next -> stack offset 10
    return MemorySnapshot({"stack": bytes(stack), "heap": bytes(heap)})


def show(title, result, memory):
    print(f"-- {title}")
    print(f"   value : {result.value.data!r} (type {result.value.type})")
    for read in result.reads:
        chunk = memory.read(read.region_id, read.offset, read.length)
        print(f"   read  : region={read.region_id!r} offset={read.offset} "
              f"length={read.length} bytes={chunk.hex()}")
    if not result.reads:
        print("   read  : <none>")


def main() -> None:
    memory = build_memory()
    packet = PtrConst("stack", 10)

    print("== layout-driven field/index evaluation (offline, deterministic) ==")
    show("unaligned field packet.delta (s32 @11, not 4-aligned)",
         evaluate(FieldOf(packet, PACKET, "delta"), {}, memory, REGION_TABLE), memory)
    show("nested subscript packet.samples[2] (u16)",
         evaluate(IndexOf(FieldOf(packet, PACKET, "samples"), SAMPLES, Const(2, U8)),
                  {}, memory, REGION_TABLE), memory)
    show("cross-region pointer chain link.next.kind (heap -> stack)",
         evaluate(FieldOf(FieldOf(PtrConst("heap", 0), LINK, "next"), PACKET, "kind"),
                  {}, memory, REGION_TABLE), memory)

    print()
    print("== triggered rejections ==")
    try:
        evaluate(IndexOf(FieldOf(packet, PACKET, "samples"), SAMPLES, Const(3, U8)),
                 {}, memory, REGION_TABLE)
    except EvalError as err:
        print(f"out-of-bounds index : {type(err).__name__}: {err}")

    bad = bytearray(memory.read("heap", 0, 32))
    bad[4] = 9  # region tag 9 does not exist in the region table
    corrupted = MemorySnapshot({"stack": memory.read("stack", 0, 64), "heap": bytes(bad)})
    try:
        evaluate(FieldOf(PtrConst("heap", 0), LINK, "next"), {}, corrupted, REGION_TABLE)
    except EvalError as err:
        print(f"bad pointer tag     : {type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
