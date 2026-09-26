"""Deterministic offline demo of layout-driven navigation (p.field / a[index]).

Builds a fixed two-region snapshot plus a caller-supplied type layout,
then evaluates real expressions:

* a nested-array subscript ``point.matrix[1][2]``
* an unaligned packed field ``point.code``
* a pointer field followed into another region ``point.heap_obj->value``

For each, the demo prints the typed value and the actual read log,
proving that field selection itself reads nothing and only the final
fetch touches memory.  It finishes with two genuine rejections (an
out-of-bounds subscript on an empty array and a pointer into an unknown
region), each carrying a locatable path.

Run:  python demo_iteration.py     (finishes in well under 90 seconds)
"""

from debugexpr import (
    ArrayLayout,
    Const,
    Deref,
    EvalError,
    FieldRef,
    IndexRef,
    IntLayout,
    MemorySnapshot,
    PtrConst,
    PtrLayout,
    StructField,
    StructLayout,
    U8,
    U16,
    U32,
    evaluate,
)

POINT = StructLayout(
    (
        StructField("tag", 0, IntLayout(U8)),
        StructField("code", 1, IntLayout(U32, align=1)),       # packed: unaligned
        StructField("matrix", 8, ArrayLayout(ArrayLayout(IntLayout(U16), 3), 2)),
        StructField("heap_obj", 24, PtrLayout("heap")),       # -> another region
    ),
    align=1,
)
HEAP_OBJ = StructLayout((StructField("value", 0, IntLayout(U16)),))


def build_memory() -> MemorySnapshot:
    stack = bytearray(64)
    stack[0] = 0x7F                                            # tag
    stack[1:5] = (0x01020304).to_bytes(4, "little")           # code (packed)
    for i in range(2):                                         # matrix[i][j] = 100+10i+j
        for j in range(3):
            stack[8 + i * 6 + j * 2 : 10 + i * 6 + j * 2] = (100 + 10 * i + j).to_bytes(2, "little")
    stack[24:32] = (40).to_bytes(8, "little")                 # heap_obj -> heap @ 40
    heap = bytearray(64)
    heap[40:42] = (0xCAFE).to_bytes(2, "little")              # heap_obj.value
    return MemorySnapshot({"stack": bytes(stack), "heap": bytes(heap)})


def show(title, expr, memory):
    result = evaluate(expr, {}, memory)
    print(f"-- {title}")
    print(f"   value : {result.value.data} (type {result.value.type})")
    if result.reads:
        for r in result.reads:
            chunk = memory.read(r.region_id, r.offset, r.length)
            print(f"   read  : region={r.region_id!r} offset={r.offset} length={r.length} bytes={chunk.hex()}")
    else:
        print("   read  : <none> (lvalue address only, no memory touched)")
    print()


def main() -> None:
    memory = build_memory()
    base = PtrConst("stack", 0)

    print("== lvalue vs value: field selection reads nothing ==")
    show("address of point.code (no fetch)", FieldRef(base, POINT, "code"), memory)

    print("== unaligned packed field: point.code ==")
    show("point.code (u32 @ offset 1, align 1)",
         Deref(FieldRef(base, POINT, "code"), IntLayout(U32, align=1)), memory)

    print("== nested arrays: point.matrix[1][2] ==")
    inner = ArrayLayout(IntLayout(U16), 3)
    show("point.matrix[1][2] (u16)",
         Deref(IndexRef(IndexRef(FieldRef(base, POINT, "matrix"),
                                 ArrayLayout(inner, 2), Const(1, U8)),
                        inner, Const(2, U8)),
               IntLayout(U16)), memory)

    print("== pointer field into another region: point.heap_obj->value ==")
    show("point.heap_obj->value (u16 in 'heap')",
         Deref(FieldRef(Deref(FieldRef(base, POINT, "heap_obj"), PtrLayout("heap")),
                        HEAP_OBJ, "value"),
               IntLayout(U16)), memory)

    print("== triggered rejections ==")
    empty = ArrayLayout(IntLayout(U8), 0)
    try:
        evaluate(IndexRef(base, empty, Const(0, U8)), {}, memory)
    except EvalError as err:
        print(f"empty-array subscript : {type(err).__name__}: {err}")
    bad = StructLayout((StructField("p", 0, PtrLayout("missing")),))
    try:
        evaluate(Deref(FieldRef(base, bad, "p"), PtrLayout("missing")), {}, memory)
    except EvalError as err:
        print(f"unknown-region pointer: {type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
