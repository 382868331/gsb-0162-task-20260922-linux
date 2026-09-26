"""Tests for layout-driven field/index access (FieldOf / IndexOf).

Expected addresses and byte values are derived independently in this
file (plain offset arithmetic and int.to_bytes/from_bytes on the test
data), never by calling the layout machinery under test.  The recorded
``reads`` are compared against the exact byte intervals the independent
calculation says must be touched — proving unrelated fields are never
read.
"""

from __future__ import annotations

import unittest

from debugexpr import (
    POINTER,
    ArrayLayout,
    Const,
    EvalError,
    FieldInfo,
    FieldOf,
    IndexOf,
    InputError,
    MemorySnapshot,
    Pointer,
    PointerError,
    PtrConst,
    ReadInterval,
    StructLayout,
    TypeMismatchError,
    S16,
    S32,
    U8,
    U16,
    U32,
    evaluate,
)

REGION_TABLE = ("stack", "heap")


def put(data: bytearray, offset: int, value: int, size: int, signed: bool = False) -> None:
    data[offset:offset + size] = value.to_bytes(size, "little", signed=signed)


def put_ptr(data: bytearray, offset: int, region_tag: int, ptr_offset: int) -> None:
    data[offset] = region_tag
    put(data, offset + 1, ptr_offset, 8)


# ---------------------------------------------------------------- layouts

# struct Point { u8 tag @0; s32 x @1 (unaligned!); u16 y @5 } size 7
POINT = StructLayout("Point", 7, (
    FieldInfo("tag", 0, U8),
    FieldInfo("x", 1, S32),
    FieldInfo("y", 5, U16),
))

# struct Node { u32 id @0; ptr next @4 } size 13
NODE = StructLayout("Node", 13, (
    FieldInfo("id", 0, U32),
    FieldInfo("next", 4, POINTER),
))

# matrix: 3 x (4 x u16)
ROW = ArrayLayout(U16, 4)
MATRIX = ArrayLayout(ROW, 3)

# struct Wrapper { u8 flag @0; matrix m @2 } size 2 + 24 = 26
WRAPPER = StructLayout("Wrapper", 26, (
    FieldInfo("flag", 0, U8),
    FieldInfo("m", 2, MATRIX),
))

EMPTY_ARRAY = ArrayLayout(U32, 0)


def build_memory() -> MemorySnapshot:
    stack = bytearray(64)
    # Point at stack offset 8: tag=0xAB, x=-2 (unaligned @9), y=0xBEEF @13
    put(stack, 8, 0xAB, 1)
    put(stack, 9, -2, 4, signed=True)
    put(stack, 13, 0xBEEF, 2)
    # Wrapper at stack offset 32: flag=7 @32, matrix @34
    put(stack, 32, 7, 1)
    for i in range(3):
        for j in range(4):
            put(stack, 34 + (i * 4 + j) * 2, 1000 + i * 10 + j, 2)
    heap = bytearray(32)
    # Node A at heap 0: id=42, next -> heap offset 16 (Node B)
    put(heap, 0, 42, 4)
    put_ptr(heap, 4, 1, 16)
    # Node B at heap 16: id=43, next -> stack offset 8 (cross-region)
    put(heap, 16, 43, 4)
    put_ptr(heap, 20, 0, 8)
    return MemorySnapshot({"stack": bytes(stack), "heap": bytes(heap)})


MEMORY = build_memory()


def ev(ast, memory=MEMORY, region_table=REGION_TABLE):
    return evaluate(ast, {}, memory, region_table=region_table)


class TestFieldOf(unittest.TestCase):
    def test_scalar_field_reads_only_its_bytes(self):
        # p->x where p = stack+8: address 9, 4 bytes, value -2
        result = ev(FieldOf(PtrConst("stack", 8), POINT, "x"))
        self.assertEqual(result.value.type, S32)
        self.assertEqual(result.value.data, -2)
        self.assertEqual(result.reads, (ReadInterval("stack", 9, 4),))

    def test_unaligned_field_is_byte_exact(self):
        # x @9 is not 4-aligned; layout reads must not enforce alignment
        result = ev(FieldOf(PtrConst("stack", 8), POINT, "x"))
        self.assertEqual(result.value.data, -2)
        # independent byte check
        raw = MEMORY.read("stack", 9, 4)
        self.assertEqual(int.from_bytes(raw, "little", signed=True), -2)

    def test_every_field_independently(self):
        expectations = {"tag": (8, 1, 0xAB), "x": (9, 4, -2), "y": (13, 2, 0xBEEF)}
        for name, (off, size, value) in expectations.items():
            with self.subTest(field=name):
                result = ev(FieldOf(PtrConst("stack", 8), POINT, name))
                self.assertEqual(result.value.data, value)
                self.assertEqual(result.reads, (ReadInterval("stack", off, size),))

    def test_unknown_field_rejected_with_path(self):
        with self.assertRaises(InputError) as cm:
            ev(FieldOf(PtrConst("stack", 8), POINT, "nope"))
        self.assertEqual(cm.exception.path, ())

    def test_field_of_non_pointer_base_rejected(self):
        with self.assertRaises(TypeMismatchError) as cm:
            ev(FieldOf(Const(0, U8), POINT, "x"))
        self.assertEqual(cm.exception.path, ("base",))


class TestIndexOf(unittest.TestCase):
    def test_simple_elements(self):
        # matrix row 0 starts at stack 34; element j at 34 + 2*j
        for j in range(4):
            with self.subTest(j=j):
                result = ev(IndexOf(PtrConst("stack", 34), ROW, Const(j, U8)))
                self.assertEqual(result.value.data, 1000 + j)
                self.assertEqual(result.reads, (ReadInterval("stack", 34 + 2 * j, 2),))

    def test_nested_arrays(self):
        # wrapper.m[2][3]: 34 + (2*4 + 3) * 2 = 56, value 1023
        base = FieldOf(PtrConst("stack", 32), WRAPPER, "m")
        row = IndexOf(base, MATRIX, Const(2, U8))
        leaf = IndexOf(row, ROW, Const(3, U8))
        result = ev(leaf)
        self.assertEqual(result.value.type, U16)
        self.assertEqual(result.value.data, 1023)
        # exactly one 2-byte read at the independently computed address
        self.assertEqual(result.reads, (ReadInterval("stack", 56, 2),))

    def test_aggregate_selection_reads_nothing(self):
        # selecting wrapper.m (aggregate) then m[1] (aggregate): no reads
        # until the final scalar leaf; the whole struct is never loaded.
        base = FieldOf(PtrConst("stack", 32), WRAPPER, "m")
        row = IndexOf(base, MATRIX, Const(1, U8))
        leaf = IndexOf(row, ROW, Const(0, U8))
        result = ev(leaf)
        self.assertEqual(result.value.data, 1010)
        self.assertEqual(result.reads, (ReadInterval("stack", 34 + 4 * 2, 2),))

    def test_out_of_bounds_index_rejected_with_path(self):
        with self.assertRaises(PointerError) as cm:
            ev(IndexOf(PtrConst("stack", 34), ROW, Const(4, U8)))
        self.assertEqual(cm.exception.path, ("index",))

    def test_negative_index_rejected_with_path(self):
        with self.assertRaises(PointerError) as cm:
            ev(IndexOf(PtrConst("stack", 34), ROW, Const(-1, S16)))
        self.assertEqual(cm.exception.path, ("index",))

    def test_empty_array_rejects_every_index(self):
        for index in (0, 1):
            with self.subTest(index=index):
                with self.assertRaises(PointerError) as cm:
                    ev(IndexOf(PtrConst("stack", 0), EMPTY_ARRAY, Const(index, U8)))
                self.assertEqual(cm.exception.path, ("index",))

    def test_non_integer_index_rejected(self):
        with self.assertRaises(TypeMismatchError) as cm:
            ev(IndexOf(PtrConst("stack", 34), ROW, PtrConst("stack", 0)))
        self.assertEqual(cm.exception.path, ("index",))


class TestPointerFields(unittest.TestCase):
    def test_pointer_chain_across_regions(self):
        # nodeA.next -> heap Node B; nodeB.id == 43
        next_ptr = FieldOf(PtrConst("heap", 0), NODE, "next")
        result = ev(FieldOf(next_ptr, NODE, "id"))
        self.assertEqual(result.value.data, 43)
        self.assertEqual(
            result.reads,
            (ReadInterval("heap", 4, 9), ReadInterval("heap", 16, 4)),
        )

    def test_pointer_to_another_region(self):
        # nodeB.next -> stack offset 8 (a Point); read its tag
        next_ptr = FieldOf(PtrConst("heap", 16), NODE, "next")
        self.assertEqual(ev(next_ptr).value.data, Pointer("stack", 8))
        result = ev(FieldOf(next_ptr, POINT, "tag"))
        self.assertEqual(result.value.data, 0xAB)
        self.assertEqual(
            result.reads,
            (ReadInterval("heap", 20, 9), ReadInterval("stack", 8, 1)),
        )

    def test_bad_region_tag_rejected_with_path(self):
        heap = bytearray(MEMORY.read("heap", 0, 32))
        put_ptr(heap, 4, 7, 0)  # tag 7 not in the 2-entry region table
        memory = MemorySnapshot({"stack": MEMORY.read("stack", 0, 64), "heap": bytes(heap)})
        with self.assertRaises(PointerError) as cm:
            ev(FieldOf(PtrConst("heap", 0), NODE, "next"), memory=memory)
        self.assertEqual(cm.exception.path, ())

    def test_pointer_field_to_unknown_region_rejected(self):
        # region table names a region missing from the snapshot
        heap = bytearray(MEMORY.read("heap", 0, 32))
        put_ptr(heap, 4, 2, 0)
        memory = MemorySnapshot({"stack": MEMORY.read("stack", 0, 64), "heap": bytes(heap)})
        with self.assertRaises(PointerError):
            ev(FieldOf(PtrConst("heap", 0), NODE, "next"),
               memory=memory, region_table=("stack", "heap", "ghost"))


class TestLayoutValidation(unittest.TestCase):
    def test_plain_struct_rejects_overlap(self):
        with self.assertRaises(ValueError):
            StructLayout("Bad", 8, (
                FieldInfo("a", 0, U32),
                FieldInfo("b", 2, U32),
            ))

    def test_union_allows_overlap(self):
        union = StructLayout("U", 4, (
            FieldInfo("word", 0, U32),
            FieldInfo("bytes", 0, ArrayLayout(U8, 4)),
        ), union=True)
        stack = bytearray(64)
        put(stack, 0, 0x01020304, 4)
        memory = MemorySnapshot({"stack": bytes(stack)})
        word = evaluate(FieldOf(PtrConst("stack", 0), union, "word"), {}, memory)
        self.assertEqual(word.value.data, 0x01020304)
        byte1 = evaluate(
            IndexOf(FieldOf(PtrConst("stack", 0), union, "bytes"),
                    ArrayLayout(U8, 4), Const(1, U8)),
            {}, memory,
        )
        self.assertEqual(byte1.value.data, 0x03)  # little-endian
        self.assertEqual(byte1.reads, (ReadInterval("stack", 1, 1),))

    def test_field_beyond_struct_size_rejected(self):
        with self.assertRaises(ValueError):
            StructLayout("Bad", 4, (FieldInfo("a", 2, U32),))

    def test_duplicate_field_name_rejected(self):
        with self.assertRaises(ValueError):
            StructLayout("Bad", 8, (
                FieldInfo("a", 0, U8),
                FieldInfo("a", 1, U8),
            ))

    def test_negative_length_rejected(self):
        with self.assertRaises(ValueError):
            ArrayLayout(U8, -1)

    def test_bool_offset_rejected(self):
        with self.assertRaises(ValueError):
            FieldInfo("a", True, U8)


class TestLValueBoundary(unittest.TestCase):
    def test_aggregate_top_level_rejected(self):
        with self.assertRaises(TypeMismatchError):
            ev(FieldOf(PtrConst("stack", 32), WRAPPER, "m"))

    def test_node_budget_counts_new_nodes(self):
        leaf = IndexOf(
            IndexOf(FieldOf(PtrConst("stack", 32), WRAPPER, "m"), MATRIX, Const(0, U8)),
            ROW, Const(0, U8),
        )
        self.assertEqual(ev(leaf).value.data, 1000)


class TestScalarApiPreserved(unittest.TestCase):
    """The original scalar API works unchanged with the new keyword absent."""

    def test_positional_scalar_eval(self):
        from debugexpr import Load, S32
        stack = bytearray(64)
        put(stack, 16, 0x00DEAD00, 4)
        memory = MemorySnapshot({"stack": bytes(stack)})
        result = evaluate(Load(PtrConst("stack", 16), S32), {}, memory)
        self.assertEqual(result.value.data, 0x00DEAD00)
        self.assertEqual(result.reads, (ReadInterval("stack", 16, 4),))


if __name__ == "__main__":
    unittest.main()
