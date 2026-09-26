"""Tests for layout-driven field/subscript navigation (p.field, a[index]).

Expected addresses and byte values are computed independently in this
file (plain integer arithmetic and bytes slicing, not via the core under
test) and cross-checked against the evaluator's results and its actual
read log, proving that unrelated fields are never read.
"""

from __future__ import annotations

import unittest

from debugexpr import (
    ArrayLayout,
    Const,
    Deref,
    EvalError,
    FieldRef,
    IndexRef,
    InputError,
    IntLayout,
    MemorySnapshot,
    Pointer,
    PointerError,
    PtrConst,
    PtrLayout,
    StructField,
    StructLayout,
    TypeMismatchError,
    U8,
    U16,
    U32,
    evaluate,
)


def region_bytes(size: int, writes: dict[int, bytes]) -> bytes:
    data = bytearray(size)
    for offset, chunk in writes.items():
        data[offset : offset + len(chunk)] = chunk
    return bytes(data)


# ---- independent reference (does not import the core's address logic) ----

def ref_field_addr(base: int, offset: int) -> int:
    return base + offset


def ref_elem_addr(base: int, index: int, elem_size: int) -> int:
    return base + index * elem_size


def ref_int_value(data: bytes, offset: int, size: int, signed: bool) -> int:
    return int.from_bytes(data[offset : offset + size], "little", signed=signed)


class LayoutDefinitionTests(unittest.TestCase):
    def test_plain_struct_rejects_overlap(self):
        with self.assertRaises(ValueError):
            StructLayout((
                StructField("a", 0, IntLayout(U32)),
                StructField("b", 2, IntLayout(U16)),  # overlaps a
            ))

    def test_union_allows_overlap(self):
        layout = StructLayout(
            (
                StructField("word", 0, IntLayout(U32)),
                StructField("byte0", 0, IntLayout(U8)),
                StructField("byte3", 3, IntLayout(U8)),
            ),
            union=True,
        )
        self.assertEqual(layout.size, 4)

    def test_struct_size_rounds_up_to_align(self):
        layout = StructLayout((StructField("a", 0, IntLayout(U8)),), align=4)
        self.assertEqual(layout.size, 4)

    def test_explicit_size_must_cover_fields(self):
        with self.assertRaises(ValueError):
            StructLayout((StructField("a", 4, IntLayout(U32)),), size=4)

    def test_empty_array_is_legal_with_zero_size(self):
        layout = ArrayLayout(IntLayout(U16), 0)
        self.assertEqual(layout.size, 0)

    def test_bad_definitions_rejected(self):
        with self.assertRaises(ValueError):
            ArrayLayout(IntLayout(U8), -1)
        with self.assertRaises(ValueError):
            StructField("x", -1, IntLayout(U8))
        with self.assertRaises(ValueError):
            IntLayout(U8, align=0)
        with self.assertRaises(ValueError):
            StructLayout((
                StructField("x", 0, IntLayout(U8)),
                StructField("x", 1, IntLayout(U8)),
            ))


class NavigationTests(unittest.TestCase):
    """Nested arrays + struct fields over a deterministic two-region snapshot."""

    # struct Point { u8 tag @0; u32 code @1 (packed, align 1);
    #                u16 matrix[2][3] @8; ptr heap_obj @24 (align 8) }
    MATRIX_OFFSET = 8
    PTR_OFFSET = 24

    @classmethod
    def setUpClass(cls):
        cls.point = StructLayout(
            (
                StructField("tag", 0, IntLayout(U8)),
                StructField("code", 1, IntLayout(U32, align=1)),
                StructField("matrix", cls.MATRIX_OFFSET, ArrayLayout(ArrayLayout(IntLayout(U16), 3), 2)),
                StructField("heap_obj", cls.PTR_OFFSET, PtrLayout("heap")),
            ),
            align=1,
        )
        matrix_writes = {
            cls.MATRIX_OFFSET + i * 6 + j * 2: (100 + 10 * i + j).to_bytes(2, "little")
            for i in range(2)
            for j in range(3)
        }
        cls.stack_data = region_bytes(
            64,
            {
                0: (0x7F).to_bytes(1, "little"),
                1: (0x01020304).to_bytes(4, "little"),
                cls.PTR_OFFSET: (40).to_bytes(8, "little"),  # -> heap @ 40
                **matrix_writes,
            },
        )
        cls.heap_data = region_bytes(64, {40: (0xCAFE).to_bytes(2, "little")})
        cls.memory = MemorySnapshot({"stack": cls.stack_data, "heap": cls.heap_data})
        cls.base = PtrConst("stack", 0)

    def ev(self, ast):
        return evaluate(ast, {}, self.memory)

    def test_field_selection_reads_nothing(self):
        result = self.ev(FieldRef(self.base, self.point, "code"))
        self.assertEqual(result.reads, ())
        self.assertEqual(result.value.data, Pointer("stack", ref_field_addr(0, 1)))

    def test_unaligned_field_reads_only_its_bytes(self):
        result = self.ev(Deref(FieldRef(self.base, self.point, "code"), IntLayout(U32, align=1)))
        self.assertEqual(result.value.data, ref_int_value(self.stack_data, 1, 4, False))
        # exactly one read: bytes [1, 5) -- tag@0 and matrix@8 untouched
        self.assertEqual(len(result.reads), 1)
        read = result.reads[0]
        self.assertEqual((read.region_id, read.offset, read.length), ("stack", 1, 4))

    def test_nested_array_subscript(self):
        inner = ArrayLayout(IntLayout(U16), 3)
        outer = ArrayLayout(inner, 2)
        i, j = 1, 2
        expr = Deref(
            IndexRef(
                IndexRef(FieldRef(self.base, self.point, "matrix"), outer, Const(i, U8)),
                inner,
                Const(j, U8),
            ),
            IntLayout(U16),
        )
        result = self.ev(expr)
        expected_addr = ref_elem_addr(
            ref_elem_addr(ref_field_addr(0, self.MATRIX_OFFSET), i, 3 * 2), j, 2
        )
        self.assertEqual(
            result.value.data, ref_int_value(self.stack_data, expected_addr, 2, False)
        )
        self.assertEqual(result.value.data, 100 + 10 * i + j)
        # only the final 2-byte element was read; no other matrix element touched
        self.assertEqual(
            [(r.region_id, r.offset, r.length) for r in result.reads],
            [("stack", expected_addr, 2)],
        )

    def test_pointer_field_into_another_region(self):
        expr = Deref(
            Deref(FieldRef(self.base, self.point, "heap_obj"), PtrLayout("heap")),
            IntLayout(U16),
        )
        result = self.ev(expr)
        self.assertEqual(result.value.data, 0xCAFE)
        # read log: 8-byte pointer in 'stack', then 2 bytes in 'heap' at 40
        self.assertEqual(
            [(r.region_id, r.offset, r.length) for r in result.reads],
            [("stack", self.PTR_OFFSET, 8), ("heap", 40, 2)],
        )

    def test_pointer_to_unknown_region_rejected_with_path(self):
        bad = StructLayout((StructField("p", 0, PtrLayout("missing")),))
        memory = MemorySnapshot({"stack": region_bytes(8, {0: (0).to_bytes(8, "little")})})
        with self.assertRaises(PointerError) as cm:
            evaluate(Deref(FieldRef(PtrConst("stack", 0), bad, "p"), PtrLayout("missing")), {}, memory)
        self.assertEqual(cm.exception.path, ())
        self.assertIn("missing", str(cm.exception))

    def test_pointer_past_target_region_rejected_with_path(self):
        memory = MemorySnapshot(
            {"stack": region_bytes(8, {0: (999).to_bytes(8, "little")}), "heap": bytes(16)}
        )
        with self.assertRaises(PointerError) as cm:
            evaluate(
                Deref(FieldRef(PtrConst("stack", 0), self.point.__class__(
                    (StructField("heap_obj", 0, PtrLayout("heap")),)), "heap_obj"), PtrLayout("heap")),
                {},
                memory,
            )
        self.assertEqual(cm.exception.path, ())

    def test_out_of_bounds_subscript_rejected_with_path(self):
        inner = ArrayLayout(IntLayout(U16), 3)
        expr = IndexRef(FieldRef(self.base, self.point, "matrix"), ArrayLayout(inner, 2), Const(2, U8))
        with self.assertRaises(PointerError) as cm:
            self.ev(expr)
        self.assertEqual(cm.exception.path, ("index",))

    def test_empty_array_rejects_every_subscript_with_path(self):
        empty = ArrayLayout(IntLayout(U8), 0)
        expr = IndexRef(PtrConst("stack", 0), empty, Const(0, U8))
        with self.assertRaises(PointerError) as cm:
            self.ev(expr)
        self.assertEqual(cm.exception.path, ("index",))
        self.assertIn("empty array", str(cm.exception))

    def test_unknown_field_rejected_with_path(self):
        with self.assertRaises(InputError) as cm:
            self.ev(FieldRef(self.base, self.point, "nosuch"))
        self.assertEqual(cm.exception.path, ())
        self.assertIn("nosuch", str(cm.exception))

    def test_union_fields_interpret_same_bytes(self):
        union = StructLayout(
            (
                StructField("word", 0, IntLayout(U32)),
                StructField("low", 0, IntLayout(U8)),
            ),
            union=True,
        )
        memory = MemorySnapshot({"r": region_bytes(4, {0: (0x01020304).to_bytes(4, "little")})})
        word = evaluate(Deref(FieldRef(PtrConst("r", 0), union, "word"), IntLayout(U32)), {}, memory)
        low = evaluate(Deref(FieldRef(PtrConst("r", 0), union, "low"), IntLayout(U8)), {}, memory)
        self.assertEqual(word.value.data, 0x01020304)
        self.assertEqual(low.value.data, 0x04)

    def test_deref_of_aggregate_layout_rejected(self):
        with self.assertRaises(InputError):
            self.ev(Deref(FieldRef(self.base, self.point, "matrix"), self.point))

    def test_deref_respects_declared_alignment(self):
        # same packed field, but declared align 4 -> offset 1 is misaligned
        with self.assertRaises(PointerError) as cm:
            self.ev(Deref(FieldRef(self.base, self.point, "code"), IntLayout(U32, align=4)))
        self.assertIn("misaligned", str(cm.exception))

    def test_subscript_base_must_be_pointer(self):
        with self.assertRaises(TypeMismatchError):
            self.ev(IndexRef(Const(0, U8), ArrayLayout(IntLayout(U8), 1), Const(0, U8)))


class ScalarApiRegressionTests(unittest.TestCase):
    """The original scalar API keeps working alongside the new nodes."""

    def test_load_and_field_unchanged(self):
        from debugexpr import Field, Load, S16

        memory = MemorySnapshot({"stack": region_bytes(64, {32: (-7).to_bytes(2, "little", signed=True)})})
        result = evaluate(Load(PtrConst("stack", 32), S16), {}, memory)
        self.assertEqual(result.value.data, -7)
        result2 = evaluate(Field(PtrConst("stack", 0), 32, S16), {}, memory)
        self.assertEqual(result2.value.data, -7)


if __name__ == "__main__":
    unittest.main()
