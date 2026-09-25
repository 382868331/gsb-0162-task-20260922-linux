"""Tests for the one-shot edit transaction (undoable edit preview).

Expected memory images are assembled by hand with independent bytearrays
in this file — they never call the implementation under test to derive
expectations.
"""

from __future__ import annotations

import unittest

from debugexpr import (
    AccessRecord,
    And,
    BinOp,
    Const,
    DivisionByZeroError,
    EvalError,
    InputError,
    Load,
    MemorySnapshot,
    Or,
    Pointer,
    PointerError,
    PtrConst,
    RangeError,
    Store,
    TransactionError,
    S8,
    S16,
    S32,
    U8,
    U16,
    U32,
    begin_edit,
    evaluate,
)


def snap_a() -> MemorySnapshot:
    """8-byte region 'a' and 4-byte region 'b', both zeroed."""
    return MemorySnapshot({"a": bytes(8), "b": bytes(4)})


class OverlapTests(unittest.TestCase):
    def test_u8_then_u32_overwrite_in_statement_order(self):
        txn = begin_edit(snap_a())
        txn.store(Store(Pointer("a", 0), U8, 0xAA))
        txn.store(Store(Pointer("a", 0), U32, 0x11223344))  # covers the u8
        committed = txn.commit()

        expected = bytearray(8)
        expected[0] = 0xAA
        expected[0:4] = (0x11223344).to_bytes(4, "little")  # later store wins
        self.assertEqual(committed.read("a", 0, 8), bytes(expected))

    def test_u32_then_u8_partial_overlap(self):
        txn = begin_edit(snap_a())
        txn.store(Store(Pointer("a", 0), U32, 0x11223344))
        txn.store(Store(Pointer("a", 1), U8, 0xFF))  # punches one byte
        committed = txn.commit()

        expected = bytearray(8)
        expected[0:4] = (0x11223344).to_bytes(4, "little")
        expected[1] = 0xFF
        self.assertEqual(committed.read("a", 0, 8), bytes(expected))

    def test_overlapping_store_visible_to_pending_read(self):
        txn = begin_edit(snap_a())
        txn.store(Store(Pointer("a", 2), U16, 0xBEEF))
        txn.store(Store(Pointer("a", 0), U32, 0x11223344))  # overwrites it
        result = txn.evaluate(Load(PtrConst("a", 2), U16))

        expected = bytearray(8)
        expected[2:4] = (0xBEEF).to_bytes(2, "little")
        expected[0:4] = (0x11223344).to_bytes(4, "little")
        want = int.from_bytes(expected[2:4], "little")
        self.assertEqual(result.value.data, want)  # 0x2211
        self.assertEqual(result.value.type, U16)

    def test_cross_region_same_offset_isolated(self):
        txn = begin_edit(snap_a())
        txn.store(Store(Pointer("a", 0), U32, 0xDEADBEEF))
        txn.store(Store(Pointer("b", 0), U8, 0x7F))
        committed = txn.commit()

        expected_a = bytearray(8)
        expected_a[0:4] = (0xDEADBEEF).to_bytes(4, "little")
        expected_b = bytearray(4)
        expected_b[0] = 0x7F
        self.assertEqual(committed.read("a", 0, 8), bytes(expected_a))
        self.assertEqual(committed.read("b", 0, 4), bytes(expected_b))


class StoreValidationTests(unittest.TestCase):
    def test_one_past_end_store_rejected_and_rolls_back(self):
        memory = snap_a()
        txn = begin_edit(memory)
        txn.store(Store(Pointer("a", 0), U8, 0x11))  # staged, then lost
        with self.assertRaises(PointerError):
            txn.store(Store(Pointer("a", 8), U8, 0x22))  # offset == size
        self.assertEqual(txn.state, "rolled-back")
        with self.assertRaises(TransactionError):
            txn.commit()
        # Trace keeps the attempted accesses; no partial memory leaks out.
        self.assertEqual(txn.trace, (AccessRecord("write", "a", 0, 1),))
        self.assertEqual(memory.read("a", 0, 8), bytes(8))

    def test_store_spanning_past_end_rejected(self):
        txn = begin_edit(snap_a())
        with self.assertRaises(PointerError):
            txn.store(Store(Pointer("b", 4), U32, 0))  # one-past-end of 'b'
        txn = begin_edit(snap_a())
        with self.assertRaises(PointerError):
            txn.store(Store(Pointer("a", 6), U32, 0))  # 6+4 > 8

    def test_exact_fit_store_accepted(self):
        txn = begin_edit(snap_a())
        txn.store(Store(Pointer("a", 4), U32, 0x01020304))  # 4+4 == size, fits
        txn.store(Store(Pointer("b", 0), U32, 0x05060708))  # fills 'b' exactly
        committed = txn.commit()
        expected_a = bytearray(8)
        expected_a[4:8] = (0x01020304).to_bytes(4, "little")
        self.assertEqual(committed.read("a", 0, 8), bytes(expected_a))
        self.assertEqual(committed.read("b", 0, 4), (0x05060708).to_bytes(4, "little"))

    def test_negative_offset_rejected(self):
        txn = begin_edit(snap_a())
        with self.assertRaises(PointerError):
            txn.store(Store(Pointer("a", -8), U32, 0))

    def test_misaligned_store_rejected(self):
        txn = begin_edit(snap_a())
        with self.assertRaises(PointerError):
            txn.store(Store(Pointer("a", 1), U32, 0))
        txn = begin_edit(snap_a())
        with self.assertRaises(PointerError):
            txn.store(Store(Pointer("a", 2), U32, 0))  # 2 % 4 != 0

    def test_unknown_region_rejected(self):
        txn = begin_edit(snap_a())
        with self.assertRaises(PointerError):
            txn.store(Store(Pointer("nope", 0), U8, 0))

    def test_signed_boundary_values(self):
        # -128 and 127 are the s8 extremes: both accepted.
        txn = begin_edit(snap_a())
        txn.store(Store(Pointer("a", 0), S8, -128))
        txn.store(Store(Pointer("a", 1), S8, 127))
        txn.store(Store(Pointer("a", 2), S8, -1))
        committed = txn.commit()
        expected = bytearray(8)
        expected[0:1] = (-128).to_bytes(1, "little", signed=True)
        expected[1:2] = (127).to_bytes(1, "little", signed=True)
        expected[2:3] = (-1).to_bytes(1, "little", signed=True)
        self.assertEqual(committed.read("a", 0, 8), bytes(expected))

    def test_signed_boundary_overflow_rejected(self):
        for bad in (128, -129):
            txn = begin_edit(snap_a())
            with self.assertRaises(RangeError):
                txn.store(Store(Pointer("a", 0), S8, bad))
            self.assertEqual(txn.state, "rolled-back")

    def test_unsigned_boundary_overflow_rejected(self):
        for bad in (256, -1):
            txn = begin_edit(snap_a())
            with self.assertRaises(RangeError):
                txn.store(Store(Pointer("a", 0), U8, bad))

    def test_non_int_value_rejected(self):
        for bad in (True, 1.5, float("nan"), float("inf"), "1"):
            txn = begin_edit(snap_a())
            with self.assertRaises(InputError):
                txn.store(Store(Pointer("a", 0), U8, bad))

    def test_malformed_statement_rejected(self):
        txn = begin_edit(snap_a())
        with self.assertRaises(InputError):
            txn.store("not a store")
        txn = begin_edit(snap_a())
        with self.assertRaises(InputError):
            txn.store(Store(("a", 0), U8, 1))  # ptr must be a Pointer
        txn = begin_edit(snap_a())
        with self.assertRaises(InputError):
            txn.store(Store(Pointer("a", 0), "u8", 1))  # type must be IntType
        txn = begin_edit(snap_a())
        with self.assertRaises(InputError):
            txn.store(Store(Pointer("a", True), U8, 1))  # bool offset


class WriteReadFailRollbackTests(unittest.TestCase):
    def test_write_read_then_fail_rolls_everything_back(self):
        memory = snap_a()
        memory.read("a", 0, 8)  # touch it once; content must survive intact
        txn = begin_edit(memory)

        txn.store(Store(Pointer("a", 0), S32, -2))
        seen = txn.evaluate(Load(PtrConst("a", 0), S32))
        self.assertEqual(seen.value.data, -2)  # expression read the pending write

        with self.assertRaises(PointerError):
            txn.evaluate(Load(PtrConst("a", 8), U8))  # one-past-end load fails

        self.assertEqual(txn.state, "rolled-back")
        # Trace: the staged write, then the successful s32 read. The failed
        # load performed no access, so nothing further is recorded.
        self.assertEqual(
            txn.trace,
            (
                AccessRecord("write", "a", 0, 4),
                AccessRecord("read", "a", 0, 4),
            ),
        )
        # No partial new memory can leak: commit is impossible, and the
        # caller's snapshot still holds the original bytes.
        with self.assertRaises(TransactionError):
            txn.commit()
        with self.assertRaises(TransactionError):
            txn.evaluate(Load(PtrConst("a", 0), U8))
        with self.assertRaises(TransactionError):
            txn.store(Store(Pointer("a", 0), U8, 1))
        self.assertEqual(memory.read("a", 0, 8), bytes(8))

    def test_failing_expression_with_earlier_reads_keeps_trace(self):
        txn = begin_edit(snap_a())
        txn.store(Store(Pointer("a", 0), U8, 9))
        # and(load<u8>(a,0) != 0, 1/0): the read happens, then div-by-zero.
        expr = And(
            BinOp("bitor", Load(PtrConst("a", 0), U8), Const(0, U8)),
            BinOp("div", Const(1, U8), Const(0, U8)),
        )
        with self.assertRaises(DivisionByZeroError):
            txn.evaluate(expr)
        kinds = [rec.kind for rec in txn.trace]
        self.assertEqual(kinds, ["write", "read"])


class ShortCircuitTests(unittest.TestCase):
    def test_short_circuited_load_has_no_side_effect(self):
        txn = begin_edit(snap_a())
        # or(1, load at one-past-end): right side never runs, no read, no error.
        result = txn.evaluate(Or(Const(1, U8), Load(PtrConst("a", 8), U8)))
        self.assertEqual(result.value.data, 1)
        self.assertEqual(txn.trace, ())  # no access was performed
        self.assertEqual(txn.state, "open")  # transaction still usable
        committed = txn.commit()
        self.assertEqual(committed.read("a", 0, 8), bytes(8))

    def test_unexecuted_store_has_no_side_effect(self):
        # A Store object that is never applied changes nothing.
        pending = Store(Pointer("a", 0), U32, 0xFFFFFFFF)
        txn = begin_edit(snap_a())
        committed = txn.commit()
        self.assertEqual(committed.read("a", 0, 8), bytes(8))
        # And after a rollback, applying further stores is impossible:
        txn2 = begin_edit(snap_a())
        with self.assertRaises(RangeError):
            txn2.store(Store(Pointer("a", 0), U8, 300))
        with self.assertRaises(TransactionError):
            txn2.store(pending)


class CommitDiscardTests(unittest.TestCase):
    def test_commit_returns_new_snapshot_source_untouched(self):
        source_bytes = bytearray(8)
        source_bytes[4] = 0x55
        memory = MemorySnapshot({"a": bytes(source_bytes)})
        txn = begin_edit(memory)
        txn.store(Store(Pointer("a", 4), U8, 0x99))
        committed = txn.commit()

        expected = bytearray(source_bytes)
        expected[4] = 0x99
        self.assertEqual(committed.read("a", 0, 8), bytes(expected))
        # The caller's snapshot and its original byte source are unchanged.
        self.assertEqual(memory.read("a", 0, 8), bytes(source_bytes))
        self.assertEqual(source_bytes[4], 0x55)

    def test_commit_is_one_shot(self):
        txn = begin_edit(snap_a())
        txn.commit()
        with self.assertRaises(TransactionError):
            txn.commit()
        with self.assertRaises(TransactionError):
            txn.discard()
        with self.assertRaises(TransactionError):
            txn.store(Store(Pointer("a", 0), U8, 1))

    def test_discard_drops_everything(self):
        memory = snap_a()
        txn = begin_edit(memory)
        txn.store(Store(Pointer("a", 0), U32, 0xCAFEBABE))
        txn.discard()
        self.assertEqual(txn.state, "discarded")
        self.assertEqual(txn.trace, (AccessRecord("write", "a", 0, 4),))
        with self.assertRaises(TransactionError):
            txn.commit()
        self.assertEqual(memory.read("a", 0, 8), bytes(8))

    def test_readonly_evaluate_unaffected_by_open_transaction(self):
        memory = snap_a()
        txn = begin_edit(memory)
        txn.store(Store(Pointer("a", 0), S16, -5))
        # The plain read-only evaluator on the original snapshot sees old bytes.
        plain = evaluate(Load(PtrConst("a", 0), S16), {}, memory)
        self.assertEqual(plain.value.data, 0)
        # The transaction view sees the pending write.
        preview = txn.evaluate(Load(PtrConst("a", 0), S16))
        self.assertEqual(preview.value.data, -5)
        txn.discard()

    def test_begin_edit_rejects_non_snapshot(self):
        with self.assertRaises(InputError):
            begin_edit({"a": bytes(8)})

    def test_error_paths_are_locatable_eval_errors(self):
        txn = begin_edit(snap_a())
        try:
            txn.store(Store(Pointer("a", 0), S8, 999))
        except EvalError as err:
            self.assertEqual(err.path, ())
        else:  # pragma: no cover - defensive
            self.fail("expected EvalError")


if __name__ == "__main__":
    unittest.main()
