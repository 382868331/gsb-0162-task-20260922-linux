"""Tests for the one-shot edit transaction (``run_transaction``).

Expected memory contents are built by hand with independent bytearray
manipulation in this file — the expectations never call the code under
test.  Every test that runs a transaction also verifies the caller's
source snapshot is byte-for-byte unchanged afterwards.
"""

from __future__ import annotations

import unittest

from debugexpr import (
    MAX_STATEMENTS,
    AccessRecord,
    And,
    BinOp,
    Const,
    InputError,
    Load,
    MemorySnapshot,
    PointerError,
    PtrConst,
    RangeError,
    Reg,
    S8,
    S16,
    S32,
    Store,
    TypeMismatchError,
    U8,
    U16,
    U32,
    evaluate,
    run_transaction,
)

# Deterministic source regions: "mem" = bytes 0..15, "other" = 32 zero bytes.
MEM_BYTES = bytes(range(16))
OTHER_BYTES = bytes(32)


def make_snapshot() -> MemorySnapshot:
    return MemorySnapshot({"mem": MEM_BYTES, "other": OTHER_BYTES})


def snapshot_bytes(snapshot: MemorySnapshot, region_id: str) -> bytes:
    return snapshot.read(region_id, 0, snapshot.region_size(region_id))


class TransactionOverlapTests(unittest.TestCase):
    def test_u8_overlapping_u32_same_region_statement_order_wins(self):
        # store u32 0xAABBCCDD at mem+0, then u8 0xEE at mem+1 (overlaps).
        snapshot = make_snapshot()
        result = run_transaction(
            [
                Store(PtrConst("mem", 0), U32, Const(0xAABBCCDD, U32)),
                Store(PtrConst("mem", 1), U8, Const(0xEE, U8)),
            ],
            {},
            snapshot,
        )
        self.assertTrue(result.ok, result.error)
        # Independent expectation: patch a plain bytearray by hand.
        expected = bytearray(MEM_BYTES)
        expected[0:4] = (0xAABBCCDD).to_bytes(4, "little")
        expected[1] = 0xEE  # later statement overwrites the overlapping byte
        committed = result.commit()
        self.assertEqual(snapshot_bytes(committed, "mem"), bytes(expected))
        self.assertEqual(snapshot_bytes(committed, "other"), OTHER_BYTES)
        # Source snapshot untouched.
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)
        # Trace: two writes, in statement order.
        self.assertEqual(
            result.trace,
            (
                AccessRecord("write", "mem", 0, 4),
                AccessRecord("write", "mem", 1, 1),
            ),
        )

    def test_u32_store_after_u8_overwrites_all_four_bytes(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [
                Store(PtrConst("mem", 4), U8, Const(0x7F, U8)),
                Store(PtrConst("mem", 4), U32, Const(0x01020304, U32)),
            ],
            {},
            snapshot,
        )
        self.assertTrue(result.ok, result.error)
        expected = bytearray(MEM_BYTES)
        expected[4] = 0x7F
        expected[4:8] = (0x01020304).to_bytes(4, "little")  # full overwrite
        self.assertEqual(snapshot_bytes(result.commit(), "mem"), bytes(expected))
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_cross_region_same_offset_does_not_interfere(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [
                Store(PtrConst("mem", 8), U32, Const(0x11111111, U32)),
                Store(PtrConst("other", 8), U32, Const(0x22222222, U32)),
            ],
            {},
            snapshot,
        )
        self.assertTrue(result.ok, result.error)
        expected_mem = bytearray(MEM_BYTES)
        expected_mem[8:12] = (0x11111111).to_bytes(4, "little")
        expected_other = bytearray(OTHER_BYTES)
        expected_other[8:12] = (0x22222222).to_bytes(4, "little")
        committed = result.commit()
        self.assertEqual(snapshot_bytes(committed, "mem"), bytes(expected_mem))
        self.assertEqual(snapshot_bytes(committed, "other"), bytes(expected_other))
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)
        self.assertEqual(snapshot_bytes(snapshot, "other"), OTHER_BYTES)


class TransactionBoundaryTests(unittest.TestCase):
    def test_write_ending_exactly_at_region_end_is_legal(self):
        snapshot = make_snapshot()
        # u32 at offset 12 of a 16-byte region ends exactly at the end.
        result = run_transaction(
            [Store(PtrConst("mem", 12), U32, Const(0xCAFEBABE, U32))], {}, snapshot
        )
        self.assertTrue(result.ok, result.error)
        expected = bytearray(MEM_BYTES)
        expected[12:16] = (0xCAFEBABE).to_bytes(4, "little")
        self.assertEqual(snapshot_bytes(result.commit(), "mem"), bytes(expected))
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_one_past_end_store_is_rejected_and_rolls_back(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [
                Store(PtrConst("mem", 0), U8, Const(0x55, U8)),
                Store(PtrConst("mem", 16), U8, Const(0x66, U8)),  # one-past-end
            ],
            {},
            snapshot,
        )
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, PointerError)
        # Trace shows the attempted accesses: the first write, nothing more.
        self.assertEqual(result.trace, (AccessRecord("write", "mem", 0, 1),))
        # No partial memory leaks: commit is unavailable.
        with self.assertRaises(InputError):
            result.commit()
        with self.assertRaises(InputError):
            result.discard()
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_store_crossing_region_end_is_rejected(self):
        snapshot = make_snapshot()
        # u32 at offset 14 needs bytes 14..17, region ends at 16.
        result = run_transaction(
            [Store(PtrConst("mem", 14), U32, Const(1, U32))], {}, snapshot
        )
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, PointerError)
        self.assertEqual(result.trace, ())
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_misaligned_store_is_rejected(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [Store(PtrConst("mem", 2), U32, Const(1, U32))], {}, snapshot
        )
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, PointerError)
        self.assertIn("alignment", str(result.error))
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_unknown_region_store_is_rejected(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [Store(PtrConst("nope", 0), U8, Const(1, U8))], {}, snapshot
        )
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, PointerError)
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)


class TransactionSignednessTests(unittest.TestCase):
    def test_signed_boundaries_accepted(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [
                Store(PtrConst("mem", 0), S8, Const(127, S8)),
                Store(PtrConst("mem", 1), S8, Const(-128, S8)),
                Store(PtrConst("mem", 2), U8, Const(255, U8)),
                Store(PtrConst("mem", 3), U8, Const(0, U8)),
            ],
            {},
            snapshot,
        )
        self.assertTrue(result.ok, result.error)
        expected = bytearray(MEM_BYTES)
        expected[0] = 127
        expected[1] = (-128) & 0xFF
        expected[2] = 255
        expected[3] = 0
        self.assertEqual(snapshot_bytes(result.commit(), "mem"), bytes(expected))
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_signed_boundary_overflow_rejected(self):
        snapshot = make_snapshot()
        for bad_value in (Const(128, S16), Const(-129, S16), Const(256, U16), Const(-1, S16)):
            with self.subTest(value=bad_value):
                # Store type is s8/u8; the value AST itself is well-formed
                # but its value does not fit the declared store type.
                store_type = U8 if bad_value.value in (256, -1) else S8
                result = run_transaction(
                    [Store(PtrConst("mem", 0), store_type, bad_value)], {}, snapshot
                )
                self.assertFalse(result.ok)
                self.assertIsInstance(result.error, RangeError)
                self.assertEqual(result.trace, ())
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)


class TransactionReadYourWritesTests(unittest.TestCase):
    def test_later_statement_reads_earlier_pending_write_then_failure_rolls_back(self):
        snapshot = make_snapshot()
        # 1) store u32 0x11223344 at mem+4
        # 2) store u8 at mem+0 whose value is load<u8>(mem+4) -> must see 0x44
        # 3) failing store (one-past-end) -> whole transaction rolls back
        result = run_transaction(
            [
                Store(PtrConst("mem", 4), U32, Const(0x11223344, U32)),
                Store(PtrConst("mem", 0), U8, Load(PtrConst("mem", 4), U8)),
                Store(PtrConst("mem", 16), U8, Const(0x99, U8)),
            ],
            {},
            snapshot,
        )
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, PointerError)
        # Trace: write(4,4B), the read-back load, write(0,1B); the third
        # store validated its pointer and failed before writing.
        self.assertEqual(
            result.trace,
            (
                AccessRecord("write", "mem", 4, 4),
                AccessRecord("read", "mem", 4, 1),
                AccessRecord("write", "mem", 0, 1),
            ),
        )
        # Rolled back: no memory obtainable, source untouched.
        with self.assertRaises(InputError):
            result.commit()
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_read_your_writes_commits_observed_value(self):
        snapshot = make_snapshot()
        # Same first two statements, no failure: the second store must have
        # observed the pending 0x44 byte, and the commit proves it.
        result = run_transaction(
            [
                Store(PtrConst("mem", 4), U32, Const(0x11223344, U32)),
                Store(PtrConst("mem", 0), U8, Load(PtrConst("mem", 4), U8)),
            ],
            {},
            snapshot,
        )
        self.assertTrue(result.ok, result.error)
        expected = bytearray(MEM_BYTES)
        expected[4:8] = (0x11223344).to_bytes(4, "little")
        expected[0] = 0x44  # low byte of the pending u32, read back little-endian
        self.assertEqual(snapshot_bytes(result.commit(), "mem"), bytes(expected))
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_value_expression_can_use_registers_and_arithmetic(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [
                Store(
                    PtrConst("mem", 8),
                    S16,
                    BinOp("add", Reg("base", S16), Const(5, S16)),
                )
            ],
            {"base": 100},
            snapshot,
        )
        self.assertTrue(result.ok, result.error)
        expected = bytearray(MEM_BYTES)
        expected[8:10] = (105).to_bytes(2, "little", signed=True)
        self.assertEqual(snapshot_bytes(result.commit(), "mem"), bytes(expected))
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)


class TransactionShortCircuitTests(unittest.TestCase):
    def test_short_circuited_load_has_no_side_effect_and_no_trace(self):
        snapshot = make_snapshot()
        # And(0, load at one-past-end): right side never evaluates, so the
        # invalid load neither fails nor appears in the trace; the store
        # itself proceeds with value 0.
        result = run_transaction(
            [
                Store(
                    PtrConst("mem", 0),
                    U8,
                    And(Const(0, U8), Load(PtrConst("mem", 16), U8)),
                )
            ],
            {},
            snapshot,
        )
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.trace, (AccessRecord("write", "mem", 0, 1),))
        expected = bytearray(MEM_BYTES)
        expected[0] = 0
        self.assertEqual(snapshot_bytes(result.commit(), "mem"), bytes(expected))
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_statements_after_failure_never_execute(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [
                Store(PtrConst("mem", 16), U8, Const(1, U8)),  # fails
                Store(PtrConst("mem", 0), U8, Const(2, U8)),  # never runs
            ],
            {},
            snapshot,
        )
        self.assertFalse(result.ok)
        # No write record for the statement after the failure.
        self.assertEqual(result.trace, ())
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)


class TransactionSettlementTests(unittest.TestCase):
    def test_discard_leaves_no_way_to_obtain_memory(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [Store(PtrConst("mem", 0), U8, Const(0xAB, U8))], {}, snapshot
        )
        self.assertTrue(result.ok, result.error)
        result.discard()
        with self.assertRaises(InputError):
            result.commit()
        with self.assertRaises(InputError):
            result.discard()
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_commit_is_one_shot(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [Store(PtrConst("mem", 0), U8, Const(0xAB, U8))], {}, snapshot
        )
        committed = result.commit()
        expected = bytearray(MEM_BYTES)
        expected[0] = 0xAB
        self.assertEqual(snapshot_bytes(committed, "mem"), bytes(expected))
        with self.assertRaises(InputError):
            result.commit()
        with self.assertRaises(InputError):
            result.discard()

    def test_empty_transaction_commits_an_equal_snapshot(self):
        snapshot = make_snapshot()
        result = run_transaction([], {}, snapshot)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.trace, ())
        committed = result.commit()
        self.assertEqual(snapshot_bytes(committed, "mem"), MEM_BYTES)
        self.assertEqual(snapshot_bytes(committed, "other"), OTHER_BYTES)

    def test_committed_snapshot_works_with_readonly_evaluate(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [Store(PtrConst("mem", 4), U32, Const(0xDEADBEEF, U32))], {}, snapshot
        )
        committed = result.commit()
        ev = evaluate(Load(PtrConst("mem", 4), U32), {}, committed)
        self.assertEqual(ev.value.data, 0xDEADBEEF)
        # The original snapshot still evaluates to the original bytes.
        ev_old = evaluate(Load(PtrConst("mem", 4), U32), {}, snapshot)
        self.assertEqual(
            ev_old.value.data, int.from_bytes(MEM_BYTES[4:8], "little")
        )


class TransactionInputTests(unittest.TestCase):
    def test_structural_input_violations_raise_immediately(self):
        snapshot = make_snapshot()
        with self.assertRaises(InputError):
            run_transaction("not-a-sequence", {}, snapshot)  # type: ignore[arg-type]
        with self.assertRaises(InputError):
            run_transaction([Store(PtrConst("mem", 0), U8, Const(1, U8))] * (MAX_STATEMENTS + 1), {}, snapshot)
        with self.assertRaises(InputError):
            run_transaction(["nope"], {}, snapshot)  # type: ignore[list-item]
        with self.assertRaises(InputError):
            run_transaction([Store(PtrConst("mem", 0), U8, Const(1, U8))], {"r": True}, snapshot)
        with self.assertRaises(InputError):
            run_transaction([Store(PtrConst("mem", 0), U8, Const(1, U8))], {"r": 1.5}, snapshot)
        with self.assertRaises(InputError):
            run_transaction([Store(PtrConst("mem", 0), U8, Const(1, U8))], {}, "nope")  # type: ignore[arg-type]
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)

    def test_error_path_locates_the_failing_statement(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [
                Store(PtrConst("mem", 0), U8, Const(1, U8)),
                Store(PtrConst("mem", 4), U8, Const(300, U16)),
            ],
            {},
            snapshot,
        )
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, RangeError)
        self.assertEqual(result.error.path, ("statements[1]", "value"))

    def test_pointer_expression_type_mismatch(self):
        snapshot = make_snapshot()
        result = run_transaction(
            [Store(Const(0, U8), U8, Const(1, U8))],  # ptr expr is an int
            {},
            snapshot,
        )
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, TypeMismatchError)
        self.assertEqual(snapshot_bytes(snapshot, "mem"), MEM_BYTES)


if __name__ == "__main__":
    unittest.main()
