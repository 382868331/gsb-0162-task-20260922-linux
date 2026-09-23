"""Tests for the candidate/conditional iteration.

Covers:

* registers holding up to 8 candidate integers and candidate-set
  propagation through arithmetic, comparison, cast, pointer and load
  nodes;
* the ``Select`` conditional node: path-sensitive reads for a determined
  condition and both-branch union for a {0, 1} condition;
* short-circuit semantics over candidate sets;
* the explicit "unknown" signal (``CandidateBudgetError``) for budget
  overflow and for more than 8 distinct possible values;
* input validation and the no-half-updated-state guarantee.

Candidate arithmetic expectations are enumerated with
``tests/reference.py``, which never imports the core under test.
"""

from __future__ import annotations

import random
import unittest

from debugexpr import (
    MAX_CANDIDATES,
    And,
    BinOp,
    CandidateBudgetError,
    Cast,
    Cmp,
    Const,
    DivisionByZeroError,
    EvalError,
    Field,
    InputError,
    Load,
    MemorySnapshot,
    Not,
    Or,
    Pointer,
    PointerError,
    PtrAdd,
    PtrConst,
    PtrSub,
    RangeError,
    Reg,
    Select,
    ShiftError,
    TypeMismatchError,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    evaluate,
)
from reference import ref_binop, ref_cast, ref_cmp

EMPTY = MemorySnapshot({})
MEM = MemorySnapshot({"r": bytes(range(32)), "h": bytes(8)})


def ev(ast, registers=None, memory=EMPTY, **kw):
    return evaluate(ast, registers or {}, memory, **kw)


class TestCandidateRegisters(unittest.TestCase):
    def test_plain_int_stays_determined(self):
        result = ev(BinOp("add", Reg("a", U8), Const(1, U8)), {"a": 41})
        self.assertTrue(result.is_determined)
        self.assertEqual(result.value.data, 42)
        self.assertEqual(result.values, (42,))

    def test_candidates_propagate_through_arithmetic(self):
        # r in {1,2,3}  ->  r + 1 = {2,3,4}
        result = ev(BinOp("add", Reg("r", U8), Const(1, U8)), {"r": (1, 2, 3)})
        self.assertEqual(result.values, (2, 3, 4))
        self.assertIsNone(result.value)

    def test_candidates_are_deduplicated_and_sorted(self):
        result = ev(Reg("r", S8), {"r": (3, 1, 3, 2, 1)})
        self.assertEqual(result.values, (1, 2, 3))
        self.assertEqual(len(result.candidates), 3)

    def test_list_input_accepted_tuple_or_list(self):
        self.assertEqual(ev(Reg("r", U8), {"r": [5, 6]}).values, (5, 6))
        self.assertEqual(ev(Reg("r", U8), {"r": (5,)}).values, (5,))

    def test_eight_candidates_is_the_boundary(self):
        result = ev(Reg("r", U8), {"r": tuple(range(8))})
        self.assertEqual(len(result.candidates), MAX_CANDIDATES)

    def test_nine_candidates_rejected(self):
        with self.assertRaises(InputError):
            ev(Reg("r", U8), {"r": tuple(range(9))})

    def test_empty_candidate_collection_rejected(self):
        with self.assertRaises(InputError):
            ev(Reg("r", U8), {"r": ()})
        with self.assertRaises(InputError):
            ev(Reg("r", U8), {"r": []})

    def test_bad_candidate_member_types_rejected(self):
        for bad in (1.5, float("nan"), float("inf"), True, False, "1", None, b"1"):
            with self.assertRaises(InputError, msg=f"member {bad!r}"):
                ev(Reg("r", U8), {"r": (1, bad)})

    def test_candidate_value_out_of_type_range(self):
        with self.assertRaises(RangeError):
            ev(Reg("r", U8), {"r": (255, 256)})
        with self.assertRaises(RangeError):
            ev(Reg("r", S8), {"r": (-129, 0)})

    def test_non_collection_scalar_kinds_rejected(self):
        for bad in ("12", b"12", {1: 2}, {1, 2}):
            # a set IS accepted as a collection; only str/bytes/dict rejected
            if isinstance(bad, set):
                continue
            with self.assertRaises(InputError, msg=f"value {bad!r}"):
                ev(Reg("r", U8), {"r": bad})

    def test_set_of_ints_accepted(self):
        self.assertEqual(ev(Reg("r", U8), {"r": {2, 1}}).values, (1, 2))


class TestCandidateArithmetic(unittest.TestCase):
    def test_wrapping_collected_per_combination(self):
        # u8: 250 + r, r in {0,5,6,10} -> {250,255,0,4}
        result = ev(
            BinOp("add", Const(250, U8), Reg("r", U8)), {"r": (0, 5, 6, 10)}
        )
        self.assertEqual(result.values, (0, 4, 250, 255))

    def test_signed_candidate_arithmetic(self):
        # s8: -128 + r, r in {0,1} -> {-128,-127}
        result = ev(
            BinOp("add", Const(-128, S8), Reg("r", S8)), {"r": (0, 1)}
        )
        self.assertEqual(result.values, (-128, -127))

    def test_division_by_zero_in_any_combination_is_an_error(self):
        # 10 / r with r in {0,1}: one combination divides by zero.
        with self.assertRaises(DivisionByZeroError):
            ev(BinOp("div", Const(10, U8), Reg("r", U8)), {"r": (0, 1)})
        # modulo likewise
        with self.assertRaises(DivisionByZeroError):
            ev(BinOp("mod", Const(10, U8), Reg("r", U8)), {"r": (1, 0)})

    def test_shift_amount_invalid_in_any_combination_is_an_error(self):
        with self.assertRaises(ShiftError):
            ev(BinOp("shr", Const(1, U8), Reg("r", U8)), {"r": (7, 8)})

    def test_all_valid_div_combinations_collected(self):
        # u8: 8 / r, r in {1,2,4} -> {8,4,2}
        result = ev(BinOp("div", Const(8, U8), Reg("r", U8)), {"r": (1, 2, 4)})
        self.assertEqual(result.values, (2, 4, 8))

    def test_comparison_candidates(self):
        result = ev(Cmp("lt", Reg("a", S8), Const(0, S8)), {"a": (-2, -1, 0, 1, 2)})
        self.assertEqual(result.values, (0, 1))
        # wholly true / wholly false stay determined
        self.assertEqual(ev(Cmp("lt", Reg("a", S8), Const(0, S8)), {"a": (-5,)}).values, (1,))
        self.assertEqual(ev(Cmp("ge", Reg("a", S8), Const(0, S8)), {"a": (1, 5)}).values, (1,))

    def test_cast_over_candidates(self):
        result = ev(Cast(Reg("a", U16), U8), {"a": (0x100, 0x1FF, 0x234)})
        self.assertEqual(result.values, (0, 0x34, 0xFF))

    def test_repeated_register_is_independent_per_occurrence(self):
        # No equality correlation: r - r over r in {1,2} enumerates all
        # four pairs: (1,1)->0, (1,2)->255, (2,1)->1, (2,2)->0 (u8).
        result = ev(BinOp("sub", Reg("r", U8), Reg("r", U8)), {"r": (1, 2)})
        self.assertEqual(result.values, (0, 1, 255))

    def test_against_independent_reference(self):
        rng = random.Random(0xC0DE)
        # Only ops whose 2x2 candidate product can be fully enumerated
        # within the 8-value representation limit; div/mod/shifts are
        # covered by the hand-computed cases above.
        ops = ["add", "sub", "mul", "bitand", "bitor", "bitxor"]
        for itype, bits, signed in [
            (U8, 8, False), (U16, 16, False), (S8, 8, True), (S32, 32, True)
        ]:
            for _ in range(40):
                op = rng.choice(ops)
                avals = rng.sample(range(max(itype.min, -8), min(itype.max, 8) + 1), k=2)
                bvals = rng.sample(range(max(itype.min, -8), min(itype.max, 8) + 1), k=2)
                ast = BinOp(op, Reg("a", itype), Reg("b", itype))
                got = ev(ast, {"a": avals, "b": bvals})
                want = {
                    ref_binop(op, a, b, bits, signed)
                    for a in avals for b in bvals
                }
                self.assertEqual(set(got.values), want, f"{op} {avals} {bvals} on {itype}")
                self.assertIs(got.type, itype)

    def test_casts_against_independent_reference(self):
        rng = random.Random(0xC0DF)
        for _ in range(50):
            vals = [rng.randint(-200, 200) for _ in range(4)]
            got = ev(Cast(Reg("a", S16), U8), {"a": vals})
            want = {ref_cast(v, 16, True, 8, False) for v in vals}
            self.assertEqual(set(got.values), want)

    def test_cmps_against_independent_reference(self):
        rng = random.Random(0xC0E0)
        for op in ("eq", "ne", "lt", "le", "gt", "ge"):
            avals = [rng.randint(-4, 4) for _ in range(3)]
            bvals = [rng.randint(-4, 4) for _ in range(3)]
            got = ev(Cmp(op, Reg("a", S8), Reg("b", S8)), {"a": avals, "b": bvals})
            want = {ref_cmp(op, a, b) for a in avals for b in bvals}
            self.assertEqual(set(got.values), want, op)


class TestSelect(unittest.TestCase):
    def test_determined_true_reads_only_then_branch(self):
        bad = Load(PtrConst("r", 32), U8)  # one-past-end: would raise
        good = Load(PtrConst("r", 0), U8)
        result = ev(Select(Const(1, U8), good, bad), memory=MEM)
        self.assertEqual(result.values, (0,))
        self.assertEqual([(x.region_id, x.offset, x.length) for x in result.reads], [("r", 0, 1)])

    def test_determined_false_reads_only_else_branch(self):
        boom = BinOp("div", Const(1, U8), Const(0, U8))
        result = ev(Select(Const(0, U8), boom, Const(7, U8)))
        self.assertEqual(result.values, (7,))
        self.assertEqual(result.reads, ())

    def test_determined_condition_skips_division_by_zero_branch(self):
        boom = BinOp("div", Const(1, U8), Const(0, U8))
        # cond=1 -> then taken; else's div-by-zero must not be reported
        self.assertEqual(ev(Select(Const(1, U8), Const(9, U8), boom)).values, (9,))

    def test_unknown_condition_reads_both_branches_and_unions(self):
        then_node = Load(PtrConst("r", 0), U8)   # byte 0
        else_node = Load(PtrConst("r", 1), U8)   # byte 1
        result = ev(Select(Reg("c", U8), then_node, else_node), {"c": (0, 1)}, MEM)
        self.assertEqual(result.values, (0, 1))
        self.assertEqual(
            [(x.region_id, x.offset) for x in result.reads], [("r", 0), ("r", 1)]
        )

    def test_unknown_condition_then_branch_reads_come_first(self):
        # evidence order: whole then branch, then whole else branch
        then_node = Field(PtrConst("r", 0), 4, U16)   # read offset 4
        else_node = Field(PtrConst("r", 0), 8, U16)   # read offset 8
        result = ev(Select(Reg("c", U8), then_node, else_node), {"c": (0, 1)}, MEM)
        self.assertEqual([r.offset for r in result.reads], [4, 8])

    def test_condition_must_be_zero_or_one(self):
        for bad in (2, 255, -1):
            with self.assertRaises(RangeError):
                ev(Select(Const(bad, S8 if bad < 0 else U8), Const(1, U8), Const(2, U8)))
        # a candidate set containing a forbidden value is rejected
        with self.assertRaises(RangeError):
            ev(Select(Reg("c", U8), Const(1, U8), Const(2, U8)), {"c": (0, 2)})
        # {0,1,2} likewise even though both legal values are present
        with self.assertRaises(RangeError):
            ev(Select(Reg("c", U8), Const(1, U8), Const(2, U8)), {"c": (0, 1, 2)})

    def test_condition_must_be_integer(self):
        with self.assertRaises(TypeMismatchError):
            ev(Select(PtrConst("r", 0), Const(1, U8), Const(2, U8)), memory=MEM)

    def test_branch_types_must_match_when_both_reachable(self):
        with self.assertRaises(TypeMismatchError):
            ev(
                Select(Reg("c", U8), Const(1, U8), PtrConst("r", 0)),
                {"c": (0, 1)},
                MEM,
            )

    def test_unreachable_branch_type_mismatch_not_inspected(self):
        # cond fixed: the dead branch (a pointer) is never evaluated, so
        # no type error -- only actually-executed code is read.
        result = ev(
            Select(Const(1, U8), Const(1, U8), PtrConst("r", 0)), memory=MEM
        )
        self.assertEqual(result.values, (1,))

    def test_error_in_a_reachable_branch_is_reported(self):
        boom = BinOp("div", Const(1, U8), Const(0, U8))
        with self.assertRaises(DivisionByZeroError):
            ev(Select(Reg("c", U8), boom, Const(0, U8)), {"c": (0, 1)})
        with self.assertRaises(PointerError):
            ev(
                Select(Reg("c", U8), Const(0, U8), Load(PtrConst("r", 32), U8)),
                {"c": (0, 1)},
                MEM,
            )

    def test_select_nested_determined_paths(self):
        # c=0 -> inner select; its d=1 takes the inner then branch (99)
        expr = Select(
            Reg("c", U8),
            Const(1, U8),
            Select(Reg("d", U8), Const(99, U8), Const(100, U8)),
        )
        self.assertEqual(ev(expr, {"c": 0, "d": 1}).values, (99,))
        self.assertEqual(ev(expr, {"c": 0, "d": 0}).values, (100,))
        self.assertEqual(ev(expr, {"c": 1, "d": 0}).values, (1,))

    def test_select_with_candidate_branches_unions(self):
        # then: r+1 in {2,3}; else: constant 9
        expr = Select(
            Reg("c", U8),
            BinOp("add", Reg("r", U8), Const(1, U8)),
            Const(9, U8),
        )
        result = ev(expr, {"c": (0, 1), "r": (1, 2)})
        self.assertEqual(result.values, (2, 3, 9))

    def test_select_error_path_locates_condition(self):
        try:
            ev(Select(Const(2, U8), Const(1, U8), Const(2, U8)))
        except RangeError as err:
            self.assertEqual(err.path, ("cond",))
        else:  # pragma: no cover
            self.fail("expected RangeError")


class TestCandidateShortCircuit(unittest.TestCase):
    def test_and_left_always_zero_never_evaluates_right(self):
        bad = Load(PtrConst("r", 32), U8)
        result = ev(And(Reg("g", U8), bad), {"g": (0, 0)}, MEM)
        self.assertEqual(result.values, (0,))
        self.assertEqual(result.reads, ())

    def test_and_mixed_condition_evaluates_right_and_keeps_zero(self):
        # g in {0,1}; right reads byte 0 (value 0); possible results:
        # g=0 short-circuits -> 0; g=1 -> right value 0 -> 0
        right = Load(PtrConst("r", 0), U8)
        result = ev(And(Reg("g", U8), right), {"g": (0, 1)}, MEM)
        self.assertEqual(result.values, (0,))
        self.assertEqual([r.offset for r in result.reads], [0])

    def test_and_mixed_condition_can_produce_both_outcomes(self):
        # g in {0,5}; right in {0,7} -> {0 (short), 0, 1} = {0,1}
        result = ev(
            And(Reg("g", U8), Reg("h", U8)), {"g": (0, 5), "h": (0, 7)}
        )
        self.assertEqual(result.values, (0, 1))

    def test_and_right_error_only_when_left_can_be_nonzero(self):
        bad = Load(PtrConst("r", 32), U8)
        # left may be nonzero -> the bad load is reachable -> error
        with self.assertRaises(PointerError):
            ev(And(Reg("g", U8), bad), {"g": (0, 1)}, MEM)
        # left always zero -> right never runs, no error
        self.assertEqual(ev(And(Reg("g", U8), bad), {"g": (0,)}, MEM).values, (0,))

    def test_or_left_always_nonzero_never_evaluates_right(self):
        bad = Load(PtrConst("r", 32), U8)
        result = ev(Or(Reg("g", U8), bad), {"g": (3, 5)}, MEM)
        self.assertEqual(result.values, (1,))
        self.assertEqual(result.reads, ())

    def test_or_mixed_condition(self):
        result = ev(Or(Reg("g", U8), Reg("h", U8)), {"g": (0, 5), "h": (0, 7)})
        self.assertEqual(result.values, (0, 1))
        with self.assertRaises(PointerError):
            ev(Or(Reg("g", U8), Load(PtrConst("r", 32), U8)), {"g": (0, 1)}, MEM)

    def test_not_over_candidates(self):
        self.assertEqual(ev(Not(Reg("g", U8)), {"g": (0, 5)}).values, (0, 1))
        self.assertEqual(ev(Not(Reg("g", U8)), {"g": (2, 5)}).values, (0,))


class TestCandidatePointersAndLoads(unittest.TestCase):
    def test_pointer_index_candidates_load_each_address(self):
        expr = Load(PtrAdd(PtrConst("r", 0), Reg("i", U8)), U8)
        result = ev(expr, {"i": (0, 1, 2)}, MEM)
        # region bytes are 0,1,2,...
        self.assertEqual(result.values, (0, 1, 2))
        self.assertEqual([r.offset for r in result.reads], [0, 1, 2])

    def test_aligned_multibyte_loads_over_candidates(self):
        expr = Load(PtrAdd(PtrConst("r", 0), Reg("i", U8)), U16)
        result = ev(expr, {"i": (0, 4)}, MEM)
        # u16 at 0: 0x0100; at 4: 0x0504
        self.assertEqual(result.values, (0x0100, 0x0504))

    def test_misaligned_candidate_address_rejected(self):
        expr = Load(PtrAdd(PtrConst("r", 0), Reg("i", U8)), U16)
        with self.assertRaises(PointerError):
            ev(expr, {"i": (0, 1)}, MEM)

    def test_out_of_bounds_candidate_address_rejected(self):
        expr = Load(PtrAdd(PtrConst("r", 0), Reg("i", U8)), U8)
        with self.assertRaises(PointerError):
            ev(expr, {"i": (31, 32)}, MEM)

    def test_ptradd_candidate_leaving_region_rejected(self):
        with self.assertRaises(PointerError):
            ev(
                PtrAdd(PtrConst("r", 30), Reg("i", U8)), {"i": (2, 3)}, MEM
            )

    def test_ptradd_candidates_including_one_past_end_ok(self):
        result = ev(PtrAdd(PtrConst("r", 30), Reg("i", U8)), {"i": (2, 1)}, MEM)
        self.assertEqual(
            set(result.values), {Pointer("r", 31), Pointer("r", 32)}
        )

    def test_ptrsub_candidate_deltas(self):
        result = ev(
            PtrSub(PtrConst("r", 16), PtrAdd(PtrConst("r", 0), Reg("i", U8))),
            {"i": (0, 8, 16)},
            MEM,
        )
        self.assertEqual(result.values, (0, 8, 16))
        self.assertIs(result.type, S64)

    def test_cross_region_candidate_pair_rejected(self):
        with self.assertRaises(PointerError):
            ev(
                PtrSub(PtrConst("r", 0), PtrAdd(PtrConst("h", 0), Reg("i", U8))),
                {"i": (0,)},
                MEM,
            )

    def test_field_over_base_candidates(self):
        result = ev(
            Field(PtrAdd(PtrConst("r", 0), Reg("b", U8)), 2, U8),
            {"b": (0, 4)},
            MEM,
        )
        self.assertEqual(result.values, (2, 6))


class TestUnknownBudget(unittest.TestCase):
    def test_more_than_eight_distinct_values_is_unknown(self):
        # a in 0..7, b in 0..7: a+b has 15 distinct values
        with self.assertRaises(CandidateBudgetError):
            ev(
                BinOp("add", Reg("a", U8), Reg("b", U8)),
                {"a": tuple(range(8)), "b": tuple(range(8))},
            )

    def test_exactly_eight_distinct_values_ok(self):
        result = ev(
            BinOp("mul", Reg("a", U8), Const(3, U8)), {"a": tuple(range(8))}
        )
        self.assertEqual(result.values, (0, 3, 6, 9, 12, 15, 18, 21))

    def test_combination_budget_exceeded(self):
        # each Reg charges its candidate count, the binop charges the product
        with self.assertRaises(CandidateBudgetError):
            ev(
                BinOp("add", Reg("a", U8), Reg("b", U8)),
                {"a": (0, 1), "b": (0, 1)},
                max_combinations=7,  # 2 + 2 + 4 = 8 combos needed
            )

    def test_combination_budget_exact_boundary_ok(self):
        result = ev(
            BinOp("add", Reg("a", U8), Reg("b", U8)),
            {"a": (0, 1), "b": (0, 1)},
            max_combinations=8,
        )
        self.assertEqual(result.values, (0, 1, 2))

    def test_unknown_is_an_eval_error_but_not_an_illegal_input_fault(self):
        self.assertTrue(issubclass(CandidateBudgetError, EvalError))

    def test_scalar_inputs_never_go_unknown(self):
        # Even a budget of 1 suffices: every scalar node charges exactly 1,
        # and the 500-node AST limit caps the total well below 4096.
        ast = Const(1, U8)
        for _ in range(40):
            ast = BinOp("add", ast, Const(1, U8))
        result = ev(ast, {}, max_combinations=1)
        self.assertEqual(result.value.data, 41)

    def test_bad_budget_argument_rejected(self):
        for bad in (0, -1, 1.5, True, "100"):
            with self.assertRaises(InputError):
                ev(Const(1, U8), {}, max_combinations=bad)

    def test_select_union_over_eight_is_unknown(self):
        # then lo in 0..3 (4 values), else hi in 4..8 (5 values):
        # union has exactly 8 values at first, then 9 -> unknown
        expr = Select(Reg("c", U8), Reg("lo", U8), Reg("hi", U8))
        ok = ev(expr, {"c": (0, 1), "lo": (0, 1, 2, 3), "hi": (4, 5, 6, 7)})
        self.assertEqual(len(ok.candidates), 8)
        with self.assertRaises(CandidateBudgetError):
            ev(
                expr,
                {"c": (0, 1), "lo": (0, 1, 2, 3), "hi": (4, 5, 6, 7, 8)},
            )


class TestNoPartialState(unittest.TestCase):
    def test_snapshot_and_registers_unchanged_after_success(self):
        data = bytes(range(32))
        snapshot = MemorySnapshot({"r": data})
        registers = {"a": [1, 2, 3]}
        ev(BinOp("add", Reg("a", U8), Const(1, U8)), registers, snapshot)
        self.assertEqual(registers, {"a": [1, 2, 3]})  # caller's list untouched
        self.assertEqual(snapshot.read("r", 0, 32), data)

    def test_snapshot_and_registers_unchanged_after_failure(self):
        snapshot = MemorySnapshot({"r": bytes(range(32))})
        registers = {"a": (0, 1)}
        with self.assertRaises(DivisionByZeroError):
            ev(BinOp("div", Const(1, U8), Reg("a", U8)), registers, snapshot)
        self.assertEqual(registers, {"a": (0, 1)})
        self.assertEqual(snapshot.region_ids, ("r",))
        with self.assertRaises(InputError):
            ev(Reg("a", U8), {"a": (1, 2, 3, 4, 5, 6, 7, 8, 9)}, snapshot)
        self.assertEqual(snapshot.read("r", 0, 1), b"\x00")

    def test_failed_call_returns_no_result_and_no_reads_leak(self):
        # The API returns a fresh result only on success; nothing mutable is
        # shared between calls.
        good = ev(Load(PtrConst("r", 0), U8), memory=MEM)
        before = good.reads
        with self.assertRaises(PointerError):
            ev(Load(PtrConst("r", 99), U8), memory=MEM)
        self.assertEqual(good.reads, before)


class TestSelectNodeBudget(unittest.TestCase):
    def test_select_counts_toward_node_budget(self):
        node = Const(1, U8)
        # wrap in selects: each Select adds 3 nodes; stay modestly under 500
        for _ in range(10):
            node = Select(Const(1, U8), node, Const(0, U8))
        self.assertEqual(ev(node).values, (1,))


if __name__ == "__main__":
    unittest.main()
