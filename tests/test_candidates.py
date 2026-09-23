"""Tests for candidate-set evaluation, conditional branches and budgets.

These extend (without replacing) ``test_evaluator.py``.  Expected
candidate results are enumerated independently in the tests: the
arithmetic reference ``tests/reference.py`` is applied to every register
assignment by plain ``itertools.product`` here, so expectations never
come from the implementation under test.
"""

from __future__ import annotations

import itertools
import random
import unittest

from debugexpr import (
    MAX_CANDIDATE_COMBINATIONS,
    MAX_REGISTER_CANDIDATES,
    And,
    BinOp,
    CandidateBudgetError,
    Cmp,
    Const,
    DivisionByZeroError,
    Field,
    If,
    InputError,
    Load,
    MemorySnapshot,
    Or,
    Pointer,
    PointerError,
    PossibleError,
    PtrAdd,
    PtrConst,
    RangeError,
    Reg,
    ShiftError,
    S8,
    TypeMismatchError,
    U8,
    U16,
    evaluate,
)
from reference import ref_binop

MEM = MemorySnapshot({"m": bytes(range(64)), "heap": bytes(32)})


def ev(ast, registers=None, memory=MEM, **kwargs):
    return evaluate(ast, registers or {}, memory, **kwargs)


def independently_collect(op, pool_a, pool_b, itype, bits, signed):
    """Expected distinct results + which assignments fail, computed without
    importing the evaluator's core logic (uses tests/reference.py)."""
    values: set[int] = set()
    div_zero = shift = False
    for a, b in itertools.product(pool_a, pool_b):
        if op in ("div", "mod") and b == 0:
            div_zero = True
            continue
        if op in ("shl", "shr") and not (0 <= b < bits):
            shift = True
            continue
        values.add(ref_binop(op, a, b, bits, signed))
    return values, div_zero, shift


class TestCandidateResults(unittest.TestCase):
    def test_single_register_candidate_set(self):
        result = ev(BinOp("add", Reg("a", U8), Const(10, U8)), {"a": (1, 2, 3)})
        self.assertEqual(result.status, "exact")
        self.assertEqual([v.data for v in result.results], [11, 12, 13])
        self.assertFalse(result.errors)

    def test_duplicate_candidates_are_deduplicated(self):
        result = ev(Reg("a", U8), {"a": (5, 5, 5)})
        self.assertEqual([v.data for v in result.results], [5])

    def test_plain_int_still_means_one_candidate(self):
        result = ev(Reg("a", U8), {"a": 7})
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.results[0].data, 7)
        # legacy accessor keeps working
        self.assertEqual(result.value.data, 7)

    def test_results_are_deterministically_ordered(self):
        result = ev(Reg("a", U8), {"a": (3, 1, 2)})
        self.assertEqual([v.data for v in result.results], [1, 2, 3])

    def test_multi_register_cross_product(self):
        expr = BinOp("add", Reg("a", U8), Reg("b", U8))
        result = ev(expr, {"a": (1, 100), "b": (2, 200)})
        # unsigned wrap: 1+2=3, 1+200=201, 100+2=102, 100+200=44
        self.assertEqual([v.data for v in result.results], [3, 44, 102, 201])

    def test_distinct_assignments_can_collapse_to_one_value(self):
        # all candidates are congruent mod 256 after the add wraps
        result = ev(BinOp("add", Reg("a", U8), Const(1, U8)), {"a": (0, 256 - 1)})
        # 0+1 = 1; 255+1 = 0
        self.assertEqual([v.data for v in result.results], [0, 1])

    def test_candidate_pointer_results(self):
        expr = PtrAdd(PtrConst("m", 0), Reg("i", U8))
        result = ev(expr, {"i": (0, 4, 8)})
        self.assertEqual(
            [v.data for v in result.results],
            [Pointer("m", 0), Pointer("m", 4), Pointer("m", 8)],
        )

    def test_against_independent_reference(self):
        rng = random.Random(0xC0FFEE)
        ops = ["add", "sub", "mul", "bitand", "bitor", "bitxor"]
        for itype, bits, signed in [(U8, 8, False), (S8, 8, True), (U16, 16, False)]:
            for _ in range(40):
                op = rng.choice(ops)
                pool_a = [rng.randint(itype.min, itype.max) for _ in range(4)]
                pool_b = [rng.randint(itype.min, itype.max) for _ in range(4)]
                got = ev(
                    BinOp(op, Reg("a", itype), Reg("b", itype)),
                    {"a": tuple(dict.fromkeys(pool_a)), "b": tuple(dict.fromkeys(pool_b))},
                )
                want, _, _ = independently_collect(
                    op, tuple(dict.fromkeys(pool_a)), tuple(dict.fromkeys(pool_b)),
                    itype, bits, signed,
                )
                self.assertEqual({v.data for v in got.results}, want, f"{op} {pool_a} {pool_b}")
                self.assertEqual(got.status, "exact")

    def test_loads_collect_read_evidence_across_worlds(self):
        expr = Load(PtrAdd(PtrConst("m", 0), Reg("off", U8)), U16)
        result = ev(expr, {"off": (0, 8, 16)})
        # all offsets are aligned and in range
        self.assertEqual(result.status, "exact")
        intervals = sorted((r.region_id, r.offset, r.length) for r in result.reads)
        self.assertEqual(intervals, [("m", 0, 2), ("m", 8, 2), ("m", 16, 2)])
        # independently expected little-endian values: m[i] = i
        self.assertEqual(
            sorted(v.data for v in result.results),
            sorted(int.from_bytes(bytes([i, i + 1]), "little") for i in (0, 8, 16)),
        )


class TestPossibleErrors(unittest.TestCase):
    def test_partial_division_by_zero_is_possible_error(self):
        result = ev(BinOp("div", Const(10, U8), Reg("b", U8)), {"b": (0, 2, 5)})
        self.assertEqual(result.status, "possible_error")
        self.assertEqual(sorted(v.data for v in result.results), [2, 5])
        self.assertEqual(len(result.errors), 1)
        self.assertIsInstance(result.errors[0], DivisionByZeroError)
        self.assertEqual(result.errors[0].path, ("right",))

    def test_partial_mod_by_zero_is_possible_error(self):
        result = ev(BinOp("mod", Const(10, U8), Reg("b", U8)), {"b": (0, 3)})
        self.assertEqual(result.status, "possible_error")
        self.assertEqual([v.data for v in result.results], [1])
        self.assertIsInstance(result.errors[0], DivisionByZeroError)

    def test_definite_division_by_zero_still_raises(self):
        with self.assertRaises(DivisionByZeroError):
            ev(BinOp("div", Const(10, U8), Reg("b", U8)), {"b": (0, 0)})

    def test_partial_shift_error(self):
        result = ev(BinOp("shl", Const(1, U8), Reg("n", U8)), {"n": (7, 8, 9)})
        self.assertEqual(result.status, "possible_error")
        self.assertEqual([v.data for v in result.results], [128])
        self.assertIsInstance(result.errors[0], ShiftError)

    def test_partial_pointer_out_of_bounds(self):
        expr = Load(PtrAdd(PtrConst("m", 60), Reg("d", S8)), U8)
        result = ev(expr, {"d": (0, 4, 5)})  # 60 ok, 64 OOB-deref, 65 OOB-pointer
        self.assertEqual(result.status, "possible_error")
        self.assertEqual([v.data for v in result.results], [60])
        self.assertTrue(all(isinstance(e, PointerError) for e in result.errors))

    def test_partial_register_range_error(self):
        result = ev(Reg("a", U8), {"a": (1, 256)})
        self.assertEqual(result.status, "possible_error")
        self.assertEqual([v.data for v in result.results], [1])
        self.assertIsInstance(result.errors[0], RangeError)

    def test_all_worlds_fail_uniformly_raises_the_error(self):
        with self.assertRaises(DivisionByZeroError):
            ev(BinOp("div", Const(1, U8), Reg("b", U8)), {"b": (0, 0)})

    def test_all_worlds_fail_differently_raises_possible_error(self):
        # condition always selects: then-branch divides by zero, else-branch
        # dereferences one-past-end; with c in {0,1} every world fails,
        # but the two errors are different.
        expr = If(
            Reg("c", U8),
            BinOp("div", Const(1, U8), Const(0, U8)),
            Load(PtrConst("m", 64), U8),
        )
        with self.assertRaises(PossibleError) as caught:
            ev(expr, {"c": (0, 1)})
        kinds = {type(e) for e in caught.exception.errors}
        self.assertEqual(kinds, {DivisionByZeroError, PointerError})


class TestShortCircuitWithCandidates(unittest.TestCase):
    def test_and_skips_possible_divzero_when_left_always_false(self):
        bad = BinOp("div", Const(1, U8), Reg("z", U8))
        result = ev(And(Const(0, U8), bad), {"z": (0, 1)})
        self.assertEqual(result.status, "exact")
        self.assertEqual([v.data for v in result.results], [0])
        self.assertEqual(result.errors, ())
        self.assertEqual(result.reads, ())

    def test_or_skips_possible_divzero_when_left_always_true(self):
        bad = BinOp("div", Const(1, U8), Reg("z", U8))
        result = ev(Or(Const(1, U8), bad), {"z": (0, 1)})
        self.assertEqual(result.status, "exact")
        self.assertEqual([v.data for v in result.results], [1])

    def test_and_mixed_left_collects_both_outcomes(self):
        # l=0 -> result 0, right never runs;
        # l=1, z=1 -> 6/1=6 is truthy -> result 1; l=1, z=0 -> possible div0.
        bad = BinOp("div", Const(6, U8), Reg("z", U8))
        result = ev(And(Reg("l", U8), bad), {"l": (0, 1), "z": (0, 1)})
        self.assertEqual(result.status, "possible_error")
        self.assertEqual(sorted(v.data for v in result.results), [0, 1])
        self.assertIsInstance(result.errors[0], DivisionByZeroError)

    def test_short_circuit_skips_oob_load_evidence(self):
        bad = Load(PtrConst("m", 64), U8)
        result = ev(Or(Const(1, U8), bad))
        self.assertEqual(result.reads, ())
        result = ev(And(Const(0, U8), bad))
        self.assertEqual(result.reads, ())


class TestConditional(unittest.TestCase):
    def _branch_expr(self, then_off, else_off, itype=U8):
        return If(
            Reg("c", U8),
            Load(PtrConst("m", then_off), itype),
            Load(PtrConst("m", else_off), itype),
        )

    def test_always_true_only_runs_then_branch(self):
        # else branch would dereference one-past-end
        result = ev(self._branch_expr(0, 64), {"c": (1, 2, 255)})
        self.assertEqual(result.status, "exact")
        self.assertEqual([v.data for v in result.results], [0])
        self.assertEqual([(r.offset, r.length) for r in result.reads], [(0, 1)])

    def test_always_false_only_runs_else_branch(self):
        result = ev(self._branch_expr(64, 8), {"c": (0,)})
        self.assertEqual([v.data for v in result.results], [8])
        self.assertEqual([r.offset for r in result.reads], [8])

    def test_unknown_condition_runs_both_branches(self):
        result = ev(self._branch_expr(0, 8), {"c": (0, 1)})
        self.assertEqual(result.status, "exact")
        self.assertEqual(sorted(v.data for v in result.results), [0, 8])
        self.assertEqual(sorted(r.offset for r in result.reads), [0, 8])

    def test_unknown_condition_with_failing_branch_is_possible_error(self):
        result = ev(self._branch_expr(0, 64), {"c": (0, 1)})
        self.assertEqual(result.status, "possible_error")
        self.assertEqual([v.data for v in result.results], [0])
        self.assertEqual([r.offset for r in result.reads], [0])
        self.assertIsInstance(result.errors[0], PointerError)
        self.assertEqual(result.errors[0].path, ("else_",))

    def test_condition_derived_from_comparison(self):
        # a > 4 takes the then branch; a in {0,4,5,255} splits both ways
        expr = self._branch_expr(0, 8)
        expr = If(Cmp("gt", Reg("a", U8), Const(4, U8)),
                  Load(PtrConst("m", 0), U8), Load(PtrConst("m", 8), U8))
        result = ev(expr, {"a": (0, 4, 5, 255)})
        self.assertEqual(sorted(v.data for v in result.results), [0, 8])
        self.assertEqual(sorted(r.offset for r in result.reads), [0, 8])

    def test_branches_can_compute_different_values(self):
        expr = If(Reg("c", U8), Const(10, U8), Const(20, U8))
        result = ev(expr, {"c": (0, 1)})
        self.assertEqual(sorted(v.data for v in result.results), [10, 20])

    def test_nested_conditional(self):
        expr = If(
            Reg("a", U8),
            If(Reg("b", U8), Const(1, U8), Const(2, U8)),
            Const(3, U8),
        )
        result = ev(expr, {"a": (0, 1), "b": (0, 1)})
        self.assertEqual(sorted(v.data for v in result.results), [1, 2, 3])
        # force the inner then branch to fail: a=0 never sees it
        expr_bad = If(
            Reg("a", U8),
            If(Reg("b", U8), BinOp("div", Const(1, U8), Const(0, U8)), Const(2, U8)),
            Const(3, U8),
        )
        result = ev(expr_bad, {"a": (0,), "b": (0, 1)})
        self.assertEqual(result.status, "exact")
        self.assertEqual([v.data for v in result.results], [3])

    def test_condition_must_be_integer(self):
        # pointer condition is a (definite) type error
        expr = If(PtrConst("m", 0), Const(1, U8), Const(2, U8))
        with self.assertRaises(TypeMismatchError):
            ev(expr)

    def test_untaken_branch_with_misaligned_pointer_is_not_dereferenced(self):
        # then offset misaligned for u16 but never reached
        expr = If(Const(1, U8), Load(PtrConst("m", 0), U16), Load(PtrConst("m", 1), U16))
        result = ev(expr)
        self.assertEqual(result.status, "exact")
        self.assertEqual([r.offset for r in result.reads], [0])


class TestCandidateBudget(unittest.TestCase):
    def _eight_by_eight(self):
        return BinOp("add", Reg("a", U8), Reg("b", U8)), {
            "a": tuple(range(1, 9)),
            "b": tuple(range(1, 9)),
        }

    def test_exactly_64_combinations_is_exact(self):
        ast, regs = self._eight_by_eight()  # 8 * 8 == MAX_CANDIDATE_COMBINATIONS
        self.assertEqual(8 * 8, MAX_CANDIDATE_COMBINATIONS)
        result = ev(ast, regs)
        self.assertEqual(result.status, "exact")
        self.assertEqual(len(result.results), 15)  # sums 2..16

    def test_65_or_more_returns_unknown_by_default(self):
        ast = BinOp("add", Reg("a", U8), BinOp("add", Reg("b", U8), Reg("c", U8)))
        regs = {"a": tuple(range(5)), "b": tuple(range(5)), "c": tuple(range(3))}  # 75
        result = ev(ast, regs)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.results, ())
        self.assertEqual(result.reads, ())
        self.assertEqual(result.errors, ())

    def test_raise_mode_raises_candidate_budget_error(self):
        ast = BinOp("add", BinOp("add", Reg("a", U8), Reg("b", U8)), Reg("c", U8))
        regs = {"a": tuple(range(8)), "b": tuple(range(8)), "c": (0, 1)}  # 128
        with self.assertRaises(CandidateBudgetError):
            ev(ast, regs, on_budget_overrun="raise")

    def test_bad_budget_option(self):
        with self.assertRaises(InputError):
            ev(Const(1, U8), on_budget_overrun="wat")

    def test_unreferenced_registers_do_not_count_toward_budget(self):
        regs = {
            "a": tuple(range(4)),
            "b": tuple(range(4)),
            "c": tuple(range(4)),
            "d": tuple(range(4)),
        }  # 256 total, but only 'a' is referenced
        result = ev(Reg("a", U8), regs)
        self.assertEqual(result.status, "exact")
        self.assertEqual(len(result.results), 4)

    def test_over_budget_still_rejects_structurally_illegal_input(self):
        big = {"a": tuple(range(8)), "b": tuple(range(8)), "c": (0, 1)}  # 128
        with self.assertRaises(InputError):
            ev(Const("not-an-int", U8), big)
        with self.assertRaises(InputError):
            ev(BinOp("nope", Const(1, U8), Const(2, U8)), big)
        with self.assertRaises(PointerError):
            ev(Load(PtrConst("m", 1000), U8), big)
        with self.assertRaises(InputError):
            ev(Reg("missing", U8), big)
        # an illegal op inside an untaken branch is still illegal input:
        # the static pass validates both branches even when the assignment
        # space is over budget (here the condition references c, so all
        # three candidate registers count and the space is 128).
        bad_if = If(Reg("c", U8), Const(1, U8),
                    BinOp("@@", BinOp("add", Reg("a", U8), Reg("b", U8)), Const(2, U8)))
        with self.assertRaises(InputError):
            ev(bad_if, big)

    def test_over_budget_does_not_claim_dynamic_outcomes(self):
        # one statically-legal load at a candidate-driven address; all
        # three candidate registers are referenced, so the 128-world space
        # is over budget: result is unknown and no reads are claimed.
        ast = Load(
            PtrAdd(PtrConst("m", 0),
                   BinOp("add", Reg("a", U8), BinOp("add", Reg("b", U8), Reg("c", U8)))),
            U8,
        )
        result = ev(ast, {"a": tuple(range(8)), "b": tuple(range(8)), "c": (0, 1)})
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.reads, ())


class TestCandidateInputValidation(unittest.TestCase):
    def test_more_than_8_candidates_rejected(self):
        with self.assertRaises(InputError):
            ev(Reg("a", U8), {"a": tuple(range(MAX_REGISTER_CANDIDATES + 1))})

    def test_empty_candidate_list_rejected(self):
        with self.assertRaises(InputError):
            ev(Reg("a", U8), {"a": ()})
        with self.assertRaises(InputError):
            ev(Reg("a", U8), {"a": []})

    def test_eight_candidates_accepted(self):
        result = ev(Reg("a", U8), {"a": tuple(range(MAX_REGISTER_CANDIDATES))})
        self.assertEqual(len(result.results), MAX_REGISTER_CANDIDATES)

    def test_bool_and_float_candidates_rejected(self):
        for bad in (True, 1.5, float("nan"), float("inf")):
            with self.assertRaises(InputError, msg=repr(bad)):
                ev(Reg("a", U8), {"a": bad})
            with self.assertRaises(InputError, msg=f"list {bad!r}"):
                ev(Reg("a", U8), {"a": (1, bad)})

    def test_candidate_iterable_must_be_tuple_or_list(self):
        with self.assertRaises(InputError):
            ev(Reg("a", U8), {"a": {1, 2}})
        with self.assertRaises(InputError):
            ev(Reg("a", U8), {"a": (v for v in (1, 2))})

    def test_value_accessor_requires_unique_result(self):
        multi = ev(Reg("a", U8), {"a": (1, 2)})
        with self.assertRaises(RuntimeError):
            multi.value
        unknown = ev(
            BinOp("add", BinOp("add", Reg("a", U8), Reg("b", U8)), Reg("c", U8)),
            {"a": tuple(range(8)), "b": tuple(range(8)), "c": (0, 1)},
        )
        self.assertEqual(unknown.status, "unknown")
        with self.assertRaises(RuntimeError):
            unknown.value


class TestResultTypesAndReadsPreserved(unittest.TestCase):
    def test_signed_candidates_keep_signed_type(self):
        result = ev(BinOp("add", Reg("a", S8), Const(1, S8)), {"a": (-128, 126)})
        self.assertTrue(all(v.type is S8 for v in result.results))
        self.assertEqual(sorted(v.data for v in result.results), [-127, 127])

    def test_field_evidence_with_candidates(self):
        # Field offsets are static ints; candidate-driven indexing goes
        # through PtrAdd.  Both are covered here.
        expr = Field(PtrConst("m", 0), 8, U16)
        result = ev(expr, {"unused": (1, 2, 3)})
        self.assertEqual([(r.offset, r.length) for r in result.reads], [(8, 2)])

        dyn = Load(PtrAdd(PtrConst("m", 0), Reg("off", U8)), U16)
        result = ev(dyn, {"off": (0, 8)})
        self.assertEqual(sorted((r.offset, r.length) for r in result.reads), [(0, 2), (8, 2)])

    def test_pointer_equality_candidates(self):
        expr = Cmp("eq", PtrAdd(PtrConst("m", 0), Reg("i", U8)), PtrConst("m", 4))
        result = ev(expr, {"i": (0, 4, 8)})
        self.assertEqual([v.data for v in result.results], [0, 1])


if __name__ == "__main__":
    unittest.main()
