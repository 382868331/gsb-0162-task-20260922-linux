"""Reusable bit-width and address-space semantics for debugger expressions.

Public API:

* ``evaluate(ast, registers, memory, *, on_budget_overrun=...)`` -> ``EvalResult``
* AST nodes: Const, PtrConst, Reg, BinOp, Cmp, Cast, PtrAdd, PtrSub,
  Load, Field, And, Or, Not, If
* Types: IntType (or U8..U64 / S8..S64), POINTER, Pointer, Value
* Memory: MemorySnapshot
* Results: EvalResult (status/results/errors/reads), ReadInterval
* Errors: EvalError and subclasses (incl. PossibleError,
  CandidateBudgetError), each carrying a locatable ``path``
* Limits: MAX_NODES, MAX_REGISTER_CANDIDATES, MAX_CANDIDATE_COMBINATIONS
"""

from .errors import (
    CandidateBudgetError,
    DivisionByZeroError,
    EvalError,
    InputError,
    NodeBudgetError,
    PointerError,
    PossibleError,
    RangeError,
    ShiftError,
    TypeMismatchError,
)
from .evaluator import (
    MAX_CANDIDATE_COMBINATIONS,
    MAX_NODES,
    MAX_REGISTER_CANDIDATES,
    EvalResult,
    ReadInterval,
    evaluate,
)
from .memory import MemorySnapshot
from .nodes import And, BinOp, Cast, Cmp, Const, Field, If, Load, Node, Not, Or, PtrAdd, PtrConst, PtrSub, Reg
from .types import (
    POINTER,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    IntType,
    Pointer,
    PointerType,
    Value,
)

__all__ = [
    "evaluate",
    "MAX_NODES",
    "MAX_REGISTER_CANDIDATES",
    "MAX_CANDIDATE_COMBINATIONS",
    "EvalResult",
    "ReadInterval",
    "MemorySnapshot",
    "Node",
    "Const",
    "PtrConst",
    "Reg",
    "BinOp",
    "Cmp",
    "Cast",
    "PtrAdd",
    "PtrSub",
    "Load",
    "Field",
    "And",
    "Or",
    "Not",
    "If",
    "IntType",
    "PointerType",
    "POINTER",
    "Pointer",
    "Value",
    "U8",
    "U16",
    "U32",
    "U64",
    "S8",
    "S16",
    "S32",
    "S64",
    "EvalError",
    "InputError",
    "TypeMismatchError",
    "RangeError",
    "ShiftError",
    "DivisionByZeroError",
    "PointerError",
    "NodeBudgetError",
    "CandidateBudgetError",
    "PossibleError",
]
