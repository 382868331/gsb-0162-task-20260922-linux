"""Reusable bit-width and address-space semantics for debugger expressions.

Public API:

* ``evaluate(ast, registers, memory, *, max_combinations=...)`` -> ``EvalResult``
* AST nodes: Const, PtrConst, Reg, BinOp, Cmp, Cast, PtrAdd, PtrSub,
  Load, Field, And, Or, Not, Select
* Types: IntType (or U8..U64 / S8..S64), POINTER, Pointer, Value,
  CandidateSet
* Memory: MemorySnapshot
* Results: EvalResult, ReadInterval, MaybeValue
* Errors: EvalError and subclasses, each carrying a locatable ``path``
* Limits: MAX_NODES, MAX_CANDIDATES, MAX_COMBINATIONS
"""

from .errors import (
    CandidateBudgetError,
    DivisionByZeroError,
    EvalError,
    InputError,
    NodeBudgetError,
    PointerError,
    RangeError,
    ShiftError,
    TypeMismatchError,
)
from .evaluator import (
    MAX_COMBINATIONS,
    MAX_NODES,
    EvalResult,
    MaybeValue,
    ReadInterval,
    evaluate,
)
from .memory import MemorySnapshot
from .nodes import And, BinOp, Cast, Cmp, Const, Field, Load, Node, Not, Or, PtrAdd, PtrConst, PtrSub, Reg, Select
from .types import (
    MAX_CANDIDATES,
    POINTER,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    CandidateSet,
    IntType,
    Pointer,
    PointerType,
    Value,
)

__all__ = [
    "evaluate",
    "MAX_NODES",
    "MAX_CANDIDATES",
    "MAX_COMBINATIONS",
    "EvalResult",
    "MaybeValue",
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
    "Select",
    "IntType",
    "PointerType",
    "POINTER",
    "Pointer",
    "Value",
    "CandidateSet",
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
]
