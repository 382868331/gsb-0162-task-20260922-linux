"""Reusable bit-width and address-space semantics for debugger expressions.

Public API:

* ``evaluate(ast, registers, memory)`` -> ``EvalResult``
* AST nodes: Const, PtrConst, Reg, BinOp, Cmp, Cast, PtrAdd, PtrSub,
  Load, Field, And, Or, Not
* Types: IntType (or U8..U64 / S8..S64), POINTER, Pointer, Value
* Memory: MemorySnapshot
* Results: EvalResult, ReadInterval
* Edit transactions: Store, AccessRecord, TransactionResult,
  run_transaction, MAX_STATEMENTS
* Errors: EvalError and subclasses, each carrying a locatable ``path``
* Limit: MAX_NODES
"""

from .errors import (
    DivisionByZeroError,
    EvalError,
    InputError,
    NodeBudgetError,
    PointerError,
    RangeError,
    ShiftError,
    TypeMismatchError,
)
from .evaluator import MAX_NODES, EvalResult, ReadInterval, evaluate
from .memory import MemorySnapshot
from .nodes import And, BinOp, Cast, Cmp, Const, Field, Load, Node, Not, Or, PtrAdd, PtrConst, PtrSub, Reg
from .transaction import MAX_STATEMENTS, AccessRecord, Store, TransactionResult, run_transaction
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
    "EvalResult",
    "ReadInterval",
    "MemorySnapshot",
    "run_transaction",
    "MAX_STATEMENTS",
    "Store",
    "AccessRecord",
    "TransactionResult",
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
]
