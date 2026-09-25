"""Reusable bit-width and address-space semantics for debugger expressions.

Public API:

* ``evaluate(ast, registers, memory)`` -> ``EvalResult``
* ``begin_edit(memory)`` -> one-shot ``EditTransaction`` with
  ``store(Store(...))`` / ``evaluate(...)`` / ``commit()`` / ``discard()``
  and an access ``trace`` (undoable edit preview over a snapshot)
* AST nodes: Const, PtrConst, Reg, BinOp, Cmp, Cast, PtrAdd, PtrSub,
  Load, Field, And, Or, Not
* Types: IntType (or U8..U64 / S8..S64), POINTER, Pointer, Value
* Memory: MemorySnapshot
* Results: EvalResult, ReadInterval, AccessRecord
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
    TransactionError,
    TypeMismatchError,
)
from .evaluator import MAX_NODES, EvalResult, ReadInterval, evaluate
from .memory import MemorySnapshot
from .nodes import And, BinOp, Cast, Cmp, Const, Field, Load, Node, Not, Or, PtrAdd, PtrConst, PtrSub, Reg
from .transaction import AccessRecord, EditTransaction, Store, begin_edit
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
    "begin_edit",
    "EditTransaction",
    "Store",
    "AccessRecord",
    "MAX_NODES",
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
    "TransactionError",
]
