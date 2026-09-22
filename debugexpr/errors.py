"""Error hierarchy for the debug-expression evaluator.

Every error carries a ``path``: a tuple of field names locating the AST
node that caused the failure, e.g. ``("left", "right")`` means
``root.left.right``.  The empty tuple ``()`` denotes the root node or a
top-level input (registers / memory / ast argument itself).
"""

from __future__ import annotations


def format_path(path: tuple[str, ...]) -> str:
    if not path:
        return "<root>"
    return "<root>." + ".".join(path)


class EvalError(Exception):
    """Base class for all evaluation failures.

    Attributes:
        path: tuple of field names locating the offending AST node.
        message: human-readable description without the path prefix.
    """

    def __init__(self, message: str, path: tuple[str, ...] = ()) -> None:
        self.message = message
        self.path = tuple(path)
        super().__init__(f"{message} (at {format_path(self.path)})")


class InputError(EvalError):
    """Python-level input contract violation.

    Raised for: non-int / bool / float (incl. NaN/Infinity) numeric inputs,
    unknown AST node objects, unknown operator names, unknown register
    names, wrong argument kinds to ``evaluate``.
    """


class TypeMismatchError(EvalError):
    """Operand types differ where the contract requires identical types."""


class RangeError(EvalError):
    """A literal or register value does not fit its declared integer type."""


class ShiftError(EvalError):
    """Shift amount is negative or >= the operand bit width."""


class DivisionByZeroError(EvalError):
    """Integer division or modulo with a zero divisor."""


class PointerError(EvalError):
    """Pointer/address-space violation.

    Covers: unknown region id, pointer arithmetic leaving the region
    (beyond the one-past-end position), dereference of the one-past-end
    position, misaligned access, cross-region pointer subtraction or
    ordering comparison.
    """


class NodeBudgetError(EvalError):
    """The AST exceeds the maximum allowed node count."""
