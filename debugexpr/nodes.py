"""Public AST node definitions.

All nodes are immutable dataclasses.  Structural validation (operator
names, operand kinds, literal ranges) happens inside ``evaluate`` so that
every violation is reported with a locatable ``path``; constructing a
node never silently coerces anything.

Node inventory
--------------
Const(value, int_type)          integer literal, must fit int_type
PtrConst(region_id, offset)       pointer literal (validated against memory)
Reg(name, int_type)               typed register read
BinOp(op, left, right)            add/sub/mul/div/mod/shl/shr/bitand/bitor/bitxor
Cmp(op, left, right)              eq/ne/lt/le/gt/ge -> u8 0/1
Cast(node, target)                integer -> integer modular reinterpretation
PtrAdd(ptr, index)                pointer + integer (byte units, same region)
PtrSub(left, right)               pointer - pointer, same region -> s64
Load(ptr, int_type)               typed aligned load from a memory snapshot
Field(ptr, offset, int_type)      load at ptr + offset (typed field access)
And(left, right) / Or(left, right) short-circuit logic -> u8 0/1
Not(node)                         logical negation -> u8 0/1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .types import IntType


class Node:
    """Base class of every AST node."""

    __slots__ = ()


@dataclass(frozen=True)
class Const(Node):
    value: int
    type: IntType


@dataclass(frozen=True)
class PtrConst(Node):
    region_id: str
    offset: int


@dataclass(frozen=True)
class Reg(Node):
    name: str
    type: IntType


@dataclass(frozen=True)
class BinOp(Node):
    op: str
    left: Node
    right: Node


@dataclass(frozen=True)
class Cmp(Node):
    op: str
    left: Node
    right: Node


@dataclass(frozen=True)
class Cast(Node):
    node: Node
    target: IntType


@dataclass(frozen=True)
class PtrAdd(Node):
    ptr: Node
    index: Node


@dataclass(frozen=True)
class PtrSub(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Load(Node):
    ptr: Node
    type: IntType


@dataclass(frozen=True)
class Field(Node):
    ptr: Node
    offset: int
    type: IntType


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Not(Node):
    node: Node




@dataclass(frozen=True)
class FieldOf(Node):
    """``base.field``: field selection on a struct/union at ``base``.

    ``base`` evaluates to a pointer (or an aggregate lvalue address);
    ``layout`` is the caller-supplied ``StructLayout`` describing it.
    Selection itself only computes an address — bytes are read only for
    scalar (IntType / POINTER) fields, never for the whole struct.
    Aggregate-typed fields yield an lvalue usable by further
    ``FieldOf``/``IndexOf``.
    """

    base: Node
    layout: object  # StructLayout
    field: str


@dataclass(frozen=True)
class IndexOf(Node):
    """``base[index]``: subscript into an array at ``base``.

    ``base`` evaluates to a pointer (or an aggregate lvalue address);
    ``layout`` is the caller-supplied ``ArrayLayout``; ``index`` is an
    integer-typed node.  Out-of-bounds indices (including any index into
    an empty array) are rejected with a path.  Scalar elements are read
    byte-exactly; aggregate elements yield an lvalue.
    """

    base: Node
    layout: object  # ArrayLayout
    index: Node


AnyNode = Union[Const, PtrConst, Reg, BinOp, Cmp, Cast, PtrAdd, PtrSub, Load, Field, And, Or, Not, FieldOf, IndexOf]
