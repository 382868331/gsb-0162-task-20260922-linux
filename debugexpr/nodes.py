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
FieldRef(base, layout, name)      lvalue field address per StructLayout (no read)
IndexRef(base, layout, index)     lvalue element address per ArrayLayout (no read)
Deref(ptr, layout)                final scalar fetch per IntLayout/PtrLayout
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
class FieldRef(Node):
    """Lvalue field selection ``base.name`` per a caller-supplied StructLayout.

    Produces the *address* of the field (a pointer); it never reads the
    enclosing struct.  Use ``Deref`` to fetch the value.
    """

    base: Node
    layout: object  # StructLayout, validated in evaluate
    name: str


@dataclass(frozen=True)
class IndexRef(Node):
    """Lvalue subscript ``base[index]`` per a caller-supplied ArrayLayout.

    Produces the *address* of the element; the subscript is bounds-checked
    against the layout length (empty arrays reject every index).
    """

    base: Node
    layout: object  # ArrayLayout, validated in evaluate
    index: Node


@dataclass(frozen=True)
class Deref(Node):
    """Final value fetch: read exactly ``layout.size`` bytes at ``ptr``.

    Only scalar layouts (IntLayout / PtrLayout) are dereferenceable.
    Alignment is the layout's declared alignment, not the natural one.
    """

    ptr: Node
    layout: object  # IntLayout | PtrLayout, validated in evaluate


AnyNode = Union[Const, PtrConst, Reg, BinOp, Cmp, Cast, PtrAdd, PtrSub, Load, Field, And, Or, Not,
           FieldRef, IndexRef, Deref]
