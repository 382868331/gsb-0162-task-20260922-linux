"""dbgx —— 离线调试器表达式：位宽与地址空间语义。

公开接口：

* :func:`evaluate` —— 唯一求值入口。
* 数据结构：:class:`Value`、:class:`IntType`、:class:`PtrType`、
  :class:`Region`、:class:`ReadRange`、:class:`EvalResult`。
* 错误：:class:`DbgxError` 及其子类
  (:class:`InputError`, :class:`ASTError`, :class:`UndefinedRegisterError`,
  :class:`TypeMismatchError`, :class:`ShiftError`,
  :class:`DivisionByZeroError`, :class:`AddressError`)。
* 常量：:data:`MAX_NODES`（500）。
"""

from __future__ import annotations

from .ast_nodes import MAX_NODES, validate
from .engine import EvalResult, evaluate
from .errors import (
    AddressError,
    ASTError,
    DbgxError,
    DivisionByZeroError,
    InputError,
    ShiftError,
    TypeMismatchError,
    UndefinedRegisterError,
)
from .types import IntType, PtrType, ReadRange, Region, Value

__all__ = [
    "evaluate",
    "validate",
    "EvalResult",
    "Value",
    "IntType",
    "PtrType",
    "Region",
    "ReadRange",
    "DbgxError",
    "InputError",
    "ASTError",
    "UndefinedRegisterError",
    "TypeMismatchError",
    "ShiftError",
    "DivisionByZeroError",
    "AddressError",
    "MAX_NODES",
]
