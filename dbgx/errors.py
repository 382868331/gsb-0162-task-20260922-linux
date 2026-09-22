"""可定位的错误类型层次。

所有由本库主动抛出的错误均继承 :class:`DbgxError`，并携带：

* ``path``：从表达式根节点到出错节点的**子节点下标路径**（元组）。
  例如 ``(2, 0)`` 表示「根的第 2 个操作数的第 0 个操作数」；根节点自身
  出错时路径为 ``()``。
* 标准 ``args[0]`` 为人类可读的中文说明，路径信息已拼入消息。
"""

from __future__ import annotations


class DbgxError(Exception):
    """全部 dbgx 错误的基类。"""

    def __init__(self, message: str, path: tuple[int, ...] = ()):
        if path:
            text = "节点路径 " + "/".join(str(i) for i in path) + ": " + message
        else:
            text = message
        super().__init__(text)
        self.message = message
        self.path: tuple[int, ...] = tuple(path)


class InputError(DbgxError):
    """evaluate 的 registers/memory 入参本身不合法。"""


class ASTError(DbgxError):
    """AST 结构/静态参数/字面量不合法，或节点数超过 500 上限。"""


class UndefinedRegisterError(DbgxError):
    """表达式引用了 registers 中不存在的寄存器。"""


class TypeMismatchError(DbgxError):
    """操作数类型或位宽/符号性不符合算子契约。"""


class ShiftError(DbgxError):
    """移位量为负或不小于左操作数位宽。"""


class DivisionByZeroError(DbgxError):
    """整数除法/取模的除数为 0。"""


class AddressError(DbgxError):
    """地址空间违例：未知区域、越界（含尾后解引用）、未对齐、跨区域等。"""
