"""带类型的运行期值、类型描述符与区域快照。

公开数据结构（契约的一部分，见 README）：

* :class:`IntType`：固定宽度整数类型，宽度 ∈ {8,16,32,64}，显式有/无符号。
* :class:`PtrType`：指针类型，指向某个区域与某种负载整数宽度。
* :class:`Value`：运行期值。整数值为无符号机器表示（0..2**w-1），
  其符号性仅由所附带的 ``IntType`` 决定；指针值为 ``(region_id, offset)``。
* :class:`Region`：只读内存区域快照，访问区间按半开区间 ``[base, end)``
  校验，``offset == end`` 是合法的尾后位置但不可解引用。
* :class:`ReadRange`：一次 ``load`` 实际读取的字节区间证据。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import AddressError, InputError

ALLOWED_WIDTHS: frozenset[int] = frozenset({8, 16, 32, 64})


@dataclass(frozen=True)
class IntType:
    """固定宽度整数类型。

    :param width: 位宽，必须是 8/16/32/64。
    :param signed: True 为有符号，False 为无符号。
    """

    width: int
    signed: bool

    def __post_init__(self) -> None:
        if self.width not in ALLOWED_WIDTHS:
            raise InputError(f"不支持的整数位宽 {self.width!r}（允许 8/16/32/64）")
        if not isinstance(self.signed, bool):
            raise InputError("signed 必须是 bool")

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1

    @property
    def nbytes(self) -> int:
        return self.width // 8

    def signed_value(self, raw: int) -> int:
        """把无符号机器表示解释为有符号 Python int。"""
        if raw >= (1 << (self.width - 1)):
            return raw - (1 << self.width)
        return raw

    def __str__(self) -> str:  # pragma: no cover - 仅用于展示
        return ("i" if self.signed else "u") + str(self.width)


@dataclass(frozen=True)
class PtrType:
    """指针类型：指向区域 ``region_id`` 中宽度 ``pointee_width`` 的负载。

    :param region_id: 区域标识（非空字符串）。
    :param pointee_width: 解引用/load 时的负载位宽 8/16/32/64。
    """

    region_id: str
    pointee_width: int

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id:
            raise InputError("region_id 必须是非空字符串")
        if self.pointee_width not in ALLOWED_WIDTHS:
            raise InputError(
                f"不支持的指针负载位宽 {self.pointee_width!r}（允许 8/16/32/64）"
            )

    @property
    def pointee_nbytes(self) -> int:
        return self.pointee_width // 8

    def __str__(self) -> str:  # pragma: no cover - 仅用于展示
        return f"*{self.pointee_width}@{self.region_id}"


# 类型联合，供类型标注使用
DType = IntType | PtrType


@dataclass(frozen=True)
class Value:
    """带类型的运行期值。

    构造时会按整数类型做机器表示合法性校验：

    * ``vtype`` 为 :class:`IntType` 时 ``raw`` 必须是非负 Python int、
      不被允许为 ``bool``，且不超过该宽度无符号上限；
    * ``vtype`` 为 :class:`PtrType` 时 ``raw`` 必须是
      ``(region_id: str, offset: int)``，区域标识须与指针类型一致，
      offset 非负。指针是否在界内由区域边界校验负责，不由 Value 负责。
    """

    vtype: DType
    raw: object

    def __post_init__(self) -> None:
        t = self.vtype
        if isinstance(t, IntType):
            raw = self.raw
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise InputError("整数 Value.raw 必须是 int（拒绝 bool、float 等）")
            if raw < 0:
                raise InputError(f"整数机器表示必须非负，得到 {raw}")
            if raw > t.mask:
                raise InputError(
                    f"整数值 {raw} 超出 {t} 的机器表示范围 [0, {t.mask}]"
                )
        elif isinstance(t, PtrType):
            raw = self.raw
            if (
                not isinstance(raw, tuple)
                or len(raw) != 2
                or not isinstance(raw[0], str)
                or isinstance(raw[1], bool)
                or not isinstance(raw[1], int)
            ):
                raise InputError(
                    "指针 Value.raw 必须是 (region_id: str, offset: int)"
                )
            if raw[0] != t.region_id:
                raise InputError(
                    f"指针值区域 {raw[0]!r} 与指针类型区域 {t.region_id!r} 不一致"
                )
            if raw[1] < 0:
                raise InputError(f"指针偏移必须非负，得到 {raw[1]}")
        else:  # pragma: no cover - 类型联合已封闭
            raise InputError(f"未知值类型 {t!r}")

    # -- 便利访问器 -----------------------------------------------------
    @property
    def is_int(self) -> bool:
        return isinstance(self.vtype, IntType)

    @property
    def is_ptr(self) -> bool:
        return isinstance(self.vtype, PtrType)

    def as_int(self) -> int:
        """无符号机器表示。"""
        if not self.is_int:
            raise InputError(f"期望整数值，实际为 {self.vtype}")
        return self.raw  # type: ignore[return-value]

    def as_signed(self) -> int:
        """有符号解释。"""
        t = self.vtype
        if not isinstance(t, IntType):
            raise InputError(f"期望整数值，实际为 {self.vtype}")
        return t.signed_value(self.raw)  # type: ignore[arg-type]

    def as_ptr(self) -> tuple[str, int]:
        if not self.is_ptr:
            raise InputError(f"期望指针值，实际为 {self.vtype}")
        return self.raw  # type: ignore[return-value]

    @staticmethod
    def int_(width: int, signed: bool, value: int) -> "Value":
        """便捷构造：先按目标宽度模截断 ``value`` 再装箱。"""
        t = IntType(width, signed)
        return Value(t, value & t.mask)

    def __str__(self) -> str:
        if self.is_int:
            t = self.vtype
            assert isinstance(t, IntType)
            if t.signed:
                return f"{self.as_signed()}:{t}"
            return f"{self.raw}:{t}"
        rid, off = self.as_ptr()
        return f"({rid!r},{off}):{self.vtype}"


@dataclass(frozen=True)
class ReadRange:
    """一次 load 实际读取的字节区间（读取证据）。

    区间为半开 ``[start, end)``，按字节地址计。
    """

    region_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise InputError(
                f"读取区间非法 [{self.start}, {self.end})，要求 start < end"
            )

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    def __str__(self) -> str:  # pragma: no cover - 仅用于展示
        return f"{self.region_id}[{self.start}:{self.end}]"


class Region:
    """只读内存区域快照。

    :param region_id: 区域标识（非空 str）。
    :param data: 字节内容，bytes 或 bytearray（bytearray 会被拷贝一份，
        调用方在求值期间/之后改动原缓冲不影响快照）。
    :param base: 区域起始字节地址，必须是非负 int（拒绝 bool）。
        区域尾后地址固定为 ``base + len(data)``。
    """

    __slots__ = ("region_id", "data", "base", "end")

    def __init__(self, region_id: str, data: bytes | bytearray, base: int = 0) -> None:
        if not isinstance(region_id, str) or not region_id:
            raise InputError("region_id 必须是非空字符串")
        if not isinstance(data, (bytes, bytearray)):
            raise InputError("region data 必须是 bytes/bytearray")
        if isinstance(base, bool) or not isinstance(base, int) or base < 0:
            raise InputError("region base 必须是非负 int（拒绝 bool）")
        self.region_id = region_id
        # 拷贝并固定为 bytes：快照不可变
        self.data = bytes(data)
        self.base = base
        self.end = base + len(self.data)

    def check_read(self, start: int, nbytes: int) -> ReadRange:
        """校验半开区间 ``[start, start+nbytes)`` 可读，返回读取区间。

        起点必须满足 ``base <= start < end``（尾后位置 end 不可解引用），
        且整条区间不得越过 end。
        """
        stop = start + nbytes
        if start < self.base or not (start < self.end and stop <= self.end):
            raise AddressError(
                f"区域 {self.region_id!r} 越界读取 [{start}, {stop})，"
                f"区域范围 [{self.base}, {self.end})（尾后位置不可解引用）"
            )
        return ReadRange(self.region_id, start, stop)
