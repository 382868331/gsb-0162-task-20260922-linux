# debugexpr — 调试表达式的位宽与地址空间语义

为离线调试器表达式工程提供的可复用求值模块：显式位宽/符号性的整数语义、
带区域（region）的指针模型、内存快照加载与读取证据。

* Python 3.14.7，仅标准库，无需安装任何额外依赖。
* 全部离线：无网络、账号、数据库、Docker 或外部服务。
* 核心算法（模截断算术、截断除法、指针边界/对齐检查）在本仓库内从零实现，
  未调用第三方同类实现。

## 环境准备（隔离 venv）

无需任何第三方依赖；venv 仅为隔离解释器：

```bash
python3 -m venv .venv
. .venv/bin/activate          # 或直接使用 .venv/bin/python
python -m unittest discover -s tests -v
```

## 接口

```python
from debugexpr import evaluate, MemorySnapshot, Const, BinOp, Load, PtrConst, U8, S32

result = evaluate(ast, registers, memory)
result.value   # Value(type, data)：带类型的结果
result.reads   # tuple[ReadInterval]：实际读取的内存区间（求值证据，按求值顺序）
```

* `ast`：由 `debugexpr` 节点类构成的 AST：`Const / PtrConst / Reg / BinOp /
  Cmp / Cast / PtrAdd / PtrSub / Load / Field / And / Or / Not`，以及布局导航
  节点 `FieldRef / IndexRef / Deref`（见下节）。
* `registers`：`dict[str, int]`，寄存器名到整数值。
* `memory`：`MemorySnapshot({"region_id": b"..."})`，不可变区域快照；
  `Load`/`Field` 只能读取传入快照中的区域。

类型：`IntType(bits, signed)`（bits ∈ {8,16,32,64}），预置 `U8..U64 / S8..S64`；
指针类型 `POINTER`，指针值 `Pointer(region_id, offset)`。

## 语义契约（固定）

* 整数显式 signed/unsigned、8/16/32/64 位；字面量与寄存器值必须落在声明
  类型范围内，否则 `RangeError`。
* 二元算术（`add/sub/mul/div/mod/shl/shr/bitand/bitor/bitxor`）要求两个操作数
  **类型完全相同**，结果按模 2^bits 截断（回绕，不饱和、不隐式提升）。
* `div/mod` 向零截断（C 语义）；除零抛 `DivisionByZeroError`；
  `INT_MIN / -1` 等溢出同样按模回绕。
* 移位量必须满足 `0 <= amount < 位宽`，否则 `ShiftError`；`shr` 对有符号
  为算术右移、对无符号为逻辑右移。
* 比较（`eq/ne/lt/le/gt/ge`）按符号性解释，要求相同整数类型；结果为 `u8` 0/1。
  指针 `eq/ne` 可跨区域；大小比较与 `PtrSub` 要求同一区域，跨区域抛
  `PointerError`。`PtrSub` 结果为 `s64` 字节差。
* 指针 = (region_id, offset)。`PtrAdd` 以字节为单位加整数，结果必须落在
  同一区域内、**含尾后位置**（offset == region_size 合法）；越出抛
  `PointerError`。解引用（`Load`/`Field`）拒绝尾后位置与自然对齐不满足的
  地址（offset % size != 0）。加载为小端。
* `Cast` 仅整数→整数：可表示时保值，否则按目标宽度模截断。
* `And/Or` 短路：左操作数已决定结果时右操作数不求值（其越界读取等副作用
  不会触发）。`Not` 逻辑取反。三者结果均为 `u8` 0/1。
* 每次 `Load`/`Field` 向结果 `reads` 追加一条 `ReadInterval(region_id,
  offset, length)`。

## 类型布局与结构化访问（p.field / a[index]）

调用方提供一份**有限类型布局**（`debugexpr.layout`），表达式即可沿结构体
字段和数组下标取值。布局完全由调用方给出——字段偏移、数组长度、元素类型
及对齐——模块不猜测任何真实 C 编译器布局：

```python
from debugexpr import (
    StructLayout, StructField, ArrayLayout, IntLayout, PtrLayout,
    FieldRef, IndexRef, Deref, PtrConst, Const, U8, U16, U32,
)

point = StructLayout((
    StructField("tag", 0, IntLayout(U8)),
    StructField("code", 1, IntLayout(U32, align=1)),        # 未对齐（packed）字段
    StructField("matrix", 8, ArrayLayout(ArrayLayout(IntLayout(U16), 3), 2)),
    StructField("heap_obj", 24, PtrLayout("heap")),        # 指向另一区域的指针
))

# point.matrix[1][2]
expr = Deref(
    IndexRef(IndexRef(FieldRef(PtrConst("stack", 0), point, "matrix"),
                      ArrayLayout(ArrayLayout(IntLayout(U16), 3), 2), Const(1, U8)),
             ArrayLayout(IntLayout(U16), 3), Const(2, U8)),
    IntLayout(U16),
)
```

布局种类：`IntLayout(int_type, align)`（align 缺省为类型自然宽度）、
`PtrLayout(target_region, align=8)`（8 字节小端偏移量，目标区域由布局指名）、
`StructLayout(fields, align, union=False, size=None)`、
`ArrayLayout(element, length, align)`（`length=0` 为空数组，合法）。

语义要点：

* **左值与值分离**：`FieldRef` / `IndexRef` 只计算字段/元素的*地址*，
  本身不读取任何内存（`reads` 为空）；只有最终的 `Deref` 按布局读取
  恰好 `layout.size` 个字节。读取证据（`result.reads`）可对照验证无关
  字段从未被访问。
* **对齐按布局声明**：`Deref` 检查 `offset % layout.align == 0`，因此
  声明 `align=1` 的未对齐字段可以读取，而声明更大对齐的字段在错对齐
  地址上仍被拒绝。
* **越界下标带路径拒绝**：下标须满足 `0 <= index < length`（空数组拒绝
  一切下标），违反抛 `PointerError`，`path` 定位到 `(..., "index")`。
* **跨地址空间指针**：`PtrLayout` 解出 `(target_region, offset)` 后按原
  地址空间规则校验——目标区域不存在或偏移越界时抛带 `path` 的
  `PointerError`；合法时可直接继续 `FieldRef`/`IndexRef`/`Deref` 导航。
* **重叠字段**：普通结构体重叠布局在定义时即被 `ValueError` 拒绝；
  显式 `union=True` 允许重叠，表示对同一段字节的多种解释。
* 布局定义错误（负偏移、负长度、非法对齐、字段重名等）在构造时抛
  `ValueError`；求值期错误仍是带 `path` 的 `EvalError` 子类。

原标量求值 API（`evaluate(ast, registers, memory)` 及全部既有节点）
保持不变，新旧节点可在同一表达式中混用，节点预算 `MAX_NODES` 同样覆盖
新节点。

## 错误语义

所有失败抛出 `EvalError` 子类（`InputError / TypeMismatchError / RangeError /
ShiftError / DivisionByZeroError / PointerError / NodeBudgetError`），均携带
`path`（字段名元组，如 `("right",)` 定位到 `<root>.right`）。不静默纠正任何
非法输入：

* 整数参数拒绝 `bool`；数值输入拒绝一切非 `int`，包括 float 及
  NaN/Infinity（`InputError`）。
* `evaluate` 是纯函数：失败即抛异常，没有任何部分变更需要回滚
  （寄存器/快照均只读，结果在成功后才构造）。

## 输入上限

* AST 最多 `MAX_NODES = 500` 个不同节点（共享子树按一个节点计），超出抛
  `NodeBudgetError`。
* 内存区域数量与大小由调用方构造的快照决定，模块不另设上限。

## 浮点口径

本模块**不支持浮点**：表达式值、寄存器、字面量一律为显式位宽整数或指针，
任何 float（含 NaN/Infinity）输入直接拒绝。因此不存在浮点误差口径问题；
所有整数结果按上述模截断规则精确确定，无容差。

## 设计取舍

* 小端固定、指针算术以字节为单位（无类型缩放）、不支持整数↔指针互转——
  按本题契约实现的最小语义子集，不宣称完整 C 或某调试器表达式语言兼容。
* 共享子树在节点预算中只计一次，但求值时按出现逐次求值（无副作用，结果一致）。
* `MemorySnapshot` 构造时防御性拷贝，保证求值期间快照不可变。

## 已知限制

* 无浮点、无数组/结构体字面量、无指针↔整数 cast、无变量赋值。
* 布局导航只支持标量终值（`Deref` 仅接受 `IntLayout`/`PtrLayout`），
  不整体读取结构体/数组；内存中的指针固定为 8 字节小端区域偏移量。
* 区域为平坦字节序列，不支持权限位或空洞（gap）映射。
* 寄存器值在 `Reg` 节点处按声明类型做范围检查，未提供寄存器类型推断。

## 验证命令

```bash
python -m unittest discover -s tests -v   # 61 个测试（41 个基线回归 + 20 个布局导航）
python demo.py                            # 基线：正常求值 + 读取证据 + 一次实际拒绝
python demo_iteration.py                  # 布局导航演示（真实离线计算，远低于 90 秒）
```

`tests/test_layout.py` 中的期望地址与字节值由测试文件内的独立整数/切片
参考计算给出（不调用被测核心的地址逻辑），并与实际读取日志逐条对照，
证明无关字段未被读取。

测试中的算术/比较/转型期望值由 `tests/reference.py` 的独立整数掩码参考
实现给出（不导入被测核心），核心结果与之逐项交叉一致。
