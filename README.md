# debugexpr — 调试表达式的位宽与地址空间语义

为离线调试器表达式工程提供的可复用求值模块：显式位宽/符号性的整数语义、
带区域（region）的指针模型、内存快照加载与读取证据。

* Python 3.14.7，仅标准库，无需安装任何额外依赖。
* 全部离线：无网络、账号、数据库、Docker 或外部服务。
* 核心算法（模截断算术、截断除法、指针边界/对齐检查）在本仓库内从零实现，
  未调用第三方同类实现。

## 接口

```python
from debugexpr import evaluate, MemorySnapshot, Const, BinOp, Load, PtrConst, U8, S32

result = evaluate(ast, registers, memory)
result.value   # Value(type, data)：带类型的结果
result.reads   # tuple[ReadInterval]：实际读取的内存区间（求值证据，按求值顺序）
```

* `ast`：由 `debugexpr` 节点类构成的 AST：`Const / PtrConst / Reg / BinOp /
  Cmp / Cast / PtrAdd / PtrSub / Load / Field / And / Or / Not`。
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
* 区域为平坦字节序列，不支持权限位或空洞（gap）映射。
* 寄存器值在 `Reg` 节点处按声明类型做范围检查，未提供寄存器类型推断。

## 验证命令

```bash
python -m unittest discover -s tests -v   # 41 个测试
python demo.py                            # 正常求值 + 读取证据 + 一次实际拒绝
```

测试中的算术/比较/转型期望值由 `tests/reference.py` 的独立整数掩码参考
实现给出（不导入被测核心），核心结果与之逐项交叉一致。

## 结构化访问增量（FieldOf / IndexOf）

在原有标量求值 API（`evaluate(ast, registers, memory)` 签名与语义不变）之上，
新增调用方提供的**显式类型布局**与两个 AST 节点：

```python
from debugexpr import (
    FieldInfo, StructLayout, ArrayLayout, FieldOf, IndexOf, evaluate,
)

PACKET = StructLayout("Packet", 11, (
    FieldInfo("kind", 0, U8),
    FieldInfo("delta", 1, S32),          # 未对齐字段：按字节精确读取
    FieldInfo("samples", 5, ArrayLayout(U16, 3)),
))
result = evaluate(
    FieldOf(PtrConst("stack", 10), PACKET, "delta"),
    {}, memory, region_table=("stack", "heap"),
)
```

* `FieldOf(base, struct_layout, field)` 即 `p.field`；`IndexOf(base,
  array_layout, index)` 即 `a[index]`。`base` 为指针或聚合左值。
* 布局完全由调用方给出（字段偏移、数组长度、元素类型、对齐元数据），
  **不推导任何真实 C 编译器布局**；`align` 仅作记录，不强制。
* **左值与值分离**：选中聚合类型（结构体/数组）字段或元素时只计算地址
  （返回内部 `LValue`），不读取任何字节；只有到达标量叶子（`IntType` 或
  `POINTER`）才按布局精确读取所需字节。`reads` 证据可证明无关字段从未
  被读取。布局驱动的读取按字节精确，不要求自然对齐（未对齐字段可读）。
* 越界下标（含空数组的任意下标）抛 `PointerError` 并带 `path`；
  内存中的指针编码为 9 字节（1 字节区域标签 + 8 字节小端偏移），区域标签
  索引 `evaluate(..., region_table=...)`（默认取快照区域顺序）；标签越界或
  指向快照中不存在的区域时抛 `PointerError` 并带 `path`。
* 普通结构体布局在构造时（`ValueError`）拒绝字段重叠；显式
  `union=True` 才允许重叠字段按 union 解释。
* 顶层表达式若仍是聚合左值（未选到标量），抛 `TypeMismatchError`。

### 环境准备（隔离 venv）

```sh
python3 -m venv .venv          # Python 3.14.7，仅标准库，无第三方依赖
. .venv/bin/activate
python -m unittest discover -s tests -v   # 全部测试（基线回归 + 布局增量）
python demo.py                 # 基线标量演示
python demo_iteration.py       # 布局增量演示（真实离线计算，远低于 90 秒）
```

### 适用范围与已知限制

* 布局是有限的、调用方显式给出的；不支持位域、变长数组、浮点类型。
* 内存指针编码（1 字节标签 + 8 字节偏移）是本模块自定义格式，
  `region_table` 最多 256 个区域；不宣称与任何目标 ABI 的指针表示兼容。
* 聚合左值只能继续被 `FieldOf`/`IndexOf` 消费，不能参与算术或直接作为
  最终结果。
