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
* `memory`：`MemorySnapshot`，不可变的多段映射快照。每个区域可给一段
  `bytes`（即位于偏移 0 的单一可读段），也可给 `MemorySegment(offset, data,
  readable)` 或 `(offset, data[, readable])` 的序列（传入顺序任意，允许
  相邻、拒绝重叠）。没有任何段覆盖的偏移是未映射空洞；`readable=False`
  表示“已映射但不可读”。`Load`/`Field` 只能读取传入快照中的区域。

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
  `PointerError`，region_size 为该区域最后一个映射段的末端；指针可指向
  空洞上方的有效位置，但解引用会按下方规则失败。解引用（`Load`/`Field`）
  拒绝尾后位置与自然对齐不满足的地址（offset % size != 0）。加载为小端。
  一次读取可以跨越多个**相邻可读段**（结果自动拼接），但整个读取是原子
  的：只要有一个字节落在未映射空洞，抛 `UnmappedAddressError`；全部已映射
  却有任一字节落在 `readable=False` 的段上，抛 `UnreadableMemoryError`。
  失败时不返回半截数值，也不追加任何读取记录。
* `Cast` 仅整数→整数：可表示时保值，否则按目标宽度模截断。
* `And/Or` 短路：左操作数已决定结果时右操作数不求值（其越界读取等副作用
  不会触发）。`Not` 逻辑取反。三者结果均为 `u8` 0/1。
* 每次 `Load`/`Field` 按其实际接触到的每个可读段，向结果 `reads` 追加一条
  `ReadInterval(region_id, offset, length)`（偏移升序）：单段读取仍是一条，
  跨相邻段的读取每段一条，来源记录精确到段。

## 错误语义

所有失败抛出 `EvalError` 子类（`InputError / TypeMismatchError / RangeError /
ShiftError / DivisionByZeroError / PointerError / NodeBudgetError`），均携带
`path`（字段名元组，如 `("right",)` 定位到 `<root>.right`）。读取失败细分为
`UnmappedAddressError`（地址没有任何映射段，即空洞；尾后越界也归此）与
`UnreadableMemoryError`（地址已映射但段的 `readable=False`），二者都是
`PointerError` 子类，因此只关心地址空间错误的调用方仍可统一捕获。不静默
纠正任何非法输入：

* 整数参数拒绝 `bool`；数值输入拒绝一切非 `int`，包括 float 及
  NaN/Infinity（`InputError`）。
* `evaluate` 是纯函数：失败即抛异常，没有任何部分变更需要回滚
  （寄存器/快照均只读，结果在成功后才构造）。

## 输入上限

* AST 最多 `MAX_NODES = 500` 个不同节点（共享子树按一个节点计），超出抛
  `NodeBudgetError`。
* 内存区域数量、段数量与大小由调用方构造的快照决定，模块不另设上限；
  段在构造时排序并校验不重叠，空洞本身不占存储。

## 浮点口径

本模块**不支持浮点**：表达式值、寄存器、字面量一律为显式位宽整数或指针，
任何 float（含 NaN/Infinity）输入直接拒绝。因此不存在浮点误差口径问题；
所有整数结果按上述模截断规则精确确定，无容差。

## 设计取舍

* 小端固定、指针算术以字节为单位（无类型缩放）、不支持整数↔指针互转——
  按本题契约实现的最小语义子集，不宣称完整 C 或某调试器表达式语言兼容。
* 共享子树在节点预算中只计一次，但求值时按出现逐次求值（无副作用，结果一致）。
* `MemorySnapshot` 构造时防御性拷贝并排序段，保证求值期间快照不可变。
  空洞（无映射）与不可读（有映射、无权限）是两类不同错误：同一次读取
  只要范围全部已映射，权限位才参与判定，因此“空洞 + 不可读”混合范围
  一律报空洞；这样权限错误只代表地址确实存在于地址空间。

## 已知限制

* 无浮点、无数组/结构体字面量、无指针↔整数 cast、无变量赋值。
* 段只携带一个读权限位，不区分写/执行权限；段不可重叠；指针不携带
  权限信息，权限在每次读取时按当时快照判定。
* 寄存器值在 `Reg` 节点处按声明类型做范围检查，未提供寄存器类型推断。

## 验证命令

```bash
python -m unittest discover -s tests -v   # 52 个测试
python demo.py   # 跨段正常读取 + 读取证据 + 空洞/不可读两类实际拒绝
```

测试中的算术/比较/转型期望值由 `tests/reference.py` 的独立整数掩码参考
实现给出（不导入被测核心），核心结果与之逐项交叉一致。
