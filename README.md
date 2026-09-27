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
* `memory`：`MemorySnapshot({"region_id": b"..."})` 或
  `MemorySnapshot({"region_id": [MemorySegment(offset, data, readable), ...]})`，
  不可变地址空间快照；`Load`/`Field` 只能读取传入快照中的区域。
  每个区域由若干**互不重叠**的映射段组成，段间空隙即未映射空洞；
  每段有独立的 `readable` 读取权限。

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
  同一区域的映射范围内、**含尾后位置**（offset == region_size 合法）；
  越出抛 `PointerError`。解引用（`Load`/`Field`）拒绝尾后位置与自然对齐
  不满足的地址（offset % size != 0）。加载为小端。
* 多段读取语义：一次加载跨越若干**相邻且可读**的段时整体成功，字节按地址
  顺序拼接；只要区间跨入未映射空洞（含越过映射范围）就抛
  `UnmappedAddressError`，跨入已映射但 `readable=False` 的段就抛
  `ReadPermissionError`。整段区间先校验后取数，因此失败是**原子**的：
  绝不返回半个数值，也绝不留下部分读取记录。
* `Cast` 仅整数→整数：可表示时保值，否则按目标宽度模截断。
* `And/Or` 短路：左操作数已决定结果时右操作数不求值（其越界读取等副作用
  不会触发）。`Not` 逻辑取反。三者结果均为 `u8` 0/1。
* 每次 `Load`/`Field` 向结果 `reads` 追加读取证据：对实际触及的每个后端段
  各追加一条 `ReadInterval(region_id, offset, length)`（即
  `debugexpr.memory.ReadSource`，按地址升序、裁剪到实际用到的字节）。
  单段加载仍是恰好一条；跨相邻多段的加载则每段一条，来源精确可查。

## 错误语义

所有失败抛出 `EvalError` 子类（`InputError / TypeMismatchError / RangeError /
ShiftError / DivisionByZeroError / PointerError / UnmappedAddressError /
ReadPermissionError / NodeBudgetError`），均携带 `path`（字段名元组，如
`("right",)` 定位到 `<root>.right`）。`UnmappedAddressError` 与
`ReadPermissionError` 都是 `PointerError` 的子类，用于区分“地址根本没有映射
的空洞”和“有映射但无读取权限”这两种不同故障。不静默纠正任何非法输入：

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
* 区域支持多段映射、段间空洞与逐段读取权限；段在构造后不可变，不支持
  写权限/执行权限等其它保护位。
* 寄存器值在 `Reg` 节点处按声明类型做范围检查，未提供寄存器类型推断。

## 验证命令

```bash
python -m unittest discover -s tests -v   # 53 个测试
python demo.py                            # 正常求值 + 跨段读取证据 + 两类实际拒绝
```

测试中的算术/比较/转型期望值由 `tests/reference.py` 的独立整数掩码参考
实现给出（不导入被测核心），核心结果与之逐项交叉一致。
