# debugexpr — 调试表达式的位宽、地址空间与候选值语义

为离线调试器表达式工程提供的可复用求值模块：显式位宽/符号性的整数语义、
带区域（region）的指针模型、内存快照加载与读取证据；本轮新增**寄存器候选
整数**、表达式的**候选结果集合**、路径敏感的**条件分支 `Select`**，以及
候选组合超预算时明确的 **unknown** 信号。

* Python 3.14.7，仅标准库，无需安装任何额外依赖。
* 全部离线：无网络、账号、数据库、Docker 或外部服务。
* 核心算法（模截断算术、截断除法、指针边界/对齐检查、候选集合枚举）在本
  仓库内从零实现，未调用第三方同类实现。

## 接口

```python
from debugexpr import evaluate, MemorySnapshot, Const, Reg, BinOp, Load, Select, U8

# registers 的值可以是单个 int（旧用法），也可以是 1..8 个候选 int
result = evaluate(ast, {"r": (1, 2, 3)}, memory)
result.type            # 所有候选共享的类型
result.values          # 去重后按确定顺序排列的候选数据 (int 或 Pointer)
result.candidates      # CandidateSet（1..8 个成员）
result.is_determined   # 恰好 1 个候选时为 True
result.value           # 确定时返回 Value(type, data)，否则为 None
result.reads           # tuple[ReadInterval]：实际可能读取的内存区间
```

* `ast`：由 `debugexpr` 节点类构成的 AST：`Const / PtrConst / Reg / BinOp /
  Cmp / Cast / PtrAdd / PtrSub / Load / Field / And / Or / Not / Select`。
* `registers`：`dict[str, int | tuple[int,...] | list[int] | set[int]]`。
  普通 `int` 与旧行为完全一致；序列给出该寄存器 1..8 个可能值（自动去重、
  排序），`bool/float`（含 NaN/Infinity）成员一律拒绝。
* `memory`：`MemorySnapshot({"region_id": b"..."})`，不可变区域快照；
  `Load`/`Field` 只能读取传入快照中的区域。
* `evaluate(..., *, max_combinations=4096)`：候选枚举的组合预算，必须为
  正整数；超出抛 `CandidateBudgetError`（即 unknown）。

类型：`IntType(bits, signed)`（bits ∈ {8,16,32,64}），预置 `U8..U64 / S8..S64`；
指针类型 `POINTER`，指针值 `Pointer(region_id, offset)`；候选集合
`CandidateSet`（`MAX_CANDIDATES = 8`）。

## 候选值语义（本轮新增）

* 每个寄存器最多 8 个候选整数；子表达式按**候选集合**求值，集合始终
  去重、确定性排序，最多保留 8 个不同值。
* 二元运算/比较对左右候选做笛卡尔积逐组合计算：
  * 每个组合都满足原有标量契约（模截断、移位范围、指针对齐/边界等）；
  * **任一组合**除零、移位越界、指针越界等，即按对应错误整体抛出——
    可能真实发生的执行错误不会被"其他组合正常"掩盖；
  * 派生出超过 8 个不同值时无法用候选集合表示，抛
    `CandidateBudgetError`（unknown），绝不静默丢值。
* 同一寄存器在表达式中出现多次时，各次出现按**独立**候选组合处理
  （不做"同一寄存器必然相等"的关联分析——本轮明确不交付指针来源/相关性
  分析）。
* 纯标量输入（所有寄存器都是单值）下候选集合恒为单元素，枚举是线性的，
  因此**永远不会**产生 unknown；`max_combinations` 只在真正读到多候选
  寄存器后才开始计费。
* `CandidateBudgetError` 是 `EvalError` 子类，但它表示"算不动（unknown）"
  而非"输入非法"：表达式本身合法，调大 `max_combinations` 后可能可枚举。

## Select 条件分支（本轮新增）

`Select(cond, then, else_)`，条件为整数：

* 条件取值被严格限制为 **0 或 1**：出现任何其他可能值（含候选集合中混入
  2/-1/255 等）抛 `RangeError`。
* 条件**确定为 1** 时只求值 `then`，确定为 0 时只求值 `else_`：另一支的
  内存读取、除零、越界指针等完全不发生（读取证据也不包含它）。
* 条件候选为 `{0,1}`（可能走任一支）时两支都实际求值：
  * 结果候选为两支候选的并集（仍受 8 个上限约束）；
  * 两支类型必须相同，否则 `TypeMismatchError`；
  * 读取证据先列 `then` 支全部读取，再列 `else_` 支全部读取；
  * 任一支存在可真实触发的错误（如除零），即按该错误抛出。
* 未被选中的分支不做类型检查——只有实际可能执行的代码才被读取。

## 标量语义契约（保持不变）

* 整数显式 signed/unsigned、8/16/32/64 位；字面量与寄存器的每个候选值
  必须落在声明类型范围内，否则 `RangeError`。
* 二元算术（`add/sub/mul/div/mod/shl/shr/bitand/bitor/bitxor`）要求两个
  操作数 **类型完全相同**，结果按模 2^bits 截断（回绕，不饱和、不隐式提升）。
* `div/mod` 向零截断（C 语义）；除零抛 `DivisionByZeroError`；
  `INT_MIN / -1` 等溢出同样按模回绕。
* 移位量必须满足 `0 <= amount < 位宽`，否则 `ShiftError`；`shr` 对有符号
  为算术右移、对无符号为逻辑右移。
* 比较（`eq/ne/lt/le/gt/ge`）按符号性解释，要求相同整数类型；结果为 `u8`
  0/1。指针 `eq/ne` 可跨区域；大小比较与 `PtrSub` 要求同一区域，跨区域抛
  `PointerError`。`PtrSub` 结果为 `s64` 字节差。
* 指针 = (region_id, offset)。`PtrAdd` 以字节为单位加整数，结果必须落在
  同一区域内、**含尾后位置**（offset == region_size 合法）；越出抛
  `PointerError`。解引用（`Load`/`Field`）拒绝尾后位置与自然对齐不满足的
  地址。加载为小端。
* `Cast` 仅整数→整数：可表示时保值，否则按目标宽度模截断。
* `And/Or` 短路按候选逐路径解释：左操作数在所有路径上都已决定结果时，
  右操作数不求值；否则右操作数在可到达路径上求值，结果集合合并
  （如左候选 `{0,5}`、右候选 `{0,7}` 的 `and` 结果为 `{0,1}`）。
  `Not` 逻辑取反。三者结果均为 `u8` 0/1。
* 每次 `Load`/`Field` 向结果 `reads` 追加一条
  `ReadInterval(region_id, offset, length)`；候选指针会按不同地址分别追加。

## 错误语义

所有失败抛出 `EvalError` 子类（`InputError / TypeMismatchError / RangeError /
ShiftError / DivisionByZeroError / PointerError / NodeBudgetError /
CandidateBudgetError`），均携带 `path`（字段名元组，如 `("right",)` 定位到
`<root>.right`；`Select` 分支路径名为 `cond / then / else_`）。不静默纠正
任何非法输入：

* 整数参数拒绝 `bool`；数值输入拒绝一切非 `int`，包括 float 及
  NaN/Infinity（`InputError`）。
* `evaluate` 是纯函数：寄存器映射与快照只读（候选序列也不会被修改），
  结果只在成功后构造；失败时没有任何半更新状态可观察。

## 输入上限

* AST 最多 `MAX_NODES = 500` 个不同节点（共享子树按一个节点计），超出抛
  `NodeBudgetError`。
* 每个寄存器最多 `MAX_CANDIDATES = 8` 个候选；每个子表达式最多 8 个不同
  可能值。
* 候选组合预算 `max_combinations = MAX_COMBINATIONS = 4096`（默认，可按
  关键字调大）；预算耗尽或结果值超过 8 个均为 `CandidateBudgetError`。
* 内存区域数量与大小由调用方构造的快照决定，模块不另设上限。

## 浮点口径

本模块**不支持浮点**：表达式值、寄存器、字面量一律为显式位宽整数或指针，
任何 float（含 NaN/Infinity）输入直接拒绝。因此不存在浮点误差口径问题；
所有整数结果按上述模截断规则精确确定，无容差。

## 设计取舍

* 候选语义是"可能性集合"的可靠枚举，不是通用符号执行：不区分集合中各值
  的出现次数、不做寄存器同名关联、不做指针来源分析或增量精化——均为本轮
  明确不交付的能力。
* 小端固定、指针算术以字节为单位（无类型缩放）、不支持整数↔指针互转——
  按本题契约实现的最小语义子集，不宣称完整 C 或某调试器表达式语言兼容。
* 共享子树在节点预算中只计一次；候选求值时按出现逐次枚举。
* `MemorySnapshot` 构造时防御性拷贝，保证求值期间快照不可变。

## 已知限制

* 无浮点、无数组/结构体字面量、无指针↔整数 cast、无变量赋值。
* 区域为平坦字节序列，不支持权限位或空洞（gap）映射。
* 寄存器值在 `Reg` 节点处按声明类型做范围检查，未提供寄存器类型推断。
* 同名寄存器多次出现不关联，可能高估可能值集合（可靠但不精确）；不做
  路径条件关联，因此 unknown 可能偏保守。
* 候选集合上限 8 与组合预算是表示/搜索边界，扩大需要不同的数据结构，
  本模块不尝试。

## 验证命令

```bash
python -m unittest discover -s tests -v   # 104 个测试
python demo.py                            # 标量/候选/分支/边界/真实拒绝
```

测试中的算术/比较/转型期望值由 `tests/reference.py` 的独立整数掩码参考
实现给出（不导入被测核心），标量逐项交叉一致，候选用例以笛卡尔积方式与
该参考实现及手算例子双重核对。
