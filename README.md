# debugexpr — 调试表达式的位宽与地址空间语义

为离线调试器表达式工程提供的可复用求值模块：显式位宽/符号性的整数语义、
带区域（region）的指针模型、内存快照加载与读取证据、**寄存器多候选值的
候选结果集合求值**与**条件分支的可达性分析**。

* Python 3.14.7，仅标准库，无需安装任何额外依赖。
* 全部离线：无网络、账号、数据库、Docker 或外部服务。
* 核心算法（模截断算术、截断除法、指针边界/对齐检查、候选世界枚举）在本
  仓库内从零实现，未调用第三方同类实现。

## 接口

```python
from debugexpr import evaluate, MemorySnapshot, Const, BinOp, Load, PtrConst, If, U8

result = evaluate(ast, registers, memory)
result.status   # "exact" | "possible_error" | "unknown"
result.results  # tuple[Value]：全部可行的带类型结果（确定序）
result.errors   # tuple[EvalError]：只有部分赋值会触发的错误（带 path）
result.reads    # tuple[ReadInterval]：某些可行赋值实际读取的内存区间
result.value    # 旧入口：结果唯一时的 Value；不唯一时抛 RuntimeError
```

* `ast`：由 `debugexpr` 节点类构成的 AST：`Const / PtrConst / Reg / BinOp /
  Cmp / Cast / PtrAdd / PtrSub / Load / Field / And / Or / Not / If`。
* `registers`：`dict[str, int | tuple[int, ...] | list[int]]`。普通 `int`
  表示单一候选值；tuple/list 给出该寄存器的候选整数集合（自动去重、保留
  首次出现顺序），每个寄存器至多 `MAX_REGISTER_CANDIDATES = 8` 个不同候选。
  `bool`、`float`（含 NaN/Infinity）候选一律拒绝；空候选列表拒绝。
* `memory`：`MemorySnapshot({"region_id": b"..."})`，不可变区域快照；
  `Load`/`Field` 只能读取传入快照中的区域。
* 可选关键字 `on_budget_overrun`：`"unknown"`（默认，超预算返回
  `status="unknown"`）或 `"raise"`（超预算抛 `CandidateBudgetError`）。

类型：`IntType(bits, signed)`（bits ∈ {8,16,32,64}），预置 `U8..U64 / S8..S64`；
指针类型 `POINTER`，指针值 `Pointer(region_id, offset)`。

## 候选值与结果状态

模块对寄存器候选的**全部具体赋值（候选世界）做精确枚举**（无指针来源分析、
无通用符号求解，仅小规模枚举）：

* 每个世界按既有标量语义独立求值；`results` 是所有世界产出的不同值集合，
  `reads` 是所有世界实际读取区间的并集（去重、确定序）。只有被 AST 引用的
  寄存器才计入候选空间；未引用的候选寄存器不参与枚举。
* `status`：
  * `"exact"`：所有世界都成功，`results` 即完整可行值集合，`errors` 为空；
  * `"possible_error"`：部分世界成功、部分世界失败；成功值在 `results`，
    去重后的失败在 `errors`（每个都是带 `path` 的 `EvalError`）；
  * `"unknown"`：候选世界数超过 `MAX_CANDIDATE_COMBINATIONS = 64`（且使用
    默认 `"unknown"` 模式）。此时不声称任何动态结果：`results/errors/reads`
    均为空；但**与赋值无关的非法输入仍然抛错**（见下"超预算下的校验"）。
* 若所有世界都失败：失败形态唯一时直接抛该错误（确定性错误）；失败形态不
  唯一时抛 `PossibleError`（其 `.errors` 汇总各形态）。
* 候选空间恰为 64 时仍精确枚举；第 65 个世界起进入超预算处理。

## 条件分支与短路

* `If(cond, then, else_)`：条件为整数（0 假、非 0 真）。枚举是精确的——
  每个世界只求值被选中的分支，因此未选中分支的读取不会进入 `reads`、其
  除零/越界等错误也不会触发。条件未知（部分世界为 0、部分非 0）时两条分支
  都会被探索，但各自只在能到达它的世界上求值。条件可以是任意整数表达式
  （如 `Cmp`）；条件为指针是 `TypeMismatchError`。
* `And/Or` 的短路在每个世界内独立生效：被短路掉的右侧表达式不读取内存、
  不产生错误。若左侧候选值真假混杂，则"短路结果"与"右侧求值结果"都会
  进入候选集合，右侧可能发生的错误按部分失败处理。
* `Not` 与逻辑运算结果仍为 `u8` 0/1。

## 语义契约（固定）

* 整数显式 signed/unsigned、8/16/32/64 位；字面量必须落在声明类型范围内，
  寄存器候选值在对应 `Reg` 节点处按声明类型逐个检查（不合法的候选产生该
  世界的 `RangeError`，即部分失败）。
* 二元算术（`add/sub/mul/div/mod/shl/shr/bitand/bitor/bitxor`）要求两个操作数
  **类型完全相同**，结果按模 2^bits 截断（回绕，不饱和、不隐式提升）。
  除零/移位越界只对实际到达该运算的世界构成错误。
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
* 每次 `Load`/`Field` 向结果 `reads` 追加一条 `ReadInterval(region_id,
  offset, length)`；候选模式下为跨世界去重后的读取证据。

### 超预算下的校验

进入 `unknown` 路径后，仍然检查所有与具体赋值无关的输入合法性并直接抛错：
AST 节点字段类型、运算符名、`IntType` 注解、整数字面量的 int 性与范围、
寄存器名是否存在、`PtrConst` 字面量是否在区域内、`Field` 偏移是否为 int、
未知条件下两条分支的结构合法性（常量条件则与标量惰性一致，只检查会选中的
分支；`And/Or` 常量左值同理）。所有依赖具体寄存器值的判定（动态除零、动态
指针边界/对齐、候选值范围、世界间类型一致性）在该路径下不做声称。

## 错误语义

所有失败抛出 `EvalError` 子类（`InputError / TypeMismatchError / RangeError /
ShiftError / DivisionByZeroError / PointerError / NodeBudgetError /
CandidateBudgetError / PossibleError`），均携带 `path`（字段名元组，如
`("right",)` 定位到 `<root>.right`；`If` 的 else 分支路径为 `("else_",)`）。
不静默纠正任何非法输入：

* 整数参数拒绝 `bool`；数值输入拒绝一切非 `int`，包括 float 及
  NaN/Infinity（`InputError`）。
* `evaluate` 是纯函数：失败即抛异常，没有任何部分变更需要回滚
  （寄存器/快照均只读，结果在所有世界求值完成后才一次性构造）。

## 输入上限

* AST 最多 `MAX_NODES = 500` 个不同节点（共享子树按一个节点计），超出抛
  `NodeBudgetError`。
* 每个寄存器最多 `MAX_REGISTER_CANDIDATES = 8` 个不同候选整数，超出
  `InputError`。
* 精确枚举的候选世界上限 `MAX_CANDIDATE_COMBINATIONS = 64`（按 AST 实际
  引用的寄存器候选数连乘）；超出返回 `unknown`，或经
  `on_budget_overrun="raise"` 抛 `CandidateBudgetError`。
* 内存区域数量与大小由调用方构造的快照决定，模块不另设上限。

## 浮点口径

本模块**不支持浮点**：表达式值、寄存器、字面量、候选值一律为显式位宽整数
或指针，任何 float（含 NaN/Infinity）输入直接拒绝。因此不存在浮点误差口径
问题；所有整数结果按上述模截断规则精确确定，无容差。

## 设计取舍

* 候选语义采用"逐世界精确枚举"而非区间/集合上近似：在 64 世界预算内结果、
  读取证据与可能错误都是精确的，寄存器间相关性天然保持（不会把 `x+x`
  错算成跨候选组合）。超出预算即保守地整体 `unknown`，不报告任何无法证明
  的动态结论；本次迭代不做指针来源分析、镜像增量精化或通用符号求解。
* 小端固定、指针算术以字节为单位（无类型缩放）、不支持整数↔指针互转——
  按本题契约实现的最小语义子集，不宣称完整 C 或某调试器表达式语言兼容。
* 共享子树在节点预算中只计一次，但求值时按出现逐次求值（无副作用，结果一致）。
* `MemorySnapshot` 构造时防御性拷贝，保证求值期间快照不可变。
* `EvalResult.value` 作为旧入口保留：结果唯一（单候选、或多候选恰好坍缩为
  同一值）时可用，否则显式要求调用方改用 `.results`。

## 已知限制

* 无浮点、无数组/结构体字面量、无指针↔整数 cast、无变量赋值。
* 区域为平坦字节序列，不支持权限位或空洞（gap）映射。
* 寄存器值在 `Reg` 节点处按声明类型做范围检查，未提供寄存器类型推断。
* 超预算（>64 世界）只保留与赋值无关的静态校验，动态结果一律 `unknown`；
  不会对部分子表达式单独求精。
* 不追踪候选值的来源/名字，`results`/`errors` 不附带"由哪个赋值产生"的
  归因信息。

## 验证命令

```bash
python -m unittest discover -s tests -v   # 87 个测试（41 旧 + 46 候选/分支/预算）
python demo.py                            # 候选集合/未知分支/可能错误/确定拒绝/超预算
```

测试中的算术期望值由 `tests/reference.py` 的独立整数掩码参考实现给出
（不导入被测核心）；候选测试在测试内用 `itertools.product` 对每个具体寄存器
赋值独立调用该参考实现汇总期望值，与被测的逐世界枚举互不共用代码。
