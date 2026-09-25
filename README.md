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

## 一次性编辑事务（可撤销的内存编辑预览）

在只读 `evaluate` 之外，`begin_edit(memory)` 开启一个**一次性**编辑事务，
把若干条带类型的 store 暂存在私有覆盖层上，供表达式预览，最后由调用方
二选一：`commit()` 生成新快照，或 `discard()` 丢弃。底层始终是调用方传入
的区域字节快照，不涉及本机地址空间。

```python
from debugexpr import begin_edit, Store, Pointer, Load, PtrConst, S32

txn = begin_edit(memory)                    # memory 是 MemorySnapshot，永不被修改
txn.store(Store(Pointer("stack", 16), S32, -7))   # 暂存一条 typed store
preview = txn.evaluate(Load(PtrConst("stack", 16), S32))  # 读到待写值 -7
txn.trace                                 # 已执行的读/写访问轨迹（AccessRecord）
new_memory = txn.commit()                 # 返回应用了全部写入的新快照
# 或者 txn.discard() 丢弃全部暂存写入
```

事务语义（固定契约）：

* 每条 `store` 先验证：指针区域存在、偏移在区域内且 `offset + width` 不越过
  区域末尾（**尾后位置拒绝写入**）、按宽度自然对齐、值为非 bool 的 int 且
  落在声明类型范围内（小端写入）。任一不满足抛对应 `EvalError` 子类。
* 重叠写按语句顺序覆盖（后写逐字节覆盖先写）；不同 region 即使偏移相同
  也互不影响。
* `txn.evaluate` 复用只读求值器，读到本事务此前暂存的写入；`And/Or`
  短路未执行的部分（包括其中可能触及的读取）不产生任何访问记录。
* **任何一步失败（store 被拒或表达式求值失败）整个事务回滚**：暂存层被
  丢弃，状态变为 `rolled-back`，`commit()` 不再可能——不会漏出部分新内存。
  已尝试的访问轨迹仍可通过 `txn.trace` 读取（`AccessRecord(kind,
  region_id, offset, length)`，`kind` 为 `"read"`/`"write"`，按执行顺序）。
* 事务为一次性：`commit`/`discard`/回滚后所有操作抛 `TransactionError`，
  不能复用。本模块不是调试协议，也不做持久化——一次只有一个预览事务。

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
ShiftError / DivisionByZeroError / PointerError / NodeBudgetError /
TransactionError`），均携带
`path`（字段名元组，如 `("right",)` 定位到 `<root>.right`）。不静默纠正任何
非法输入：

* 整数参数拒绝 `bool`；数值输入拒绝一切非 `int`，包括 float 及
  NaN/Infinity（`InputError`）。
* `evaluate` 是纯函数：失败即抛异常，没有任何部分变更需要回滚
  （寄存器/快照均只读，结果在成功后才构造）。编辑事务同样不修改调用方
  快照：写入只落在事务私有的字节覆盖层，`commit()` 返回全新快照。

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
* 编辑事务的覆盖层按 region 懒拷贝（首次写某 region 时复制其字节），
  读取经只读视图透传并记录访问轨迹；回滚只是丢弃覆盖层，代价为 O(1)。

## 已知限制

* 无浮点、无数组/结构体字面量、无指针↔整数 cast、无变量赋值。
* 区域为平坦字节序列，不支持权限位或空洞（gap）映射。
* 寄存器值在 `Reg` 节点处按声明类型做范围检查，未提供寄存器类型推断。
* 编辑事务是一次性内存预览：不支持并发事务、断点续作、调试协议交互
  或任何持久化；`commit` 之后想要新编辑需重新 `begin_edit`。

## 验证命令

```bash
python -m unittest discover -s tests -v   # 66 个测试
python demo.py                            # 正常求值 + 读取证据 + 实际拒绝 + 编辑事务演示
```

测试中的算术/比较/转型期望值由 `tests/reference.py` 的独立整数掩码参考
实现给出（不导入被测核心），核心结果与之逐项交叉一致；编辑事务的期望
内存映像由 `tests/test_transaction.py` 内独立的字节数组手工拼出，并验证
调用方传入的源快照在 commit/rollback 后内容不变。
