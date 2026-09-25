# debugexpr — 调试表达式的位宽与地址空间语义

为离线调试器表达式工程提供的可复用求值模块：显式位宽/符号性的整数语义、
带区域（region）的指针模型、内存快照加载与读取证据，以及只读求值之上的
一次性编辑事务（可提交/可丢弃的内存编辑预览）。

* Python 3.14.7，仅标准库，无需安装任何额外依赖。
* 全部离线：无网络、账号、数据库、Docker 或外部服务。
* 核心算法（模截断算术、截断除法、指针边界/对齐检查、编辑事务 overlay）
  在本仓库内从零实现，未调用第三方同类实现。

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

### 一次性编辑事务

```python
from debugexpr import run_transaction, Store, PtrConst, Const, Load, U8, U32

txn = run_transaction(
    [
        Store(PtrConst("stack", 16), U32, Const(0x11223344, U32)),  # 带类型的写
        Store(PtrConst("stack", 0), U8, Load(PtrConst("stack", 16), U8)),  # 读到上面的待写值
    ],
    registers,
    memory,          # 源快照；事务期间与之后都绝不被修改
)
txn.ok               # 全部语句成功则为 True
txn.trace            # tuple[AccessRecord]：已执行的读/写访问轨迹（按执行顺序）
txn.error            # 失败时为第一个失败语句的 EvalError，成功时为 None
new_memory = txn.commit()   # 成功结果二选一：提交，返回应用了写入的新 MemorySnapshot
txn.discard()               # 或：丢弃，什么都不发生
```

* `Store(ptr, type, value)`：`ptr`/`value` 为普通求值 AST。`ptr` 须求值为
  指针，`value` 须求值为能装入 `type`（`IntType`）的整数；写入为
  `type.size` 字节小端。
* 语句按顺序在源快照的**私有 overlay** 上执行：后续语句的指针/值表达式
  通过 overlay 读到前面语句的待写值（read-your-writes）。
* 每条 store 依次验证：指针范围（拒绝尾后写）、声明类型的自然对齐、
  整数值落在声明类型表示范围内；任一不满足即失败。
* 同一区域内的重叠写按语句顺序覆盖（后写赢）；不同 region 即使偏移相同
  也互不影响。
* 任一步失败 → 整个事务回滚：结果携带已尝试的访问轨迹（`trace`），
  但**不暴露任何部分新内存**——失败结果上 `commit()`/`discard()` 都会
  抛 `InputError`。失败语句之后的语句从不执行，无副作用；`And`/`Or`
  短路未求值的子表达式同样无读取、无副作用。
* 提交与丢弃都是一次性的：成功结果上二者合计只能调用一次，再次调用抛
  `InputError`。`commit()` 返回的新快照由全新字节构造，源快照保持不变。
* 事务只在 Python 堆上复制调用方传入的区域字节，不触碰本机地址空间；
  它是一次性编辑预览，不是调试协议，也不做任何持久化。

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
* `run_transaction` 的语句级失败（指针/对齐/范围/类型等）**不抛出**，
  而是回滚后放入返回结果的 `error` 字段（同为 `EvalError` 子类，含
  `path`，如 `("statements[1]", "value")`）；只有 Python 级输入契约
  违规（非 Store 序列、寄存器值是 bool/float、非 MemorySnapshot 等）
  才立即抛 `InputError`。无论成败，源快照都不会被修改。

## 输入上限

* AST 最多 `MAX_NODES = 500` 个不同节点（共享子树按一个节点计），超出抛
  `NodeBudgetError`；事务中每条 store 的 `ptr`/`value` 表达式分别受此限制。
* 一个事务最多 `MAX_STATEMENTS = 128` 条 store 语句，超出抛 `InputError`。
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
* 编辑事务的 overlay 采用惰性 copy-on-write：只有被写的 region 才会被
  复制，未触碰的 region 在 `commit()` 时直接复用源快照字节。

## 已知限制

* 无浮点、无数组/结构体字面量、无指针↔整数 cast、无变量赋值。
* 区域为平坦字节序列，不支持权限位或空洞（gap）映射。
* 寄存器值在 `Reg` 节点处按声明类型做范围检查，未提供寄存器类型推断。
* 编辑事务为单次、进程内、非持久：无嵌套/并发事务，无调试协议接线，
  提交结果只是一个新的 `MemorySnapshot`，由调用方自行处置。

## 验证命令

```bash
python -m unittest discover -s tests -v   # 63 个测试（41 个只读求值回归 + 22 个编辑事务）
python demo.py                            # 正常求值 + 读取证据 + 实际拒绝 + 事务回滚/提交演示
```

测试中的算术/比较/转型期望值由 `tests/reference.py` 的独立整数掩码参考
实现给出（不导入被测核心）；事务测试的期望内存内容在
`tests/test_transaction.py` 内用独立 bytearray 手工拼出，并逐项验证
调用方源快照在事务后字节不变。核心结果与两类参考交叉一致。
