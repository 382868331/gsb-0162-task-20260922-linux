# dbgx — 离线调试器表达式：位宽与地址空间语义

一个从零实现的、可导入的 Python 库，对离线调试器表达式 AST 求值，严格实现
**显式位宽/符号性的整数语义**与**基于区域（region）的地址空间语义**。

* Python 版本：**Python 3.14.7**（仅用标准库，3.10+ 的语法亦可运行，但验收环境为 3.14.7）
* 依赖：**无第三方依赖**，`pip install` 后无需安装任何其他包
* 网络/账号/硬件/数据库/中间件/Docker：均不需要
* 测试与演示全部离线，示例数据由 `demo.py` 内确定性代码生成

## 运行命令

```bash
# 必要测试（96 个用例，含 800 组独立参考差分）
python -m unittest discover -s tests -v

# 演示（固定输入，真实计算，约 0.05 秒，远小于 8 秒）
python demo.py
```

## 公开接口

```python
from dbgx import evaluate, Value, IntType, PtrType, Region, ReadRange

result = evaluate(ast, registers, memory)
# result.value      : Value   —— 带类型的结果值
# result.reads      : tuple[ReadRange, ...] —— 实际执行的 load 读取区间
# result.node_count : int     —— AST 静态节点数（<= 500）
```

* `registers: dict[str, Value]`：只读寄存器表。
* `memory: dict[str, Region]`：只读**区域快照**；`load` 只能读这些区域，
  不会也无法访问任何其他地址空间。构造 `Region` 时数据被拷贝为不可变
  `bytes`，调用方之后改动原缓冲不影响求值。
* 求值是纯函数：不修改入参；失败时抛异常，不返回半成品，不存在部分变更。

### 数据结构

| 类型 | 含义 |
|---|---|
| `IntType(width, signed)` | 固定宽度整数类型，`width ∈ {8,16,32,64}`，`signed: bool` |
| `PtrType(region_id, pointee_width)` | 指针类型：指向某区域、某负载宽度 |
| `Value(IntType, raw)` | 整数值；`raw` 为无符号机器表示 `0..2^w-1`，符号性只由类型决定 |
| `Value(PtrType, (region_id, offset))` | 指针值；`offset` 为绝对字节地址，非负 |
| `Region(region_id, data, base=0)` | 区域快照；尾后地址为 `base + len(data)` |
| `ReadRange(region_id, start, end)` | 半开字节区间 `[start, end)`，读取证据 |

`Value` 便捷方法：`as_int()`（无符号视角）、`as_signed()`（有符号视角）、
`as_ptr()`、`Value.int_(w, signed, x)`（构造时按模截断）。

### AST 节点格式（tuple，首元素为算子名）

| 节点 | 语义 |
|---|---|
| `("int", value:int, width, signed)` | 整数字面量，构造时按模 2^w 截断 |
| `("reg", name:str)` | 读寄存器 |
| `("add","sub","mul", l, r)` | 回绕算术，结果位模式 `(l∘r) mod 2^w` |
| `("udiv","urem", l, r)` | 无符号除/余，除零抛 `DivisionByZeroError` |
| `("sdiv","srem", l, r)` | 有符号除/余，**向零截断**（C 语义）；`INT_MIN / -1` 回绕为 `INT_MIN` |
| `("and","or","xor", l, r)` | 位运算 |
| `("shl","lsr", l, amt)` | 左移/逻辑右移；移位量按机器表示必须 `0 <= amt < w` |
| `("asr", l, amt)` | 算术右移（高位补符号位）；**仅允许有符号整数类型** |
| `("eq","ne","lt","le","gt","ge", l, r)` | 比较；有符号类型按有符号排序，结果为 **u8** 0/1 |
| `("lnot", x)` | 逻辑非（整数 != 0），结果 u8 |
| `("band", l, r)` / `("bor", l, r)` | **短路**与/或，结果 u8；跳过的子树不执行、不读内存 |
| `("cast", x, width, signed)` | 位宽/符号转换：保留位模式，变窄截断、变宽零扩展、同宽换符号性即重解释 |
| `("field", x, lsb, width, signed)` | 位域提取：取 `x` 的 `[lsb, lsb+width)` 位，结果宽度 ∈ {8,16,32,64} |
| `("ptr", region_id, offset, pointee_width)` | 指针常量 |
| `("padd", p, i)` / `("psub", p, i)` | 指针 ± **有符号解释**的整数（字节单位） |
| `("pdiff", p1, p2)` | 同区域指针字节差，结果 **i64**；跨区域抛 `AddressError` |
| `("load", p)` | 对齐小端解引用；结果为负载宽度的**无符号**整数（需有符号请用同宽 `cast`） |

## 契约要点（语义取舍）

1. **二元整数算术要求两侧完全同类型**：位宽与符号性都必须一致，否则
   `TypeMismatchError`，不做隐式提升。
2. **一切算术按模截断**：中间用 Python 大整数精确计算，输出按 `mod 2^w`
   得到非负机器表示，因此不存在溢出标志或异常（除零除外）。
3. **比较按符号性**：同一模式 `0xFFFF`，u16 下 65535，i16 下 -1；
   `lt/gt/...` 按各侧类型的符号性解释。等不等价仅看位模式（仍要求同类型）。
4. **移位量必须小于位宽**：按右侧机器表示检查 `amt < w`；`amt == w`、
   `amt > w` 一律 `ShiftError`。有符号 `-1` 的机器表示为 `2^w-1`，
   自然被拒。
5. **指针与区域**：
   * 指针带 `region_id`；`padd/psub/load` 引用的区域必须存在于传入快照。
   * `padd/psub` 结果必须落在 `[base, end]`，**允许恰好尾后 `end`**；
     再越过（含负地址）抛 `AddressError`。
   * `load` 拒绝尾后指针（起点必须 `< end`）、拒绝越过尾后的区间。
   * **对齐**按绝对地址的自然访问宽度：`offset % (w/8) == 0`，否则
     `AddressError`（u8 恒对齐）。
   * 字节序固定为**小端**。
   * `pdiff` 只在同一区域内有定义，返回字节差 i64（不按负载宽度缩放）。
6. **load 只能读传入快照**：无全局内存、无隐式区域。
7. **短路语义**：`band` 左假不评估右侧，`bor` 左真不评估右侧；被跳过的
   越界 `load` 不会报错，也不会出现在 `reads` 证据中。
8. **无浮点运算**：类型系统只有固定宽度整数与指针；任何 `float`
   （含 `NaN`、`+Infinity`、`-Infinity`）作为字面量入参一律被静态拒绝，
   因此不存在浮点结果，也就不存在浮点误差口径问题——整数按模、内存按
   字节重组，全部精确。

## 输入上限

* AST 节点总数（tuple 节点计数）**最多 500**，超出在静态校验阶段抛
  `ASTError`，不执行任何求值；该限制是结构性的，与短路是否跳过子树无关。
* 整数位宽仅 8/16/32/64；区域/寄存器名为非空字符串；
  指针偏移非负，且受区域边界约束。

## 错误语义（全部可定位）

所有本库错误继承 `DbgxError`，带 `e.path`：从根到出错节点的子节点下标
元组（根为 `()`），消息中也有 `节点路径 ...:` 前缀。

| 错误 | 触发场景 |
|---|---|
| `InputError` | registers/memory/Value/Region 入参结构非法；整数参数传了 **bool**；float/NaN/Infinity |
| `ASTError` | AST 结构非法、未知算子、静态参数非法、节点数 > 500 |
| `UndefinedRegisterError` | 引用未提供的寄存器 |
| `TypeMismatchError` | 操作数类型/位宽/符号性不符、asr 用于无符号、field 越界等 |
| `ShiftError` | 移位量 >= 位宽 |
| `DivisionByZeroError` | udiv/urem/sdiv/srem 除数为零 |
| `AddressError` | 未知区域、越界（含尾后解引用）、未对齐、跨区域相减 |

非法输入一律显式报错，不做静默纠正。

## 设计取舍

* **机器表示用非负 int + 附带类型**：避免在 Python 层同时维护有符号/
  无符号两个值；符号性只在比较、asr、有符号除法和指针步长处解释。
* **递归下降求值器**：AST 上限 500 节点，最坏递归深度 500；导入时把
  Python 递归下限抬到 4000（只在当前更低时生效），保证 500 节点的
  左/右旋链都能求值，同时避免迭代式机器带来的复杂度。
* **快照不可变**：`Region` 拷贝输入并固定为 `bytes`，从根上保证失败
  求值与多次求值不会相互污染。
* **对齐按绝对地址**：这是硬件对“错对齐访问”的通常定义。
* **布尔结果统一为 u8**：比较、`lnot`、`band`、`bor` 的结果类型固定。
* 本模块是题目契约的**有限数学/协议子集**：不实现隐式类型提升、指针
  相等比较、浮点、大端、非自然对齐访问、符号扩展 load；不宣称兼容任何
  完整行业标准（DWARF/C 表达式等）。

## “预算耗尽 vs 无解”说明

本库不含搜索/求解算法，因此不存在“搜索预算耗尽”与“证明无解”的区分；
唯一的预算是 500 节点的结构性上限（`ASTError`），它与语义违例
（`TypeMismatchError`/`AddressError` 等“该表达式在此输入下无合法值”）
是不同的错误类型，不会相互混淆。

## 测试与独立参考

* `tests/reference.py`：**独立**的 Python 整数掩码参考，不导入 `dbgx`，
  按契约独立重算期望值与错误类别。
* `tests/test_differential.py`：800 组确定性随机 AST（纯整数 400 组 +
  含 load/指针/短路 400 组）与参考逐项比对值、类型、读取证据、错误类别。
* 其余测试覆盖符号右移、模截断、有符号/无符号比较、移位边界、cast/field、
  尾后指针、未对齐、跨区域相减、短路越界读取、NaN/Infinity/bool 拒绝、
  500/501 节点边界、错误定位路径。
* 往返一致（cast 往返、位域重组、load-字节重组、快照隔离）仅作为
  **补充**测试（`tests/test_supplementary.py`），不用于产生期望值。

## 已知限制

* AST 必须是嵌套 tuple（无 span/源码位置）；定位信息是节点下标路径而非
  行列号。
* 无解析器：输入直接是已构造的 AST tuple。
* 最大递归深度受 500 节点契约约束；刻意构造的 500 层深树依赖导入时设置
  的递归下限。
* 指针算术以字节为单位，不提供“按元素宽度缩放”的步进；需要时由调用方
  自行乘上宽度。
