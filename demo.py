#!/usr/bin/env python3
"""dbgx 离线演示（固定输入，真实计算，无网络/无 sleep/无预录结果）。

场景：一个确定性构造的 32 字节外设区域快照，演示
  1. 有符号算术右移（符号位填充）；
  2. 指针算术 + 对齐 load + field 位域，打印「读取证据」ReadRange；
  3. 短路 and 跳过一个必然越界的尾后 load（读取证据为空）；
  4. 边界拒绝：尾后指针解引用、跨区域指针相减 —— 真实抛出 AddressError。

运行：python demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dbgx import (
    AddressError,
    DbgxError,
    IntType,
    Region,
    Value,
    evaluate,
)

BASE = 0x4000_0000


def build_snapshot() -> dict[str, Region]:
    """确定性生成外设寄存器块（32 字节）。

    偏移  内容
    0x00  u32 magic  = 0x574F4C46
    0x04  u32 status：bit[0]=enable, bit[8:15]=有符号温度修正 (-10)
    0x08  i16 temp   = -25（0xFFE7）
    0x0C  u16
    0x10  u32 count  = 0x00001234
    """
    buf = bytearray(32)
    buf[0x00:0x04] = (0x574F4C46).to_bytes(4, "little")
    status = (1 & 0xFF) | (((-10) & 0xFF) << 8)  # enable=1, 修正=-10
    buf[0x04:0x08] = status.to_bytes(4, "little")
    buf[0x08:0x0A] = (-25 & 0xFFFF).to_bytes(2, "little", signed=False)
    buf[0x10:0x14] = (0x00001234).to_bytes(4, "little")
    return {
        "periph": Region("periph", bytes(buf), base=BASE),
        # 另一个独立地址空间，仅用于演示跨区域拒绝
        "rom": Region("rom", b"\xAA" * 16, base=0x8000_0000),
    }


def show(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> int:
    mem = build_snapshot()
    regs = {"shift_in": Value(IntType(32, True), (-100) & 0xFFFFFFFF)}

    show("1) 有符号算术右移：i32(-100) asr 3")
    node = ("asr", ("reg", "shift_in"), ("int", 3, 32, True))
    r = evaluate(node, regs, mem)
    print(f"   结果 = {r.value.as_signed()}（类型 {r.value.vtype}），"
          f"机器表示 0x{r.value.as_int():08X}，节点数 {r.node_count}")
    print("   期望：-100 >> 3 = -13（向 -∞ 方向，高位补 1）")

    show("2) 指针 + 对齐 load + field：读 magic 并提取 status 位域")
    # load32(periph + 0) == MAGIC，且 field(load32(periph+4), lsb=8, i8) == -10
    magic_load = ("load", ("padd", ("ptr", "periph", BASE, 32),
                           ("int", 0, 32, True)))
    status_load = ("load", ("padd", ("ptr", "periph", BASE, 32),
                            ("int", 4, 32, True)))
    temp_corr = ("field", status_load, 8, 8, True)
    node = ("band",
            ("eq", magic_load, ("int", 0x574F4C46, 32, False)),
            ("eq", temp_corr, ("int", -10, 8, True)))
    r = evaluate(node, regs, mem)
    print(f"   magic 匹配 AND 温度修正 == -10  =>  {r.value.as_int()}（u8 布尔）")
    print("   实际读取区间证据：")
    for rr in r.reads:
        print(f"     - {rr.region_id}[0x{rr.start:08X}:0x{rr.end:08X})"
              f"（{rr.nbytes} 字节）")

    show("3) 短路求值：false AND <尾后越界 load> —— 右侧不执行")
    one_past_end = BASE + 32
    node = ("band",
            ("eq", ("int", 1, 8, False), ("int", 2, 8, False)),
            ("load", ("ptr", "periph", one_past_end, 8)))
    r = evaluate(node, regs, mem)
    print(f"   结果 = {r.value.as_int()}；读取区间数 = {len(r.reads)}"
          f"（被短路，未触发尾后越界）")

    show("4a) 边界拒绝【真实异常】：尾后指针解引用")
    node = ("load", ("ptr", "periph", one_past_end, 8))
    try:
        evaluate(node, regs, mem)
    except AddressError as e:
        print(f"   捕获 {type(e).__name__}（定位路径 {e.path}）：")
        print(f"   {e.message}")

    show("4b) 边界拒绝【真实异常】：跨区域指针相减")
    node = ("pdiff",
            ("ptr", "periph", BASE + 8, 8),
            ("ptr", "rom", 0x8000_0000, 8))
    try:
        evaluate(node, regs, mem)
    except AddressError as e:
        print(f"   捕获 {type(e).__name__}（定位路径 {e.path}）：")
        print(f"   {e.message}")

    show("4c) 边界拒绝【真实异常】：未对齐 u32 访问")
    node = ("load", ("ptr", "periph", BASE + 1, 32))
    try:
        evaluate(node, regs, mem)
    except AddressError as e:
        print(f"   捕获 {type(e).__name__}：{e.message}")

    show("4d) 边界拒绝【真实异常】：移位量 == 位宽")
    node = ("shl", ("int", 1, 8, False), ("int", 8, 8, False))
    try:
        evaluate(node, regs, mem)
    except DbgxError as e:
        print(f"   捕获 {type(e).__name__}：{e.message}")

    print("\n演示完成：以上结果均由 dbgx.evaluate 当场计算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
