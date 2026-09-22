"""独立参考实现：纯 Python 整数掩码语义。

本模块**不导入 dbgx**，只按题目契约用 Python 任意精度整数重新实现期望值
计算，供 unittest 交叉核对。它是测试夹具，不是产品代码：

* 整数值表示：``("i", width, signed, raw)``，raw 为无符号机器表示。
* 指针值表示：``("p", region_id, offset, pointee_width)``。
* 区域：``{"region_id": (base: int, data: bytes)}``。
* 返回：``(value, reads)``，reads 为 ``(region_id, start, end)`` 列表。
"""

from __future__ import annotations

WIDTHS = (8, 16, 32, 64)


def mask(width: int) -> int:
    return (1 << width) - 1


def wrap(width: int, x: int) -> int:
    """按模 2**width 截断（Python % 对负数同样给出非负余数）。"""
    return x % (1 << width)


def to_signed(width: int, raw: int) -> int:
    if raw >= 1 << (width - 1):
        return raw - (1 << width)
    return raw


def _err(tag: str, msg: str) -> Exception:
    return _RefError(f"{tag}: {msg}")


class _RefError(Exception):
    """参考实现按类别区分错误：tag 用于与 dbgx 错误类型对应。"""

    def __init__(self, text: str):
        self.tag, self.msg = text.split(": ", 1)
        super().__init__(text)


def ref_eval(node, regs, mem):
    """与 dbgx.evaluate 平行的参考求值。

    合法程序返回值/读取证据；违例抛 :class:`_RefError`，tag ∈
    {type, shift, div0, addr, undef, ast}。
    """
    reads: list[tuple[str, int, int]] = []

    def ev(n):
        op = n[0]

        if op == "int":
            _, v, w, s = n
            return ("i", w, s, wrap(w, v))

        if op == "reg":
            if n[1] not in regs:
                raise _err("undef", f"register {n[1]}")
            v = regs[n[1]]
            # regs 中直接用参考表示
            return v

        if op == "lnot":
            x = ev(n[1])
            _assert_int(x, op)
            return ("i", 8, False, 1 if x[3] == 0 else 0)

        if op in ("band", "bor"):
            l = ev(n[1])
            _assert_int(l, op)
            ltrue = l[3] != 0
            if op == "band" and not ltrue:
                return ("i", 8, False, 0)
            if op == "bor" and ltrue:
                return ("i", 8, False, 1)
            r = ev(n[2])
            _assert_int(r, op)
            return ("i", 8, False, 1 if r[3] != 0 else 0)

        if op == "cast":
            x = ev(n[1])
            _assert_int(x, op)
            _, _, w2, s2 = n
            return ("i", w2, s2, wrap(w2, x[3]))

        if op == "field":
            x = ev(n[1])
            _assert_int(x, op)
            _, _, lsb, fw, fs = n
            if lsb + fw > x[1]:
                raise _err("type", f"field overflow {lsb}+{fw}>{x[1]}")
            raw = (x[3] >> lsb) & ((1 << fw) - 1)
            return ("i", fw, fs, raw)

        if op == "ptr":
            _, rid, off, pw = n
            return ("p", rid, off, pw)

        if op in ("padd", "psub"):
            p = ev(n[1])
            i = ev(n[2])
            if p[0] != "p":
                raise _err("type", f"{op} lhs not ptr")
            _assert_int(i, op)
            rid, off, pw = p[1], p[2], p[3]
            if rid not in mem:
                raise _err("addr", f"unknown region {rid}")
            base, data = mem[rid]
            end = base + len(data)
            delta = to_signed(i[1], i[3])
            newoff = off + delta if op == "padd" else off - delta
            if not (base <= newoff <= end):
                raise _err("addr", f"{op} OOB {newoff} not in [{base},{end}]")
            return ("p", rid, newoff, pw)

        if op == "pdiff":
            p1 = ev(n[1])
            p2 = ev(n[2])
            if p1[0] != "p" or p2[0] != "p":
                raise _err("type", "pdiff needs pointers")
            if p1[1] != p2[1]:
                raise _err("addr", f"cross-region diff {p1[1]} vs {p2[1]}")
            if p1[1] not in mem:
                raise _err("addr", f"unknown region {p1[1]}")
            base, data = mem[p1[1]]
            end = base + len(data)
            for off in (p1[2], p2[2]):
                if not (base <= off <= end):
                    raise _err("addr", f"diff ptr OOB {off} not in [{base},{end}]")
            return ("i", 64, True, wrap(64, p1[2] - p2[2]))

        if op == "load":
            p = ev(n[1])
            if p[0] != "p":
                raise _err("type", "load needs pointer")
            rid, off, pw = p[1], p[2], p[3]
            if rid not in mem:
                raise _err("addr", f"unknown region {rid}")
            base, data = mem[rid]
            nb = pw // 8
            if off % nb != 0:
                raise _err("addr", f"misaligned {off} % {nb}")
            end = base + len(data)
            if not (base <= off < end and off + nb <= end):
                raise _err("addr", f"OOB load [{off},{off + nb}) in [{base},{end})")
            idx = off - base
            chunk = data[idx : idx + nb]
            raw = int.from_bytes(chunk, "little")
            reads.append((rid, off, off + nb))
            return ("i", pw, False, raw)

        if op in ("eq", "ne", "lt", "le", "gt", "ge"):
            l = ev(n[1])
            r = ev(n[2])
            _assert_int(l, op)
            _assert_int(r, op)
            if (l[1], l[2]) != (r[1], r[2]):
                raise _err("type", f"compare {l[1:3]} vs {r[1:3]}")
            if op == "eq":
                c = l[3] == r[3]
            elif op == "ne":
                c = l[3] != r[3]
            else:
                lv = to_signed(l[1], l[3]) if l[2] else l[3]
                rv = to_signed(r[1], r[3]) if r[2] else r[3]
                c = {"lt": lv < rv, "le": lv <= rv, "gt": lv > rv, "ge": lv >= rv}[
                    op
                ]
            return ("i", 8, False, 1 if c else 0)

        # 二元算术
        return _binop(n, ev)

    out = ev(node)
    return out, reads


def _assert_int(v, op: str) -> None:
    if v[0] != "i":
        raise _err("type", f"{op} expects int")


def _binop(n, ev):
    op = n[0]
    l = ev(n[1])
    r = ev(n[2])
    _assert_int(l, op)
    _assert_int(r, op)
    if (l[1], l[2]) != (r[1], r[2]):
        raise _err("type", f"{op} mixed types {l[1:3]} vs {r[1:3]}")
    w, s = l[1], l[2]
    a, b = l[3], r[3]
    m = mask(w)

    if op in ("add", "sub", "mul", "and", "or", "xor"):
        x = {
            "add": a + b,
            "sub": a - b,
            "mul": a * b,
            "and": a & b,
            "or": a | b,
            "xor": a ^ b,
        }[op]
        return ("i", w, s, x & m)

    if op in ("udiv", "urem"):
        if b == 0:
            raise _err("div0", op)
        return ("i", w, s, a // b if op == "udiv" else a % b)

    if op in ("sdiv", "srem"):
        sa, sb = to_signed(w, a), to_signed(w, b)
        if sb == 0:
            raise _err("div0", op)
        q = abs(sa) // abs(sb)
        if (sa < 0) != (sb < 0):
            q = -q
        if op == "sdiv":
            return ("i", w, s, wrap(w, q))
        return ("i", w, s, wrap(w, sa - q * sb))

    if op in ("shl", "lsr", "asr"):
        if b >= w:
            raise _err("shift", f"{b} >= {w}")
        if op == "shl":
            return ("i", w, s, (a << b) & m)
        if op == "lsr":
            return ("i", w, s, a >> b)
        if not s:
            raise _err("type", "asr on unsigned")
        return ("i", w, s, wrap(w, to_signed(w, a) >> b))

    raise _err("ast", f"unknown op {op}")
