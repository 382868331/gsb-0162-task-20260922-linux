"""One-shot edit transactions on top of the read-only evaluator.

A transaction is a short sequence of typed ``Store`` statements executed
in order against a private overlay of the caller's ``MemorySnapshot``.
Later statements (pointer and value expressions of subsequent stores) read
earlier pending writes through the overlay.  When all statements succeed,
the caller inspects the access trace and then chooses exactly once:
``commit()`` (a new ``MemorySnapshot`` with the writes applied) or
``discard()`` (nothing happens).

Contract (fixed, see README):

* Every store validates, in order: pointer bounds (one-past-end writes are
  rejected), natural alignment of the declared type, and that the
  evaluated integer fits the declared type's representation.
* Overlapping writes within one region are resolved by statement order:
  the later statement's bytes win.  Different regions never interfere,
  even at identical offsets.
* Any failing statement rolls the whole transaction back: the result
  carries the trace of accesses attempted so far, but no partial memory
  is exposed (``commit`` is unavailable on a failed result).
* Statements after a failing one never execute (no side effects), and
  short-circuited subexpressions perform no loads.
* The caller's source snapshot is never mutated; committed snapshots are
  built from fresh bytes.

The overlay lives entirely in this process's Python heap as copies of the
caller-supplied region bytes; no native address space is touched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import EvalError, InputError, PointerError, RangeError, TypeMismatchError
from .evaluator import MAX_NODES, _Ctx, _count_nodes, _eval
from .memory import MemorySnapshot
from .nodes import Node
from .types import IntType, Pointer, PointerType

MAX_STATEMENTS = 128


@dataclass(frozen=True)
class Store:
    """One typed store statement.

    ``ptr`` and ``value`` are evaluator ASTs.  ``ptr`` must evaluate to a
    pointer, ``value`` to an integer that fits ``type``; the store then
    writes ``type.size`` little-endian bytes at that pointer.
    """

    ptr: Node
    type: IntType
    value: Node


@dataclass(frozen=True)
class AccessRecord:
    """One memory access actually performed during the transaction.

    ``kind`` is ``"read"`` (expression loads) or ``"write"`` (validated
    stores).  Records appear in execution order; a store is recorded only
    after it has passed all validation and been applied to the overlay.
    """

    kind: str
    region_id: str
    offset: int
    length: int


class _OverlayMemory:
    """Lazy copy-on-write view over a ``MemorySnapshot``.

    Same duck-typed interface the evaluator uses (``region_size`` /
    ``read`` / ``has_region``), plus ``write``.  Only regions actually
    written are copied; the source snapshot is never touched.
    """

    __slots__ = ("_source", "_dirty")

    def __init__(self, source: MemorySnapshot) -> None:
        self._source = source
        self._dirty: dict[str, bytearray] = {}

    def has_region(self, region_id: str) -> bool:
        return self._source.has_region(region_id)

    def region_size(self, region_id: str) -> int:
        return self._source.region_size(region_id)  # raises PointerError if unknown

    def read(self, region_id: str, offset: int, length: int) -> bytes:
        buf = self._dirty.get(region_id)
        if buf is None:
            return self._source.read(region_id, offset, length)
        if offset < 0 or length < 0 or offset + length > len(buf):
            raise PointerError(
                f"read of {length} bytes at offset {offset} is outside "
                f"region {region_id!r} (size {len(buf)})"
            )
        return bytes(buf[offset : offset + length])

    def write(self, region_id: str, offset: int, data: bytes) -> None:
        buf = self._dirty.get(region_id)
        if buf is None:
            buf = bytearray(self._source.read(region_id, 0, self._source.region_size(region_id)))
            self._dirty[region_id] = buf
        buf[offset : offset + len(data)] = data

    def merged_snapshot(self) -> MemorySnapshot:
        """A new snapshot: source regions with dirty overlays applied."""
        regions: dict[str, bytes] = {}
        for region_id in self._source.region_ids:
            buf = self._dirty.get(region_id)
            if buf is None:
                regions[region_id] = self._source.read(region_id, 0, self._source.region_size(region_id))
            else:
                regions[region_id] = bytes(buf)
        return MemorySnapshot(regions)


class TransactionResult:
    """Outcome of ``run_transaction``.

    On success (``ok`` is True) the caller chooses exactly once between
    ``commit()`` and ``discard()``; any further settlement call raises
    ``InputError``.  On failure (``ok`` is False, ``error`` set) the
    transaction is already rolled back: ``trace`` lists the accesses
    attempted before the failure, and ``commit()`` is unavailable, so no
    partially edited memory can leak out.
    """

    __slots__ = ("_overlay", "_settled", "error", "ok", "trace")

    def __init__(
        self,
        ok: bool,
        trace: tuple[AccessRecord, ...],
        error: EvalError | None,
        overlay: _OverlayMemory | None,
    ) -> None:
        self.ok = ok
        self.trace = trace
        self.error = error
        self._overlay = overlay if ok else None  # failed txn exposes no memory
        self._settled = not ok  # failed results are settled (rolled back) at birth

    def commit(self) -> MemorySnapshot:
        """Apply the pending writes, returning a new ``MemorySnapshot``."""
        if self._settled:
            raise InputError("transaction is already settled (commit/discard are one-shot)")
        self._settled = True
        assert self._overlay is not None
        snapshot = self._overlay.merged_snapshot()
        self._overlay = None
        return snapshot

    def discard(self) -> None:
        """Drop the pending writes; the source snapshot stays as it was."""
        if self._settled:
            raise InputError("transaction is already settled (commit/discard are one-shot)")
        self._settled = True
        self._overlay = None


def _check_statement(stmt: object, index: int) -> Store:
    where = f"statements[{index}]"
    if not isinstance(stmt, Store):
        raise InputError(f"{where} must be a Store, got {stmt!r}")
    if not isinstance(stmt.ptr, Node):
        raise InputError(f"{where}.ptr must be a debugexpr Node, got {stmt.ptr!r}")
    if not isinstance(stmt.value, Node):
        raise InputError(f"{where}.value must be a debugexpr Node, got {stmt.value!r}")
    if not isinstance(stmt.type, IntType):
        raise InputError(f"{where}.type must be an IntType, got {stmt.type!r}")
    for field_name, ast in (("ptr", stmt.ptr), ("value", stmt.value)):
        count = _count_nodes(ast)
        if count > MAX_NODES:
            raise InputError(
                f"{where}.{field_name} has {count} distinct nodes, exceeding the limit of {MAX_NODES}"
            )
    return stmt


def _exec_store(stmt: Store, ctx: _Ctx, overlay: _OverlayMemory, path: tuple[str, ...]) -> AccessRecord:
    """Validate and apply one store to the overlay; return its write record."""
    ptr_value = _eval(stmt.ptr, ctx, path + ("ptr",))
    if not isinstance(ptr_value.type, PointerType):
        raise TypeMismatchError(
            f"store: pointer expression must evaluate to a pointer, got {ptr_value.type}",
            path + ("ptr",),
        )
    value = _eval(stmt.value, ctx, path + ("value",))
    if not isinstance(value.type, IntType):
        raise TypeMismatchError(
            f"store: value expression must evaluate to an integer, got {value.type}",
            path + ("value",),
        )
    itype = stmt.type
    assert isinstance(value.data, int)
    if not itype.contains(value.data):
        raise RangeError(
            f"store: value {value.data} does not fit {itype} [{itype.min}, {itype.max}]",
            path + ("value",),
        )
    ptr = ptr_value.data
    assert isinstance(ptr, Pointer)
    size = itype.size
    region_size = overlay.region_size(ptr.region_id)  # raises PointerError if unknown
    if ptr.offset < 0 or ptr.offset + size > region_size:
        raise PointerError(
            f"store: {itype} write at offset {ptr.offset} in region {ptr.region_id!r} "
            f"needs {size} bytes but the region ends at {region_size} "
            f"(one-past-end writes are rejected)",
            path + ("ptr",),
        )
    if ptr.offset % size != 0:
        raise PointerError(
            f"store: misaligned {itype} write at offset {ptr.offset} in region "
            f"{ptr.region_id!r} (requires {size}-byte alignment)",
            path + ("ptr",),
        )
    raw = (value.data & ((1 << itype.bits) - 1)).to_bytes(size, "little")
    overlay.write(ptr.region_id, ptr.offset, raw)
    return AccessRecord("write", ptr.region_id, ptr.offset, size)


def run_transaction(
    statements: Sequence[Store],
    registers: Mapping[str, int],
    memory: MemorySnapshot,
) -> TransactionResult:
    """Run a one-shot edit transaction; never raises for statement failures.

    Args:
        statements: an ordered sequence of ``Store`` statements (at most
            ``MAX_STATEMENTS``).  Structural violations (non-Store items,
            non-Node ASTs, oversized ASTs) raise ``InputError`` immediately.
        registers: mapping of register name -> Python int, same contract
            as ``evaluate``.
        memory: the source ``MemorySnapshot``; never mutated.

    Returns:
        ``TransactionResult``.  On failure ``ok`` is False, ``error`` holds
        the ``EvalError`` from the first failing statement, and ``trace``
        lists every access attempted up to that point; statements after
        the failing one never ran.  On success the caller settles the
        transaction exactly once via ``commit()`` or ``discard()``.
    """
    if isinstance(statements, (str, bytes)) or not isinstance(statements, Sequence):
        raise InputError(f"statements must be a sequence of Store, got {type(statements).__name__}")
    if len(statements) > MAX_STATEMENTS:
        raise InputError(f"transaction has {len(statements)} statements, exceeding the limit of {MAX_STATEMENTS}")
    if not isinstance(registers, Mapping):
        raise InputError(f"registers must be a mapping of str -> int, got {type(registers).__name__}")
    clean_registers: dict[str, int] = {}
    for name, value in registers.items():
        if type(name) is not str or not name:
            raise InputError(f"register name must be a non-empty str, got {name!r}")
        if type(value) is bool or not isinstance(value, int):
            raise InputError(
                f"register {name!r} value must be an int (bool and float are rejected), got {value!r}"
            )
        clean_registers[name] = value
    if not isinstance(memory, MemorySnapshot):
        raise InputError(f"memory must be a MemorySnapshot, got {type(memory).__name__}")

    checked = [_check_statement(stmt, i) for i, stmt in enumerate(statements)]

    overlay = _OverlayMemory(memory)
    ctx = _Ctx(clean_registers, overlay)
    trace: list[AccessRecord] = []
    for i, stmt in enumerate(checked):
        try:
            record = _exec_store(stmt, ctx, overlay, (f"statements[{i}]",))
        except EvalError as err:
            # Roll back: report the accesses attempted so far, expose no memory.
            trace.extend(
                AccessRecord("read", r.region_id, r.offset, r.length) for r in ctx.reads
            )
            return TransactionResult(False, tuple(trace), err, None)
        trace.extend(
            AccessRecord("read", r.region_id, r.offset, r.length) for r in ctx.reads
        )
        ctx.reads.clear()
        trace.append(record)
    return TransactionResult(True, tuple(trace), None, overlay)
