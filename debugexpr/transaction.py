"""One-shot undoable memory-edit transaction (edit preview).

An ``EditTransaction`` layers typed ``Store`` statements on top of an
immutable ``MemorySnapshot`` without ever touching it:

* ``txn.store(Store(...))`` validates pointer range, width, alignment and
  the integer representation, then stages the bytes in a private overlay.
  Overlapping stores overwrite earlier staged bytes in statement order;
  stores to different regions never interact, even at identical offsets.
* ``txn.evaluate(ast, registers)`` runs the ordinary read-only evaluator
  against the overlay view, so expressions observe earlier pending writes.
  Short-circuited operands are never evaluated, hence never read.
* ``txn.commit()`` returns a **new** ``MemorySnapshot`` with all staged
  writes applied; ``txn.discard()`` drops them.  The caller's original
  snapshot is never mutated, and no host memory is involved — everything
  lives in plain byte strings.

Any failed step (a rejected store or a failing expression) rolls the whole
transaction back: the overlay is dropped, no partial new memory can be
observed, and the transaction is closed.  The access trace of everything
attempted up to that point remains available via ``txn.trace``.

A transaction is one-shot: after commit / discard / rollback every further
operation raises ``TransactionError``.  This is deliberately not a debug
protocol and not a persistent store — one preview transaction at a time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import EvalError, InputError, PointerError, RangeError, TransactionError
from .evaluator import EvalResult, evaluate
from .memory import MemorySnapshot
from .nodes import Node
from .types import IntType, Pointer

_OPEN = "open"
_COMMITTED = "committed"
_DISCARDED = "discarded"
_ROLLED_BACK = "rolled-back"


@dataclass(frozen=True)
class Store:
    """One typed store statement: write ``value`` as ``type`` at ``ptr``.

    ``ptr`` is a ``Pointer(region_id, offset)``; ``type`` an ``IntType``
    (the width to write, little-endian); ``value`` a Python int that must
    fit the declared type.  Validation happens in ``EditTransaction.store``.
    """

    ptr: Pointer
    type: IntType
    value: int


@dataclass(frozen=True)
class AccessRecord:
    """One memory access actually performed inside a transaction.

    ``kind`` is ``"read"`` (a load served by the overlay view) or
    ``"write"`` (a staged store).  Records are in transaction order.
    """

    kind: str
    region_id: str
    offset: int
    length: int


class _OverlayView(MemorySnapshot):
    """Read-through view: base snapshot plus the staged write overlay.

    Every served read is appended to the shared transaction log, so the
    trace survives even when a later step fails and the overlay is dropped.
    """

    __slots__ = ("_base", "_overlay", "_log")

    def __init__(
        self,
        base: MemorySnapshot,
        overlay: dict[str, bytearray],
        log: list[AccessRecord],
    ) -> None:
        self._base = base
        self._overlay = overlay
        self._log = log

    @property
    def region_ids(self) -> tuple[str, ...]:
        return self._base.region_ids

    def has_region(self, region_id: str) -> bool:
        return self._base.has_region(region_id)

    def region_size(self, region_id: str) -> int:
        return self._base.region_size(region_id)

    def read(self, region_id: str, offset: int, length: int) -> bytes:
        staged = self._overlay.get(region_id)
        if staged is None:
            data = self._base.read(region_id, offset, length)  # enforces bounds
        else:
            if offset < 0 or length < 0 or offset + length > len(staged):
                raise PointerError(
                    f"read of {length} bytes at offset {offset} is outside "
                    f"region {region_id!r} (size {len(staged)})"
                )
            data = bytes(staged[offset : offset + length])
        self._log.append(AccessRecord("read", region_id, offset, length))
        return data


class EditTransaction:
    """A one-shot edit preview over a caller-supplied memory snapshot."""

    def __init__(self, memory: MemorySnapshot) -> None:
        if not isinstance(memory, MemorySnapshot):
            raise InputError(f"memory must be a MemorySnapshot, got {type(memory).__name__}")
        self._base = memory
        self._overlay: dict[str, bytearray] = {}
        self._log: list[AccessRecord] = []
        self._view = _OverlayView(memory, self._overlay, self._log)
        self._state = _OPEN

    @property
    def state(self) -> str:
        """``"open"``, ``"committed"``, ``"discarded"`` or ``"rolled-back"``."""
        return self._state

    @property
    def trace(self) -> tuple[AccessRecord, ...]:
        """Every read/write actually performed so far, in order.

        Available in every state, including after a rollback — this is the
        evidence of what was attempted before the failure.
        """
        return tuple(self._log)

    def _require_open(self) -> None:
        if self._state != _OPEN:
            raise TransactionError(
                f"transaction is {self._state}; edit transactions are one-shot "
                f"and cannot be reused"
            )

    def _rollback(self) -> None:
        # Drop every staged byte: no partial new memory may leak out.
        self._overlay.clear()
        self._state = _ROLLED_BACK

    def store(self, store: Store) -> None:
        """Validate and stage one typed store.

        Raises:
            TransactionError: the transaction is no longer open.
            InputError: malformed statement (wrong kinds, bool/float value).
            RangeError: the value does not fit the declared integer type.
            PointerError: unknown region, misaligned offset, or the write
                would not fit inside the region (one-past-end included).
            EvalError: any failure also rolls the whole transaction back.
        """
        self._require_open()
        try:
            self._apply(store)
        except EvalError:
            self._rollback()
            raise

    def _apply(self, store: Store) -> None:
        if not isinstance(store, Store):
            raise InputError(f"expected a Store statement, got {store!r}")
        ptr, itype, value = store.ptr, store.type, store.value
        if not isinstance(ptr, Pointer):
            raise InputError(f"store target must be a Pointer, got {ptr!r}")
        if type(ptr.region_id) is not str or not ptr.region_id:
            raise InputError(f"store region_id must be a non-empty str, got {ptr.region_id!r}")
        if type(ptr.offset) is bool or not isinstance(ptr.offset, int):
            raise InputError(f"store offset must be an int (bool is rejected), got {ptr.offset!r}")
        if not isinstance(itype, IntType):
            raise InputError(f"store type must be an IntType, got {itype!r}")
        if type(value) is bool or not isinstance(value, int):
            raise InputError(f"store value must be an int (bool and float are rejected), got {value!r}")
        if not itype.contains(value):
            raise RangeError(
                f"store value {value} does not fit {itype} [{itype.min}, {itype.max}]"
            )
        size = self._base.region_size(ptr.region_id)  # PointerError if unknown
        width = itype.size
        if ptr.offset % width != 0:
            raise PointerError(
                f"misaligned {itype} store at offset {ptr.offset} in region "
                f"{ptr.region_id!r} (requires {width}-byte alignment)"
            )
        if ptr.offset < 0 or ptr.offset + width > size:
            raise PointerError(
                f"{itype} store at offset {ptr.offset} in region {ptr.region_id!r} "
                f"needs {width} bytes but the region ends at {size} "
                f"(one-past-end writes are rejected)"
            )
        staged = self._overlay.get(ptr.region_id)
        if staged is None:
            staged = bytearray(self._base.read(ptr.region_id, 0, size))
            self._overlay[ptr.region_id] = staged
        staged[ptr.offset : ptr.offset + width] = value.to_bytes(width, "little", signed=itype.signed)
        self._log.append(AccessRecord("write", ptr.region_id, ptr.offset, width))

    def evaluate(self, ast: Node, registers: Mapping[str, int] | None = None) -> EvalResult:
        """Evaluate ``ast`` against the pending-write view of memory.

        Uses the unchanged read-only evaluator; reads observe earlier staged
        stores of this transaction.  A failing expression rolls the whole
        transaction back (the trace keeps the accesses attempted so far).
        """
        self._require_open()
        try:
            return evaluate(ast, registers or {}, self._view)
        except EvalError:
            self._rollback()
            raise

    def commit(self) -> MemorySnapshot:
        """Apply all staged writes and return a **new** snapshot.

        The caller's original snapshot is left untouched.  Closes the
        transaction; it cannot be used again.
        """
        self._require_open()
        regions = {}
        for region_id in self._base.region_ids:
            staged = self._overlay.get(region_id)
            if staged is None:
                regions[region_id] = self._base.read(region_id, 0, self._base.region_size(region_id))
            else:
                regions[region_id] = bytes(staged)
        self._overlay.clear()
        self._state = _COMMITTED
        return MemorySnapshot(regions)

    def discard(self) -> None:
        """Drop all staged writes and close the transaction."""
        self._require_open()
        self._overlay.clear()
        self._state = _DISCARDED


def begin_edit(memory: MemorySnapshot) -> EditTransaction:
    """Open a one-shot edit transaction over ``memory`` (a snapshot, never
    mutated; nothing here touches the host address space)."""
    return EditTransaction(memory)
