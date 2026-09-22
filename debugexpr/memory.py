"""Immutable memory snapshot passed to ``evaluate``.

A snapshot is a fixed set of named regions, each a ``bytes`` object.
``Load``/``Field`` may only read from regions present in the snapshot
handed to ``evaluate``; there is no global or mutable memory.
"""

from __future__ import annotations

from collections.abc import Mapping

from .errors import InputError, PointerError


class MemorySnapshot:
    """An immutable, explicitly constructed view of address-space regions."""

    __slots__ = ("_regions",)

    def __init__(self, regions: Mapping[str, bytes]) -> None:
        if not isinstance(regions, Mapping):
            raise InputError(f"regions must be a mapping of str -> bytes, got {type(regions).__name__}")
        frozen: dict[str, bytes] = {}
        for region_id, data in regions.items():
            if type(region_id) is not str or not region_id:
                raise InputError(f"region id must be a non-empty str, got {region_id!r}")
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise InputError(
                    f"region {region_id!r}: data must be bytes-like, got {type(data).__name__}"
                )
            frozen[region_id] = bytes(data)  # defensive copy: snapshot semantics
        self._regions = frozen

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(self._regions)

    def has_region(self, region_id: str) -> bool:
        return region_id in self._regions

    def region_size(self, region_id: str) -> int:
        try:
            return len(self._regions[region_id])
        except KeyError:
            raise PointerError(f"unknown memory region {region_id!r}") from None

    def read(self, region_id: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes at ``offset``.  Bounds are enforced here too,
        so callers can never observe memory outside the snapshot."""
        data = self._regions.get(region_id)
        if data is None:
            raise PointerError(f"unknown memory region {region_id!r}")
        if offset < 0 or length < 0 or offset + length > len(data):
            raise PointerError(
                f"read of {length} bytes at offset {offset} is outside "
                f"region {region_id!r} (size {len(data)})"
            )
        return data[offset : offset + length]
