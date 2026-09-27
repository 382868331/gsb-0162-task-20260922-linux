"""Immutable memory snapshot passed to evaluate().

A snapshot is a fixed set of named regions.  Each region is an address
space assembled from zero or more segments; every segment has an
absolute base offset inside its region, payload bytes and a readable
permission bit:

* addresses covered by a readable=True segment are readable;
* addresses covered by a readable=False segment are mapped but the read
  permission is absent (a permission fault, never a hole);
* addresses covered by no segment at all are unmapped holes.

Load/Field may only read from regions present in the snapshot handed to
evaluate(); there is no global or mutable memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import InputError, PointerError, UnmappedAddressError, UnreadableMemoryError


@dataclass(frozen=True)
class MemorySegment:
    """One mapped run of bytes inside a region.

    Attributes:
        offset: absolute byte offset of the segment inside its region.
        data: bytes backing [offset, offset + len(data)).
        readable: read permission bit; False means mapped-but-unreadable.
    """

    offset: int
    data: bytes
    readable: bool = True

    @property
    def end(self) -> int:
        return self.offset + len(self.data)


class MemorySnapshot:
    """An immutable, explicitly constructed view of address-space regions.

    Two construction shapes are accepted per region:

    * bytes -- a single readable segment at offset 0;
    * a sequence of MemorySegment / (offset, data) /
      (offset, data, readable) tuples in any order.  Adjacent segments
      (end == next offset) are allowed; overlapping ones are rejected.
    """

    __slots__ = ("_segments", "_extents")

    def __init__(self, regions: Mapping[str, object]) -> None:
        if not isinstance(regions, Mapping):
            raise InputError(
                f"regions must be a mapping of str -> bytes/segments, got {type(regions).__name__}"
            )
        segmented: dict[str, tuple[MemorySegment, ...]] = {}
        extents: dict[str, int] = {}
        for region_id, spec in regions.items():
            if type(region_id) is not str or not region_id:
                raise InputError(f"region id must be a non-empty str, got {region_id!r}")
            if isinstance(spec, (bytes, bytearray, memoryview)):
                data = bytes(spec)
                segmented[region_id] = ((MemorySegment(0, data, True),) if data else ())
                extents[region_id] = len(data)
                continue
            if not isinstance(spec, Sequence):
                raise InputError(
                    f"region {region_id!r}: value must be bytes-like or a sequence of segments, "
                    f"got {type(spec).__name__}"
                )
            segments = [self._parse_segment(region_id, index, item)
                        for index, item in enumerate(spec)]
            segments.sort(key=lambda seg: seg.offset)
            extent = 0
            for seg in segments:
                if seg.offset < extent:
                    raise InputError(
                        f"region {region_id!r}: segments overlap at offset {seg.offset} "
                        f"(previous segment ends at {extent})"
                    )
                extent = seg.end
            segmented[region_id] = tuple(segments)
            extents[region_id] = extent
        self._segments = segmented
        self._extents = extents

    @staticmethod
    def _parse_segment(region_id: str, index: int, item: object) -> MemorySegment:
        if isinstance(item, MemorySegment):
            offset, data, readable = item.offset, item.data, item.readable
        elif isinstance(item, tuple) and 2 <= len(item) <= 3:
            offset, data = item[0], item[1]
            readable = item[2] if len(item) == 3 else True
        else:
            raise InputError(
                f"region {region_id!r} segment {index}: must be a MemorySegment or an "
                f"(offset, data[, readable]) tuple, got {item!r}"
            )
        if type(offset) is bool or not isinstance(offset, int):
            raise InputError(
                f"region {region_id!r} segment {index}: offset must be an int, got {offset!r}"
            )
        if offset < 0:
            raise InputError(
                f"region {region_id!r} segment {index}: offset must be >= 0, got {offset}"
            )
        if type(readable) is not bool:
            raise InputError(
                f"region {region_id!r} segment {index}: readable must be a bool, got {readable!r}"
            )
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise InputError(
                f"region {region_id!r} segment {index}: data must be bytes-like, "
                f"got {type(data).__name__}"
            )
        return MemorySegment(offset, bytes(data), readable)

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(self._segments)

    def has_region(self, region_id: str) -> bool:
        return region_id in self._segments

    def segments(self, region_id: str) -> tuple[MemorySegment, ...]:
        try:
            return self._segments[region_id]
        except KeyError:
            raise PointerError(f"unknown memory region {region_id!r}") from None

    def region_size(self, region_id: str) -> int:
        """Extent of the region: the end of its last mapped segment."""
        try:
            return self._extents[region_id]
        except KeyError:
            raise PointerError(f"unknown memory region {region_id!r}") from None

    def classify(self, region_id: str, offset: int) -> str:
        """Classify one byte as readable, unreadable or unmapped."""
        for seg in self._segments.get(region_id, ()):
            if seg.offset <= offset < seg.end:
                return "readable" if seg.readable else "unreadable"
        return "unmapped"

    def access(
        self, region_id: str, offset: int, length: int
    ) -> tuple[bytes, tuple[tuple[int, int], ...]]:
        """Atomically read length bytes starting at offset.

        The whole range must be mapped and readable or the read fails as a
        unit (no partial result):

        * any unmapped byte raises UnmappedAddressError (a hole);
        * a fully mapped range touching an unreadable segment raises
          UnreadableMemoryError.

        Returns the assembled bytes and the source records: the
        (offset, length) intersections with the readable segments actually
        touched, in ascending offset order.
        """
        try:
            segments = self._segments[region_id]
        except KeyError:
            raise PointerError(f"unknown memory region {region_id!r}") from None
        if type(offset) is not int or type(length) is not int or offset < 0 or length < 0:
            raise PointerError(f"invalid read offset={offset!r} length={length!r}")
        end = offset + length

        covered = 0  # number of prefix bytes of [offset, end) proven mapped
        for seg in segments:
            if seg.end <= offset + covered:
                continue
            if seg.offset > offset + covered:
                break  # a hole before this segment: the prefix cannot extend
            covered = min(end, seg.end) - offset
            if covered == length:
                break
        mapped = covered == length
        if not mapped:
            raise UnmappedAddressError(
                f"read of {length} bytes at offset {offset} in region {region_id!r} "
                f"crosses an unmapped hole"
            )

        for seg in segments:
            lo = max(seg.offset, offset)
            hi = min(seg.end, end)
            if lo < hi and not seg.readable:
                raise UnreadableMemoryError(
                    f"read of {length} bytes at offset {offset} in region {region_id!r} "
                    f"touches an unreadable segment [{seg.offset}, {seg.end})"
                )

        raw = bytearray(length)
        sources: list[tuple[int, int]] = []
        for seg in segments:
            lo = max(seg.offset, offset)
            hi = min(seg.end, end)
            if lo < hi:
                raw[lo - offset : hi - offset] = seg.data[lo - seg.offset : hi - seg.offset]
                sources.append((lo, hi - lo))
        return bytes(raw), tuple(sources)

    def read(self, region_id: str, offset: int, length: int) -> bytes:
        """Backward-compatible read honoring segments, permissions and holes."""
        return self.access(region_id, offset, length)[0]
