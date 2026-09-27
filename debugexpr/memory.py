"""Immutable memory snapshot passed to ``evaluate``.

A snapshot is a fixed set of named *regions*.  Each region is an address
space built from one or more non-overlapping :class:`MemorySegment`
mappings.  Every segment carries its own ``readable`` permission:

* an address covered by a ``readable=True`` segment is ordinary memory;
* an address covered by a ``readable=False`` segment is *mapped but not
  readable* (e.g. a protected page);
* any address not covered by any segment is an *unmapped hole* (a gap).

The two failure modes are reported with distinct error types
(:class:`UnmappedAddressError` vs :class:`ReadPermissionError`).

``Load``/``Field`` may only read from regions present in the snapshot
handed to ``evaluate``; there is no global or mutable memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import InputError, PointerError, ReadPermissionError, UnmappedAddressError


@dataclass(frozen=True)
class MemorySegment:
    """One mapped byte interval inside a region.

    The segment maps ``data`` at ``[offset, offset + len(data))`` within
    its region.  Segments must not overlap; gaps between them are
    unmapped holes.  ``readable=False`` models a mapped but protected
    interval (e.g. a guard page).
    """

    offset: int
    data: bytes
    readable: bool = True

    def __post_init__(self) -> None:
        if type(self.offset) is bool or not isinstance(self.offset, int):
            raise InputError(f"segment offset must be an int, got {self.offset!r}")
        if self.offset < 0:
            raise InputError(f"segment offset must be non-negative, got {self.offset}")
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise InputError(
                f"segment data must be bytes-like, got {type(self.data).__name__}"
            )
        if type(self.readable) is not bool:
            raise InputError(f"segment readable must be a bool, got {self.readable!r}")

    @property
    def end(self) -> int:
        return self.offset + len(self.data)


@dataclass(frozen=True)
class ReadSource:
    """Provenance of one contiguous portion of a successful read.

    A single typed load that spans several adjacent readable segments
    yields one ``ReadSource`` per backing segment (each clipped to the
    bytes actually used).
    """

    region_id: str
    offset: int
    length: int


class MemorySnapshot:
    """An immutable, explicitly constructed view of address-space regions.

    Two construction forms are accepted per region:

    * ``bytes``-like -- a single readable segment mapped at offset 0 (the
      classic flat region, kept for backwards compatibility);
    * a sequence of :class:`MemorySegment` -- a sparse mapping with
      per-segment read permission and possible unmapped holes.
    """

    __slots__ = ("_segments", "_extents")

    def __init__(
        self,
        regions: Mapping[str, bytes | bytearray | memoryview | Sequence[MemorySegment]],
    ) -> None:
        if not isinstance(regions, Mapping):
            raise InputError(
                f"regions must be a mapping of str -> bytes/segments, got {type(regions).__name__}"
            )
        frozen_segments: dict[str, tuple[MemorySegment, ...]] = {}
        extents: dict[str, int] = {}
        for region_id, spec in regions.items():
            if type(region_id) is not str or not region_id:
                raise InputError(f"region id must be a non-empty str, got {region_id!r}")
            if isinstance(spec, (bytes, bytearray, memoryview)):
                segments = (MemorySegment(0, bytes(spec), True),)
            elif isinstance(spec, Sequence):
                raw_segments = list(spec)
                if not raw_segments:
                    raise InputError(f"region {region_id!r}: segment list must not be empty")
                normalized: list[MemorySegment] = []
                for seg in raw_segments:
                    if not isinstance(seg, MemorySegment):
                        raise InputError(
                            f"region {region_id!r}: segments must be MemorySegment, "
                            f"got {type(seg).__name__}"
                        )
                    normalized.append(MemorySegment(seg.offset, bytes(seg.data), seg.readable))
                segments = tuple(normalized)
            else:
                raise InputError(
                    f"region {region_id!r}: must be bytes-like or a sequence of MemorySegment, "
                    f"got {type(spec).__name__}"
                )
            ordered = sorted(segments, key=lambda s: (s.offset, s.end))
            prev_end = 0
            extent = 0
            for seg in ordered:
                if seg.offset < prev_end:
                    raise InputError(
                        f"region {region_id!r}: segments overlap at offset {seg.offset}"
                    )
                prev_end = seg.end
                if seg.end > extent:
                    extent = seg.end
            frozen_segments[region_id] = tuple(ordered)
            extents[region_id] = extent
        self._segments = frozen_segments
        self._extents = extents

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(self._segments)

    def has_region(self, region_id: str) -> bool:
        return region_id in self._segments

    def region_size(self, region_id: str) -> int:
        """Region extent: one past the end of the last mapped segment."""
        try:
            return self._extents[region_id]
        except KeyError:
            raise PointerError(f"unknown memory region {region_id!r}") from None

    def segments(self, region_id: str) -> tuple[MemorySegment, ...]:
        """The region's mapped segments in ascending address order."""
        try:
            return self._segments[region_id]
        except KeyError:
            raise PointerError(f"unknown memory region {region_id!r}") from None

    def resolve_read(
        self, region_id: str, offset: int, length: int
    ) -> tuple[bytes, tuple[ReadSource, ...]]:
        """Atomically read ``length`` bytes starting at ``offset``.

        The whole interval must be mapped *and* readable; a request that
        reaches an unmapped hole raises :class:`UnmappedAddressError`, one
        that reaches a mapped-but-unreadable segment raises
        :class:`ReadPermissionError`.  Validation completes before any
        bytes are returned, so a failure never yields a partial value.

        Returns the assembled byte string plus one :class:`ReadSource`
        per backing segment actually touched (clipped to the requested
        interval), in ascending address order -- the precise provenance.
        """
        segments = self._segments.get(region_id)
        if segments is None:
            raise PointerError(f"unknown memory region {region_id!r}")
        if type(offset) is bool or not isinstance(offset, int):
            raise InputError(f"read offset must be an int, got {offset!r}")
        if type(length) is bool or not isinstance(length, int):
            raise InputError(f"read length must be an int, got {length!r}")
        if offset < 0 or length < 0:
            raise PointerError(
                f"read at offset {offset} length {length} has negative components "
                f"in region {region_id!r}"
            )
        end = offset + length
        extent = self._extents[region_id]
        if end > extent:
            raise UnmappedAddressError(
                f"read of {length} bytes at offset {offset} runs past the mapped extent "
                f"{extent} of region {region_id!r}"
            )

        pieces: list[bytes] = []
        sources: list[ReadSource] = []
        cursor = offset
        if length == 0:
            # A zero-length read is legal only when its address lies inside
            # a readable mapped segment (its end position is irrelevant).
            for seg in segments:
                if seg.offset <= offset < seg.end:
                    if not seg.readable:
                        raise ReadPermissionError(
                            f"zero-length read at offset {offset} touches a mapped but "
                            f"non-readable interval in region {region_id!r}"
                        )
                    return b"", ()
                if seg.offset > offset:
                    break
            raise UnmappedAddressError(
                f"zero-length read at offset {offset} is not inside a mapped segment "
                f"of region {region_id!r}"
            )

        for seg in segments:
            if seg.end <= cursor:
                continue
            if seg.offset >= end:
                break
            if seg.offset > cursor:
                raise UnmappedAddressError(
                    f"read of {length} bytes at offset {offset} crosses an unmapped hole "
                    f"at offset {cursor} in region {region_id!r}"
                )
            if not seg.readable:
                raise ReadPermissionError(
                    f"read of {length} bytes at offset {offset} touches a mapped but "
                    f"non-readable interval at offset {cursor} in region {region_id!r}"
                )
            clip_end = min(seg.end, end)
            local_start = cursor - seg.offset
            local_end = clip_end - seg.offset
            pieces.append(seg.data[local_start:local_end])
            sources.append(ReadSource(region_id, cursor, clip_end - cursor))
            cursor = clip_end
            if cursor == end:
                break

        if cursor != end:
            raise UnmappedAddressError(
                f"read of {length} bytes at offset {offset} crosses an unmapped hole "
                f"at offset {cursor} in region {region_id!r}"
            )

        return b"".join(pieces), tuple(sources)

    def read(self, region_id: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes at ``offset``; same atomic rules as
        :meth:`resolve_read`, returning only the assembled bytes."""
        data, _sources = self.resolve_read(region_id, offset, length)
        return data
