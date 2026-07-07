"""0x4C 'L' MapData decoder.

The MapData message is a whole-map snapshot streamed when the player
opens the strategic map. JS handler is ``Ig`` (V.L). The body packs
two sections: a run-length list of fuel-dot positions followed by a
flat array of tank slots.

The fuel-dot RLE section is the map's yellow-pixel fuel atlas
(server-cached per session, byte-identical across map opens; ~40% of
dots still hold fuel when visited, and every verified dot held
high-volume fuel -- see wiki [[map-data-decode]]). Restored 2026-07-03
for dot-hop restocking and dot-relay travel; it was decoded for
length only between 2026-06-22 and then.

Trace-verified from ``tpclient.js`` Ig.h. Layout details live on
:class:`tankpit_bot.protocol.types.MapDataDict`.
"""

from __future__ import annotations

from tankpit_bot.protocol.helpers import DecodeError, require_min_length, x16
from tankpit_bot.protocol.types import MapDataDict, MapTankEntry

# Each tank slot is 5 bytes: x, y, tank_id (LE u16), packed byte.
_TANK_ENTRY_BYTES = 5


def _decode_fuel_dots(body: bytes, rle_count: int) -> list[tuple[int, int]]:
    """Decode the skip-RLE fuel-dot section.

    Mirrors JS ``Ig.h`` exactly: a 2-D cursor starts at ``(1, 1)``;
    each byte advances x by its value, wrapping to ``y += 1,
    x %= 256`` whenever x exceeds 255. A byte of 255 is a pure skip
    (cursor advance, no dot); every other byte emits the cursor
    position as a fuel dot.

    Args:
        body: Full XOR-decoded MapData body.
        rle_count: RLE byte count from the LE u16 header.

    Returns:
        Fuel-dot ``(x, y)`` positions in stream order.
    """
    dots: list[tuple[int, int]] = []
    x = 1
    y = 1
    for offset in range(2, 2 + rle_count):
        step = body[offset]
        x += step
        if x > 255:
            y += 1
            x %= 256
        if step != 255:
            dots.append((x, y))
    return dots


def _decode_tank_entries(body: bytes, start: int) -> list[MapTankEntry]:
    """Decode the trailing tank-slot section.

    Args:
        body: Full XOR-decoded MapData body.
        start: Body offset where the tank section begins.

    Returns:
        One :class:`MapTankEntry` per 5-byte slot to end of body.

    Raises:
        DecodeError: If the trailing section is not a clean multiple
            of 5 bytes -- the wire format guarantees it; a remainder
            means the body is truncated or corrupt and is never
            silently rounded down.
    """
    tail = len(body) - start
    if tail % _TANK_ENTRY_BYTES != 0:
        raise DecodeError(
            f"MapData.tank_entries: trailing section of {tail} bytes is not a "
            f"multiple of {_TANK_ENTRY_BYTES}; body is truncated or corrupt"
        )
    entries: list[MapTankEntry] = []
    cursor = start
    while cursor < len(body):
        x = body[cursor]
        y = body[cursor + 1]
        tank_id = x16(body[cursor + 2], body[cursor + 3])
        packed = body[cursor + 4]
        rank = (packed >> 4) & 0xF
        damage = (packed >> 2) & 0x3
        team = packed & 0x3
        entries.append(
            MapTankEntry(
                x=x,
                y=y,
                tank_id=tank_id,
                rank=rank,
                damage=damage,
                team=team,
            )
        )
        cursor += _TANK_ENTRY_BYTES
    return entries


def decode_map_data(data: bytes) -> MapDataDict:
    """Decode the 0x4C 'L' MapData blob into fuel dots and the tank list.

    See :class:`MapDataDict` for layout. The header is two bytes (LE
    u16) carrying the run-length region's byte count; bytes
    ``2..2+count`` hold the skip-RLE fuel-dot atlas. Everything past
    the RLE region is a packed list of 5-byte tank slots.

    Args:
        data: XOR-decoded message body (without the 0x4C prefix).

    Returns:
        Decoded :class:`MapDataDict`.

    Raises:
        DecodeError: If decoding fails (header truncated, RLE region
            extends past body, or tank-entry section length is not a
            clean multiple of 5 bytes).
    """
    require_min_length(data, 2, "MapData")
    rle_count = x16(data[0], data[1])
    require_min_length(data, 2 + rle_count, "MapData.rle_section")
    fuel_dots = _decode_fuel_dots(data, rle_count)
    tanks = _decode_tank_entries(data, 2 + rle_count)
    return MapDataDict(
        msg_type=0x4C,
        fuel_dots=fuel_dots,
        tanks=tanks,
    )


__all__ = ["decode_map_data"]
