"""0x4C MapData encoder — exact byte inverse of ``decoders.map_data``."""

from __future__ import annotations

from tankpit_bot.protocol.helpers import pack16
from tankpit_bot.protocol.types import MapDataDict

_MAP_WIDTH = 256


def _encode_fuel_dot_rle(dots: list[tuple[int, int]]) -> bytes:
    """Emit the skip-RLE fuel-dot atlas (inverse of ``_decode_fuel_dots``).

    The decoder walks a 2-D cursor from ``(1, 1)`` where each byte
    advances x (wrapping x>255 to the next row) and every byte except
    255 emits a dot. In linear space (``y * 256 + x``) a step byte
    advances the cursor by exactly its value, so the greedy encoding —
    255-skips until the remaining delta fits one byte — reproduces the
    server's atlas byte-for-byte (3,797/3,797 corpus bodies).

    Args:
        dots: Fuel-dot positions in stream order.

    Returns:
        RLE section bytes.
    """
    out = bytearray()
    cursor = _MAP_WIDTH + 1
    for x, y in dots:
        target = y * _MAP_WIDTH + x
        delta = target - cursor
        while delta >= 255:
            out.append(255)
            delta -= 255
        out.append(delta)
        cursor = target
    return bytes(out)


def encode_map_data(message: MapDataDict) -> bytes:
    """Encode a 0x4C MapData payload (inverse of ``decode_map_data``).

    Args:
        message: Decoded map data.

    Returns:
        Payload bytes without the 0x4C prefix: LE u16 RLE byte count,
        the fuel-dot atlas, then one 5-byte slot per tank.
    """
    rle = _encode_fuel_dot_rle(message["fuel_dots"])
    out = bytearray(pack16(len(rle)) + rle)
    for tank in message["tanks"]:
        packed = (tank["team"] & 3) | ((tank["damage"] & 3) << 2) | ((tank["rank"] & 15) << 4)
        out += bytes([tank["x"], tank["y"]]) + pack16(tank["tank_id"]) + bytes([packed])
    return bytes(out)


__all__ = [
    "encode_map_data",
]
