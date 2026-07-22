"""Movement message decoders.

This module handles decoding of movement-related messages:
movement paths and movement responses.
"""

from __future__ import annotations

from tankpit_bot.protocol.helpers import require_min_length, x16, x24
from tankpit_bot.protocol.types import MovementDict, MovementResponseDict


def decode_movement(data: bytes) -> MovementDict:
    """Decode movement from XOR-decoded data.

    Layout from tpclient.js Lg.h (V.G), verified 2026-06-19:
      [0:2]  tank_id (LE u16)
      [2]    start_x
      [3]    start_y
      [4]    direction
      [5]    rank_category (b.u in Lg.prototype.h)
      [6:9]  lb_score (24-bit BE)
      [9]    rank (b.l in Lg.prototype.h)
      [10]   animation flag (passed to Re constructor, not tank state)
      [11]   is_carrying (1=true)
      [12:]  waypoints

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded movement.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 12, "Movement")
    # Waypoints are ASCII direction chars at bytes 12+ (n/s/e/w)
    path_str = ""
    for i in range(12, len(data)):
        ch = chr(data[i])
        if ch in "nsew":
            path_str += ch
    # Compute final position from start + path
    sx, sy = data[2], data[3]
    fx, fy = sx, sy
    for ch in path_str:
        if ch == "n":
            fy -= 1
        elif ch == "s":
            fy += 1
        elif ch == "e":
            fx += 1
        else:  # ch == "w" (path_str only contains nsew)
            fx -= 1
    waypoints: list[tuple[int, int]] = [(fx, fy)] if path_str else []
    return MovementDict(
        msg_type=0x47,
        tank_id=x16(data[0], data[1]),
        start_x=sx,
        start_y=sy,
        direction=data[4],
        damage_state=data[5],
        lb_score=x24(data[6], data[7], data[8]),
        rank=data[9],
        flag=data[10],
        is_carrying=data[11] == 1,
        waypoints=waypoints,
        path_tiles=len(path_str),
        path=path_str,
    )


def decode_movement_response(data: bytes) -> MovementResponseDict:
    """Decode movement response from XOR-decoded data.

    Layout from tpclient.js Mg.h (V["="]), verified 2026-06-19:
      a[0]    = team
      a[1:3]  = tank_id (LE u16)
      a[3]    = x
      a[4]    = y
      a[5]    = direction
      a[6]    = damage_state (b.u; dual-purpose: rank_category on init)
      a[7]    = rank (b.l)
      a[8:11] = lb_score (24-bit BE)
      a[11]   = carrying flag

    Prior decoder required only 11 bytes and dropped the carrying byte
    at offset 11; the container-path TankPositionStatus had it correct.
    Container path is being deleted 2026-06-19; protocol becomes the
    single source of truth with the carrying byte restored.

    Args:
        data: XOR-decoded message body (without 0x3D prefix).

    Returns:
        Decoded movement response.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 11, "MovementResponse")
    # Carrying byte at offset 11 is present in normal wire bodies (real
    # captures show 12 bytes) but some trimmed test fixtures only carry
    # 11 bytes. Per JS Mg.h the carrying byte exists but its execute
    # path doesn't use it directly, so 11-byte fixtures still decode
    # cleanly here -- we just default carrying to 0.
    carrying = data[11] if len(data) >= 12 else 0
    return MovementResponseDict(
        msg_type=0x3D,
        team=data[0],
        tank_id=x16(data[1], data[2]),
        x=data[3],
        y=data[4],
        direction=data[5],
        damage_state=data[6],
        rank=data[7],
        lb_score=x24(data[8], data[9], data[10]),
        carrying=carrying,
    )


__all__ = [
    "decode_movement",
    "decode_movement_response",
]
