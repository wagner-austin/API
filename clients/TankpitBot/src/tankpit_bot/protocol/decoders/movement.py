"""Movement message decoders.

This module handles decoding of movement-related messages:
movement paths and movement responses.
"""

from __future__ import annotations

from tankpit_bot.protocol.helpers import require_min_length, x16, x24
from tankpit_bot.protocol.types import MovementDict, MovementResponseDict


def decode_movement(data: bytes) -> MovementDict:
    """Decode movement from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded movement.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 9, "Movement")
    # Waypoints are ASCII direction chars at bytes 11+ (n/s/e/w)
    path_str = ""
    for i in range(11, len(data)):
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
        elif ch == "w":
            fx -= 1
    return MovementDict(
        msg_type=0x47,
        tank_id=x16(data[0], data[1]),
        start_x=sx,
        start_y=sy,
        direction=data[4],
        flag=data[5],
        leaderboard_position=x24(data[6], data[7], data[8]),
        waypoints=[(fx, fy)],
    )


def decode_movement_response(data: bytes) -> MovementResponseDict:
    """Decode movement response from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x3D prefix).

    Returns:
        Decoded movement response.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 11, "MovementResponse")
    return MovementResponseDict(
        msg_type=0x3D,
        team=data[0],
        tank_id=x16(data[1], data[2]),
        x=data[3],
        y=data[4],
        direction=data[5],
        rank=data[7],
        leaderboard_position=x24(data[8], data[9], data[10]),
    )


__all__ = [
    "decode_movement",
    "decode_movement_response",
]
