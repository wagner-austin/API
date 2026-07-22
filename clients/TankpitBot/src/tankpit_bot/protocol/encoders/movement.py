"""Movement message encoders — exact byte inverses of ``decoders.movement``."""

from __future__ import annotations

from tankpit_bot.protocol.helpers import pack16, pack24
from tankpit_bot.protocol.types import MovementDict, MovementResponseDict


def encode_movement(message: MovementDict) -> bytes:
    """Encode a 0x47 Movement payload (inverse of ``decode_movement``).

    The nsew path string is emitted verbatim as ASCII waypoint bytes;
    ``waypoints`` and ``path_tiles`` are derived fields and not
    re-encoded.

    Args:
        message: Decoded movement.

    Returns:
        Payload bytes without the 0x47 prefix.
    """
    return (
        pack16(message["tank_id"])
        + bytes(
            [
                message["start_x"],
                message["start_y"],
                message["direction"],
                message["damage_state"],
            ]
        )
        + pack24(message["lb_score"])
        + bytes([message["rank"], message["flag"], 1 if message["is_carrying"] else 0])
        + message["path"].encode("ascii")
    )


def encode_movement_response(message: MovementResponseDict) -> bytes:
    """Encode a 0x3D MovementResponse payload (inverse of ``decode_movement_response``).

    Args:
        message: Decoded movement response.

    Returns:
        Payload bytes without the 0x3D prefix (12 bytes, carrying
        byte included).
    """
    return (
        bytes([message["team"]])
        + pack16(message["tank_id"])
        + bytes(
            [
                message["x"],
                message["y"],
                message["direction"],
                message["damage_state"],
                message["rank"],
            ]
        )
        + pack24(message["lb_score"])
        + bytes([message["carrying"]])
    )


__all__ = [
    "encode_movement",
    "encode_movement_response",
]
