"""Combat message encoders — exact byte inverses of ``decoders.combat``."""

from __future__ import annotations

from tankpit_bot.protocol.helpers import pack16
from tankpit_bot.protocol.types import DeactivationDict, ShootEventDict

_MINE_KILLER_BASE = 65530


def encode_shoot_event(message: ShootEventDict) -> bytes:
    """Encode a 0x53 ShootEvent payload (inverse of ``decode_shoot_event``).

    Args:
        message: Decoded shoot event.

    Returns:
        Payload bytes without the 0x53 prefix.
    """
    return (
        bytes([message["team"]])
        + pack16(message["shooter_id"])
        + bytes(
            [
                message["source_x"],
                message["source_y"],
                message["target_x"],
                message["target_y"],
                message["aim_x"],
                message["aim_y"],
                message["weapon"],
            ]
        )
    )


def encode_deactivation(message: DeactivationDict) -> bytes:
    """Encode a 0x41 Deactivation payload (inverse of ``decode_deactivation``).

    Args:
        message: Decoded deactivation.

    Returns:
        Payload bytes without the 0x41 prefix. Mine kills re-add the
        65530 killer-id offset the decoder strips.
    """
    raw_killer = (
        message["killer_id"] + _MINE_KILLER_BASE
        if message["is_mine_kill"]
        else message["killer_id"]
    )
    return (
        bytes([message["status"]])
        + pack16(message["victim_id"])
        + bytes([1 if message["promo_eligible"] else 0])
        + pack16(raw_killer)
    )


__all__ = [
    "encode_deactivation",
    "encode_shoot_event",
]
