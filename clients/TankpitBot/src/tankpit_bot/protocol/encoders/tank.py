"""Tank message encoders — exact byte inverses of ``decoders.tank``.

Each function produces the XOR-decoded payload (without the type or
envelope byte) for one tank-family message. Byte layouts live on the
matching decoder's docstring; every encoder here was graded
byte-identical against the full capture archive on 2026-07-21
(wiki [[physics-module-roadmap]] Phase 4 step a).
"""

from __future__ import annotations

from tankpit_bot.protocol.types import (
    TankEntryDict,
    TankExitDict,
    TankInfoDict,
    TankRemoveDict,
    TankStatusDict,
    TankStatusSyncDict,
)
from tankpit_bot.wire.helpers import pack16, pack24


def encode_tank_info(message: TankInfoDict) -> bytes:
    """Encode a 0x21 TankInfo payload (inverse of ``decode_tank_info``).

    Args:
        message: Decoded tank info.

    Returns:
        Payload bytes without the 0x21 prefix.
    """
    return (
        bytes([message["team"]])
        + pack16(message["tank_id"])
        + message["decoration_state"]
        + pack24(message["persistent_tank_id"])
        + message["name"].encode("utf-8")
    )


def encode_tank_entry(message: TankEntryDict) -> bytes:
    """Encode a 0x28 TankEntry payload (inverse of ``decode_tank_entry``).

    The leading flags byte is not carried by the dict; the corpus
    (2026-07-21, 6/6 tunneled samples) shows it always equals the team,
    so it is re-derived from ``team``.

    Args:
        message: Decoded tank entry.

    Returns:
        Payload bytes without the 0x28 prefix.
    """
    packed = (
        (message["team"] & 3) | ((message["damage_state"] & 3) << 2) | ((message["rank"] & 15) << 4)
    )
    return (
        bytes([message["team"]])
        + pack16(message["tank_id"])
        + bytes([packed])
        + pack24(message["score"])
        + bytes([message["x"], message["y"]])
    )


def encode_tank_status_sync(message: TankStatusSyncDict) -> bytes:
    """Encode a 0x2E TankStatusSync body (inverse of ``decode_tank_status_sync``).

    Length variant follows the optional fields: 8 bytes bare, 9 with
    ``promo_state``, 12 with ``promo_bar_lit`` and ``fuel``. That byte
    used to be hardcoded to 1 here on a corpus claim that has since
    been falsified — 219 archived long-form bodies carry 0 — so it is
    now carried through from the decode
    ([[session-state-deglobalisation]]).

    Args:
        message: Decoded status sync.

    Returns:
        Body bytes without the envelope's 0x2E subtype byte.
    """
    out = (
        bytes([message["subtype"]])
        + pack16(message["tank_id"])
        + bytes([message["damage_state"], message["rank"]])
        + pack24(message["lb_score"])
    )
    if message["promo_state"] is not None:
        out += bytes([message["promo_state"]])
        if message["fuel"] is not None:
            out += bytes([1 if message["promo_bar_lit"] else 0]) + pack16(message["fuel"])
    return out


def encode_tank_status(message: TankStatusDict) -> bytes:
    """Encode a 0x3E TankStatus payload (inverse of ``decode_tank_status``).

    Args:
        message: Decoded full tank status.

    Returns:
        Payload bytes without the 0x3E prefix.
    """
    info = (
        (message["team"] & 3) | ((message["damage_state"] & 3) << 2) | ((message["rank"] & 15) << 4)
    )
    return (
        bytes([info])
        + pack16(message["tank_id"])
        + message["decoration_state"]
        + pack24(message["leaderboard_score"])
        + pack24(message["leaderboard_position"])
        + message["name"].encode("utf-8")
    )


def encode_tank_exit(message: TankExitDict) -> bytes:
    """Encode a 0x29 TankExit payload (inverse of ``decode_tank_exit``).

    Args:
        message: Decoded tank exit announcement.

    Returns:
        Payload bytes without the 0x29 prefix.
    """
    return (
        bytes([message["team"]])
        + pack16(message["tank_id"])
        + bytes(
            [
                1 if message["was_silent"] else 0,
                1 if message["was_eliminated"] else 0,
            ]
        )
    )


def encode_tank_remove(message: TankRemoveDict) -> bytes:
    """Encode a 0x58 TankRemove payload (inverse of ``decode_tank_remove``).

    Args:
        message: Decoded tank remove.

    Returns:
        Payload bytes without the 0x58 prefix.
    """
    return pack16(message["tank_id"])


__all__ = [
    "encode_tank_entry",
    "encode_tank_exit",
    "encode_tank_info",
    "encode_tank_remove",
    "encode_tank_status",
    "encode_tank_status_sync",
]
