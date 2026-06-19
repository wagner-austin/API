"""Combat message decoders.

This module handles decoding of combat-related messages:
shoot events, deactivations.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.protocol.helpers import (
    require_min_length,
    x16,
)
from tankpit_bot.protocol.types import (
    DeactivationDict,
    ShootEventDict,
)

log = get_logger(__name__)


def decode_shoot_event(data: bytes) -> ShootEventDict:
    """Decode shooting / hit event from XOR-decoded data.

    Layout from tpclient.js Gg.h (V.S), re-verified 2026-06-19 against
    real wire bytes from runs/bot/bot-20260619-050303 msg t+25.47s
    `53 02 15 05 b2 7d b2 7e b2 7e 01`:
      [0]    team (red=0, purple=1, blue=2, orange=3)
      [1:3]  shooter_id (LE u16)
      [3]    source_x  (shooter's tile X -- live position of attacker)
      [4]    source_y
      [5]    target_x  (shot's landing tile X)
      [6]    target_y
      [7]    unk1 (often duplicates target -- semantics TBD)
      [8]    unk2
      [9]    weapon (0=single, 1=dual, 2=missile, 3=homing)

    Prior decoder swapped target/source meanings and misnamed bytes 7-9.
    Three-way validated against enemy position tracking, homing target
    tile, and wire damage transitions.

    Args:
        data: XOR-decoded message body (without 0x53 prefix).

    Returns:
        Decoded shoot event.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 10, "ShootEvent")
    return ShootEventDict(
        msg_type=0x53,
        team=data[0],
        shooter_id=x16(data[1], data[2]),
        source_x=data[3],
        source_y=data[4],
        target_x=data[5],
        target_y=data[6],
        unk1=data[7],
        unk2=data[8],
        weapon=data[9],
    )


def decode_deactivation(data: bytes) -> DeactivationDict:
    """Decode deactivation event from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x41 prefix).

    Returns:
        Decoded deactivation.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 6, "Deactivation")
    # Layout from tpclient.js Pg.h (V.A):
    # [status:1] [victim_id:2 LE] [promo_eligible:1] [killer_id:2 LE]
    raw_killer = x16(data[4], data[5])
    is_mine = raw_killer >= 65530
    killer_id = raw_killer - 65530 if is_mine else raw_killer
    return DeactivationDict(
        msg_type=0x41,
        status=data[0],
        victim_id=x16(data[1], data[2]),
        promo_eligible=data[3] == 1,
        killer_id=killer_id,
        is_mine_kill=is_mine,
    )


# 0x4B MinePlacement and 0x45 MineDetonation live in
# tankpit_bot.container (decode_mine_placement, decode_mine_detonation).
# These wire formats only arrive as subtypes inside 0x2E containers --
# the protocol path was redundant and was deleted 2026-06-19.


__all__ = [
    "decode_deactivation",
    "decode_shoot_event",
]
