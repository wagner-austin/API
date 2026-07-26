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
      [7]    aim_x     (aim tile X -- where the gun is pointed)
      [8]    aim_y     (aim tile Y)
      [9]    weapon (0=single, 1=dual, 2=missile, 3=homing)

    Bytes 7-8 promoted from ``unk1``/``unk2`` to ``aim_x``/``aim_y``
    2026-06-20 after JS source confirmed they are the projectile
    animation's PIXEL CENTRE source (``24*z+12, 16*O+8`` in ``yf``).
    Real-combat corpus (practice-vs-real-20260620-150138) shows
    aim==target on every observed straight shot (single/dual);
    homing/missile weapons may diverge.

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
        aim_x=data[7],
        aim_y=data[8],
        weapon=data[9],
    )


def decode_deactivation(data: bytes) -> DeactivationDict:
    """Decode a 0x41 Deactivation from XOR-decoded data.

    The 0x41 type byte is overloaded on the wire: the six-byte
    XOR-encoded Deactivation vs the two-byte PLAINTEXT autoscroll
    ack (``"A0"``/``"A1"``, un-XORed). The ack must be discriminated
    at the framing layer before any XOR decode — see
    :func:`~tankpit_bot.protocol.decoders.text.try_decode_plaintext_ack`.
    This decoder handles only the binary Deactivation form.

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
