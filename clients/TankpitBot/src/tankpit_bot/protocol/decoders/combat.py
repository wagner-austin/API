"""Combat message decoders.

This module handles decoding of combat-related messages:
shoot events, hit confirmations, deactivations, mine placement/detonation.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.protocol.helpers import (
    DecodeError,
    require_exact_length,
    require_min_length,
    x16,
    x24,
)
from tankpit_bot.protocol.types import (
    DeactivationDict,
    HitConfirmationDict,
    MineDetonationDict,
    MinePlacementDict,
    ShootEventDict,
)

log = get_logger(__name__)


def decode_shoot_event(data: bytes) -> ShootEventDict:
    """Decode shooting event from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x53 prefix).

    Returns:
        Decoded shoot event.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 12, "ShootEvent")
    return ShootEventDict(
        msg_type=0x53,
        shooter_id=x16(data[0], data[1]),
        target_x=data[2],
        target_y=data[3],
        projectile_x=data[4],
        projectile_y=data[5],
        fuel=x24(data[6], data[7], data[8]),
        weapon=data[9],
        ammo=data[10],
        friendly_fire=data[11] == 1,
    )


def decode_hit_confirmation(data: bytes, xor_table: bytes) -> HitConfirmationDict:
    """Decode HIT message from raw body.

    Args:
        data: Raw message body (12 bytes, starts with 0x2E).
        xor_table: XOR table for decoding.

    Returns:
        Decoded hit confirmation.

    Raises:
        DecodeError: If decoding fails.
    """
    require_exact_length(data, 12, "HitConfirmation")
    if data[0] != 0x2E:
        raise DecodeError("HitConfirmation: expected 0x2E prefix")

    decoded = bytearray(len(data) - 1)
    for i in range(len(decoded)):
        decoded[i] = data[i + 1] ^ xor_table[i]

    return HitConfirmationDict(
        msg_type=0x2E,
        target_y=decoded[5],
        target_x=decoded[6],
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
    # Layout: [pad:1] [victim_id:2 LE] [pad:1] [killer_id:2 LE]
    return DeactivationDict(
        msg_type=0x41,
        victim_id=x16(data[1], data[2]),
        killer_id=x16(data[4], data[5]),
        rank=0,
        points=0,
    )


def decode_mine_placement(data: bytes) -> MinePlacementDict:
    """Decode mine placement from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded mine placement.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 4, "MinePlacement")
    mine_type = data[0]
    tank_id = x16(data[1], data[2])
    count = data[3]
    positions: list[tuple[int, int]] = []
    idx = 4
    for _ in range(count):
        if idx + 1 >= len(data):
            break
        positions.append((data[idx], data[idx + 1]))
        idx += 2
    return MinePlacementDict(
        msg_type=0x4B, mine_type=mine_type, tank_id=tank_id, positions=positions
    )


def decode_mine_detonation(data: bytes) -> MineDetonationDict:
    """Decode mine detonation from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded mine detonation.
    """
    positions: list[tuple[int, int]] = []
    for i in range(0, len(data) - 1, 2):
        positions.append((data[i], data[i + 1]))
    return MineDetonationDict(msg_type=0x45, positions=positions)


__all__ = [
    "decode_deactivation",
    "decode_hit_confirmation",
    "decode_mine_detonation",
    "decode_mine_placement",
    "decode_shoot_event",
]
