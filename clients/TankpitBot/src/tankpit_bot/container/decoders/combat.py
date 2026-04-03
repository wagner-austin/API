"""Combat-related container message decoders.

This module provides decoders for combat hit and deactivation messages.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.container.helpers import (
    ContainerDecodeError,
    extract_uint16_le,
    require_exact_length,
)
from tankpit_bot.container.types import (
    CombatHitDict,
    DeactivationDeathDict,
    DeactivationKillDict,
    MineDetonationDict,
    MinePlacementDict,
)

log = get_logger(__name__)


def is_combat_hit_structure(data: bytes) -> bool:
    """Check if data matches combat hit message structure.

    Combat hit criteria:
    - Exactly 11 bytes

    Note: Terminator pattern (first==last) varies per session due to XOR encoding.
    We identify solely by length since 11 bytes is unique to combat hits.

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches combat hit pattern.
    """
    return len(data) == 11


def is_mine_placement_structure(data: bytes) -> bool:
    """Check if data matches tunneled mine placement structure.

    Criteria:
    - Exactly 15 bytes
    - First decoded byte is 0x4B
    - Count byte at index 4 yields an exact payload length

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches tunneled mine placement.
    """
    if len(data) != 15 or data[0] != 0x4B:
        return False
    count = data[4]
    return len(data) == 5 + count * 2


def is_mine_detonation_structure(data: bytes) -> bool:
    """Check if data matches tunneled mine detonation structure."""
    return len(data) >= 3 and data[0] == 0x45 and (len(data) - 1) % 2 == 0


def decode_mine_detonation(data: bytes) -> MineDetonationDict:
    """Decode tunneled mine detonation from container body."""
    if not is_mine_detonation_structure(data):
        raise ContainerDecodeError(
            f"MineDetonation: expected subtype 0x45 with coordinate pairs, got {data.hex()}"
        )
    positions: list[tuple[int, int]] = []
    for offset in range(1, len(data), 2):
        positions.append((data[offset], data[offset + 1]))
    return MineDetonationDict(msg_type=0x45, positions=positions)


def decode_mine_placement(data: bytes) -> MinePlacementDict:
    """Decode tunneled mine placement from container body.

    Structure (15 bytes):
      [subtype:1] [mine_type:1] [tank_id:2 LE] [count:1] [positions: count*2]

    Args:
        data: Decoded container body bytes.

    Returns:
        Decoded mine placement data.
    """
    require_exact_length(data, 15, "MinePlacement")
    mine_type = data[1]
    tank_id = extract_uint16_le(data, 2, "MinePlacement.tank_id")
    count = data[4]
    positions: list[tuple[int, int]] = []
    for i in range(count):
        offset = 5 + i * 2
        positions.append((data[offset], data[offset + 1]))
    return MinePlacementDict(
        msg_type=0x4B,
        mine_type=mine_type,
        tank_id=tank_id,
        positions=positions,
    )


def decode_combat_hit(data: bytes) -> CombatHitDict:
    """Decode combat hit message from container body.

    Structure (11 bytes):
      [subtype:1] [direction:1] [attacker_id:2 LE] [combat_data:7]

    Args:
        data: Decoded container body bytes (must be 11 bytes).

    Returns:
        Decoded combat hit data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 11, "CombatHit")

    direction = data[1]
    attacker_id = extract_uint16_le(data, 2, "CombatHit.attacker_id")
    combat_data = bytes(data[4:11])  # Last 7 bytes
    is_outgoing = direction == 0x09

    return CombatHitDict(
        msg_type="combat_hit",
        direction=direction,
        attacker_id=attacker_id,
        combat_data=combat_data,
        is_outgoing=is_outgoing,
    )


def is_deactivation_kill_structure(data: bytes) -> bool:
    """Check if data matches deactivation kill message structure.

    Criteria:
    - Exactly 5 bytes
    - First decoded byte is 0x41 ('A')

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches deactivation kill pattern.
    """
    return len(data) == 5 and data[0] == 0x41


def decode_deactivation_kill(data: bytes) -> DeactivationKillDict:
    """Decode deactivation kill message from container body.

    Structure (5 bytes):
      [subtype:1] [victim_id:2 LE] [killer_id:2 LE]

    Args:
        data: Decoded container body bytes (must be 5 bytes).

    Returns:
        Decoded deactivation kill data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 5, "DeactivationKill")

    log.info(
        "DEACTIVATION_KILL RAW: hex=%s bytes=[%s]",
        data.hex(),
        ", ".join(str(b) for b in data),
    )

    victim_id = extract_uint16_le(data, 1, "DeactivationKill.victim_id")
    killer_id = extract_uint16_le(data, 3, "DeactivationKill.killer_id")

    return DeactivationKillDict(
        msg_type="deactivation_kill",
        victim_id=victim_id,
        killer_id=killer_id,
    )


def is_deactivation_death_structure(data: bytes) -> bool:
    """Check if data matches deactivation death message structure.

    Criteria:
    - Exactly 7 bytes
    - First decoded byte is 0x43 ('C')

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches deactivation death pattern.
    """
    return len(data) == 7 and data[0] == 0x43


def decode_deactivation_death(data: bytes) -> DeactivationDeathDict:
    """Decode deactivation death message from container body.

    Structure (7 bytes):
      [subtype:1] [flags:1] [killer_id:2 LE] [extra:3]

    Args:
        data: Decoded container body bytes (must be 7 bytes).

    Returns:
        Decoded deactivation death data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 7, "DeactivationDeath")

    flags = data[1]
    killer_id = extract_uint16_le(data, 2, "DeactivationDeath.killer_id")
    extra_data = bytes(data[4:7])

    return DeactivationDeathDict(
        msg_type="deactivation_death",
        flags=flags,
        killer_id=killer_id,
        extra_data=extra_data,
    )


__all__ = [
    "decode_combat_hit",
    "decode_deactivation_death",
    "decode_deactivation_kill",
    "decode_mine_detonation",
    "decode_mine_placement",
    "is_combat_hit_structure",
    "is_deactivation_death_structure",
    "is_deactivation_kill_structure",
    "is_mine_detonation_structure",
    "is_mine_placement_structure",
]
