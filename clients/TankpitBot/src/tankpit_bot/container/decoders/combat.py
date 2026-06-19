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
    DeactivationDeathDict,
    MineDetonationDict,
    MinePlacementDict,
)

log = get_logger(__name__)


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
    "decode_deactivation_death",
    "decode_mine_detonation",
    "decode_mine_placement",
    "is_deactivation_death_structure",
    "is_mine_detonation_structure",
    "is_mine_placement_structure",
]
