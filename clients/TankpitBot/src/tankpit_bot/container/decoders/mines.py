"""Tunneled mine decoders carried inside the 0x2E container envelope.

Two subtypes are container-only (no protocol counterpart):
- 0x4B MinePlacement (15 bytes): [subtype][mine_type][tank_id:LE16][count][positions]
- 0x45 MineDetonation (3+ odd bytes): [subtype][(x, y) pairs]
"""

from __future__ import annotations

from tankpit_bot.container.helpers import (
    ContainerDecodeError,
    extract_uint16_le,
    require_min_length,
)
from tankpit_bot.container.types import (
    MineDetonationDict,
    MinePlacementDict,
)


def is_mine_placement_structure(data: bytes) -> bool:
    """Check if data matches tunneled mine placement structure.

    Criteria:
    - First decoded byte is 0x4B
    - Header is 5 bytes ([subtype][mine_type][tank_id:LE16][count])
    - Total length equals ``5 + count * 2`` (each position is 2 bytes)

    Real-combat wire evidence (capture 2026-06-20 15:02:56,
    practice-vs-real-20260620-150138): count=7 -> 19 bytes. The prior
    ``len(data) == 15`` gate silently dropped every non-5-position
    placement to UnknownContainer.

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches tunneled mine placement.
    """
    if len(data) < 5 or data[0] != 0x4B:
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

    Structure (5 + count*2 bytes):
      [subtype:1=0x4B] [mine_type:1] [tank_id:2 LE] [count:1] [positions: count*2]

    Real-combat capture 2026-06-20 confirms count varies from 5 (the
    only value seen in solo practice) up to at least 7 (placement
    around an enemy tank during 1v1 combat).

    Args:
        data: Decoded container body bytes.

    Returns:
        Decoded mine placement data.

    Raises:
        ContainerDecodeError: If the body header is shorter than 5
            bytes or the trailing payload doesn't match the count byte.
    """
    require_min_length(data, 5, "MinePlacement")
    if data[0] != 0x4B:
        raise ContainerDecodeError(f"MinePlacement: expected subtype 0x4B, got 0x{data[0]:02X}")
    count = data[4]
    expected_length = 5 + count * 2
    if len(data) != expected_length:
        raise ContainerDecodeError(
            f"MinePlacement: count={count} requires {expected_length} bytes, got {len(data)}"
        )
    mine_type = data[1]
    tank_id = extract_uint16_le(data, 2, "MinePlacement.tank_id")
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


__all__ = [
    "decode_mine_detonation",
    "decode_mine_placement",
    "is_mine_detonation_structure",
    "is_mine_placement_structure",
]
