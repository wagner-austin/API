"""Radar-related container message decoders.

This module provides decoders for radar response messages.
"""

from __future__ import annotations

from tankpit_bot.container.helpers import require_min_length
from tankpit_bot.container.types import (
    RadarContainerDict,
    RadarMineDict,
    RadarResponseDict,
)


def is_radar_response_structure(data: bytes) -> bool:
    """Check if data matches radar response structure.

    Criteria:
    - Minimum 3 bytes (header only)
    - Subtype byte (first byte) is 0x4F
    - Length >= 3 + (container_count * 4)
    - Remaining bytes after containers divisible by 3 (mines)

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches radar response pattern.
    """
    if len(data) < 3:
        return False
    # Must have radar response subtype
    if data[0] != 0x4F:
        return False
    container_count = data[1]
    # Must have at least 3 + (count * 4) bytes for containers
    expected_container_len = 3 + (container_count * 4)
    if len(data) < expected_container_len:
        return False
    # Remaining bytes must be divisible by 3 (mines are 3 bytes each)
    remaining = len(data) - expected_container_len
    return remaining % 3 == 0


def decode_radar_response(data: bytes) -> RadarResponseDict:
    """Decode radar response message.

    Parses containers (4 bytes each) then mines (3 bytes each) from remaining.

    Args:
        data: Decoded container body bytes.

    Returns:
        Decoded radar response with containers and mines.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_min_length(data, 3, "RadarResponse")

    container_count = data[1]
    # byte 2 is flags/padding (always 0?)
    expected_container_len = 3 + (container_count * 4)
    require_min_length(data, expected_container_len, "RadarResponse")

    # Parse containers
    containers: list[RadarContainerDict] = []
    offset = 3
    for _ in range(container_count):
        x = data[offset]
        y = data[offset + 1]
        raw_volume = data[offset + 2] | (data[offset + 3] << 8)
        volume = -1 if raw_volume == 0xFFFF else raw_volume

        containers.append(
            RadarContainerDict(
                x=x,
                y=y,
                volume=volume,
            )
        )
        offset += 4

    # Parse mines from remaining bytes (3 bytes each)
    mines: list[RadarMineDict] = []
    remaining = len(data) - offset
    mine_count = remaining // 3
    for _ in range(mine_count):
        x = data[offset]
        y = data[offset + 1]
        team = data[offset + 2]
        mines.append(RadarMineDict(x=x, y=y, team=team))
        offset += 3

    return RadarResponseDict(
        msg_type="radar_response",
        container_count=container_count,
        containers=containers,
        mines=mines,
    )


__all__ = [
    "decode_radar_response",
    "is_radar_response_structure",
]
