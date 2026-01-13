"""Miscellaneous container message decoders.

This module provides decoders for teleport, container pickup, player list,
tip notification, chunk data, world state, and unknown container messages.
"""

from __future__ import annotations

from tankpit_bot.container.helpers import (
    require_exact_length,
    require_length_range,
    require_min_length,
)
from tankpit_bot.container.types import (
    ChunkDataDict,
    ContainerPickupDict,
    PlayerListExtendedDict,
    PlayerListShortDict,
    TeleportLandedDict,
    TipNotificationDict,
    UnknownContainerDict,
    WorldStateDict,
)


def is_teleport_landed_structure(data: bytes) -> bool:
    """Check if data matches teleport landed structure.

    Teleport landed criteria:
    - Exactly 1 byte

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches teleport landed pattern.
    """
    return len(data) == 1


def decode_teleport_landed(data: bytes) -> TeleportLandedDict:
    """Decode teleport landed container message.

    Args:
        data: Decoded container body bytes (must be 1 byte).

    Returns:
        Decoded teleport landed data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 1, "TeleportLanded")

    return TeleportLandedDict(
        msg_type="teleport_landed",
        subtype=data[0],
    )


def is_container_pickup_structure(data: bytes) -> bool:
    """Check if data matches container pickup structure.

    Criteria:
    - Exactly 5 bytes
    - Subtype byte (first byte) is 0x43

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches container pickup pattern.
    """
    if len(data) != 5:
        return False
    return data[0] == 0x43


def decode_container_pickup(data: bytes) -> ContainerPickupDict:
    """Decode container pickup message.

    Args:
        data: Decoded container body bytes (must be 5 bytes with 0x43 subtype).

    Returns:
        Decoded container pickup data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 5, "ContainerPickup")

    x = data[1]
    y = data[2]
    volume = data[3] | (data[4] << 8)

    return ContainerPickupDict(
        msg_type="container_pickup",
        x=x,
        y=y,
        volume=volume,
        is_fuel=volume > 0,
    )


def is_player_list_short_structure(data: bytes) -> bool:
    """Check if data matches short player list response structure.

    Criteria:
    - Exactly 4 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches player list short pattern.
    """
    return len(data) == 4


def decode_player_list_short(data: bytes) -> PlayerListShortDict:
    """Decode short player list response from container body.

    Args:
        data: Decoded container body bytes (must be 4 bytes).

    Returns:
        Decoded player list response.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 4, "PlayerListShort")

    return PlayerListShortDict(
        msg_type="player_list_short",
        response_data=bytes(data[1:]),
    )


def is_player_list_extended_structure(data: bytes) -> bool:
    """Check if data matches extended player list response structure.

    Criteria:
    - Exactly 7 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches player list extended pattern.
    """
    return len(data) == 7


def decode_player_list_extended(data: bytes) -> PlayerListExtendedDict:
    """Decode extended player list response from container body.

    Args:
        data: Decoded container body bytes (must be 7 bytes).

    Returns:
        Decoded player list response.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 7, "PlayerListExtended")

    return PlayerListExtendedDict(
        msg_type="player_list_extended",
        response_data=bytes(data[1:4]),
        extended_data=bytes(data[4:7]),
    )


def is_tip_notification_structure(data: bytes) -> bool:
    """Check if data matches tip notification structure.

    Tip notification criteria:
    - Length 29-79 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches tip notification pattern.
    """
    return 29 <= len(data) <= 79


def decode_tip_notification(data: bytes) -> TipNotificationDict:
    """Decode tip notification container message.

    Args:
        data: Decoded container body bytes (must be 29-79 bytes).

    Returns:
        Decoded tip notification data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_length_range(data, 29, 79, "TipNotification")

    return TipNotificationDict(
        msg_type="tip_notification",
        subtype=data[0],
        length=len(data),
        notification_data=bytes(data[1:]),
    )


def is_chunk_data_structure(data: bytes) -> bool:
    """Check if data matches chunk data structure.

    Chunk data criteria:
    - Length 80-130 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches chunk data pattern.
    """
    return 80 <= len(data) <= 130


def decode_chunk_data(data: bytes) -> ChunkDataDict:
    """Decode chunk data container message.

    Args:
        data: Decoded container body bytes (must be 80-130 bytes).

    Returns:
        Decoded chunk data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_length_range(data, 80, 130, "ChunkData")

    return ChunkDataDict(
        msg_type="chunk_data",
        subtype=data[0],
        length=len(data),
        chunk_data=bytes(data[1:]),
    )


def is_world_state_structure(data: bytes) -> bool:
    """Check if data matches world state structure.

    World state criteria:
    - Length >= 500 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches world state pattern.
    """
    return len(data) >= 500


def decode_world_state(data: bytes) -> WorldStateDict:
    """Decode world state container message.

    Args:
        data: Decoded container body bytes (must be >= 500 bytes).

    Returns:
        Decoded world state data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_min_length(data, 500, "WorldState")

    return WorldStateDict(
        msg_type="world_state",
        subtype=data[0],
        length=len(data),
        world_data=bytes(data[1:]),
    )


def decode_unknown_container(data: bytes) -> UnknownContainerDict:
    """Create unknown container result for unrecognized structures.

    Args:
        data: Decoded container body bytes.

    Returns:
        Unknown container data for debugging.

    Raises:
        ContainerDecodeError: If data is empty.
    """
    require_min_length(data, 1, "UnknownContainer")

    return UnknownContainerDict(
        msg_type="unknown_container",
        subtype=data[0],
        length=len(data),
        data=bytes(data),
    )


__all__ = [
    "decode_chunk_data",
    "decode_container_pickup",
    "decode_player_list_extended",
    "decode_player_list_short",
    "decode_teleport_landed",
    "decode_tip_notification",
    "decode_unknown_container",
    "decode_world_state",
    "is_chunk_data_structure",
    "is_container_pickup_structure",
    "is_player_list_extended_structure",
    "is_player_list_short_structure",
    "is_teleport_landed_structure",
    "is_tip_notification_structure",
    "is_world_state_structure",
]
