"""Container message type identification.

This module provides functions to identify container message types
by their structure (length, byte patterns) rather than XOR-dependent
subtype values.
"""

from __future__ import annotations

from tankpit_bot.container.decoders.combat import (
    is_deactivation_death_structure,
    is_mine_detonation_structure,
    is_mine_placement_structure,
)
from tankpit_bot.container.decoders.misc import (
    is_chunk_data_structure,
    is_container_pickup_structure,
    is_player_list_extended_structure,
    is_player_list_short_structure,
    is_teleport_landed_structure,
    is_tip_notification_structure,
    is_world_state_structure,
)
from tankpit_bot.container.decoders.position import is_position_update_structure
from tankpit_bot.container.decoders.tank import (
    is_tank_leave_structure,
    is_tank_registry_structure,
    is_tank_status_short_structure,
    is_tank_update_compact_structure,
    is_tank_update_full_structure,
)
from tankpit_bot.container.types import ContainerMessageType


def _identify_tank_update_type(data: bytes) -> ContainerMessageType:
    """Identify tank update message types by structure.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified tank update type, or UNKNOWN if not a tank update.
    """
    # Tank update compact: exactly 10 bytes
    if is_tank_update_compact_structure(data):
        return ContainerMessageType.TANK_UPDATE_COMPACT
    # Tunneled mine placement: exactly 15 bytes with subtype 0x4B
    if is_mine_placement_structure(data):
        return ContainerMessageType.MINE_PLACEMENT
    # Tank update full: exactly 15 bytes
    if is_tank_update_full_structure(data):
        return ContainerMessageType.TANK_UPDATE_FULL
    return ContainerMessageType.UNKNOWN


def _identify_player_list_type(data: bytes) -> ContainerMessageType:
    """Identify player list message types by structure.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified player list type, or UNKNOWN if not a player list.
    """
    # Player list short: 4 bytes
    if is_player_list_short_structure(data):
        return ContainerMessageType.PLAYER_LIST_SHORT
    # Player list extended: 7 bytes
    # Note: 7 bytes conflicts with deactivation_death, so check deactivation first
    if is_player_list_extended_structure(data):
        return ContainerMessageType.PLAYER_LIST_EXTENDED
    return ContainerMessageType.UNKNOWN


def _identify_deactivation_type(data: bytes) -> ContainerMessageType:
    """Identify deactivation_death messages by structure.

    0x41 DeactivationKill was deleted from the container path
    2026-06-19; the protocol path is the single source of truth.

    Args:
        data: Decoded container body bytes.

    Returns:
        DEACTIVATION_DEATH if matched, otherwise UNKNOWN.
    """
    if is_deactivation_death_structure(data):
        return ContainerMessageType.DEACTIVATION_DEATH
    return ContainerMessageType.UNKNOWN


def _identify_single_length_type(data: bytes) -> ContainerMessageType:
    """Identify message types that have a single exact length.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified type, or UNKNOWN if not matched.
    """
    # Teleport landed: exactly 1 byte
    if is_teleport_landed_structure(data):
        return ContainerMessageType.TELEPORT_LANDED
    # 2-3 byte short bodies are now UNKNOWN_CONTAINER (was a catch-all
    # misidentifying them as TankStatusSync; the real TankStatusSync is
    # 8+ bytes and lives on the protocol path).
    # Tank leave: 6 bytes
    if is_tank_leave_structure(data):
        return ContainerMessageType.TANK_LEAVE
    # Tank status short: exactly 9 bytes
    if is_tank_status_short_structure(data):
        return ContainerMessageType.TANK_STATUS_SHORT
    # 13-byte position update: subtype 0x24. The 13-byte 0x2E nested
    # self-status form moved to the protocol path 2026-06-19 (handled
    # as TankStatusSync with fuel in decode_tank_status_sync).
    if is_position_update_structure(data):
        return ContainerMessageType.POSITION_UPDATE
    return ContainerMessageType.UNKNOWN


def _identify_subtype_specific(data: bytes) -> ContainerMessageType:
    """Identify message types by first-byte subtype.

    Checks for types that require a specific subtype byte:
    - Container pickup: 0x43 subtype, 5 bytes
    - Radar response: 0x4F subtype, 7+ bytes with count-based length

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified type, or UNKNOWN if not matched.
    """
    # Mine detonation: 0x45 subtype with repeated coordinate pairs
    if is_mine_detonation_structure(data):
        return ContainerMessageType.MINE_DETONATION
    # Container pickup: exactly 5 bytes with 0x43 subtype
    if is_container_pickup_structure(data):
        return ContainerMessageType.CONTAINER_PICKUP
    # Radar response: 7+ bytes with 0x4F subtype
    return ContainerMessageType.UNKNOWN


def _identify_range_type(data: bytes) -> ContainerMessageType:
    """Identify message types by length ranges.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified range type, or UNKNOWN if not matched.
    """
    # Tip notification: 29-79 bytes
    if is_tip_notification_structure(data):
        return ContainerMessageType.TIP_NOTIFICATION
    # Chunk data: 80-130 bytes
    if is_chunk_data_structure(data):
        return ContainerMessageType.CHUNK_DATA
    # World state: 500+ bytes
    if is_world_state_structure(data):
        return ContainerMessageType.WORLD_STATE
    return ContainerMessageType.UNKNOWN


def identify_container_type(data: bytes) -> ContainerMessageType:
    """Identify the type of container message by structure.

    Order of checks matters - more specific patterns first.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified message type.
    """
    if len(data) < 1:
        return ContainerMessageType.UNKNOWN

    # Subtype-specific types first (container_pickup=0x43, radar_response=0x4F)
    subtype_type = _identify_subtype_specific(data)
    if subtype_type != ContainerMessageType.UNKNOWN:
        return subtype_type
    # Single-length types (11, 13, 9, 2-3, 6 bytes)
    single_type = _identify_single_length_type(data)
    if single_type != ContainerMessageType.UNKNOWN:
        return single_type
    # 0x47 Movement handled by protocol tunnel path; falls through if
    # the protocol min_len gate doesn't pass (rare). Tank registry next.
    if is_tank_registry_structure(data):
        return ContainerMessageType.TANK_REGISTRY
    # Tank update types (10, 14, 15 bytes)
    tank_update = _identify_tank_update_type(data)
    if tank_update != ContainerMessageType.UNKNOWN:
        return tank_update
    # Deactivation types (5, 7 bytes) - check before player list due to 7-byte conflict
    deactivation = _identify_deactivation_type(data)
    if deactivation != ContainerMessageType.UNKNOWN:
        return deactivation
    # Player list types (4 bytes only - 7 bytes handled by deactivation_death)
    player_list = _identify_player_list_type(data)
    if player_list != ContainerMessageType.UNKNOWN:
        return player_list
    # Range-based types (21+, 29+, 80+, 500+ bytes)
    range_type = _identify_range_type(data)
    if range_type != ContainerMessageType.UNKNOWN:
        return range_type

    return ContainerMessageType.UNKNOWN


__all__ = [
    "identify_container_type",
]
