"""Container message decoders.

This module provides the main decode_container_message dispatcher and
re-exports all decoder functions from submodules.
"""

from __future__ import annotations

from tankpit_bot.container.decoders.combat import (
    decode_combat_hit,
    decode_deactivation_death,
    decode_deactivation_kill,
    decode_mine_placement,
    is_combat_hit_structure,
    is_deactivation_death_structure,
    is_deactivation_kill_structure,
    is_mine_placement_structure,
)
from tankpit_bot.container.decoders.misc import (
    decode_chunk_data,
    decode_container_pickup,
    decode_player_list_extended,
    decode_player_list_short,
    decode_teleport_landed,
    decode_tip_notification,
    decode_unknown_container,
    decode_world_state,
    is_chunk_data_structure,
    is_container_pickup_structure,
    is_player_list_extended_structure,
    is_player_list_short_structure,
    is_teleport_landed_structure,
    is_tip_notification_structure,
    is_world_state_structure,
)
from tankpit_bot.container.decoders.position import (
    decode_movement,
    decode_position_update,
    is_movement_structure,
    is_position_update_structure,
)
from tankpit_bot.container.decoders.radar import (
    decode_radar_response,
    is_radar_response_structure,
)
from tankpit_bot.container.decoders.tank import (
    DIRECTION_BYTES,
    DIRECTION_EAST,
    DIRECTION_NORTH,
    DIRECTION_SOUTH,
    DIRECTION_WEST,
    SUBTYPE_MOVEMENT,
    SUBTYPE_TANK_REGISTRY,
    decode_tank_leave,
    decode_tank_registry,
    decode_tank_status_short,
    decode_tank_status_sync,
    decode_tank_update_compact,
    decode_tank_update_extended,
    decode_tank_update_full,
    is_tank_leave_structure,
    is_tank_registry_structure,
    is_tank_status_short_structure,
    is_tank_status_sync_structure,
    is_tank_update_compact_structure,
    is_tank_update_extended_structure,
    is_tank_update_full_structure,
)
from tankpit_bot.container.identification import identify_container_type
from tankpit_bot.container.types import (
    ContainerMessage,
    ContainerMessageType,
)


def _decode_tank_update(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode tank update message types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a tank update type.
    """
    if msg_type == ContainerMessageType.TANK_UPDATE_COMPACT:
        return decode_tank_update_compact(data)
    if msg_type == ContainerMessageType.TANK_UPDATE_EXTENDED:
        return decode_tank_update_extended(data)
    if msg_type == ContainerMessageType.TANK_UPDATE_FULL:
        return decode_tank_update_full(data)
    return None


def _decode_player_list(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode player list message types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a player list type.
    """
    if msg_type == ContainerMessageType.PLAYER_LIST_SHORT:
        return decode_player_list_short(data)
    if msg_type == ContainerMessageType.PLAYER_LIST_EXTENDED:
        return decode_player_list_extended(data)
    return None


def _decode_deactivation(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode deactivation message types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a deactivation type.
    """
    if msg_type == ContainerMessageType.DEACTIVATION_KILL:
        return decode_deactivation_kill(data)
    if msg_type == ContainerMessageType.DEACTIVATION_DEATH:
        return decode_deactivation_death(data)
    return None


def _decode_single_type(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode single-length container types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a single-length type.
    """
    if msg_type == ContainerMessageType.COMBAT_HIT:
        return decode_combat_hit(data)
    if msg_type == ContainerMessageType.MINE_PLACEMENT:
        return decode_mine_placement(data)
    if msg_type == ContainerMessageType.TANK_REGISTRY:
        return decode_tank_registry(data)
    if msg_type == ContainerMessageType.MOVEMENT:
        return decode_movement(data)
    if msg_type == ContainerMessageType.POSITION_UPDATE:
        return decode_position_update(data)
    if msg_type == ContainerMessageType.TANK_STATUS_SHORT:
        return decode_tank_status_short(data)
    if msg_type == ContainerMessageType.TANK_STATUS_SYNC:
        return decode_tank_status_sync(data)
    if msg_type == ContainerMessageType.TANK_LEAVE:
        return decode_tank_leave(data)
    return None


def _decode_range_type(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode range-based and miscellaneous container types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a range/misc type.
    """
    if msg_type == ContainerMessageType.TELEPORT_LANDED:
        return decode_teleport_landed(data)
    if msg_type == ContainerMessageType.CONTAINER_PICKUP:
        return decode_container_pickup(data)
    if msg_type == ContainerMessageType.RADAR_RESPONSE:
        return decode_radar_response(data)
    if msg_type == ContainerMessageType.TIP_NOTIFICATION:
        return decode_tip_notification(data)
    if msg_type == ContainerMessageType.CHUNK_DATA:
        return decode_chunk_data(data)
    if msg_type == ContainerMessageType.WORLD_STATE:
        return decode_world_state(data)
    return None


def decode_container_message(data: bytes) -> ContainerMessage:
    """Decode a 0x2E container message by structure.

    Uses structure-based pattern matching rather than subtype bytes,
    making it robust across sessions with different XOR keys.

    Args:
        data: Decoded container body bytes (after XOR decode, without 0x2E prefix).

    Returns:
        Decoded message as appropriate TypedDict.

    Raises:
        ContainerDecodeError: If known structure fails validation.
    """
    msg_type = identify_container_type(data)

    # Single-length types (1, 2-3, 5, 6, 9, 11, 13, 16-20 bytes)
    single_msg = _decode_single_type(msg_type, data)
    if single_msg is not None:
        return single_msg
    # Tank update types (10, 14, 15 bytes)
    tank_update = _decode_tank_update(msg_type, data)
    if tank_update is not None:
        return tank_update
    # Deactivation types (5, 7 bytes)
    deactivation = _decode_deactivation(msg_type, data)
    if deactivation is not None:
        return deactivation
    # Player list types (4, 7 bytes)
    player_list = _decode_player_list(msg_type, data)
    if player_list is not None:
        return player_list
    # Range-based and misc types (21+, 29+, 80+, 500+ bytes)
    range_msg = _decode_range_type(msg_type, data)
    if range_msg is not None:
        return range_msg

    return decode_unknown_container(data)


__all__ = [
    "DIRECTION_BYTES",
    "DIRECTION_EAST",
    "DIRECTION_NORTH",
    "DIRECTION_SOUTH",
    "DIRECTION_WEST",
    "SUBTYPE_MOVEMENT",
    "SUBTYPE_TANK_REGISTRY",
    "decode_chunk_data",
    "decode_combat_hit",
    "decode_container_message",
    "decode_container_pickup",
    "decode_deactivation_death",
    "decode_deactivation_kill",
    "decode_mine_placement",
    "decode_movement",
    "decode_player_list_extended",
    "decode_player_list_short",
    "decode_position_update",
    "decode_radar_response",
    "decode_tank_leave",
    "decode_tank_registry",
    "decode_tank_status_short",
    "decode_tank_status_sync",
    "decode_tank_update_compact",
    "decode_tank_update_extended",
    "decode_tank_update_full",
    "decode_teleport_landed",
    "decode_tip_notification",
    "decode_unknown_container",
    "decode_world_state",
    "identify_container_type",
    "is_chunk_data_structure",
    "is_combat_hit_structure",
    "is_container_pickup_structure",
    "is_deactivation_death_structure",
    "is_deactivation_kill_structure",
    "is_mine_placement_structure",
    "is_movement_structure",
    "is_player_list_extended_structure",
    "is_player_list_short_structure",
    "is_position_update_structure",
    "is_radar_response_structure",
    "is_tank_leave_structure",
    "is_tank_registry_structure",
    "is_tank_status_short_structure",
    "is_tank_status_sync_structure",
    "is_tank_update_compact_structure",
    "is_tank_update_extended_structure",
    "is_tank_update_full_structure",
    "is_teleport_landed_structure",
    "is_tip_notification_structure",
    "is_world_state_structure",
]
