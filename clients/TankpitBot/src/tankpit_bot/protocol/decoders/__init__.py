"""Protocol message decoders.

This module provides the main decode_message dispatcher and re-exports
all decoder functions from submodules.
"""

from __future__ import annotations

from tankpit_bot.protocol.decoders.combat import (
    decode_deactivation,
    decode_shoot_event,
)
from tankpit_bot.protocol.decoders.map_data import decode_map_data
from tankpit_bot.protocol.decoders.misc import (
    decode_action_done,
    decode_active_forces,
    decode_build_pickup,
    decode_chat_message,
    decode_decoration,
    decode_promotion,
    decode_statistics,
)
from tankpit_bot.protocol.decoders.movement import (
    decode_movement,
    decode_movement_response,
)
from tankpit_bot.protocol.decoders.radar import (
    decode_enemy_detection,
    decode_radar_container,
    decode_radar_mine,
    decode_radar_result,
    decode_radar_scan_result,
    encode_radar_container,
    encode_radar_mine,
    encode_radar_scan_result,
    require_radar_container,
    require_radar_mine,
    require_radar_scan_result,
)
from tankpit_bot.protocol.decoders.resources import (
    decode_equipment_gain,
    decode_equipment_toggle,
    decode_fuel_deposit,
    decode_fuel_gain,
    decode_inventory,
)
from tankpit_bot.protocol.decoders.routing import (
    _decode_combat_message,
    _decode_misc_message,
    _decode_movement_message,
    _decode_radar_message,
    _decode_resource_message,
    _decode_tank_message,
    _decode_world_message,
)
from tankpit_bot.protocol.decoders.tank import (
    decode_0x2e_message,
    decode_tank_entry,
    decode_tank_exit,
    decode_tank_info,
    decode_tank_remove,
    decode_tank_status,
    decode_tank_status_sync,
)
from tankpit_bot.protocol.decoders.text import (
    decode_join_confirm,
    decode_text_message,
    decode_world_info,
)
from tankpit_bot.protocol.decoders.world import (
    decode_cache_update,
    decode_combined_tile_update,
    decode_overlay_update,
    decode_supervisor,
    decode_supervisor_text,
    decode_sync,
    decode_terrain_update,
    decode_viewport_update,
    supervisor_error_code,
    supervisor_is_cant_go,
    supervisor_is_insufficient_fuel,
    viewport_entity_has_equipment_cache,
    viewport_entity_has_fuel_cache,
    viewport_entity_has_no_cache,
)
from tankpit_bot.protocol.helpers import DecodeError
from tankpit_bot.protocol.types import (
    BinaryMessage,
    DecodedMessage,
    TextMessage,
)


def decode_message(msg_type: int, data: bytes) -> BinaryMessage:
    """Decode a BINARY message based on its type.

    NOTE: For text messages, use decode_text_message() instead.

    Args:
        msg_type: First byte of message (NOT XOR encoded).
        data: Remaining message bytes (XOR decoded).

    Returns:
        Decoded message object.

    Raises:
        DecodeError: If message type is unknown or decoding fails.
    """
    result = _decode_combat_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_resource_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_radar_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_tank_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_movement_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_world_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_misc_message(msg_type, data)
    if result is not None:
        return result

    raise DecodeError(f"decode_message: unknown type 0x{msg_type:02X}")


def try_decode_message(msg_type: int, data: bytes) -> DecodedMessage | None:
    """Try to decode a message, returning None if unsupported.

    Unlike decode_message(), this does not raise DecodeError for unknown types.
    Use this when you want to handle unknown types gracefully without exceptions.

    Args:
        msg_type: First byte of message (NOT XOR encoded).
        data: Remaining message bytes (XOR decoded).

    Returns:
        Decoded message object, or None if type is unknown/unsupported.
    """
    return try_decode_binary_message(msg_type, data)


def try_decode_binary_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Try to decode a binary message, returning None if unsupported.

    Args:
        msg_type: First byte of message (NOT XOR encoded).
        data: Remaining message bytes (XOR decoded).

    Returns:
        Decoded message object, or None if type is unknown/unsupported.
    """
    result = _decode_combat_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_resource_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_radar_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_tank_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_movement_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_world_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_misc_message(msg_type, data)
    if result is not None:
        return result

    return None


__all__ = [
    "BinaryMessage",
    "DecodedMessage",
    "TextMessage",
    "decode_0x2e_message",
    "decode_action_done",
    "decode_active_forces",
    "decode_build_pickup",
    "decode_cache_update",
    "decode_chat_message",
    "decode_combined_tile_update",
    "decode_deactivation",
    "decode_decoration",
    "decode_enemy_detection",
    "decode_equipment_gain",
    "decode_equipment_toggle",
    "decode_fuel_deposit",
    "decode_fuel_gain",
    "decode_inventory",
    "decode_join_confirm",
    "decode_map_data",
    "decode_message",
    "decode_movement",
    "decode_movement_response",
    "decode_overlay_update",
    "decode_promotion",
    "decode_radar_container",
    "decode_radar_mine",
    "decode_radar_result",
    "decode_radar_scan_result",
    "decode_shoot_event",
    "decode_statistics",
    "decode_supervisor",
    "decode_supervisor_text",
    "decode_sync",
    "decode_tank_entry",
    "decode_tank_exit",
    "decode_tank_info",
    "decode_tank_remove",
    "decode_tank_status",
    "decode_tank_status_sync",
    "decode_terrain_update",
    "decode_text_message",
    "decode_viewport_update",
    "decode_world_info",
    "encode_radar_container",
    "encode_radar_mine",
    "encode_radar_scan_result",
    "require_radar_container",
    "require_radar_mine",
    "require_radar_scan_result",
    "supervisor_error_code",
    "supervisor_is_cant_go",
    "supervisor_is_insufficient_fuel",
    "try_decode_binary_message",
    "try_decode_message",
    "viewport_entity_has_equipment_cache",
    "viewport_entity_has_fuel_cache",
    "viewport_entity_has_no_cache",
]
