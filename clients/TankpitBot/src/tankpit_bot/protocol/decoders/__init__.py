"""Protocol message decoders.

This module provides the main decode_message dispatcher and re-exports
all decoder functions from submodules.
"""

from __future__ import annotations

from tankpit_bot.container.types import ContainerMessage
from tankpit_bot.protocol.constants import (
    MSG_ACTION_DONE,
    MSG_ACTIVE_FORCES,
    MSG_CACHE_OVERLAY_UPDATE,
    MSG_CACHE_UPDATE,
    MSG_CHAT,
    MSG_DEACTIVATE,
    MSG_ENEMY_DETECT,
    MSG_EQUIP_GAIN,
    MSG_EQUIP_TOGGLE,
    MSG_FUEL_DEPOSIT,
    MSG_FUEL_GAIN,
    MSG_INVENTORY,
    MSG_MINE_DETONATE,
    MSG_MINE_PLACE,
    MSG_MOVE_RESPONSE,
    MSG_MOVEMENT,
    MSG_OVERLAY_UPDATE,
    MSG_RADAR_RESULT,
    MSG_SHOOT,
    MSG_STATISTICS,
    MSG_SUPERVISOR,
    MSG_SYNC,
    MSG_TANK_ENTRY,
    MSG_TANK_EXIT,
    MSG_TANK_INFO,
    MSG_TANK_STATS,
    MSG_TANK_STATUS_FULL,
    MSG_TERRAIN_UPDATE,
    MSG_VIEWPORT,
)
from tankpit_bot.protocol.decoders.combat import (
    decode_deactivation,
    decode_hit_confirmation,
    decode_mine_detonation,
    decode_mine_placement,
    decode_shoot_event,
)
from tankpit_bot.protocol.decoders.misc import (
    decode_action_done,
    decode_active_forces,
    decode_chat_message,
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
from tankpit_bot.protocol.decoders.tank import (
    decode_0x2e_message,
    decode_tank_entry,
    decode_tank_exit,
    decode_tank_info,
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
    decode_sync,
    decode_terrain_update,
    decode_viewport_update,
    supervisor_has_promo_kill,
    supervisor_is_promo_eligible,
    viewport_entity_has_equipment_cache,
    viewport_entity_has_fuel_cache,
    viewport_entity_has_no_cache,
)
from tankpit_bot.protocol.helpers import DecodeError
from tankpit_bot.protocol.types import (
    ActionDoneDict,
    ActiveForcesDict,
    CacheUpdateDict,
    ChatMessageDict,
    CombinedTileUpdateDict,
    DeactivationDict,
    EnemyDetectionDict,
    EquipmentGainDict,
    EquipmentToggleDict,
    FuelDepositDict,
    FuelGainDict,
    InventoryDict,
    JoinConfirmDict,
    MineDetonationDict,
    MinePlacementDict,
    MovementDict,
    MovementResponseDict,
    OverlayUpdateDict,
    RadarResultDict,
    RadarScanResultDict,
    ShootEventDict,
    StatisticsDict,
    SupervisorDict,
    SyncDict,
    TankEntryDict,
    TankExitDict,
    TankInfoDict,
    TankStatusDict,
    TankStatusSyncDict,
    TerrainUpdateDict,
    ViewportUpdateDict,
    WorldInfoDict,
)

# Text message types (no XOR decoding, ASCII format)
TextMessage = JoinConfirmDict | WorldInfoDict

# Binary message types (XOR decoded)
BinaryMessage = (
    ShootEventDict
    | DeactivationDict
    | FuelGainDict
    | FuelDepositDict
    | RadarResultDict
    | EnemyDetectionDict
    | InventoryDict
    | EquipmentGainDict
    | EquipmentToggleDict
    | MinePlacementDict
    | MineDetonationDict
    | RadarScanResultDict
    | MovementDict
    | TankInfoDict
    | MovementResponseDict
    | SyncDict
    | CacheUpdateDict
    | OverlayUpdateDict
    | CombinedTileUpdateDict
    | TankEntryDict
    | TankExitDict
    | ActionDoneDict
    | ChatMessageDict
    | StatisticsDict
    | ActiveForcesDict
    | TankStatusSyncDict
    | TankStatusDict
    | SupervisorDict
    | TerrainUpdateDict
    | ViewportUpdateDict
    | ContainerMessage
)

# Union type for all decoded messages
DecodedMessage = TextMessage | BinaryMessage


def _decode_combat_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode combat-related messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a combat message.
    """
    if msg_type == MSG_SHOOT:
        return decode_shoot_event(data)
    if msg_type == MSG_DEACTIVATE:
        return decode_deactivation(data)
    if msg_type == MSG_MINE_PLACE:
        return decode_mine_placement(data)
    if msg_type == MSG_MINE_DETONATE:
        return decode_mine_detonation(data)
    return None


def _decode_resource_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode resource-related messages (fuel, equipment).

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a resource message.
    """
    if msg_type == MSG_FUEL_GAIN:
        return decode_fuel_gain(data)
    if msg_type == MSG_FUEL_DEPOSIT:
        return decode_fuel_deposit(data)
    if msg_type == MSG_INVENTORY:
        return decode_inventory(data)
    if msg_type == MSG_EQUIP_GAIN:
        return decode_equipment_gain(data)
    if msg_type == MSG_EQUIP_TOGGLE:
        return decode_equipment_toggle(data)
    return None


def _decode_radar_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode radar and detection messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a radar message.
    """
    if msg_type == MSG_RADAR_RESULT:
        return decode_radar_result(data)
    if msg_type == MSG_ENEMY_DETECT:
        return decode_enemy_detection(data)
    return None


# Protocol types that are known to be tunneled inside 0x2E envelopes.
# The first byte of the 0x2E body identifies the inner protocol message.
# When matched, the body is unwrapped and decoded as that protocol type.
# Unmatched subtypes fall through to container structure-based identification.
_TUNNELED_SUBTYPES: frozenset[int] = frozenset(
    {
        # Tank messages
        MSG_TANK_INFO,  # 0x21 — TankInfo (10+ bytes)
        MSG_TANK_ENTRY,  # 0x28 — TankEntry (10+ bytes)
        # Movement messages
        MSG_MOVE_RESPONSE,  # 0x3D — MovementResponse (11 bytes)
        MSG_TANK_STATUS_FULL,  # 0x3E — TankStatus (13+ bytes)
        MSG_SYNC,  # 0x3F — Sync heartbeat
        MSG_DEACTIVATE,  # 0x41 — Tank deactivation (7 bytes)
        MSG_FUEL_GAIN,  # 0x44 — FuelGain (3 bytes)
        MSG_RADAR_RESULT,  # 0x46 — RadarResult (2 bytes)
        MSG_MOVEMENT,  # 0x47 — Movement (9+ bytes)
        MSG_INVENTORY,  # 0x49 — Inventory (6 bytes)
        MSG_TERRAIN_UPDATE,  # 0x4A — Terrain/structure tile updates
        MSG_SHOOT,  # 0x53 — ShootEvent (12 bytes)
        MSG_ACTION_DONE,  # 0x54 — ActionDone
        MSG_VIEWPORT,  # 0x56 — ViewportUpdate (2+ bytes)
        MSG_FUEL_DEPOSIT,  # 0x64 — FuelDeposit (2 bytes)
        MSG_EQUIP_GAIN,  # 0x67 — EquipmentGain (6 bytes)
        MSG_EQUIP_TOGGLE,  # 0x74 — EquipmentToggle (5 bytes)
    }
)


def _try_unwrap_0x2e(data: bytes) -> BinaryMessage | None:
    """Try to decode a tunneled protocol message from inside a 0x2E envelope.

    Only attempts unwrapping for a known allowlist of protocol types that
    have strong structural validation. This prevents greedy decoders (like
    Sync which accepts any data) from swallowing container messages
    (tank_update_compact, position_update, movement, etc.).

    Args:
        data: XOR-decoded container body (without 0x2E prefix).

    Returns:
        Decoded protocol message if recognized, None to fall through
        to container structure-based identification.
    """
    if len(data) < 2:
        return None
    subtype = data[0]
    if subtype == MSG_TANK_STATS:
        # Nested 0x2E — decode as TankStatusSync from data[1:]
        if len(data) >= 9:
            return decode_tank_status_sync(data[1:])
        return None
    if subtype == MSG_CACHE_OVERLAY_UPDATE:
        # 0x2E -> 0x4F is not a normal combined tile update here.
        # Radar scan results are tunneled inside the 0x2E envelope using the
        # inner 0x4F subtype, which collides with the standalone combined tile
        # patch opcode. Decode the tunneled form as radar first.
        try:
            return decode_radar_scan_result(data[1:])
        except DecodeError:
            return None
    if subtype not in _TUNNELED_SUBTYPES:
        return None
    try:
        return try_decode_binary_message(subtype, data[1:])
    except DecodeError:
        return None


def _decode_tank_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode tank status and info messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a tank message.
    """
    if msg_type == MSG_TANK_ENTRY:
        return decode_tank_entry(data)
    if msg_type == MSG_TANK_EXIT:
        return decode_tank_exit(data)
    if msg_type == MSG_TANK_STATS:
        # 0x2E is a container envelope. Try unwrapping tunneled protocol
        # messages before falling through to container structure matching.
        tunneled = _try_unwrap_0x2e(data)
        if tunneled is not None:
            return tunneled
        return decode_0x2e_message(data)
    if msg_type == MSG_TANK_STATUS_FULL:
        return decode_tank_status(data)
    if msg_type == MSG_TANK_INFO:
        return decode_tank_info(data)
    return None


def _decode_movement_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode movement-related messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a movement message.
    """
    if msg_type == MSG_MOVEMENT:
        return decode_movement(data)
    if msg_type == MSG_MOVE_RESPONSE:
        return decode_movement_response(data)
    return None


def _decode_world_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode world/environment messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a world message.
    """
    if msg_type == MSG_VIEWPORT:
        return decode_viewport_update(data)
    if msg_type == MSG_TERRAIN_UPDATE:
        return decode_terrain_update(data)
    if msg_type == MSG_CACHE_UPDATE:
        return decode_cache_update(data)
    if msg_type == MSG_OVERLAY_UPDATE:
        return decode_overlay_update(data)
    if msg_type == MSG_CACHE_OVERLAY_UPDATE:
        return decode_combined_tile_update(data)
    if msg_type == MSG_SYNC:
        return decode_sync(data)
    return None


def _decode_misc_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode miscellaneous messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a misc message.
    """
    if msg_type == MSG_CHAT:
        return decode_chat_message(data)
    if msg_type == MSG_STATISTICS:
        return decode_statistics(data)
    if msg_type == MSG_ACTIVE_FORCES:
        return decode_active_forces(data)
    if msg_type == MSG_SUPERVISOR:
        return decode_supervisor(data)
    if msg_type == MSG_ACTION_DONE:
        return decode_action_done(data)
    return None


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
    "decode_cache_update",
    "decode_chat_message",
    "decode_combined_tile_update",
    "decode_deactivation",
    "decode_enemy_detection",
    "decode_equipment_gain",
    "decode_equipment_toggle",
    "decode_fuel_deposit",
    "decode_fuel_gain",
    "decode_hit_confirmation",
    "decode_inventory",
    "decode_join_confirm",
    "decode_message",
    "decode_mine_detonation",
    "decode_mine_placement",
    "decode_movement",
    "decode_movement_response",
    "decode_overlay_update",
    "decode_radar_container",
    "decode_radar_mine",
    "decode_radar_result",
    "decode_radar_scan_result",
    "decode_shoot_event",
    "decode_statistics",
    "decode_supervisor",
    "decode_sync",
    "decode_tank_entry",
    "decode_tank_exit",
    "decode_tank_info",
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
    "supervisor_has_promo_kill",
    "supervisor_is_promo_eligible",
    "try_decode_binary_message",
    "try_decode_message",
    "viewport_entity_has_equipment_cache",
    "viewport_entity_has_fuel_cache",
    "viewport_entity_has_no_cache",
]
