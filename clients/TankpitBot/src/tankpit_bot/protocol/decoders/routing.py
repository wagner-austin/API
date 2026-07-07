"""Message type routing for protocol decoders."""

from __future__ import annotations

from tankpit_bot.protocol.constants import (
    MSG_ACTION_DONE,
    MSG_ACTIVE_FORCES,
    MSG_ACTIVE_PLAYERS,
    MSG_BUILD_PICKUP,
    MSG_CACHE_UPDATE,
    MSG_CHAT,
    MSG_DEACTIVATE,
    MSG_DECORATION,
    MSG_DISCONNECT,
    MSG_ENEMY_DETECT,
    MSG_EQUIP_GAIN,
    MSG_EQUIP_TOGGLE,
    MSG_FUEL_DEPOSIT,
    MSG_FUEL_GAIN,
    MSG_INVENTORY,
    MSG_MAP_DATA,
    MSG_MOVE_RESPONSE,
    MSG_MOVEMENT,
    MSG_OVERLAY_UPDATE,
    MSG_PING,
    MSG_PROMOTION,
    MSG_RADAR_RESULT,
    MSG_RADAR_SCAN,
    MSG_SHOOT,
    MSG_STATISTICS,
    MSG_SUPERVISOR,
    MSG_SUPERVISOR_TEXT,
    MSG_SYNC,
    MSG_TANK_ENTRY,
    MSG_TANK_EXIT,
    MSG_TANK_INFO,
    MSG_TANK_REMOVE,
    MSG_TANK_STATS,
    MSG_TANK_STATUS_FULL,
    MSG_TERRAIN_UPDATE,
    MSG_TOP10,
    MSG_VIEWPORT,
)
from tankpit_bot.protocol.decoders.combat import (
    decode_deactivation,
    decode_shoot_event,
)
from tankpit_bot.protocol.decoders.map_data import decode_map_data
from tankpit_bot.protocol.decoders.movement import (
    decode_movement,
    decode_movement_response,
)
from tankpit_bot.protocol.decoders.radar import (
    decode_enemy_detection,
    decode_radar_result,
    decode_radar_scan_result,
)
from tankpit_bot.protocol.decoders.resources import (
    decode_equipment_gain,
    decode_equipment_toggle,
    decode_fuel_deposit,
    decode_fuel_gain,
    decode_inventory,
)
from tankpit_bot.protocol.decoders.session_events import (
    decode_action_done,
    decode_active_forces,
    decode_active_players,
    decode_build_pickup,
    decode_chat_message,
    decode_connection_lost,
    decode_decoration,
    decode_ping_response,
    decode_promotion,
    decode_statistics,
    decode_top10,
)
from tankpit_bot.protocol.decoders.tank import (
    decode_0x2e_message,
    decode_tank_entry,
    decode_tank_exit,
    decode_tank_info,
    decode_tank_remove,
    decode_tank_status,
)
from tankpit_bot.protocol.decoders.world import (
    decode_cache_update,
    decode_overlay_update,
    decode_supervisor,
    decode_supervisor_text,
    decode_sync,
    decode_terrain_update,
    decode_viewport_update,
)
from tankpit_bot.protocol.types import BinaryMessage


def _decode_combat_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode combat-related messages."""
    if msg_type == MSG_SHOOT:
        return decode_shoot_event(data)
    if msg_type == MSG_DEACTIVATE:
        return decode_deactivation(data)
    return None


def _decode_resource_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode resource-related messages (fuel, equipment)."""
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
    """Decode radar and detection messages."""
    if msg_type == MSG_RADAR_RESULT:
        return decode_radar_result(data)
    if msg_type == MSG_ENEMY_DETECT:
        return decode_enemy_detection(data)
    if msg_type == MSG_RADAR_SCAN:
        # 0x4F has a single wire personality (JS handler ch / V.O): a
        # batch of per-tile cache + overlay writes, used by the server
        # as the radar response. Corpus 2026-07-03 (199 sessions): all
        # 1817 bodies arrived tunneled inside 0x2E; this top-level
        # route keeps decode_message total over every message byte.
        return decode_radar_scan_result(data)
    return None


def _decode_tank_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode tank status and info messages."""
    if msg_type == MSG_TANK_ENTRY:
        return decode_tank_entry(data)
    if msg_type == MSG_TANK_EXIT:
        return decode_tank_exit(data)
    if msg_type == MSG_TANK_REMOVE:
        return decode_tank_remove(data)
    if msg_type == MSG_TANK_STATS:
        # 0x2E container envelope. `decode_0x2e_message` is the single
        # entrypoint for body decoding (subtype-first + length fallback).
        return decode_0x2e_message(data)
    if msg_type == MSG_TANK_STATUS_FULL:
        return decode_tank_status(data)
    if msg_type == MSG_TANK_INFO:
        return decode_tank_info(data)
    return None


def _decode_movement_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode movement-related messages."""
    if msg_type == MSG_MOVEMENT:
        return decode_movement(data)
    if msg_type == MSG_MOVE_RESPONSE:
        return decode_movement_response(data)
    return None


def _decode_world_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode world/environment messages."""
    if msg_type == MSG_VIEWPORT:
        return decode_viewport_update(data)
    if msg_type == MSG_TERRAIN_UPDATE:
        return decode_terrain_update(data)
    if msg_type == MSG_CACHE_UPDATE:
        return decode_cache_update(data)
    if msg_type == MSG_OVERLAY_UPDATE:
        return decode_overlay_update(data)
    if msg_type == MSG_SYNC:
        return decode_sync(data)
    if msg_type == MSG_MAP_DATA:
        return decode_map_data(data)
    return None


def _decode_misc_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode miscellaneous messages."""
    if msg_type == MSG_CHAT:
        return decode_chat_message(data)
    if msg_type == MSG_STATISTICS:
        return decode_statistics(data)
    if msg_type == MSG_ACTIVE_FORCES:
        return decode_active_forces(data)
    if msg_type == MSG_SUPERVISOR:
        return decode_supervisor(data)
    if msg_type == MSG_SUPERVISOR_TEXT:
        return decode_supervisor_text(data)
    if msg_type == MSG_ACTION_DONE:
        return decode_action_done(data)
    if msg_type == MSG_PROMOTION:
        return decode_promotion(data)
    if msg_type == MSG_DECORATION:
        return decode_decoration(data)
    if msg_type == MSG_BUILD_PICKUP:
        return decode_build_pickup(data)
    return _decode_session_broadcast(msg_type, data)


def _decode_session_broadcast(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode session-level server broadcasts.

    Split out of :func:`_decode_misc_message` to keep that function
    under the C901 complexity ceiling. Covers ActivePlayers, Top10,
    PingResponse, and ConnectionLost -- all decoded but historically
    routed nowhere; now they fan out into the events stream via the
    dispatcher.
    """
    if msg_type == MSG_ACTIVE_PLAYERS:
        return decode_active_players(data)
    if msg_type == MSG_TOP10:
        return decode_top10(data)
    if msg_type == MSG_PING:
        return decode_ping_response(data)
    if msg_type == MSG_DISCONNECT:
        return decode_connection_lost(data)
    return None
