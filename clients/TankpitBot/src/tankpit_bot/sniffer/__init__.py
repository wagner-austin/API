"""WebSocket traffic sniffer module.

This module provides browser-based WebSocket capture and decoding for
TankPit protocol analysis, organized into submodules:

- constants: Configuration values and protocol signatures
- xor: XOR encoding/decoding utilities
- viewport: Viewport position tracking
- player_tracking: Player ID and tank name mapping
- world_state: World state from radar/movement messages
- trackers: Tracker instances and initialization
- formatters: Message formatting for human-readable output
- decoders: Message decoding functions
- core: WebSocketSniffer class and entry points
"""

from __future__ import annotations

from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.sniffer.constants import (
    DECODED_SIGS,
    DEFAULT_CAPTURE_DURATION_MS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TARGET_URL,
)
from tankpit_bot.sniffer.core import (
    SnifferError,
    WebSocketSniffer,
    main,
    run_sniffer,
)
from tankpit_bot.sniffer.decoders import (
    decode_8byte_state,
    decode_and_log_binary,
    decode_command,
    decode_join_confirm,
    decode_message,
    decode_plus_message,
    decode_received_text_message,
    decode_state_message,
    decode_text_message,
    process_received_message,
    try_decode_binary,
    try_decode_received,
    try_decode_received_text,
)
from tankpit_bot.sniffer.formatters import (
    damage_name,
    format_combat_details,
    format_combat_hit,
    format_container_details,
    format_container_pickup,
    format_container_simple,
    format_decoded_message,
    format_message_details,
    format_misc_details,
    format_movement,
    format_position_details,
    format_position_update,
    format_radar_details,
    format_radar_response,
    format_resource_details,
    format_tank_details,
    format_tank_registry_details,
    format_tank_status_short,
    format_tank_update_details,
    handle_tank_registry,
    rank_name,
    team_name,
)
from tankpit_bot.sniffer.player_tracking import (
    get_tank_name,
    record_movement_response,
    register_tank_name,
    reset_player_id_mapper,
    resolve_movement_tank,
)
from tankpit_bot.sniffer.trackers import (
    ALL_TRACKERS,
    RECEIVED_TRACKERS,
    container_tracker,
    deactivation_tracker,
    deposit_tracker,
    equip_gain_tracker,
    equip_tracker,
    exit_tracker,
    extract_magic_from_auth,
    init_trackers_with_magic,
    item_tracker,
    mine_tracker,
    position_tracker,
    radar_ack_tracker,
    radar_tracker,
    reset_all_trackers,
    tank_tracker,
)
from tankpit_bot.sniffer.viewport import (
    get_viewport_left,
    reset_viewport_tracking,
    update_viewport_origin,
)
from tankpit_bot.sniffer.world_state import (
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.sniffer.world_state_inventory import (
    get_inventory_state,
    update_inventory_from_gain,
    update_inventory_from_protocol,
    update_inventory_from_toggle,
)
from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar
from tankpit_bot.sniffer.world_state_tiles import render_world_state_ascii
from tankpit_bot.sniffer.xor import (
    build_global_xor_table,
    get_global_xor_table,
    reset_xor_state,
    xor_decode,
)

__all__ = [
    "ALL_TRACKERS",
    "DECODED_SIGS",
    "DEFAULT_CAPTURE_DURATION_MS",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TARGET_URL",
    "RECEIVED_TRACKERS",
    "PlaywrightNotInstalledError",
    "SnifferError",
    "WebSocketSniffer",
    "build_global_xor_table",
    "container_tracker",
    "damage_name",
    "deactivation_tracker",
    "decode_8byte_state",
    "decode_and_log_binary",
    "decode_command",
    "decode_join_confirm",
    "decode_message",
    "decode_plus_message",
    "decode_received_text_message",
    "decode_state_message",
    "decode_text_message",
    "deposit_tracker",
    "dispatch_world_state_update",
    "equip_gain_tracker",
    "equip_tracker",
    "exit_tracker",
    "extract_magic_from_auth",
    "format_combat_details",
    "format_combat_hit",
    "format_container_details",
    "format_container_pickup",
    "format_container_simple",
    "format_decoded_message",
    "format_message_details",
    "format_misc_details",
    "format_movement",
    "format_position_details",
    "format_position_update",
    "format_radar_details",
    "format_radar_response",
    "format_resource_details",
    "format_tank_details",
    "format_tank_registry_details",
    "format_tank_status_short",
    "format_tank_update_details",
    "get_global_xor_table",
    "get_inventory_state",
    "get_tank_name",
    "get_viewport_left",
    "handle_tank_registry",
    "init_trackers_with_magic",
    "item_tracker",
    "main",
    "mine_tracker",
    "position_tracker",
    "process_received_message",
    "radar_ack_tracker",
    "radar_tracker",
    "rank_name",
    "record_movement_response",
    "register_tank_name",
    "render_world_state_ascii",
    "reset_all_trackers",
    "reset_player_id_mapper",
    "reset_viewport_tracking",
    "reset_world_state",
    "reset_xor_state",
    "resolve_movement_tank",
    "run_sniffer",
    "tank_tracker",
    "team_name",
    "try_decode_binary",
    "try_decode_received",
    "try_decode_received_text",
    "update_inventory_from_gain",
    "update_inventory_from_protocol",
    "update_inventory_from_toggle",
    "update_viewport_origin",
    "update_world_state_from_fuel_total",
    "update_world_state_from_position",
    "update_world_state_from_radar",
    "xor_decode",
]
