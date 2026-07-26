"""Server-message encoders (Phase 4 step a — wiki [[physics-module-roadmap]]).

Byte-exact inverses of ``tankpit_bot.protocol.decoders``, organized
module-for-module (a separate package rather than same-file placement
because the tank decoder module already sits at the 400-line ceiling).
The simulator's fake server is the primary consumer; the round-trip
suite grades every encoder against the full capture archive.
"""

from __future__ import annotations

from tankpit_bot.protocol.encoders.combat import (
    encode_deactivation,
    encode_shoot_event,
)
from tankpit_bot.protocol.encoders.envelope import (
    encode_envelope_body,
    encode_message_payload,
    encode_plaintext_ack,
)
from tankpit_bot.protocol.encoders.map_data import encode_map_data
from tankpit_bot.protocol.encoders.movement import (
    encode_movement,
    encode_movement_response,
)
from tankpit_bot.protocol.encoders.radar import (
    encode_enemy_detection,
    encode_radar_container,
    encode_radar_mine,
    encode_radar_mine_clear,
    encode_radar_result,
    encode_radar_scan_result,
)
from tankpit_bot.protocol.encoders.resources import (
    encode_equipment_gain,
    encode_equipment_toggle,
    encode_fuel_deposit,
    encode_fuel_gain,
    encode_inventory,
)
from tankpit_bot.protocol.encoders.session_events import (
    encode_action_done,
    encode_active_forces,
    encode_active_players,
    encode_build_pickup,
    encode_chat_message,
    encode_connection_lost,
    encode_decoration,
    encode_ping_response,
    encode_promotion,
    encode_statistics,
    encode_top10,
)
from tankpit_bot.protocol.encoders.tank import (
    encode_tank_entry,
    encode_tank_exit,
    encode_tank_info,
    encode_tank_remove,
    encode_tank_status,
    encode_tank_status_sync,
)
from tankpit_bot.protocol.encoders.world import (
    encode_cache_update,
    encode_overlay_update,
    encode_supervisor,
    encode_supervisor_text,
    encode_sync,
    encode_terrain_update,
    encode_viewport_update,
)

__all__ = [
    "encode_action_done",
    "encode_active_forces",
    "encode_active_players",
    "encode_build_pickup",
    "encode_cache_update",
    "encode_chat_message",
    "encode_connection_lost",
    "encode_deactivation",
    "encode_decoration",
    "encode_enemy_detection",
    "encode_envelope_body",
    "encode_equipment_gain",
    "encode_equipment_toggle",
    "encode_fuel_deposit",
    "encode_fuel_gain",
    "encode_inventory",
    "encode_map_data",
    "encode_message_payload",
    "encode_movement",
    "encode_movement_response",
    "encode_overlay_update",
    "encode_ping_response",
    "encode_plaintext_ack",
    "encode_promotion",
    "encode_radar_container",
    "encode_radar_mine",
    "encode_radar_mine_clear",
    "encode_radar_result",
    "encode_radar_scan_result",
    "encode_shoot_event",
    "encode_statistics",
    "encode_supervisor",
    "encode_supervisor_text",
    "encode_sync",
    "encode_tank_entry",
    "encode_tank_exit",
    "encode_tank_info",
    "encode_tank_remove",
    "encode_tank_status",
    "encode_tank_status_sync",
    "encode_terrain_update",
    "encode_top10",
    "encode_viewport_update",
]
