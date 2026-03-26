"""Bot module for TankPit automation.

This module provides the Bot class and supporting types for building
automated TankPit players with AI-driven autonomous control.

Submodules:
- ai: AI behavior system (evaluators, actions, pathfinding, threats)
- base: Bot class extending WebSocketSniffer with AI game loop
- commands: Command encoding utilities
- states: State machine enum and transition logic
- types: Bot-specific TypedDicts for commands
"""

from tankpit_bot.bot.ai import (
    ai_tick,
    make_default_ai_config,
    make_initial_ai_state,
    select_best_behavior,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.bot.base import Bot, BotError, ProtocolNotDiscoveredError, main
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.commands import (
    encode_move_command,
    encode_pickup_move_command,
    encode_radar_command,
    encode_shoot_command,
    encode_teleport_command,
)
from tankpit_bot.bot.executor import apply_equipment, dispatch_command, execute
from tankpit_bot.bot.states import (
    VALID_TRANSITIONS,
    BotState,
    BotStateDataDict,
    StateName,
    is_valid_transition,
    make_initial_state_data,
    set_fuel_threshold,
    set_target,
    transition_to,
    validate_transition,
)
from tankpit_bot.bot.tick_loop import run_tick_loop
from tankpit_bot.bot.tick_loop_types import (
    TickDecisionDict,
    decode_tick_decision,
    encode_tick_decision,
    make_tick_decision,
)
from tankpit_bot.bot.types import (
    BotCommand,
    MapOpenCommandDict,
    MoveCommandDict,
    PickupMoveCommandDict,
    RadarCommandDict,
    ShootCommandDict,
    TeleportCommandDict,
    make_map_open_command,
    make_move_command,
    make_pickup_move_command,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.bot.vision import (
    ContainerEntryDict,
    PositionEntryDict,
    TankRegistryEntryDict,
    VisionStateDict,
    add_fuel_delta,
    add_tank_to_registry,
    decode_container_entry,
    decode_position_entry,
    decode_tank_registry_entry,
    decode_vision_state,
    encode_container_entry,
    encode_position_entry,
    encode_tank_registry_entry,
    encode_vision_state,
    get_merged_fuel,
    get_merged_fuel_containers,
    make_container_entry,
    make_empty_vision_state,
    make_position_entry,
    make_tank_registry_entry,
    pickup_container_vision,
    remove_container,
    render_vision_ascii,
    render_vision_debug,
    set_self_tank_id,
    update_container,
    update_self_fuel_vision,
    update_tank_position,
)
from tankpit_bot.bot.world_sync import drain_js_messages, install_ws_hook

__all__ = [
    "VALID_TRANSITIONS",
    "Bot",
    "BotCommand",
    "BotError",
    "BotState",
    "BotStateDataDict",
    "CombatFeedback",
    "ContainerEntryDict",
    "MapOpenCommandDict",
    "MoveCommandDict",
    "PickupMoveCommandDict",
    "PositionEntryDict",
    "ProtocolNotDiscoveredError",
    "RadarCommandDict",
    "ShootCommandDict",
    "StateName",
    "TankRegistryEntryDict",
    "TeleportCommandDict",
    "TickDecisionDict",
    "VisionStateDict",
    "add_fuel_delta",
    "add_tank_to_registry",
    "ai_tick",
    "apply_equipment",
    "decide",
    "decode_container_entry",
    "decode_position_entry",
    "decode_tank_registry_entry",
    "decode_tick_decision",
    "decode_vision_state",
    "dispatch_command",
    "drain_js_messages",
    "encode_container_entry",
    "encode_move_command",
    "encode_pickup_move_command",
    "encode_position_entry",
    "encode_radar_command",
    "encode_shoot_command",
    "encode_tank_registry_entry",
    "encode_teleport_command",
    "encode_tick_decision",
    "encode_vision_state",
    "execute",
    "get_merged_fuel",
    "get_merged_fuel_containers",
    "install_ws_hook",
    "is_valid_transition",
    "main",
    "make_container_entry",
    "make_default_ai_config",
    "make_empty_vision_state",
    "make_initial_ai_state",
    "make_initial_state_data",
    "make_map_open_command",
    "make_move_command",
    "make_pickup_move_command",
    "make_position_entry",
    "make_radar_command",
    "make_shoot_command",
    "make_tank_registry_entry",
    "make_teleport_command",
    "make_tick_decision",
    "pickup_container_vision",
    "remove_container",
    "render_vision_ascii",
    "render_vision_debug",
    "run_tick_loop",
    "select_best_behavior",
    "set_fuel_threshold",
    "set_self_tank_id",
    "set_target",
    "transition_to",
    "update_container",
    "update_self_fuel_vision",
    "update_tank_position",
    "validate_transition",
]
