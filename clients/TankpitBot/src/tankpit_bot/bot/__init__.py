"""Bot module for TankPit automation.

This module provides the Bot class and supporting types for building
automated TankPit players with state machine control.

Submodules:
- base: Bot class extending WebSocketSniffer with state machine
- commands: Command encoding utilities
- states: State machine enum and transition logic
- types: Bot-specific TypedDicts for commands
"""

import sys

from tankpit_bot import _test_hooks
from tankpit_bot.bot.base import Bot, BotError, ProtocolNotDiscoveredError
from tankpit_bot.bot.commands import (
    encode_move_command,
    encode_pickup_move_command,
    encode_radar_command,
    encode_shoot_command,
    encode_teleport_command,
)
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
from tankpit_bot.bot.types import (
    BotCommand,
    MoveCommandDict,
    PickupMoveCommandDict,
    RadarCommandDict,
    ShootCommandDict,
    TeleportCommandDict,
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


def main() -> None:
    """Entry point for tankpit-bot command.

    Currently displays usage instructions since the protocol must be
    captured first using the sniffer.
    """
    capture_path = _test_hooks.get_env("TANKPIT_CAPTURE")
    if capture_path is None:
        capture_path = "capture_session.json"

    output_lines = [
        "TankpitBot - Automated Tankpit.com player",
        "",
        "Before running the bot, you must first capture the game protocol:",
        "",
        "  1. Run the sniffer to capture WebSocket traffic:",
        "     make sniff",
        "",
        "  2. Log into tankpit.com in the browser window that opens",
        "",
        "  3. Join a game and play for a bit",
        "",
        "  4. The sniffer will save the captured protocol to:",
        f"     {capture_path}",
        "",
        "  5. Analyze the captured messages to understand the protocol",
        "",
        "Once the protocol is understood, this bot will be updated to:",
        "  - Connect to the WebSocket server",
        "  - Authenticate and join games",
        "  - Control tank movement and shooting",
        "  - Implement AI strategy",
        "",
    ]

    for line in output_lines:
        sys.stdout.write(line + "\n")


__all__ = [
    "VALID_TRANSITIONS",
    "Bot",
    "BotCommand",
    "BotError",
    "BotState",
    "BotStateDataDict",
    "ContainerEntryDict",
    "MoveCommandDict",
    "PickupMoveCommandDict",
    "PositionEntryDict",
    "ProtocolNotDiscoveredError",
    "RadarCommandDict",
    "ShootCommandDict",
    "StateName",
    "TankRegistryEntryDict",
    "TeleportCommandDict",
    "VisionStateDict",
    "add_fuel_delta",
    "add_tank_to_registry",
    "decode_container_entry",
    "decode_position_entry",
    "decode_tank_registry_entry",
    "decode_vision_state",
    "encode_container_entry",
    "encode_move_command",
    "encode_pickup_move_command",
    "encode_position_entry",
    "encode_radar_command",
    "encode_shoot_command",
    "encode_tank_registry_entry",
    "encode_teleport_command",
    "encode_vision_state",
    "get_merged_fuel",
    "get_merged_fuel_containers",
    "is_valid_transition",
    "main",
    "make_container_entry",
    "make_empty_vision_state",
    "make_initial_state_data",
    "make_move_command",
    "make_pickup_move_command",
    "make_position_entry",
    "make_radar_command",
    "make_shoot_command",
    "make_tank_registry_entry",
    "make_teleport_command",
    "pickup_container_vision",
    "remove_container",
    "render_vision_ascii",
    "render_vision_debug",
    "set_fuel_threshold",
    "set_self_tank_id",
    "set_target",
    "transition_to",
    "update_container",
    "update_self_fuel_vision",
    "update_tank_position",
    "validate_transition",
]
