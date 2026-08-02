"""Bot-specific TypedDicts and types.

This module provides TypedDicts for bot command encoding and state management.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

# =============================================================================
# Command Types
# =============================================================================


class MoveCommandDict(TypedDict):
    """Move command parameters.

    Attributes:
        cmd_type: Command type identifier.
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).
    """

    cmd_type: Literal["move"]
    target_x: int
    target_y: int


class ShootCommandDict(TypedDict):
    """Shoot command parameters.

    Attributes:
        cmd_type: Command type identifier.
        target_x: Target X coordinate.
        target_y: Target Y coordinate.
        target_id: Tank ID of the target (0 if no specific target).
    """

    cmd_type: Literal["shoot"]
    target_x: int
    target_y: int
    target_id: int


class RadarCommandDict(TypedDict):
    """Radar command parameters.

    Attributes:
        cmd_type: Command type identifier.
    """

    cmd_type: Literal["radar"]


class PickupFuelCommandDict(TypedDict):
    """Fuel pickup command parameters.

    Attributes:
        cmd_type: Command type identifier.
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).
    """

    cmd_type: Literal["pickup_fuel"]
    target_x: int
    target_y: int


class PickupEquipmentCommandDict(TypedDict):
    """Equipment pickup command parameters.

    Attributes:
        cmd_type: Command type identifier.
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).
    """

    cmd_type: Literal["pickup_equipment"]
    target_x: int
    target_y: int


# =============================================================================
# Factory Functions
# =============================================================================


def make_move_command(target_x: int, target_y: int) -> MoveCommandDict:
    """Create a move command.

    Args:
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).

    Returns:
        MoveCommandDict with the specified target.
    """
    return MoveCommandDict(cmd_type="move", target_x=target_x, target_y=target_y)


def make_shoot_command(
    target_x: int,
    target_y: int,
    target_id: int = 0,
) -> ShootCommandDict:
    """Create a shoot command.

    Args:
        target_x: Target X coordinate.
        target_y: Target Y coordinate.
        target_id: Tank ID of the target (0 if no specific target).

    Returns:
        ShootCommandDict with the specified target.
    """
    return ShootCommandDict(
        cmd_type="shoot",
        target_x=target_x,
        target_y=target_y,
        target_id=target_id,
    )


def make_radar_command() -> RadarCommandDict:
    """Create a radar command.

    Returns:
        RadarCommandDict for radar scan.
    """
    return RadarCommandDict(cmd_type="radar")


def make_pickup_fuel_command(target_x: int, target_y: int) -> PickupFuelCommandDict:
    """Create a fuel pickup command.

    Args:
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).

    Returns:
        PickupFuelCommandDict with the specified target.
    """
    return PickupFuelCommandDict(cmd_type="pickup_fuel", target_x=target_x, target_y=target_y)


def make_pickup_equipment_command(
    target_x: int,
    target_y: int,
) -> PickupEquipmentCommandDict:
    """Create an equipment pickup command.

    Args:
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).

    Returns:
        PickupEquipmentCommandDict with the specified target.
    """
    return PickupEquipmentCommandDict(
        cmd_type="pickup_equipment",
        target_x=target_x,
        target_y=target_y,
    )


class MapOpenCommandDict(TypedDict):
    """Map open command parameters (reveals global enemy positions).

    Attributes:
        cmd_type: Command type identifier.
    """

    cmd_type: Literal["map_open"]


def make_map_open_command() -> MapOpenCommandDict:
    """Create a map open command.

    Returns:
        MapOpenCommandDict for opening the map.
    """
    return MapOpenCommandDict(cmd_type="map_open")


class TeleportCommandDict(TypedDict):
    """Teleport command parameters.

    Attributes:
        cmd_type: Command type identifier.
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).
    """

    cmd_type: Literal["teleport"]
    target_x: int
    target_y: int


def make_teleport_command(target_x: int, target_y: int) -> TeleportCommandDict:
    """Create a teleport command.

    Args:
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).

    Returns:
        TeleportCommandDict with the specified target.
    """
    return TeleportCommandDict(cmd_type="teleport", target_x=target_x, target_y=target_y)


class ChatCommandDict(TypedDict):
    """Chat command parameters (preset message send).

    The wire frame carries the sender's current tile alongside the
    message ID — the page client fills its own position into every
    chat send (wiki [[chat-messages]], wire-verified 2026-07-29).

    Attributes:
        cmd_type: Command type identifier.
        message_id: Preset chat message ID (0-64, JS E[] table).
        target_x: Sender's X tile at decision time (0-255).
        target_y: Sender's Y tile at decision time (0-255).
    """

    cmd_type: Literal["chat"]
    message_id: int
    target_x: int
    target_y: int


def make_chat_command(message_id: int, target_x: int, target_y: int) -> ChatCommandDict:
    """Create a chat command.

    Args:
        message_id: Preset chat message ID (0-64).
        target_x: Sender's X tile at decision time (0-255).
        target_y: Sender's Y tile at decision time (0-255).

    Returns:
        ChatCommandDict with the specified message and position.
    """
    return ChatCommandDict(
        cmd_type="chat",
        message_id=message_id,
        target_x=target_x,
        target_y=target_y,
    )


class ScopeShiftCommandDict(TypedDict):
    """Scope-extend command: shift the stored viewport window.

    The wire ``Rb`` frame ``[3,'Z',direction]`` — free (no fuel, no
    queue slot), answered by a fresh ``0x5A`` whose origin follows the
    measured ANCHOR law: the tank pins to the window edge trailing the
    requested compass direction ([[viewport-shift-protocol]],
    wire-measured 2026-08-01 from the 2026-07-10 human capture).

    Attributes:
        cmd_type: Command type identifier.
        direction: Compass byte, clockwise from north (``SCOPE_*``
            constants in ``protocol.commands``: 0=N .. 7=NW).
    """

    cmd_type: Literal["scope_shift"]
    direction: int


def make_scope_shift_command(direction: int) -> ScopeShiftCommandDict:
    """Create a scope-shift command.

    Args:
        direction: Compass byte (0=N clockwise through 7=NW).

    Returns:
        ScopeShiftCommandDict for the requested shift.
    """
    return ScopeShiftCommandDict(cmd_type="scope_shift", direction=direction)


class HoldCommandDict(TypedDict):
    """No-op command: the tick executes but dispatches nothing.

    Produced by the durable owner arbitrator when the SPA pins
    ``manual_mode = "UNSET"``. The executor recognises ``cmd_type ==
    "hold"`` and returns without touching the wire. The tick still runs
    to completion so ai_state persists, status publishes, and the
    scorecard sees the beat — the tank simply holds its position.

    Attributes:
        cmd_type: Command type identifier.
    """

    cmd_type: Literal["hold"]


def make_hold_command() -> HoldCommandDict:
    """Create the no-op hold command.

    Returns:
        :class:`HoldCommandDict` — a stateless command used for manual
        idle ticks.
    """
    return HoldCommandDict(cmd_type="hold")


# Union of all bot command types
BotCommand = (
    MoveCommandDict
    | ShootCommandDict
    | RadarCommandDict
    | PickupFuelCommandDict
    | PickupEquipmentCommandDict
    | MapOpenCommandDict
    | TeleportCommandDict
    | ChatCommandDict
    | ScopeShiftCommandDict
    | HoldCommandDict
)


__all__ = [
    "BotCommand",
    "ChatCommandDict",
    "HoldCommandDict",
    "MapOpenCommandDict",
    "MoveCommandDict",
    "PickupEquipmentCommandDict",
    "PickupFuelCommandDict",
    "RadarCommandDict",
    "ScopeShiftCommandDict",
    "ShootCommandDict",
    "TeleportCommandDict",
    "make_chat_command",
    "make_hold_command",
    "make_map_open_command",
    "make_move_command",
    "make_pickup_equipment_command",
    "make_pickup_fuel_command",
    "make_radar_command",
    "make_scope_shift_command",
    "make_shoot_command",
    "make_teleport_command",
]
