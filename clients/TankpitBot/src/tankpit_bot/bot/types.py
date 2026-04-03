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


# Union of all bot command types
BotCommand = (
    MoveCommandDict
    | ShootCommandDict
    | RadarCommandDict
    | PickupFuelCommandDict
    | PickupEquipmentCommandDict
    | MapOpenCommandDict
    | TeleportCommandDict
)


__all__ = [
    "BotCommand",
    "MapOpenCommandDict",
    "MoveCommandDict",
    "PickupEquipmentCommandDict",
    "PickupFuelCommandDict",
    "RadarCommandDict",
    "ShootCommandDict",
    "TeleportCommandDict",
    "make_map_open_command",
    "make_move_command",
    "make_pickup_equipment_command",
    "make_pickup_fuel_command",
    "make_radar_command",
    "make_shoot_command",
    "make_teleport_command",
]
