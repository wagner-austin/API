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
    """

    cmd_type: Literal["shoot"]
    target_x: int
    target_y: int


class RadarCommandDict(TypedDict):
    """Radar command parameters.

    Attributes:
        cmd_type: Command type identifier.
    """

    cmd_type: Literal["radar"]


class PickupMoveCommandDict(TypedDict):
    """Pickup move command parameters (move to container and pick it up).

    Attributes:
        cmd_type: Command type identifier.
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).
    """

    cmd_type: Literal["pickup_move"]
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


def make_shoot_command(target_x: int, target_y: int) -> ShootCommandDict:
    """Create a shoot command.

    Args:
        target_x: Target X coordinate.
        target_y: Target Y coordinate.

    Returns:
        ShootCommandDict with the specified target.
    """
    return ShootCommandDict(cmd_type="shoot", target_x=target_x, target_y=target_y)


def make_radar_command() -> RadarCommandDict:
    """Create a radar command.

    Returns:
        RadarCommandDict for radar scan.
    """
    return RadarCommandDict(cmd_type="radar")


def make_pickup_move_command(target_x: int, target_y: int) -> PickupMoveCommandDict:
    """Create a pickup move command.

    Args:
        target_x: Target X coordinate (0-255).
        target_y: Target Y coordinate (0-255).

    Returns:
        PickupMoveCommandDict with the specified target.
    """
    return PickupMoveCommandDict(cmd_type="pickup_move", target_x=target_x, target_y=target_y)


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
    | PickupMoveCommandDict
    | TeleportCommandDict
)


__all__ = [
    "BotCommand",
    "MoveCommandDict",
    "PickupMoveCommandDict",
    "RadarCommandDict",
    "ShootCommandDict",
    "TeleportCommandDict",
    "make_move_command",
    "make_pickup_move_command",
    "make_radar_command",
    "make_shoot_command",
    "make_teleport_command",
]
