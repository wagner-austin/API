"""Bot command encoder.

This module provides functions to encode BotCommand TypedDicts into
wire-format bytes ready for WebSocket transmission.
"""

from __future__ import annotations

from tankpit_bot.bot.types import (
    MoveCommandDict,
    PickupMoveCommandDict,
    RadarCommandDict,
    ShootCommandDict,
    TeleportCommandDict,
)
from tankpit_bot.protocol.commands import (
    CMD_RADAR,
    build_move_command,
    build_pickup_command,
    build_query_command,
    build_shoot_command,
    build_teleport_command,
)


def encode_move_command(command: MoveCommandDict) -> bytes:
    """Encode a move command to wire-format bytes.

    Args:
        command: The move command to encode.

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    return build_move_command(command["target_x"], command["target_y"])


def encode_pickup_move_command(command: PickupMoveCommandDict) -> bytes:
    """Encode a pickup move command to wire-format bytes.

    Args:
        command: The pickup move command to encode.

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    return build_pickup_command(command["target_x"], command["target_y"])


def encode_shoot_command(command: ShootCommandDict) -> bytes:
    """Encode a shoot command to wire-format bytes.

    Args:
        command: The shoot command to encode.

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    return build_shoot_command(command["target_x"], command["target_y"])


def encode_radar_command(command: RadarCommandDict) -> bytes:
    """Encode a radar command to wire-format bytes.

    Args:
        command: The radar command to encode (unused, but validates type).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    _ = command  # Validate type, command has no payload
    return build_query_command(CMD_RADAR)


def encode_teleport_command(command: TeleportCommandDict) -> bytes:
    """Encode a teleport command to wire-format bytes.

    Args:
        command: The teleport command to encode.

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    return build_teleport_command(command["target_x"], command["target_y"])


__all__ = [
    "encode_move_command",
    "encode_pickup_move_command",
    "encode_radar_command",
    "encode_shoot_command",
    "encode_teleport_command",
]
