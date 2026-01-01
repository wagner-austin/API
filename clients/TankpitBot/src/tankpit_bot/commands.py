"""Structured command types for Tankpit game protocol.

Game commands are XOR-encoded and sent over WebSocket. This module provides
typed structures for encoding and decoding commands.

Command format: [2-byte length LE] + '!' + [type_byte] + [cmd_byte] + [data...]

The type_byte and cmd_byte are XOR-encoded using the session's XOR table.
Command IDs must be discovered by decoding captured sessions.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_int,
    require_str,
)

# Command prefix byte (this is known from protocol analysis)
COMMAND_PREFIX = ord("!")

# =============================================================================
# Known Command IDs (discovered from protocol analysis)
# =============================================================================

# XOR-encoded commands (type=2, start with '!')
CMD_ENTER_GAME = 63  # 0x3f - Click to enter game
CMD_RADAR = 102  # 0x66 - 's' key - Toggle radar display
CMD_MINE = 107  # 0x6b - 'd' key - Drop mine
CMD_MAP_OPEN = 108  # 0x6c - 'f' key - Open map view

# Plain commands (no XOR encoding, just length header + body)
PLAIN_QUIT = b"-"  # 'q' key - Quit game and return to lobby


# =============================================================================
# Command TypedDicts
# =============================================================================


CommandType = Literal["query", "action"]


class QueryCommand(TypedDict):
    """A query command (3 bytes: ! + type + cmd).

    Query commands request information from the server without data payload.

    Attributes:
        kind: Always "query" for query commands.
        cmd_id: Command identifier byte (discovered from protocol).
    """

    kind: Literal["query"]
    cmd_id: int


class ActionCommand(TypedDict):
    """An action command (variable: ! + type + cmd + data).

    Action commands send data to the server (movement, shooting, etc.).

    Attributes:
        kind: Always "action" for action commands.
        cmd_id: Command identifier byte (discovered from protocol).
        data: Command payload as hex string.
    """

    kind: Literal["action"]
    cmd_id: int
    data: str  # Hex-encoded payload


# =============================================================================
# Encode/Decode Functions
# =============================================================================


def encode_query_command(cmd: QueryCommand) -> JSONObject:
    """Encode QueryCommand to JSON-serializable dict.

    Args:
        cmd: QueryCommand to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "kind": cmd["kind"],
        "cmd_id": cmd["cmd_id"],
    }


def decode_query_command(data: JSONObject) -> QueryCommand:
    """Decode QueryCommand from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated QueryCommand.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    kind = require_str(data, "kind")
    if kind != "query":
        raise JSONTypeError(f"Expected kind='query', got '{kind}'")

    return QueryCommand(
        kind="query",
        cmd_id=require_int(data, "cmd_id"),
    )


def encode_action_command(cmd: ActionCommand) -> JSONObject:
    """Encode ActionCommand to JSON-serializable dict.

    Args:
        cmd: ActionCommand to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "kind": cmd["kind"],
        "cmd_id": cmd["cmd_id"],
        "data": cmd["data"],
    }


def decode_action_command(data: JSONObject) -> ActionCommand:
    """Decode ActionCommand from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ActionCommand.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    kind = require_str(data, "kind")
    if kind != "action":
        raise JSONTypeError(f"Expected kind='action', got '{kind}'")

    return ActionCommand(
        kind="action",
        cmd_id=require_int(data, "cmd_id"),
        data=require_str(data, "data"),
    )


# =============================================================================
# Command Factory Functions
# =============================================================================


def make_query_command(cmd_id: int) -> QueryCommand:
    """Create a query command.

    Args:
        cmd_id: Command identifier byte.

    Returns:
        QueryCommand instance.
    """
    return QueryCommand(kind="query", cmd_id=cmd_id)


def make_action_command(cmd_id: int, data: bytes) -> ActionCommand:
    """Create an action command.

    Args:
        cmd_id: Command identifier byte.
        data: Command payload bytes.

    Returns:
        ActionCommand instance.
    """
    return ActionCommand(kind="action", cmd_id=cmd_id, data=data.hex())


# =============================================================================
# Command Serialization
# =============================================================================


def serialize_query_command(cmd: QueryCommand, type_byte: int) -> bytes:
    """Serialize a query command to wire format (before XOR encoding).

    Wire format: '!' + type_byte + cmd_id (3 bytes total)

    Args:
        cmd: QueryCommand to serialize.
        type_byte: Session-specific type byte.

    Returns:
        3-byte command ready for XOR encoding.
    """
    return bytes([COMMAND_PREFIX, type_byte, cmd["cmd_id"]])


def serialize_action_command(cmd: ActionCommand, type_byte: int) -> bytes:
    """Serialize an action command to wire format (before XOR encoding).

    Wire format: '!' + type_byte + cmd_id + data (variable length)

    Args:
        cmd: ActionCommand to serialize.
        type_byte: Session-specific type byte.

    Returns:
        Command bytes ready for XOR encoding.
    """
    data_bytes = bytes.fromhex(cmd["data"])
    return bytes([COMMAND_PREFIX, type_byte, cmd["cmd_id"]]) + data_bytes


def deserialize_command(data: bytes, type_byte: int) -> QueryCommand | ActionCommand:
    """Deserialize a command from wire format (after XOR decoding).

    Args:
        data: Decoded command bytes (must start with '!').
        type_byte: Session-specific type byte for validation.

    Returns:
        QueryCommand or ActionCommand depending on data length.

    Raises:
        ValueError: If data format is invalid.
    """
    if len(data) < 3:
        raise ValueError(f"Command too short: {len(data)} bytes, need at least 3")

    if data[0] != COMMAND_PREFIX:
        raise ValueError(
            f"Invalid command prefix: 0x{data[0]:02x}, expected 0x{COMMAND_PREFIX:02x}"
        )

    received_type = data[1]
    if received_type != type_byte:
        raise ValueError(f"Type byte mismatch: 0x{received_type:02x}, expected 0x{type_byte:02x}")

    cmd_id = data[2]

    if len(data) == 3:
        return QueryCommand(kind="query", cmd_id=cmd_id)
    payload = data[3:]
    return ActionCommand(kind="action", cmd_id=cmd_id, data=payload.hex())


__all__ = [
    "CMD_ENTER_GAME",
    "CMD_MAP_OPEN",
    "CMD_MINE",
    "CMD_RADAR",
    "COMMAND_PREFIX",
    "PLAIN_QUIT",
    "ActionCommand",
    "CommandType",
    "QueryCommand",
    "decode_action_command",
    "decode_query_command",
    "deserialize_command",
    "encode_action_command",
    "encode_query_command",
    "make_action_command",
    "make_query_command",
    "serialize_action_command",
    "serialize_query_command",
]
