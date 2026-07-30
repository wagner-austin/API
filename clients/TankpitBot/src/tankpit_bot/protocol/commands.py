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

# Server tick rate (verified by fire spam testing)
TICK_RATE_MS = 2000  # Commands processed every 2 seconds

# =============================================================================
# Known Command IDs (discovered from protocol analysis)
# =============================================================================

# XOR-encoded commands (type=2, start with '!')
CMD_ENTER_GAME = 63  # 0x3f - Click to enter game
CMD_ACTIVE_FORCES = 42  # 0x2a - 'x' key - Show active forces
CMD_ACTIVE_PLAYERS = 47  # 0x2f - '/' key - Show active players
CMD_RADAR = 102  # 0x66 - 's' key - Toggle radar display
CMD_NEAREST_ENEMY = 104  # 0x68 - 'e' key - Target nearest enemy
CMD_INVENTORY = 105  # 0x69 - 'i' key - Show inventory
CMD_MINE = 107  # 0x6b - 'd' key - Drop mine
CMD_MAP_OPEN = 108  # 0x6c - 'f' key - Open map view
CMD_STATISTICS = 118  # 0x76 - 'c' key - Show statistics

# XOR-encoded commands (type=2, start with '!')
CMD_PING = 46  # 0x2e - 'F6' key - Ping server, returns latency in ms

# XOR-encoded commands (type=4, start with '!') - Movement
CMD_MOVE = 112  # 0x70 - Mouse click - Tank movement (click to move)
# Movement payload: X (1 byte) + Y (1 byte) = target coordinates

CMD_PICKUP_FUEL = 100  # 0x64 / 'd' - Long press - Get fuel
# Pickup payload: X (1 byte) + Y (1 byte) = target coordinates

CMD_PICKUP_EQUIPMENT = 106  # 0x6a / 'j' - Long press - Get equipment
# Pickup payload: X (1 byte) + Y (1 byte) = target coordinates

CMD_MAP_TELEPORT = 116  # 0x74 - Map click - Teleport via map (fuel cost varies by distance)
# Teleport payload: X (1 byte) + Y (1 byte) = destination coordinates
# Requires map to be open first (CMD_MAP_OPEN)

# Movable concrete blocks (wiki [[movable-blocks]], wire-cracked
# 2026-07-20): ONE command for both pickup and drop — the server
# decides from carry state. Client binds it as "Click and Hold:
# Pick Up / Drop".
CMD_BLOCK = 98  # 0x62 / 'b' - Long press - Pick up / drop a movable block

# XOR-encoded commands (type=6, start with '!') - Combat
CMD_SHOOT = 115  # 0x73 - Spacebar - Fire at target position
# Shoot payload: X (1 byte) + Y (1 byte) + target_id_lo (1 byte) + target_id_hi (1 byte)
# - Bytes 0-1: Target coordinates (where mouse clicked)
# - Bytes 2-3: Target entity ID (little-endian, 0x0000 if no specific target)
# Shot type (regular/dual/missile/homing) determined by enabled equipment state

# XOR-encoded commands (type=3, start with '!')
CMD_TOP10 = 49  # 0x31 - 't/r/p/b/o' keys - Leaderboard (extra byte: ff=all, 00-03=team)
# Leaderboard response: rank(1) + mystery(1) + score(2) + team(1) + 0x08 + namelen(1) + name

CMD_SCOPE = 90  # 0x5a - Arrow/Page keys - Pan camera view
# Extra byte: 00=N, 02=E, 03=SE, 05=SW, 06=W, 07=NW
# Response: Z + viewport data with entity positions

CMD_TOGGLE_EQUIPMENT = 114  # 0x72 - '1-5' keys - Toggle equipment
# Extra byte: 0x31='1'=armor, 0x32='2'=dual, 0x33='3'=missile, 0x34='4'=homing, 0x35='5'=radar
# Response: t(1) + armor(1) + dual(1) + missile(1) + homing(1) + radar(1) - each 0=off, 1=on

# Scope direction codes (extra byte for CMD_SCOPE)
SCOPE_NORTH = 0x00  # ArrowUp
SCOPE_EAST = 0x02  # ArrowRight
SCOPE_SOUTHEAST = 0x03  # PageDown
SCOPE_SOUTHWEST = 0x05  # End
SCOPE_WEST = 0x06  # ArrowLeft
SCOPE_NORTHWEST = 0x07  # Home

# Plain commands (no XOR encoding, just length header + body)
PLAIN_QUIT = b"-"  # 'q' key - Quit game and return to lobby
PLAIN_SOUND_ON = b"V140"  # 'l' key - Sound on
PLAIN_SOUND_OFF = b"V040"  # 'l' key - Sound off
PLAIN_AUTOSCROLL_ON = b"A1"  # 'a' key - Autoscroll on (JS: "A" + Number(true))
PLAIN_AUTOSCROLL_OFF = b"A0"  # 'a' key - Autoscroll off (JS: "A" + Number(false))


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


# =============================================================================
# Wire Format Command Builders (NO XOR encoding for sent commands)
# =============================================================================
#
# Based on protocol analysis of captured traffic, sent commands use:
#   Type byte = 0x20 | type_number (NOT XOR encoded)
#   Format: ! + type_byte + cmd_id + [payload]
#
# Type numbers:
#   2 = Query (hotkeys like radar, inventory)
#   3 = UI (scope, leaderboard, equipment toggle)
#   4 = Movement (move, pickup, teleport)
#   6 = Combat (shoot)

TYPE_QUERY = 2  # Query commands (radar, mine, inventory, etc.)
TYPE_UI = 3  # UI commands (scope, leaderboard, equipment toggle)
TYPE_MOVEMENT = 4  # Movement commands (move, pickup, teleport)
TYPE_COMBAT = 6  # Combat commands (shoot)


def build_query_command(cmd_id: int) -> bytes:
    """Build a query command ready to send (with length header).

    Query commands: radar, mine, map open, inventory, etc.
    Format: [len_lo, len_hi] + ! + 0x22 + cmd_id (5 bytes total)

    Args:
        cmd_id: Command ID (e.g., CMD_RADAR=102, CMD_MINE=107).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    body = bytes([COMMAND_PREFIX, TYPE_QUERY, cmd_id])
    # Add 2-byte little-endian length header
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


def build_quit_command() -> bytes:
    """Build the graceful-quit command ready to send (with length header).

    The plain ``q``-key command (:data:`PLAIN_QUIT`): no XOR encoding,
    just the 2-byte little-endian length header + ``-``. The server
    treats it as a deliberate exit to the lobby, unlike the abrupt
    socket drop a bare browser close produces.

    Returns:
        Framed quit command bytes ready to send via WebSocket.
    """
    length = len(PLAIN_QUIT)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + PLAIN_QUIT


def build_move_command(x: int, y: int) -> bytes:
    """Build a MOVE command ready to send (with length header).

    Format: [len_lo, len_hi] + ! + 0x24 + 0x70 + X + Y (7 bytes total)

    Args:
        x: Target X coordinate (0-255).
        y: Target Y coordinate (0-255).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    body = bytes([COMMAND_PREFIX, TYPE_MOVEMENT, CMD_MOVE, x & 0xFF, y & 0xFF])
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


def build_block_command(x: int, y: int) -> bytes:
    """Build a block pickup/drop command ready to send (with header).

    One command serves both actions — the server decides pickup vs
    drop from the tank's carry state (wiki [[movable-blocks]]).

    Args:
        x: Target block/drop tile X (0-255).
        y: Target block/drop tile Y (0-255).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    body = bytes([COMMAND_PREFIX, TYPE_MOVEMENT, CMD_BLOCK, x & 0xFF, y & 0xFF])
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


def build_pickup_fuel_command(x: int, y: int) -> bytes:
    """Build a fuel pickup command ready to send (with length header).

    Format: [len_lo, len_hi] + ! + 0x24 + 0x64 + X + Y (7 bytes total)

    Args:
        x: Target X coordinate (0-255).
        y: Target Y coordinate (0-255).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    body = bytes([COMMAND_PREFIX, TYPE_MOVEMENT, CMD_PICKUP_FUEL, x & 0xFF, y & 0xFF])
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


def build_pickup_equipment_command(x: int, y: int) -> bytes:
    """Build an equipment pickup command ready to send (with length header).

    Format: [len_lo, len_hi] + ! + 0x24 + 0x6a + X + Y (7 bytes total)

    Args:
        x: Target X coordinate (0-255).
        y: Target Y coordinate (0-255).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    body = bytes([COMMAND_PREFIX, TYPE_MOVEMENT, CMD_PICKUP_EQUIPMENT, x & 0xFF, y & 0xFF])
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


def build_teleport_command(x: int, y: int) -> bytes:
    """Build a MAP_TELEPORT command ready to send (with length header).

    Requires map to be open first (send CMD_MAP_OPEN).
    Format: [len_lo, len_hi] + ! + 0x24 + 0x74 + X + Y (7 bytes total)

    Args:
        x: Destination X coordinate (0-255).
        y: Destination Y coordinate (0-255).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    body = bytes([COMMAND_PREFIX, TYPE_MOVEMENT, CMD_MAP_TELEPORT, x & 0xFF, y & 0xFF])
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


def build_shoot_command(x: int, y: int, target_id: int = 0) -> bytes:
    """Build a SHOOT command ready to send (with length header).

    Format: [len_lo, len_hi] + ! + 0x26 + 0x73 + X + Y + id_lo + id_hi (9 bytes total)

    Args:
        x: Target X coordinate (0-255).
        y: Target Y coordinate (0-255).
        target_id: Target entity ID (0 if no specific target).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    body = bytes(
        [
            COMMAND_PREFIX,
            TYPE_COMBAT,
            CMD_SHOOT,
            x & 0xFF,
            y & 0xFF,
            target_id & 0xFF,
            (target_id >> 8) & 0xFF,
        ]
    )
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


def build_scope_command(direction: int) -> bytes:
    """Build a SCOPE (pan camera) command ready to send.

    Format: [len_lo, len_hi] + ! + 0x23 + 0x5a + direction (6 bytes total)

    Args:
        direction: Scope direction (SCOPE_NORTH, SCOPE_EAST, etc.).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    body = bytes([COMMAND_PREFIX, TYPE_UI, CMD_SCOPE, direction])
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


def build_toggle_equipment_command(slot: int) -> bytes:
    """Build a TOGGLE_EQUIPMENT command ready to send.

    Format: [len_lo, len_hi] + ! + 0x23 + 0x72 + slot_char (6 bytes total)

    Args:
        slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile, 4=homing, 5=radar.

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    # Slot is sent as ASCII digit: '1'=0x31, '2'=0x32, etc.
    slot_char = 0x30 + slot
    body = bytes([COMMAND_PREFIX, TYPE_UI, CMD_TOGGLE_EQUIPMENT, slot_char])
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


__all__ = [
    "CMD_ACTIVE_FORCES",
    "CMD_ACTIVE_PLAYERS",
    "CMD_BLOCK",
    "CMD_ENTER_GAME",
    "CMD_INVENTORY",
    "CMD_MAP_OPEN",
    "CMD_MAP_TELEPORT",
    "CMD_MINE",
    "CMD_MOVE",
    "CMD_NEAREST_ENEMY",
    "CMD_PICKUP_EQUIPMENT",
    "CMD_PICKUP_FUEL",
    "CMD_PING",
    "CMD_RADAR",
    "CMD_SCOPE",
    "CMD_SHOOT",
    "CMD_STATISTICS",
    "CMD_TOGGLE_EQUIPMENT",
    "CMD_TOP10",
    "COMMAND_PREFIX",
    "PLAIN_AUTOSCROLL_OFF",
    "PLAIN_AUTOSCROLL_ON",
    "PLAIN_QUIT",
    "PLAIN_SOUND_OFF",
    "PLAIN_SOUND_ON",
    "SCOPE_EAST",
    "SCOPE_NORTH",
    "SCOPE_NORTHWEST",
    "SCOPE_SOUTHEAST",
    "SCOPE_SOUTHWEST",
    "SCOPE_WEST",
    "TICK_RATE_MS",
    "TYPE_COMBAT",
    "TYPE_MOVEMENT",
    "TYPE_QUERY",
    "TYPE_UI",
    "ActionCommand",
    "CommandType",
    "QueryCommand",
    "build_block_command",
    "build_move_command",
    "build_pickup_equipment_command",
    "build_pickup_fuel_command",
    "build_query_command",
    "build_quit_command",
    "build_scope_command",
    "build_shoot_command",
    "build_teleport_command",
    "build_toggle_equipment_command",
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
