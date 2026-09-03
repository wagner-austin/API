"""Wire builders: framed command bytes, ready to send.

Every function returns a complete frame — the 2-byte little-endian
length header followed by ``!``, the type byte and the command byte —
so a caller sends the result unchanged. Sent commands are NOT XOR
encoded; the cipher applies to what the server sends back.

Split out of :mod:`tankpit_bot.protocol.commands` 2026-09-03. These
are the half with behaviour; the vocabulary they assemble stayed
behind and is imported.
"""

from __future__ import annotations

from tankpit_bot.protocol.commands import (
    CMD_BLOCK,
    CMD_MAP_TELEPORT,
    CMD_MOVE,
    CMD_PICKUP_EQUIPMENT,
    CMD_PICKUP_FUEL,
    CMD_SCOPE,
    CMD_SHOOT,
    CMD_TOGGLE_EQUIPMENT,
    COMMAND_PREFIX,
    PLAIN_QUIT,
    TYPE_COMBAT,
    TYPE_MOVEMENT,
    TYPE_QUERY,
    TYPE_UI,
)


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
]
