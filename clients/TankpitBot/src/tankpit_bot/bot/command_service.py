"""Command dispatch service — owns wire encoding and WebSocket send.

Encapsulates the XOR table, CDP session reference, and WebSocket send
callback. Every game command (move, shoot, teleport, radar, etc.) is
built, encoded, and sent through this service.

The service does NOT own HFSM state transitions — callers handle state
changes after a successful send.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.bot.command_sender import SendWebSocketBytesFunc, send_command_bytes
from tankpit_bot.bot.commands import (
    encode_move_command,
    encode_pickup_equipment_command,
    encode_pickup_fuel_command,
    encode_teleport_command,
)
from tankpit_bot.bot.types import (
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_teleport_command,
)
from tankpit_bot.protocol.commands import (
    CMD_ENTER_GAME,
    CMD_INVENTORY,
    CMD_MAP_OPEN,
    CMD_NEAREST_ENEMY,
    CMD_RADAR,
    build_query_command,
    build_quit_command,
    build_shoot_command,
    build_toggle_equipment_command,
)

log = get_logger(__name__)

_SLOT_NAMES = ["armor", "dual", "missile", "homing", "radar"]


class CommandService:
    """Owns wire encoding and WebSocket send for all game commands.

    Holds the XOR table and CDP session reference. Does not own HFSM
    state transitions — the caller handles state changes after a
    successful send returns True.
    """

    def __init__(self, send_ws_bytes: SendWebSocketBytesFunc) -> None:
        """Initialize with the WebSocket send callback.

        Args:
            send_ws_bytes: Callback to send raw bytes over the WebSocket.
        """
        self._send_ws_bytes = send_ws_bytes
        self.cdp: CDPSessionProtocol | None = None
        self.xor_table: bytes | None = None

    def send_bytes(self, data: bytes, cmd_name: str) -> bool:
        """XOR encode and send command bytes via WebSocket.

        Args:
            data: Framed command bytes (with 2-byte length header).
            cmd_name: Command name for logging.

        Returns:
            True if sent, False if CDP session not available.
        """
        return send_command_bytes(
            self.cdp,
            self.xor_table,
            data,
            cmd_name,
            self._send_ws_bytes,
        )

    def quit_game(self) -> bool:
        """Send the plain graceful-quit command (``q`` key).

        Sent at session teardown before the browser closes so the
        server records a deliberate lobby exit instead of an abrupt
        socket drop.

        Returns:
            True if command was sent.
        """
        return self.send_bytes(build_quit_command(), "quit_game")

    def enter_game(self) -> bool:
        """Send CMD_ENTER_GAME to activate the tank.

        Returns:
            True if command was sent.
        """
        return self.send_bytes(build_query_command(CMD_ENTER_GAME), "enter_game")

    def move(self, x: int, y: int) -> bool:
        """Send move command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        cmd = make_move_command(x, y)
        return self.send_bytes(encode_move_command(cmd), "move")

    def pickup_fuel(self, x: int, y: int) -> bool:
        """Send fuel pickup command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        cmd = make_pickup_fuel_command(x, y)
        return self.send_bytes(encode_pickup_fuel_command(cmd), "pickup_fuel")

    def pickup_equipment(self, x: int, y: int) -> bool:
        """Send equipment pickup command.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        cmd = make_pickup_equipment_command(x, y)
        return self.send_bytes(encode_pickup_equipment_command(cmd), "pickup_equipment")

    def teleport(self, x: int, y: int) -> bool:
        """Send teleport command. Map must already be open.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent, False if CDP unavailable.
        """
        cmd = make_teleport_command(x, y)
        return self.send_bytes(encode_teleport_command(cmd), f"teleport({x},{y})")

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        """Send shoot command.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            target_id: Target entity ID (0 if no specific target).

        Returns:
            True if command was sent.
        """
        return self.send_bytes(
            build_shoot_command(x, y, target_id),
            f"shoot({x},{y},id={target_id})",
        )

    def radar(self) -> bool:
        """Send radar scan command.

        Returns:
            True if command was sent.
        """
        return self.send_bytes(build_query_command(CMD_RADAR), "radar")

    def open_map(self) -> bool:
        """Send map open command.

        Returns:
            True if command was sent.
        """
        return self.send_bytes(build_query_command(CMD_MAP_OPEN), "map_open")

    def request_nearest_enemy(self) -> bool:
        """Send CMD_NEAREST_ENEMY ('e' key) to query nearest enemy position.

        Returns:
            True if command was sent.
        """
        return self.send_bytes(build_query_command(CMD_NEAREST_ENEMY), "nearest_enemy")

    def request_inventory(self) -> bool:
        """Send CMD_INVENTORY ('i') to request the inventory snapshot.

        The cheapest game action on the wire (2 bytes, free, no world
        effect) — the watch-probe heartbeat uses it to hold the
        push-on-activity stream open (wiki log 2026-07-24).

        Returns:
            True if command was sent.
        """
        return self.send_bytes(build_query_command(CMD_INVENTORY), "inventory")

    def toggle_equipment(self, slot: int) -> bool:
        """Toggle equipment slot on/off.

        Args:
            slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile,
                4=homing, 5=radar.

        Returns:
            True if command was sent.
        """
        if slot < 1 or slot > 5:
            log.warning("Invalid equipment slot: %d (must be 1-5)", slot)
            return False
        encoded = build_toggle_equipment_command(slot)
        return self.send_bytes(encoded, f"toggle_{_SLOT_NAMES[slot - 1]}")


__all__ = [
    "CommandService",
]
