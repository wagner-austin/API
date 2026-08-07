"""Position tracking from movement response messages.

This module provides the PositionTracker class for decoding position
updates from TankPit WebSocket messages.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_session_xor_table, decode_base64_safe

log = get_logger(__name__)


class PositionTracker:
    """Tracks position from movement response messages.

    Position Encoding (verified):
    - Movement responses (17-21 bytes) contain FROM position at bytes 4-5
    - XOR decoding with offset1 (skip first byte)
    - x = body[4] ^ xor_table[3], y = body[5] ^ xor_table[4]
    - Subtype varies per session (0x75, 0x76, etc.) due to XOR encoding
    - Shows where you moved FROM (previous position)

    Blocked Movement:
    - 5-byte response indicates blocked path
    - Game allows partial movement (moves as far as possible)
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._current_pos: tuple[int, int] | None = None
        self._move_subtype: int | None = None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table.

        Args:
            magic: The session magic string for XOR encoding.

        Raises:
            XorStaticKeyUnavailableError: If the static key cannot be
                read. This used to return silently, leaving
                ``_xor_table`` None so every later decode ran against
                no cipher ([[session-state-deglobalisation]]).
        """
        self._xor_table = build_session_xor_table(magic)
        self._move_subtype = None

    def decode_position(self, body: bytes) -> tuple[int, int] | None:
        """Decode FROM position from movement response.

        Args:
            body: Raw message body (17-21 bytes, starts with 0x2e).

        Returns:
            (x, y) tuple of FROM position, or None if invalid.
        """
        if len(body) < 6 or body[0] != 0x2E:
            return None
        if not (17 <= len(body) <= 21):
            return None
        if self._xor_table is None or len(self._xor_table) < 5:
            return None

        x = body[4] ^ self._xor_table[3]
        y = body[5] ^ self._xor_table[4]

        if self._move_subtype is None:
            self._move_subtype = body[1]

        return (x, y)

    def is_blocked_response(self, body: bytes) -> bool:
        """Check if message indicates blocked movement.

        Args:
            body: Raw message body.

        Returns:
            True if this is a blocked movement response.
        """
        return len(body) == 5 and body[0] == 0x2E

    def update_from_move(self, target_x: int, target_y: int) -> None:
        """Update current position from MOVE command target.

        Args:
            target_x: Target X coordinate.
            target_y: Target Y coordinate.
        """
        self._current_pos = (target_x, target_y)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return position status if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Position status string, or None if not a position message.
        """
        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in position message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        if self.is_blocked_response(body):
            return "[POS:BLOCKED]"

        if len(body) < 17 or len(body) > 21 or body[0] != 0x2E:
            return None

        pos = self.decode_position(body)
        if pos is None:
            return None

        return f"[POS:FROM] ({pos[0]}, {pos[1]})"

    @property
    def current_position(self) -> tuple[int, int] | None:
        """Get current tracked position."""
        return self._current_pos


__all__ = ["PositionTracker"]
