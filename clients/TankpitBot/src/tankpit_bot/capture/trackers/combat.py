"""Combat event tracking (kills and deaths).

This module provides the DeactivationTracker class for decoding
kill and death events from TankPit WebSocket messages.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_xor_table, decode_base64_safe, load_xor_static_key

log = get_logger(__name__)


class DeactivationTracker:
    """Tracks deactivation (kill/death) events.

    Deactivation Format (verified):
    - 8-byte 0x2E message
    - XOR decode from byte 1 to get: 0x41 + victim_id + killer_id + data
    - Same format for kills and deaths - check victim_id to determine if you died
    - Death indicators: fuel spike to ~65508, fuel reset on respawn
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None
        self._my_tank_id: int | None = None
        self._kills: int = 0
        self._deaths: int = 0

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table.

        Args:
            magic: The session magic string for XOR encoding.
        """
        static_key, self._static_key = load_xor_static_key(self._static_key)
        if static_key is None:
            return
        self._xor_table = build_xor_table(static_key, magic)

    def set_my_tank_id(self, tank_id: int) -> None:
        """Set our tank ID for death detection.

        Args:
            tank_id: The tank ID to track as self.
        """
        self._my_tank_id = tank_id

    def process_message(self, payload: str) -> str | None:
        """Process a message and return deactivation status if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Deactivation status string, or None if not a deactivation message.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in deactivation message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        if len(body) != 8 or body[0] != 0x2E:
            return None

        decoded = bytearray(len(body) - 1)
        for i in range(len(decoded)):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        if decoded[0] != 0x41:
            return None

        victim_id = decoded[1] | (decoded[2] << 8)
        killer_id = decoded[3] | (decoded[4] << 8)

        if self._my_tank_id is not None and victim_id == self._my_tank_id:
            self._deaths += 1
            return f"[DEATH] You were killed by tank {killer_id} (deaths: {self._deaths})"

        self._kills += 1
        return f"[KILL] Tank {victim_id} killed by {killer_id} (kills: {self._kills})"

    @property
    def kills(self) -> int:
        """Get total kills tracked."""
        return self._kills

    @property
    def deaths(self) -> int:
        """Get total deaths tracked."""
        return self._deaths


__all__ = ["DeactivationTracker"]
