"""Fuel deposit tracking.

This module provides the FuelDepositTracker class for decoding
fuel deposit events from TankPit WebSocket messages.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_xor_table, decode_base64_safe, load_xor_static_key

log = get_logger(__name__)


class FuelDepositTracker:
    """Tracks fuel deposit from 0x64 'd' messages.

    Fuel Deposit Format (verified):
    - 4-byte 0x2E message, XOR decoded from byte 1
    - Decoded: 0x64 [amount_lo] [amount_hi]
    - Indicates fuel was deposited to base
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None
        self._total_deposited: int = 0

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table.

        Args:
            magic: The session magic string for XOR encoding.
        """
        static_key, self._static_key = load_xor_static_key(self._static_key)
        if static_key is None:
            return
        self._xor_table = build_xor_table(static_key, magic)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return fuel deposit if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Fuel deposit string, or None if not a deposit message.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in fuel deposit message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        if len(body) != 4 or body[0] != 0x2E:
            return None

        decoded = bytearray(3)
        for i in range(3):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        if decoded[0] != 0x64:
            return None

        amount = decoded[1] | (decoded[2] << 8)
        self._total_deposited += amount

        return f"[FUEL:DEPOSIT] +{amount} (total: {self._total_deposited})"

    @property
    def total_deposited(self) -> int:
        """Get total fuel deposited this session."""
        return self._total_deposited


__all__ = ["FuelDepositTracker"]
