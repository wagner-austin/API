"""Item pickup tracking.

This module provides the ItemPickupTracker class for decoding
item pickup events from TankPit WebSocket messages.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_session_xor_table, decode_base64_safe

log = get_logger(__name__)

ITEM_NAMES: tuple[str, ...] = ("armor", "dual", "missile", "homing", "radar")


class ItemPickupTracker:
    """Tracks item pickup events from 0x49 messages.

    Item Pickup Format (verified):
    - 8-byte 0x2E message with subtype 0x49 ('I')
    - XOR decode from byte 1: 67 01 [armor] [?] [missile] [homing] [?]
    - Each byte represents quantity of that item type gained
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._total_armor: int = 0
        self._total_missile: int = 0
        self._total_homing: int = 0

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

    def _decode_pickup(self, payload: str) -> tuple[int, int, int, int, int] | None:
        """Decode pickup message and extract quantities.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tuple of (armor, dual, missile, homing, radar) or None if invalid.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in pickup message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]
        if len(body) != 8 or body[0] != 0x2E:
            return None

        decoded = bytearray(7)
        for i in range(7):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        if decoded[0] != 0x67 or decoded[1] != 0x01:
            return None

        armor = decoded[2]
        dual = decoded[3]
        missile = decoded[4]
        homing = decoded[5]
        radar = decoded[6] if len(decoded) > 6 else 0

        return (armor, dual, missile, homing, radar)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return item pickup status if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Item pickup status string, or None if not an item pickup message.
        """
        quantities = self._decode_pickup(payload)
        if quantities is None:
            return None

        if all(q == 0 for q in quantities):
            return None

        armor, _, missile, homing, _ = quantities
        self._total_armor += armor
        self._total_missile += missile
        self._total_homing += homing

        items = [
            f"{qty} {name}" for qty, name in zip(quantities, ITEM_NAMES, strict=True) if qty > 0
        ]
        return f"[PICKUP] {', '.join(items)}"


__all__ = ["ITEM_NAMES", "ItemPickupTracker"]
