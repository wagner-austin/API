"""Radar scan tracking.

This module provides tracker classes for decoding radar scan results
and acknowledgements from TankPit WebSocket messages.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_xor_table, decode_base64_safe, load_xor_static_key

log = get_logger(__name__)


class RadarTracker:
    """Tracks radar scan results from 0x70 messages.

    Radar Result Format (verified):
    - 0x2E 0x70 message, XOR decoded from byte 1
    - Decoded: 0x4F [count] 0x00 [entity_records...]
    - Each record = 4 bytes: [x] [y] [fuel_lo] [fuel_hi]
    - fuel = fuel_lo | (fuel_hi << 8)
    - fuel = 0xFFFF means tank/entity, not fuel container
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table.

        Args:
            magic: The session magic string for XOR encoding.
        """
        static_key, self._static_key = load_xor_static_key(self._static_key)
        if static_key is None:
            return
        self._xor_table = build_xor_table(static_key, magic)

    def _decode_radar(self, payload: str) -> tuple[int, bytearray] | None:
        """Decode radar message and extract count and records.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tuple of (count, records_bytes) or None if invalid.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in radar message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]
        if len(body) < 4 or body[0] != 0x2E or body[1] != 0x70:
            return None

        decoded = bytearray(len(body) - 1)
        xor_table = self._xor_table
        for i in range(len(decoded)):
            decoded[i] = body[i + 1] ^ xor_table[i]

        if decoded[0] != 0x4F:
            return None

        count = decoded[1]
        records = decoded[3:]
        return (count, records)

    def _classify_entity(self, x: int, y: int, val_unsigned: int) -> tuple[str, str]:
        """Classify radar entity by value.

        Args:
            x: X coordinate.
            y: Y coordinate.
            val_unsigned: Unsigned 16-bit value.

        Returns:
            Tuple of (category, formatted_string).
        """
        if val_unsigned == 0xFFFF:
            return ("tanks", f"({x},{y})")
        if val_unsigned >= 0x8000:
            val_signed = val_unsigned - 0x10000
            return ("equip", f"({x},{y})={abs(val_signed)}")
        return ("fuel", f"({x},{y})={val_unsigned}")

    def process_message(self, payload: str) -> str | None:
        """Process a message and return radar results if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Radar results string, or None if not a radar result message.
        """
        result = self._decode_radar(payload)
        if result is None:
            return None

        count, records = result
        if count == 0:
            return "[RADAR] No entities found"

        entities: dict[str, list[str]] = {"fuel": [], "equip": [], "tanks": []}

        for i in range(0, min(len(records) - 3, count * 4), 4):
            x = records[i]
            y = records[i + 1]
            val_unsigned = records[i + 2] | (records[i + 3] << 8)
            category, formatted = self._classify_entity(x, y, val_unsigned)
            entities[category].append(formatted)

        parts = []
        for key, label in [("fuel", "fuel"), ("equip", "equip"), ("tanks", "tanks")]:
            if entities[key]:
                parts.append(f"{label}: {' '.join(entities[key])}")

        return f"[RADAR] {count} found - {'; '.join(parts)}"


class RadarAckTracker:
    """Tracks radar acknowledgement from 0x46 'F' messages.

    Radar Ack Format (verified):
    - 4-byte 0x2E message, XOR decoded from byte 1
    - Decoded: 0x46 [byte1] [byte2]
    - Appears after using radar (S key)
    - Purpose: Acknowledge radar scan was received
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None
        self._count: int = 0

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
        """Process a message and return radar ack if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Radar ack string, or None if not a radar ack message.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in radar ack message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        if len(body) != 4 or body[0] != 0x2E:
            return None

        decoded = bytearray(3)
        for i in range(3):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        if decoded[0] != 0x46:
            return None

        self._count += 1
        return f"[RADAR:ACK] #{self._count}"

    @property
    def count(self) -> int:
        """Get number of radar acknowledgements received."""
        return self._count


__all__ = ["RadarAckTracker", "RadarTracker"]
