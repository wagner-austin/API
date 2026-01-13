"""Equipment state tracking.

This module provides tracker classes for decoding equipment toggle
and gain events from TankPit WebSocket messages.
"""

from __future__ import annotations

from typing import ClassVar

from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_xor_table, decode_base64_safe, load_xor_static_key

log = get_logger(__name__)


class EquipmentToggleTracker:
    """Tracks equipment toggle state from 0x74 messages.

    Equipment Toggle Format (verified):
    - 7-byte 0x2E message, XOR decoded from byte 1
    - Decoded: 0x74 [armor] [dual] [missile] [homing] [radar]
    - Each byte is 0 (OFF) or 1 (ON)
    """

    EQUIPMENT_NAMES: ClassVar[list[str]] = ["armor", "dual", "missile", "homing", "radar"]

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None
        self._state: list[bool] = [False] * 5
        self._prev_state: list[bool] | None = None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table.

        Args:
            magic: The session magic string for XOR encoding.
        """
        static_key, self._static_key = load_xor_static_key(self._static_key)
        if static_key is None:
            return
        self._xor_table = build_xor_table(static_key, magic)

    def _decode_toggle(self, payload: str) -> list[bool] | None:
        """Decode equipment toggle message.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            List of 5 booleans for equipment state, or None if invalid.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in equipment toggle message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]
        if len(body) != 7 or body[0] != 0x2E:
            return None

        decoded = bytearray(6)
        for i in range(6):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        if decoded[0] != 0x74:
            return None

        return [bool(decoded[i + 1]) for i in range(5)]

    def _detect_changes(self, new_state: list[bool]) -> list[str]:
        """Detect equipment state changes.

        Args:
            new_state: New equipment state (5 booleans).

        Returns:
            List of change descriptions.
        """
        if self._prev_state is None:
            return []

        changes = []
        for i, (old, new) in enumerate(zip(self._prev_state, new_state, strict=True)):
            if old != new:
                status = "ON" if new else "OFF"
                changes.append(f"{self.EQUIPMENT_NAMES[i]}={status}")
        return changes

    def process_message(self, payload: str) -> str | None:
        """Process a message and return equipment toggle status if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Equipment toggle status string, or None if not a toggle message.
        """
        new_state = self._decode_toggle(payload)
        if new_state is None:
            return None

        changes = self._detect_changes(new_state)
        self._prev_state = self._state
        self._state = new_state

        if changes:
            return f"[EQUIP:TOGGLE] {', '.join(changes)}"

        active = [self.EQUIPMENT_NAMES[i] for i, on in enumerate(new_state) if on]
        if active:
            return f"[EQUIP:STATE] active: {', '.join(active)}"
        return "[EQUIP:STATE] all OFF"

    @property
    def state(self) -> dict[str, bool]:
        """Get current equipment toggle state."""
        return {name: self._state[i] for i, name in enumerate(self.EQUIPMENT_NAMES)}


class EquipmentGainTracker:
    """Tracks equipment gain from 0x67 'g' messages.

    Equipment Gain Format (verified):
    - 8-byte 0x2E message, XOR decoded from byte 1
    - Decoded: 0x67 [type] [zeros...] [equipment_flags]
    - Different from 0x49 'I' item pickup (which is confirmation)
    - Represents equipment spawned/gained
    """

    EQUIPMENT_NAMES: ClassVar[list[str]] = ["armor", "dual", "missile", "homing", "radar"]

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

    def process_message(self, payload: str) -> str | None:
        """Process a message and return equipment gain if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Equipment gain string, or None if not an equipment gain message.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in equipment gain message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        if len(body) != 8 or body[0] != 0x2E:
            return None

        decoded = bytearray(7)
        for i in range(7):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        if decoded[0] != 0x67:
            return None

        flags5 = decoded[5]
        flags6 = decoded[6]

        gained = []
        for i, name in enumerate(self.EQUIPMENT_NAMES):
            if flags5 & (1 << i) or flags6 & (1 << i):
                gained.append(name)

        if gained:
            return f"[EQUIP:GAIN] {', '.join(gained)}"
        return f"[EQUIP:GAIN] flags={flags5},{flags6}"


__all__ = ["EquipmentGainTracker", "EquipmentToggleTracker"]
