"""Container tracking.

This module provides the ContainerTracker class for decoding fuel
container updates from TankPit WebSocket messages.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_xor_table, decode_base64_safe, load_xor_static_key

log = get_logger(__name__)


class ContainerTracker:
    """Tracks fuel container updates from 0x43 'C' messages.

    Container Update Format (verified):
    - 6-byte 0x2E message, XOR decoded from byte 1
    - Decoded: 0x43 [container_id_lo] [container_id_hi] [fuel_lo] [fuel_hi]
    - Container IDs are distinct from tank IDs (no overlap)
    - Fuel value of 0 means container is depleted/empty
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None
        self._containers: dict[int, int] = {}

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
        """Process a message and return container update if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Container update string, or None if not a container message.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in container message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        if len(body) != 6 or body[0] != 0x2E:
            return None

        decoded = bytearray(5)
        for i in range(5):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        if decoded[0] != 0x43:
            return None

        container_id = decoded[1] | (decoded[2] << 8)
        fuel = decoded[3] | (decoded[4] << 8)

        prev_fuel = self._containers.get(container_id)
        self._containers[container_id] = fuel

        if prev_fuel is not None:
            if fuel == 0:
                return f"[CONTAINER:{container_id}] DEPLETED (was {prev_fuel})"
            diff = fuel - prev_fuel
            return f"[CONTAINER:{container_id}] fuel={fuel} ({diff:+d})"

        if fuel == 0:
            return f"[CONTAINER:{container_id}] EMPTY"
        return f"[CONTAINER:{container_id}] fuel={fuel}"

    @property
    def containers(self) -> dict[int, int]:
        """Get current container fuel states."""
        return dict(self._containers)


__all__ = ["ContainerTracker"]
