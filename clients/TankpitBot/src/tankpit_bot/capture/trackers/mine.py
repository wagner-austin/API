"""Mine placement and detonation tracking.

This module provides the MineTracker class for decoding mine-related
events from TankPit WebSocket messages.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_xor_table, decode_base64_safe, load_xor_static_key

log = get_logger(__name__)


class MineTracker:
    """Tracks mine placement and detonation events.

    Mine Mechanics:
    - Placing mines creates a 3x3 grid centered on player position
    - Shooting enemy mines triggers chain reaction detonations

    Message Types:
    - Mine placement confirmation: top-level 0x4B
      Format: mine_type, owner_id (u16 LE), count, then (x, y) pairs
    - Mine detonation/chain: top-level 0x45
      Format: (x, y) pairs for each detonated mine
    - Mine drop command: type=4, id=98 or id=100
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None
        self._mines_placed: int = 0
        self._mines_detonated: int = 0

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table.

        Args:
            magic: The session magic string for XOR encoding.
        """
        static_key, self._static_key = load_xor_static_key(self._static_key)
        if static_key is None:
            return
        self._xor_table = build_xor_table(static_key, magic)

    def process_message(self, payload: str, direction: str = "received") -> str | None:
        """Process a message and return mine status if relevant.

        Args:
            payload: Base64 encoded message payload.
            direction: 'sent' or 'received'.

        Returns:
            Mine status string, or None if not a mine message.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in mine message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        if direction == "sent" and len(body) >= 5 and body[0] == 0x21:
            return self._process_mine_command(body)

        msg_type = body[0]
        if msg_type not in (0x45, 0x4B):
            return None

        decoded = bytearray(len(body))
        decoded[0] = msg_type
        xor_table = self._xor_table
        for i in range(1, len(body)):
            decoded[i] = body[i] ^ xor_table[i - 1]

        if msg_type == 0x4B:
            return self._parse_mine_placed(decoded)
        return self._parse_mine_detonation(decoded)

    def _process_mine_command(self, body: bytes) -> str | None:
        """Process sent mine drop command.

        Preconditions (enforced by caller):
        - self._xor_table is not None
        - len(body) >= 5

        Args:
            body: Raw command body starting with '!' (0x21).

        Returns:
            Mine drop status string, or None if not a mine command.
        """
        xor_table = self._xor_table
        assert xor_table is not None, "_xor_table must be set before calling _process_mine_command"

        decrypted = bytearray(len(body))
        decrypted[0] = body[0]
        for i in range(1, len(body)):
            decrypted[i] = body[i] ^ xor_table[i - 1]

        cmd_type = decrypted[1]
        cmd_id = decrypted[2]

        if cmd_type == 4 and cmd_id in (98, 100):
            x = decrypted[3]
            y = decrypted[4]
            return f"[MINE:DROP] at ({x},{y}) (3x3 grid)"

        return None

    def _parse_mine_placed(self, decoded: bytearray) -> str:
        """Parse mine placement confirmation (0x4B).

        Args:
            decoded: XOR decoded message data.

        Returns:
            Mine placed string.
        """
        self._mines_placed += 1

        if len(decoded) >= 5:
            owner_id = decoded[2] | (decoded[3] << 8)
            count = decoded[4]
            positions: list[str] = []
            for i in range(count):
                offset = 5 + i * 2
                if offset + 1 >= len(decoded):
                    break
                positions.append(f"({decoded[offset]},{decoded[offset + 1]})")
            if positions:
                return f"[MINE:PLACED] owner={owner_id} count={count}: " + " ".join(positions)
            return f"[MINE:PLACED] owner={owner_id} count={count}"

        return f"[MINE:PLACED] (total: {self._mines_placed})"

    def _parse_mine_detonation(self, decoded: bytearray) -> str:
        """Parse mine detonation/chain reaction (0x45).

        Args:
            decoded: XOR decoded message data.

        Returns:
            Mine detonation string.
        """
        if len(decoded) < 3:
            return "[MINE:EXPLODE]"

        count = (len(decoded) - 1) // 2
        self._mines_detonated += count

        positions = []
        for i in range(count):
            offset = 1 + i * 2
            x = decoded[offset]
            y = decoded[offset + 1]
            positions.append(f"({x},{y})")

        chain_str = " CHAIN!" if count > 1 else ""
        return f"[MINE:EXPLODE]{chain_str} {count} mines: {' '.join(positions)}"

    @property
    def mines_placed(self) -> int:
        """Get total mines placed."""
        return self._mines_placed

    @property
    def mines_detonated(self) -> int:
        """Get total mines detonated."""
        return self._mines_detonated


__all__ = ["MineTracker"]
