"""Tank event tracking (join, leave, move, shoot, status).

This module provides tracker classes for decoding tank-related events
from TankPit WebSocket messages.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_xor_table, decode_base64_safe, load_xor_static_key
from tankpit_bot.protocol.constants import RANK_NAMES, TEAM_NAMES

log = get_logger(__name__)

TEAM_COLORS = TEAM_NAMES  # Backward-compatible alias


class TankTracker:
    """Tracks tank entry, status, movement, and shooting events.

    Message Types (from client JS):
    - 0x28 '(' Tank Entry: tank joins with rank, team, position
    - 0x3E '>' Tank Status: rank, team, equipment, fuel, points, name
    - 0x47 'G' Movement: tank_id, position, direction, path
    - 0x53 'S' Shooting: shooter_team, shooter_id, origin position
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None
        self._tanks: dict[int, dict[str, str | int]] = {}
        self._dispatch: dict[int, tuple[int, Callable[[bytearray], str | None]]] = {
            0x28: (4, self._parse_tank_join),
            0x29: (4, self._parse_tank_leave),
            0x3E: (13, self._parse_tank_status),
            0x47: (5, self._parse_movement),
            0x53: (5, self._parse_shooting),
            0x21: (12, self._parse_tank_info),
            0x4D: (6, self._parse_player_list),
            0x2F: (4, self._parse_player_update),
            0x56: (14, self._parse_statistics),
            0x2B: (3, self._parse_promotion),
            0x52: (4, self._parse_supervisor_msg),
        }

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table.

        Args:
            magic: The session magic string for XOR encoding.
        """
        static_key, self._static_key = load_xor_static_key(self._static_key)
        if static_key is None:
            return
        self._xor_table = build_xor_table(static_key, magic)

    def _decode_payload(self, payload: str) -> tuple[int, bytearray, bytes] | None:
        """Decode base64 payload and XOR decrypt.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tuple of (msg_type, decoded_bytes, raw_body) or None if invalid.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in tank message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]
        msg_type = body[0]
        max_decode = min(len(body) - 1, len(self._xor_table))
        decoded = bytearray(len(body) - 1)
        for i in range(len(decoded)):
            if i < max_decode:
                decoded[i] = body[i + 1] ^ self._xor_table[i]
            else:
                decoded[i] = body[i + 1]

        return (msg_type, decoded, bytes(body))

    def process_message(self, payload: str) -> str | None:
        """Process a message and return tank info if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tank info string, or None if not a tank-related message.
        """
        result = self._decode_payload(payload)
        if result is None:
            return None

        msg_type, decoded, raw_body = result

        if msg_type == 0x2E:
            if len(decoded) < 8:
                return None
            return self._parse_status_sync(decoded, raw_body)

        entry = self._dispatch.get(msg_type)
        if entry is None:
            return None

        min_len, handler = entry
        if len(decoded) < min_len:
            return None

        return handler(decoded)

    def _parse_tank_join(self, decoded: bytearray) -> str | None:
        """Parse tank join message (0x28 '(').

        Args:
            decoded: XOR decoded message data.

        Returns:
            Tank join string, or None if invalid.
        """
        if len(decoded) < 3:
            return None
        tank_id = decoded[1] | (decoded[2] << 8)
        name = self.get_name(tank_id)
        name_str = f"'{name}'" if name else f"id={tank_id}"
        extra = decoded[3:].hex() if len(decoded) > 3 else ""
        return f"[JOIN] {name_str} data={extra}"

    def _parse_tank_leave(self, decoded: bytearray) -> str | None:
        """Parse tank leave message (0x29 ')').

        Args:
            decoded: XOR decoded message data.

        Returns:
            Tank leave string, or None if invalid.
        """
        if len(decoded) < 3:
            return None
        tank_id = decoded[1] | (decoded[2] << 8)
        name = self.get_name(tank_id)
        name_str = f"'{name}'" if name else f"id={tank_id}"
        extra = decoded[3:].hex() if len(decoded) > 3 else ""
        return f"[LEAVE] {name_str} data={extra}"

    def _parse_tank_status(self, decoded: bytearray) -> str | None:
        """Parse tank status message (0x3E).

        Args:
            decoded: XOR decoded message data.

        Returns:
            Tank status string, or None if invalid.
        """
        if len(decoded) < 13:
            return None

        info_byte = decoded[0]
        team = info_byte & 0x03
        rank = (info_byte >> 4) & 0x0F

        tank_id = decoded[1] | (decoded[2] << 8)

        name = ""
        if len(decoded) > 13:
            name = bytes(decoded[13:]).decode("utf-8", errors="ignore").rstrip("\x00")

        team_name = TEAM_NAMES[team] if team < len(TEAM_NAMES) else f"team{team}"
        rank_name = RANK_NAMES[rank] if rank < len(RANK_NAMES) else f"rank{rank}"

        self._tanks[tank_id] = {"team": team_name, "rank": rank_name, "name": name}

        name_str = f" '{name}'" if name else ""
        return f"[TANK:STATUS] id={tank_id}{name_str} {team_name} {rank_name}"

    def _parse_movement(self, decoded: bytearray) -> str | None:
        """Parse movement message (0x47 format, 5+ bytes).

        Format: [tank_id_lo, tank_id_hi, x, y, direction]

        Args:
            decoded: XOR decoded message data.

        Returns:
            Movement string, or None if invalid.
        """
        if len(decoded) < 5:
            return None
        tank_id = decoded[0] | (decoded[1] << 8)
        x = decoded[2]
        y = decoded[3]
        direction = decoded[4]

        if tank_id in self._tanks:
            self._tanks[tank_id]["x"] = x
            self._tanks[tank_id]["y"] = y
            name = self._tanks[tank_id].get("name", "")
            if name:
                return f"[TANK:MOVE] {name} to ({x},{y}) dir={direction}"

        return f"[TANK:MOVE] tank={tank_id} to ({x},{y}) dir={direction}"

    def _parse_movement_response(self, decoded: bytearray) -> str | None:
        """Parse movement response message (0x3D format, 11 bytes).

        Format: [team, tank_id, unk, x, y, direction, unk, rank, lb_hi, lb_mid, lb_lo]

        Args:
            decoded: XOR decoded message data.

        Returns:
            Movement string, or None if invalid.
        """
        if len(decoded) < 11:
            return None
        team = decoded[0]
        tank_id = decoded[1]
        x = decoded[3]
        y = decoded[4]
        direction = decoded[5]
        rank = decoded[7]

        team_name = TEAM_NAMES[team] if team < len(TEAM_NAMES) else f"team{team}"
        rank_name = RANK_NAMES[rank] if rank < len(RANK_NAMES) else f"rank{rank}"

        if tank_id in self._tanks:
            self._tanks[tank_id]["x"] = x
            self._tanks[tank_id]["y"] = y
            name = self._tanks[tank_id].get("name", "")
            if name:
                return (
                    f"[TANK:MOVE] {name} (id={tank_id}) {team_name} {rank_name} "
                    f"to ({x},{y}) dir={direction}"
                )

        return f"[TANK:MOVE] id={tank_id} {team_name} {rank_name} to ({x},{y}) dir={direction}"

    def _parse_shooting(self, decoded: bytearray) -> str | None:
        """Parse shooting message (0x53).

        Args:
            decoded: XOR decoded message data.

        Returns:
            Shooting string, or None if invalid.
        """
        if len(decoded) < 4:
            return None
        shooter_team = decoded[0]
        shooter_id = decoded[1] | (decoded[2] << 8)
        shot_x = decoded[3]
        shot_y = decoded[4] if len(decoded) > 4 else 0

        if shooter_team < len(TEAM_NAMES):
            team_name = TEAM_NAMES[shooter_team]
        else:
            team_name = f"team{shooter_team}"

        if shooter_id in self._tanks:
            name = self._tanks[shooter_id].get("name", "")
            if name:
                return f"[TANK:SHOT] {name} ({team_name}) fired from ({shot_x},{shot_y})"

        return f"[TANK:SHOT] id={shooter_id} ({team_name}) fired from ({shot_x},{shot_y})"

    def _parse_tank_info(self, decoded: bytearray) -> str | None:
        """Parse tank info message (0x21) - contains tank_id -> name mapping.

        Args:
            decoded: XOR decoded message data.

        Returns:
            Tank info string, or None if invalid.
        """
        if len(decoded) < 11:
            return None

        tank_id = decoded[1] | (decoded[2] << 8)

        name = ""
        for b in decoded[10:]:
            if 32 <= b < 127:
                name += chr(b)
            elif name:
                break

        if not name:
            return None

        self.register_name(tank_id, name)

        return f"[TANK:INFO] id={tank_id} name='{name}'"

    def _parse_player_list(self, decoded: bytearray) -> str:
        """Parse player list message (0x4D 'M').

        Args:
            decoded: XOR decoded message data.

        Returns:
            Player list string.
        """
        tank_id = decoded[0] | (decoded[1] << 8)
        name = self.get_name(tank_id)
        b2 = decoded[2]
        b3 = decoded[3]
        b4 = decoded[4]
        name_str = f"'{name}'" if name else f"id={tank_id}"
        return f"[PLAYERS] {name_str} data={b2:02x} {b3:02x} {b4:02x}"

    def _parse_player_update(self, decoded: bytearray) -> str:
        """Parse player update message (0x2F '/').

        Args:
            decoded: XOR decoded message data.

        Returns:
            Player update string.
        """
        entries = []
        i = 0
        while i + 2 < len(decoded):
            tank_id = decoded[i] | (decoded[i + 1] << 8)
            data = decoded[i + 2]
            name = self.get_name(tank_id)
            name_str = f"'{name}'" if name else f"id={tank_id}"
            entries.append(f"{name_str}:{data}")
            i += 3
        return f"[PLAYERS] {', '.join(entries)}"

    def _parse_statistics(self, decoded: bytearray) -> str:
        """Parse statistics message (0x56 'V').

        Args:
            decoded: XOR decoded message data.

        Returns:
            Statistics string.
        """
        hours = decoded[0] | (decoded[1] << 8)
        mins = decoded[2]
        secs = decoded[3]
        destroyed = decoded[7]
        deactivated = decoded[8]
        promo_pts = (decoded[12] << 8) | decoded[13] if len(decoded) > 13 else 0
        time_str = f"{hours}h{mins}m{secs}s"
        stats_str = f"destroyed={destroyed} deactivated={deactivated} promo={promo_pts}"
        return f"[STATS] {time_str} {stats_str}"

    def _parse_promotion(self, decoded: bytearray) -> str:
        """Parse promotion message (0x2B '+').

        Args:
            decoded: XOR decoded message data.

        Returns:
            Promotion string.
        """
        rank_idx = decoded[0]
        promoted = decoded[1] == 1
        rank_name = RANK_NAMES[rank_idx] if rank_idx < len(RANK_NAMES) else f"rank{rank_idx}"
        if promoted:
            return f"[PROMOTED] to {rank_name}!"
        return f"[DEMOTED] to {rank_name}"

    def _parse_supervisor_msg(self, decoded: bytearray) -> str:
        """Parse supervisor message (0x52 'R').

        Args:
            decoded: XOR decoded message data.

        Returns:
            Supervisor message string.
        """
        status = decoded[2] if len(decoded) > 2 else 0
        return f"[SUPERVISOR] status={status}"

    def _parse_status_sync(self, decoded: bytearray, raw_body: bytes) -> str:
        """Parse tank status sync message (0x2E '.').

        Args:
            decoded: XOR decoded message data.
            raw_body: Raw body bytes (unused but kept for signature compatibility).

        Returns:
            Status sync string.
        """
        _ = raw_body  # Unused but part of interface
        subtype = decoded[0] if len(decoded) > 0 else 0
        tank_id = decoded[1] | (decoded[2] << 8) if len(decoded) > 2 else 0
        name = self.get_name(tank_id)
        name_str = f"'{name}'" if name else f"id={tank_id}"

        subtype_char = chr(subtype) if 32 <= subtype < 127 else ""
        subtype_str = f"0x{subtype:02x}"
        if subtype_char:
            subtype_str += f" '{subtype_char}'"

        return f"[STATUS:{subtype_str}] {name_str} len={len(decoded)} hex={decoded.hex()}"

    def register_name(self, tank_id: int, name: str) -> None:
        """Manually register a tank name.

        Args:
            tank_id: The tank ID.
            name: The tank name.
        """
        if tank_id not in self._tanks:
            self._tanks[tank_id] = {}
        self._tanks[tank_id]["name"] = name

    def get_name(self, tank_id: int) -> str | None:
        """Get the name for a tank ID.

        Args:
            tank_id: The tank ID.

        Returns:
            The tank name, or None if not known.
        """
        if tank_id not in self._tanks:
            return None
        name = self._tanks[tank_id].get("name")
        if isinstance(name, str):
            return name
        return None

    def get_all_names(self) -> dict[int, str]:
        """Get all known tank ID to name mappings.

        Returns:
            Dictionary of tank_id -> name.
        """
        result: dict[int, str] = {}
        for tid, info in self._tanks.items():
            name = info.get("name")
            if isinstance(name, str) and name:
                result[tid] = name
        return result


class TankExitTracker:
    """Tracks tank exit/disconnect from 0x58 'X' messages.

    Tank Exit Format (verified):
    - 4-byte 0x2E message, XOR decoded from byte 1
    - Decoded: 0x58 [tank_id_lo] [tank_id_hi]
    - Indicates player left/disconnected (not killed)
    - Tank IDs match those seen in TANK_MOVE messages
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None
        self._exited: set[int] = set()

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
        """Process a message and return tank exit if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tank exit string, or None if not an exit message.
        """
        if self._xor_table is None:
            return None

        data = decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in exit message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        if len(body) != 4 or body[0] != 0x2E:
            return None

        decoded = bytearray(3)
        for i in range(3):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        if decoded[0] != 0x58:
            return None

        tank_id = decoded[1] | (decoded[2] << 8)
        self._exited.add(tank_id)

        return f"[TANK:EXIT] id={tank_id}"

    @property
    def exited_tanks(self) -> set[int]:
        """Get set of tank IDs that have exited."""
        return set(self._exited)


__all__ = ["RANK_NAMES", "TEAM_COLORS", "TankExitTracker", "TankTracker"]
