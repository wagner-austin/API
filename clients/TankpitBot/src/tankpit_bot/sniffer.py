"""WebSocket traffic sniffer using Playwright and CDP.

Captures WebSocket messages from tankpit.com by:
1. Launching a Chromium browser via Playwright
2. Creating a CDP session to intercept Network events
3. Navigating to the target URL
4. Recording all WebSocket frames (sent and received)
5. Saving the capture session to a JSON file
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks, protocol
from tankpit_bot.browser import (
    BrowserSession,
    PlaywrightNotInstalledError,
    get_current_time_ms,
    reset_cdp_time_offset,
)
from tankpit_bot.dom_scraper import GameLogEntry
from tankpit_bot.types import (
    CapturedMessage,
    CaptureSession,
    CombatEvent,
    MessageStats,
    SessionSummary,
    encode_capture_session,
    encode_session_summary,
)

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
        self._static_key: str | None = None
        self._move_subtype: int | None = None  # Detected movement subtype

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)
        self._move_subtype = None  # Reset on new session

    def decode_position(self, body: bytes) -> tuple[int, int] | None:
        """Decode FROM position from movement response.

        Args:
            body: Raw message body (17-21 bytes, starts with 0x2e).

        Returns:
            (x, y) tuple of FROM position, or None if invalid.
        """
        # Movement responses are 17-21 bytes
        if len(body) < 6 or body[0] != 0x2E:
            return None
        if not (17 <= len(body) <= 21):
            return None
        if self._xor_table is None or len(self._xor_table) < 5:
            return None

        # Offset1 XOR: skip first byte
        x = body[4] ^ self._xor_table[3]
        y = body[5] ^ self._xor_table[4]

        # Validate coordinates are reasonable (0-255 game grid)
        if x > 255 or y > 255:
            return None

        # Track the subtype for this session
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
        # 5-byte 0x2e messages appear to indicate blocked movement
        return len(body) == 5 and body[0] == 0x2E

    def update_from_move(self, target_x: int, target_y: int) -> None:
        """Update current position from MOVE command target."""
        self._current_pos = (target_x, target_y)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return position status if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Position status string, or None if not a position message.
        """
        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in position message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Check for blocked movement (5-byte response)
        if self.is_blocked_response(body):
            return "[POS:BLOCKED]"

        # Check for movement response (17-21 bytes)
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

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def set_my_tank_id(self, tank_id: int) -> None:
        """Set our tank ID for death detection."""
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

        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in deactivation message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Deactivation messages are 8 bytes starting with 0x2E
        if len(body) != 8 or body[0] != 0x2E:
            return None

        # XOR decode from byte 1
        decoded = bytearray(len(body) - 1)
        for i in range(len(decoded)):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        # Check if first decoded byte is 'A' (0x41)
        if decoded[0] != 0x41:
            return None

        # Parse victim and killer IDs
        victim_id = decoded[1] | (decoded[2] << 8)
        killer_id = decoded[3] | (decoded[4] << 8)

        # Check if this is our death
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
        self._static_key: str | None = None
        self._total_armor: int = 0
        self._total_missile: int = 0
        self._total_homing: int = 0

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def _decode_pickup(self, payload: str) -> tuple[int, int, int, int, int] | None:
        """Decode pickup message and extract quantities.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tuple of (armor, dual, missile, homing, radar) or None if invalid.
        """
        if self._xor_table is None:
            return None

        data = _decode_base64_safe(payload)
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

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def _decode_radar(self, payload: str) -> tuple[int, bytearray] | None:
        """Decode radar message and extract count and records.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tuple of (count, records_bytes) or None if invalid.
        """
        if self._xor_table is None:
            return None

        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in radar message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]
        if len(body) < 4 or body[0] != 0x2E or body[1] != 0x70:
            return None

        decoded = bytearray(len(body) - 1)
        for i in range(len(decoded)):
            if i < len(self._xor_table):
                decoded[i] = body[i + 1] ^ self._xor_table[i]
            else:
                decoded[i] = body[i + 1]

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


# Team color names from client JS
TEAM_COLORS = ["red", "purple", "blue", "orange"]
# Rank names from client JS
RANK_NAMES = [
    "recruit",
    "private",
    "corporal",
    "sergeant",
    "lieutenant",
    "captain",
    "major",
    "general",
]


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
            0x28: (4, self._parse_tank_join),  # Tank Entry '('
            0x29: (4, self._parse_tank_leave),  # Tank Leave ')'
            0x3E: (13, self._parse_tank_status),  # Tank Status '>'
            # 0x3D '=' is TEXT (JOIN_CONFIRM), not binary - handled elsewhere
            0x47: (5, self._parse_movement),  # Movement 'G'
            0x53: (5, self._parse_shooting),  # Shooting 'S'
            0x21: (12, self._parse_tank_info),  # Tank Info '!'
            0x4D: (6, self._parse_player_list),  # Player List 'M'
            0x2F: (4, self._parse_player_update),  # Player Update '/'
            0x56: (14, self._parse_statistics),  # Statistics 'V'
            0x2B: (3, self._parse_promotion),  # Promotion '+'
            0x52: (4, self._parse_supervisor_msg),  # Supervisor 'R'
            # Note: 0x2E (Status Sync) handled separately - needs raw_body
        }

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def _decode_payload(self, payload: str) -> tuple[int, bytearray, bytes] | None:
        """Decode base64 payload and XOR decrypt.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tuple of (msg_type, decoded_bytes, raw_body) or None if invalid.
            raw_body is needed because some subtypes (0x01, 0x03) are not XOR encoded.
        """
        if self._xor_table is None:
            return None

        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in tank message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]
        if len(body) < 2:
            return None

        msg_type = body[0]
        max_decode = min(len(body) - 1, len(self._xor_table))
        decoded = bytearray(len(body) - 1)
        for i in range(len(decoded)):
            if i < max_decode:
                decoded[i] = body[i + 1] ^ self._xor_table[i]
            else:
                decoded[i] = body[i + 1]

        if len(decoded) < 1:
            return None

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

        # Handle 0x2E (Status Sync) separately - needs raw_body for subtype
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

        decoded format (0x28 NOT included):
        - [0] = subtype/flags
        - [1-2] = tank_id (u16 LE)
        - [3+] = additional data (position, rank, etc.)
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

        decoded format (0x29 NOT included):
        - [0] = subtype/flags
        - [1-2] = tank_id (u16 LE)
        - [3+] = additional data
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

        decoded format (0x3E NOT included):
        - [0] = info byte (rank/team encoded)
        - [1-2] = tank_id (u16 LE)
        - [3-12] = equipment/stats data
        - [13+] = tank name (null-terminated string)
        """
        if len(decoded) < 13:
            return None

        info_byte = decoded[0]
        # Team in lower 2 bits, rank in bits 4-6
        team = info_byte & 0x03
        rank = (info_byte >> 4) & 0x07

        tank_id = decoded[1] | (decoded[2] << 8)

        # Name starts at byte 13
        name = ""
        if len(decoded) > 13:
            name = bytes(decoded[13:]).decode("utf-8", errors="ignore").rstrip("\x00")

        team_name = TEAM_COLORS[team] if team < len(TEAM_COLORS) else f"team{team}"
        rank_name = RANK_NAMES[rank] if rank < len(RANK_NAMES) else f"rank{rank}"

        self._tanks[tank_id] = {"team": team_name, "rank": rank_name, "name": name}

        name_str = f" '{name}'" if name else ""
        return f"[TANK:STATUS] id={tank_id}{name_str} {team_name} {rank_name}"

    def _parse_movement_response(self, decoded: bytearray) -> str | None:
        """Parse movement response message (0x3D).

        decoded format (0x3D NOT included):
        - [0] = team
        - [1-2] = tank_id (u16 LE)
        - [3] = x
        - [4] = y
        - [5] = direction
        - [6] = unknown (values: 1, 2, 3)
        - [7] = rank (0-7)
        - [8-10] = score (u24 BE)
        """
        if len(decoded) < 11:
            return None

        team = decoded[0]
        tank_id = decoded[1] | (decoded[2] << 8)
        x = decoded[3]
        y = decoded[4]
        # decoded[5] is direction (unused)
        rank = decoded[7]
        lb_pos = (decoded[8] << 16) | (decoded[9] << 8) | decoded[10]

        name = self.get_name(tank_id)
        team_name = TEAM_COLORS[team] if team < len(TEAM_COLORS) else f"team{team}"
        rank_name = RANK_NAMES[rank] if rank < len(RANK_NAMES) else f"rank{rank}"

        # Update tank info
        if tank_id not in self._tanks:
            self._tanks[tank_id] = {}
        self._tanks[tank_id]["team"] = team_name
        self._tanks[tank_id]["rank"] = rank_name
        self._tanks[tank_id]["lb_pos"] = lb_pos
        self._tanks[tank_id]["x"] = x
        self._tanks[tank_id]["y"] = y

        name_str = f"'{name}'" if name else f"id={tank_id}"
        return f"[MOVE:RESP] {name_str} ({team_name}) {rank_name} #{lb_pos} at ({x},{y})"

    def _parse_movement(self, decoded: bytearray) -> str | None:
        """Parse movement message (0x47).

        decoded format (0x47 NOT included):
        - [0-1] = tank_id (u16 LE)
        - [2] = x
        - [3] = y
        - [4] = direction
        """
        if len(decoded) < 4:
            return None
        tank_id = decoded[0] | (decoded[1] << 8)
        x = decoded[2]
        y = decoded[3]
        direction = decoded[4] if len(decoded) > 4 else 0

        # Update tank position
        if tank_id in self._tanks:
            self._tanks[tank_id]["x"] = x
            self._tanks[tank_id]["y"] = y
            name = self._tanks[tank_id].get("name", "")
            if name:
                return f"[TANK:MOVE] {name} (id={tank_id}) to ({x},{y}) dir={direction}"

        return f"[TANK:MOVE] id={tank_id} to ({x},{y}) dir={direction}"

    def _parse_shooting(self, decoded: bytearray) -> str | None:
        """Parse shooting message (0x53).

        decoded format (0x53 NOT included):
        - [0] = shooter_team
        - [1-2] = shooter_id (u16 LE)
        - [3] = x
        - [4] = y
        """
        if len(decoded) < 4:
            return None
        shooter_team = decoded[0]
        shooter_id = decoded[1] | (decoded[2] << 8)
        shot_x = decoded[3]
        shot_y = decoded[4] if len(decoded) > 4 else 0

        if shooter_team < len(TEAM_COLORS):
            team_name = TEAM_COLORS[shooter_team]
        else:
            team_name = f"team{shooter_team}"

        # Get shooter name if known
        if shooter_id in self._tanks:
            name = self._tanks[shooter_id].get("name", "")
            if name:
                return f"[TANK:SHOT] {name} ({team_name}) fired from ({shot_x},{shot_y})"

        return f"[TANK:SHOT] id={shooter_id} ({team_name}) fired from ({shot_x},{shot_y})"

    def _parse_tank_info(self, decoded: bytearray) -> str | None:
        """Parse tank info message (0x21) - contains tank_id -> name mapping.

        decoded format (0x21 NOT included):
        - [0] = team (0=red, 1=purple, 2=blue, 3=orange)
        - [1-2] = tank_id (u16 LE)
        - [3-6] = decoration_state (4 bytes = 9 2-bit values)
        - [7-9] = score (24-bit BE)
        - [10+] = name (UTF-8)

        NOTE: This message does NOT contain the tank's current rank!
        """
        if len(decoded) < 11:
            return None

        tank_id = decoded[1] | (decoded[2] << 8)

        # Extract name from byte 10 onwards
        name = ""
        for b in decoded[10:]:
            if 32 <= b < 127:
                name += chr(b)
            elif name:
                break

        if not name:
            return None

        # Register the name
        self.register_name(tank_id, name)

        return f"[TANK:INFO] id={tank_id} name='{name}'"

    def _parse_player_list(self, decoded: bytearray) -> str:
        """Parse player list message (0x4D 'M').

        decoded format (0x4D NOT included):
        - [0-1] = tank_id (u16 LE)
        - [2-4] = position or rank data
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

        decoded format (0x2F NOT included):
        - Repeating entries: tank_id (u16 LE), data_byte
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

        decoded format (0x56 NOT included) - 14 bytes:
        - [0-1] = hours (u16 LE)
        - [2] = minutes
        - [3] = seconds
        - [4-6] = padding
        - [7] = destroyed (single byte)
        - [8] = deactivated (single byte)
        - [9-11] = padding
        - [12-13] = promotion points (u16 BE)
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

        decoded format (0x2B NOT included) - 2 bytes:
        - [0] = new rank level (0-8)
        - [1] = promoted flag (1 = promoted, 0 = rank set)

        Ranks: 0=recruit, 1=private, 2=corporal, 3=sergeant,
               4=lieutenant, 5=captain, 6=major, 7=colonel, 8=general
        """
        ranks = [
            "recruit",
            "private",
            "corporal",
            "sergeant",
            "lieutenant",
            "captain",
            "major",
            "colonel",
            "general",
        ]
        rank_idx = decoded[0]
        promoted = decoded[1] == 1
        rank_name = ranks[rank_idx] if rank_idx < len(ranks) else f"rank{rank_idx}"
        if promoted:
            return f"[PROMOTED] to {rank_name}!"
        return f"[DEMOTED] to {rank_name}"

    def _parse_supervisor_msg(self, decoded: bytearray) -> str:
        """Parse supervisor message (0x52 'R').

        decoded format (0x52 NOT included) - 3 bytes:
        - [0] = always 0x01
        - [1] = always 0x00
        - [2] = status value (seen: 4, 7, 8)

        JS class: xg (supervisor message)

        Status values (4, 7, 8) meaning unknown.
        """
        status = decoded[2] if len(decoded) > 2 else 0
        return f"[SUPERVISOR] status={status}"

    def _parse_status_sync(self, decoded: bytearray, raw_body: bytes) -> str:
        """Parse tank status sync message (0x2E '.').

        This catches 0x2E messages not handled by specialized trackers.
        Shows decoded subtype and hex for analysis.

        Known decoded subtypes (handled elsewhere):
        - 0x43 'C' = Container (ContainerTracker)
        - 0x41 'A' = Deactivation (DeactivationTracker)
        - 0x67 'g' = EquipmentGain (EquipmentGainTracker)
        - 0x64 'd' = FuelDeposit (FuelDepositTracker)
        - 0x74 't' = EquipmentToggle (EquipmentToggleTracker)
        - 0x58 'X' = TankExit (TankExitTracker)
        - 0x46 'F' = RadarAck (RadarAckTracker)
        """
        # XOR-decoded subtype
        subtype = decoded[0] if len(decoded) > 0 else 0
        tank_id = decoded[1] | (decoded[2] << 8) if len(decoded) > 2 else 0
        name = self.get_name(tank_id)
        name_str = f"'{name}'" if name else f"id={tank_id}"

        # Show subtype as char if printable ASCII
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


class MineTracker:
    """Tracks mine placement and detonation events.

    Mine Mechanics:
    - Placing mines creates a 3x3 grid centered on player position
    - Shooting enemy mines triggers chain reaction detonations

    Message Types:
    - Mine placement confirmation: decoded signature 0x4B 'K'
      Format: owner_id (u16 LE), positions follow
    - Mine detonation/chain: decoded signature 0x45 'E'
      Format: count, then (x, y) pairs for each detonated mine
    - Mine drop command: type=4, id=98 or id=100
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._static_key: str | None = None
        self._mines_placed: int = 0
        self._mines_detonated: int = 0

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

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

        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in mine message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Check for sent mine drop command
        if direction == "sent" and len(body) >= 5 and body[0] == 0x21:  # '!' command
            return self._process_mine_command(body)

        # Received messages
        if len(body) < 3 or body[0] != 0x2E:
            return None

        # XOR decode from byte 1
        decoded = bytearray(len(body) - 1)
        for i in range(len(decoded)):
            if i < len(self._xor_table):
                decoded[i] = body[i + 1] ^ self._xor_table[i]
            else:
                decoded[i] = body[i + 1]

        sig = decoded[0] if decoded else 0

        # Mine placement confirmation (0x4B = 'K')
        if sig == 0x4B:
            return self._parse_mine_placed(decoded)

        # Mine detonation (0x45 = 'E')
        if sig == 0x45:
            return self._parse_mine_detonation(decoded)

        return None

    def _process_mine_command(self, body: bytes) -> str | None:
        """Process sent mine drop command."""
        if self._xor_table is None:
            return None
        # Decrypt command
        decrypted = bytearray(len(body))
        decrypted[0] = body[0]  # '!'
        xor_table = self._xor_table
        for i in range(1, len(body)):
            if i - 1 < len(xor_table):
                decrypted[i] = body[i] ^ xor_table[i - 1]
            else:
                decrypted[i] = body[i]

        cmd_type = decrypted[1]
        cmd_id = decrypted[2]

        # Mine drop command: type=4, id=98 or id=100
        if cmd_type == 4 and cmd_id in (98, 100):
            if len(decrypted) >= 5:
                x = decrypted[3]
                y = decrypted[4]
                # 3x3 grid centered on player
                return f"[MINE:DROP] at ({x},{y}) (3x3 grid)"
            return "[MINE:DROP]"

        return None

    def _parse_mine_placed(self, decoded: bytearray) -> str:
        """Parse mine placement confirmation (0x4B)."""
        self._mines_placed += 1

        if len(decoded) >= 5:
            owner_id = decoded[1] | (decoded[2] << 8)
            x = decoded[3]
            y = decoded[4]
            return f"[MINE:PLACED] owner={owner_id} at ({x},{y})"

        return f"[MINE:PLACED] (total: {self._mines_placed})"

    def _parse_mine_detonation(self, decoded: bytearray) -> str:
        """Parse mine detonation/chain reaction (0x45)."""
        if len(decoded) < 2:
            return "[MINE:EXPLODE]"

        count = decoded[1]
        self._mines_detonated += count

        positions = []
        for i in range(count):
            offset = 2 + i * 2
            if offset + 1 < len(decoded):
                x = decoded[offset]
                y = decoded[offset + 1]
                positions.append(f"({x},{y})")

        if positions:
            chain_str = " CHAIN!" if count > 1 else ""
            return f"[MINE:EXPLODE]{chain_str} {count} mines: {' '.join(positions)}"

        return f"[MINE:EXPLODE] {count} mines"

    @property
    def mines_placed(self) -> int:
        """Get total mines placed."""
        return self._mines_placed

    @property
    def mines_detonated(self) -> int:
        """Get total mines detonated."""
        return self._mines_detonated


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

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def _decode_toggle(self, payload: str) -> list[bool] | None:
        """Decode equipment toggle message.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            List of 5 booleans for equipment state, or None if invalid.
        """
        if self._xor_table is None:
            return None

        data = _decode_base64_safe(payload)
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
        self._containers: dict[int, int] = {}  # container_id -> fuel

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return container update if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Container update string, or None if not a container message.
        """
        if self._xor_table is None:
            return None

        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in container message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Container messages are 6 bytes: 0x2E + encoded data
        if len(body) < 6 or body[0] != 0x2E:
            return None

        # XOR decode from byte 1
        decoded = bytearray(5)
        for i in range(5):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        # Check for 0x43 'C' signature
        if decoded[0] != 0x43:
            return None

        container_id = decoded[1] | (decoded[2] << 8)
        fuel = decoded[3] | (decoded[4] << 8)

        # Track changes
        prev = self._containers.get(container_id)
        self._containers[container_id] = fuel

        if fuel == 0:
            if prev and prev > 0:
                return f"[CONTAINER:{container_id}] DEPLETED (was {prev})"
            return f"[CONTAINER:{container_id}] EMPTY"

        if prev is not None and prev != fuel:
            diff = fuel - prev
            return f"[CONTAINER:{container_id}] fuel={fuel} ({diff:+d})"

        return f"[CONTAINER:{container_id}] fuel={fuel}"

    @property
    def containers(self) -> dict[int, int]:
        """Get current container fuel states."""
        return dict(self._containers)


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

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return tank exit if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tank exit string, or None if not an exit message.
        """
        if self._xor_table is None:
            return None

        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in exit message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Exit messages are 4 bytes: 0x2E + encoded data
        if len(body) != 4 or body[0] != 0x2E:
            return None

        # XOR decode from byte 1
        decoded = bytearray(3)
        for i in range(3):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        # Check for 0x58 'X' signature
        if decoded[0] != 0x58:
            return None

        tank_id = decoded[1] | (decoded[2] << 8)
        self._exited.add(tank_id)

        return f"[TANK:EXIT] id={tank_id}"

    @property
    def exited_tanks(self) -> set[int]:
        """Get set of tank IDs that have exited."""
        return set(self._exited)


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

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return equipment gain if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Equipment gain string, or None if not an equipment gain message.
        """
        if self._xor_table is None:
            return None

        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in equipment gain message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Equipment gain messages are 8 bytes: 0x2E + encoded data
        if len(body) != 8 or body[0] != 0x2E:
            return None

        # XOR decode from byte 1
        decoded = bytearray(7)
        for i in range(7):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        # Check for 0x67 'g' signature
        if decoded[0] != 0x67:
            return None

        # Parse equipment flags (bytes 5 and 6 contain equipment bits)
        flags5 = decoded[5]
        flags6 = decoded[6]

        # Decode which equipment was gained
        gained = []
        for i, name in enumerate(self.EQUIPMENT_NAMES):
            if flags5 & (1 << i) or flags6 & (1 << i):
                gained.append(name)

        # Also check raw values for unknown patterns
        if gained:
            return f"[EQUIP:GAIN] {', '.join(gained)}"
        return f"[EQUIP:GAIN] flags={flags5},{flags6}"


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

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return fuel deposit if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Fuel deposit string, or None if not a deposit message.
        """
        if self._xor_table is None:
            return None

        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in fuel deposit message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Deposit messages are 4 bytes: 0x2E + encoded data
        if len(body) != 4 or body[0] != 0x2E:
            return None

        # XOR decode from byte 1
        decoded = bytearray(3)
        for i in range(3):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        # Check for 0x64 'd' signature
        if decoded[0] != 0x64:
            return None

        amount = decoded[1] | (decoded[2] << 8)
        self._total_deposited += amount

        return f"[FUEL:DEPOSIT] +{amount} (total: {self._total_deposited})"

    @property
    def total_deposited(self) -> int:
        """Get total fuel deposited this session."""
        return self._total_deposited


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

    def _load_static_key(self) -> str | None:
        """Load static XOR key from file."""
        if self._static_key is not None:
            return self._static_key

        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            self._static_key = _test_hooks.read_text(static_key_path).strip()
            return self._static_key
        return None

    def set_magic(self, magic: str) -> None:
        """Set magic key and build XOR table."""
        static_key = self._load_static_key()
        if static_key is None:
            return

        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return radar ack if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Radar ack string, or None if not a radar ack message.
        """
        if self._xor_table is None:
            return None

        data = _decode_base64_safe(payload)
        if data is None:
            log.debug("Invalid base64 in radar ack message")
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Radar ack messages are 4 bytes: 0x2E + encoded data
        if len(body) != 4 or body[0] != 0x2E:
            return None

        # XOR decode from byte 1
        decoded = bytearray(3)
        for i in range(3):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        # Check for 0x46 'F' signature
        if decoded[0] != 0x46:
            return None

        self._count += 1
        return f"[RADAR:ACK] #{self._count}"

    @property
    def count(self) -> int:
        """Get number of radar acknowledgements received."""
        return self._count


# Global tracker instances
_position_tracker = PositionTracker()
_deactivation_tracker = DeactivationTracker()
_item_tracker = ItemPickupTracker()
_radar_tracker = RadarTracker()
_tank_tracker = TankTracker()
_mine_tracker = MineTracker()
_equip_tracker = EquipmentToggleTracker()
_container_tracker = ContainerTracker()
_exit_tracker = TankExitTracker()
_equip_gain_tracker = EquipmentGainTracker()
_deposit_tracker = FuelDepositTracker()
_radar_ack_tracker = RadarAckTracker()

# All trackers for bulk initialization (all have set_magic and _xor_table)
_ALL_TRACKERS = (
    _position_tracker,
    _deactivation_tracker,
    _item_tracker,
    _radar_tracker,
    _tank_tracker,
    _mine_tracker,
    _equip_tracker,
    _container_tracker,
    _exit_tracker,
    _equip_gain_tracker,
    _deposit_tracker,
    _radar_ack_tracker,
)

# Trackers for received messages (all except mine_tracker which needs direction)
_RECEIVED_TRACKERS = (
    _position_tracker,
    _deactivation_tracker,
    _item_tracker,
    _radar_tracker,
    _tank_tracker,
    _equip_tracker,
    _container_tracker,
    _exit_tracker,
    _equip_gain_tracker,
    _deposit_tracker,
    _radar_ack_tracker,
)


def _init_trackers_with_magic(magic: str) -> None:
    """Initialize all trackers with magic key if not already set."""
    for tracker in _ALL_TRACKERS:
        if tracker._xor_table is None:
            tracker.set_magic(magic)
    # Also build global XOR table for unified decoder
    _build_global_xor_table(magic)


# Module-level XOR table for unified decoder
_global_xor_table: bytes | None = None
_global_static_key: str | None = None


def _load_global_static_key() -> str | None:
    """Load static XOR key from file."""
    global _global_static_key
    if _global_static_key is not None:
        return _global_static_key
    static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
    if _test_hooks.path_exists(static_key_path):
        _global_static_key = _test_hooks.read_text(static_key_path).strip()
        return _global_static_key
    return None


def _build_global_xor_table(magic: str) -> None:
    """Build global XOR table from magic key."""
    global _global_xor_table
    static_key = _load_global_static_key()
    if static_key is None:
        return
    table = bytearray(len(static_key))
    for i in range(len(static_key)):
        table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
    _global_xor_table = bytes(table)


def _xor_decode(body: bytes) -> bytes:
    """XOR decode message body (skip first byte which is msg_type)."""
    if _global_xor_table is None or len(body) < 2:
        return body[1:] if len(body) > 1 else b""
    decoded = bytearray(len(body) - 1)
    for i in range(len(decoded)):
        if i < len(_global_xor_table):
            decoded[i] = body[i + 1] ^ _global_xor_table[i]
        else:
            decoded[i] = body[i + 1]
    return bytes(decoded)


# Message type byte -> display name mapping
_MSG_TYPE_NAMES: dict[int, str] = {
    0x21: "TankInfo",
    0x28: "TankJoin",
    0x29: "TankLeave",
    0x2E: "TankStatus",
    0x3D: "MoveResponse",
    0x3E: "TankStatus",
    0x41: "Deactivation",
    0x43: "Container",
    0x44: "FuelDeposit",
    0x45: "MineDetonate",
    0x46: "RadarAck",
    0x47: "Movement",
    0x48: "MovementShort",
    0x49: "ItemPickup",
    0x4A: "TerrainUpdate",
    0x4B: "MinePlace",
    0x4C: "WorldEntry",
    0x4D: "PlayerList",
    0x4F: "RadarResult",
    0x52: "Supervisor",
    0x53: "Shooting",
    0x54: "ActionDone",
    0x56: "Statistics",
    0x58: "TankExit",
    0x5A: "ViewportUpdate",
    0x64: "FuelDeposit",
    0x67: "EquipGain",
    0x74: "EquipToggle",
}


def _format_decoded_message(msg_type: int, decoded: protocol.BinaryMessage) -> str:
    """Format a decoded protocol message as readable string.

    Args:
        msg_type: Message type byte.
        decoded: Decoded binary protocol message.

    Returns:
        Formatted string for logging.
    """
    # For container messages, use the string msg_type from container_decoder
    actual_type = decoded["msg_type"]
    if isinstance(actual_type, str):
        # Container message - use specific type name
        type_name = actual_type.replace("_", " ").title().replace(" ", "")
    else:
        # Protocol message - use int-based lookup
        type_name = _MSG_TYPE_NAMES.get(msg_type, f"Msg0x{msg_type:02X}")
    details = _format_message_details(decoded)
    if details:
        return f"[{type_name}] {details}"
    return f"[{type_name}]"


# Message type categories for formatting dispatch
_COMBAT_MSG_TYPES: frozenset[int] = frozenset({0x53, 0x41})
_TANK_MSG_TYPES: frozenset[int] = frozenset({0x28, 0x58, 0x2E, 0x3E, 0x21, 0x47, 0x3D, 0x48})
_RESOURCE_MSG_TYPES: frozenset[int] = frozenset({0x44, 0x64, 0x49, 0x43})
_POSITION_MSG_TYPES: frozenset[int] = frozenset({0x4B, 0x45})
_RADAR_MSG_TYPES: frozenset[int] = frozenset({0x46, 0x4F, 0x5A})
_MISC_MSG_TYPES: frozenset[int] = frozenset({0x67, 0x74, 0x56, 0x52, 0x4D})


def _format_combat_details(d: protocol.BinaryMessage) -> str:
    """Format combat-related message details."""
    if d["msg_type"] == 0x53:
        return f"shooter={d['shooter_id']} tgt=({d['target_x']},{d['target_y']})"
    if d["msg_type"] == 0x41:
        return f"victim={d['victim_id']} killer={d['killer_id']}"
    return ""


# Rank number -> name mapping
_RANK_NAMES: tuple[str, ...] = (
    "recruit",
    "private",
    "corporal",
    "sergeant",
    "lieutenant",
    "captain",
    "major",
    "general",
)

# Damage state -> description
_DAMAGE_NAMES: tuple[str, ...] = ("full", "light", "medium", "critical")

# Team number -> name
_TEAM_NAMES: tuple[str, ...] = ("red", "blue", "green", "purple")


def _rank_name(rank: int) -> str:
    """Get rank name from rank number."""
    return _RANK_NAMES[rank] if 0 <= rank < len(_RANK_NAMES) else f"r{rank}"


def _damage_name(damage: int) -> str:
    """Get damage description from damage_state."""
    return _DAMAGE_NAMES[damage] if 0 <= damage < len(_DAMAGE_NAMES) else f"d{damage}"


def _team_name(team: int) -> str:
    """Get team name from team number."""
    return _TEAM_NAMES[team] if 0 <= team < len(_TEAM_NAMES) else f"t{team}"


def _format_tank_details(d: protocol.BinaryMessage) -> str:
    """Format tank status message details."""
    if d["msg_type"] == 0x28:
        # TankEntryDict: tank_id, x, y, name (no rank/team/damage)
        return f"tank={d['tank_id']} at ({d['x']},{d['y']}) name={d['name']}"
    if d["msg_type"] == 0x58:
        return f"tank={d['tank_id']} left"
    if d["msg_type"] == 0x2E:
        # TankStatusSyncDict: has damage_state and rank
        rank = _rank_name(d["rank"])
        dmg = _damage_name(d["damage_state"])
        return f"tank={d['tank_id']} {rank} hp={dmg} lb={d['leaderboard_position']}"
    if d["msg_type"] == 0x3E:
        # TankStatusDict: has leaderboard_score not score
        rank = _rank_name(d["rank"])
        team = _team_name(d["team"])
        return f"tank={d['tank_id']} {team} {rank} score={d['leaderboard_score']}"
    if d["msg_type"] == 0x21:
        # TankInfoDict: has team
        team = _team_name(d["team"])
        return f"tank={d['tank_id']} {team} name={d['name']}"
    if d["msg_type"] == 0x47:
        # MovementDict: no rank, has fuel
        x, y, dr = d["start_x"], d["start_y"], d["direction"]
        return f"tank={d['tank_id']} at ({x},{y}) dir={dr} fuel={d['fuel']}"
    if d["msg_type"] == 0x3D:
        # MovementResponseDict: has rank and leaderboard_position
        rank = _rank_name(d["rank"])
        x, y, dr = d["x"], d["y"], d["direction"]
        return f"tank={d['tank_id']} at ({x},{y}) dir={dr} {rank} lb={d['leaderboard_position']}"
    if d["msg_type"] == 0x48:
        rank = _rank_name(d["rank"])
        return f"tank={d['tank_id']} at ({d['x']},{d['y']}) {rank}"
    return ""


def _format_resource_details(d: protocol.BinaryMessage) -> str:
    """Format resource-related message details."""
    if d["msg_type"] == 0x44:
        return f"amount={d['amount']} free={d['is_free']}"
    if d["msg_type"] == 0x64:
        return f"amount={d['amount']}"
    if d["msg_type"] == 0x49:
        return f"counts={d['counts']}"
    if d["msg_type"] == 0x43:
        return f"id={d['container_id']} fuel={d['fuel']}"
    return ""


def _format_position_details(d: protocol.BinaryMessage) -> str:
    """Format position update message details."""
    if d["msg_type"] == 0x4B:
        return f"tank={d['tank_id']} count={len(d['positions'])}"
    if d["msg_type"] == 0x45:
        return f"count={len(d['positions'])}"
    return ""


def _format_radar_details(d: protocol.BinaryMessage) -> str:
    """Format radar-related message details."""
    if d["msg_type"] == 0x46:
        return f"type={d['detection_type']} found={d['found']}"
    if d["msg_type"] == 0x4F:
        return f"entities={len(d['entities'])}"
    if d["msg_type"] == 0x5A:
        return f"dir={d['direction']} entities={len(d['entities'])}"
    return ""


def _format_misc_details(d: protocol.BinaryMessage) -> str:
    """Format miscellaneous message details."""
    if d["msg_type"] == 0x67:
        return f"gained={d['gained']}"
    if d["msg_type"] == 0x74:
        return f"enabled={d['enabled']}"
    if d["msg_type"] == 0x56:
        return f"time={d['playtime_hours']}h{d['playtime_minutes']}m"
    if d["msg_type"] == 0x52:
        return f"status={d['status']} data={d['data']}"
    if d["msg_type"] == 0x4D:
        return f"sender={d['sender_id']} type={d['message_type']}"
    return ""


def _format_container_details(d: protocol.BinaryMessage) -> str:
    """Format container message details (string msg_type from container_decoder).

    Args:
        d: Decoded binary protocol message.

    Returns:
        Formatted details string, or empty string if not a container message.
    """
    match d:
        case {"msg_type": "combat_hit", "direction": int(direction), "attacker_id": int(aid)}:
            dir_str = "out" if direction == 0x09 else "in"
            return f"attacker={aid} dir={dir_str}"
        case {"msg_type": "tank_registry", "tank_id": int(tid), "flags": int(flags)}:
            return f"tank={tid} flags=0x{flags:02X}"
        case {
            "msg_type": "position_update",
            "tank_id": int(tid),
            "flags": int(f),
            "status_bytes": bytes(sb),
        }:
            return f"tank={tid} flags=0x{f:02X} data={sb.hex()}"
        case {"msg_type": "tank_status_sync", "sync_data": bytes(sd)}:
            return f"data={sd.hex()}"
        case {
            "msg_type": "tank_status_short",
            "tank_id": int(tid),
            "damage_state": int(dmg),
            "rank": int(rank),
            "leaderboard_position": int(lb),
        }:
            rank_str = _rank_name(rank)
            dmg_str = _damage_name(dmg)
            return f"tank={tid} {rank_str} hp={dmg_str} lb={lb}"
        case {
            "msg_type": "tank_update_compact",
            "tank_id": int(tid),
            "flags": int(f),
            "status_data": bytes(sd),
        }:
            return f"tank={tid} flags=0x{f:02X} data={sd.hex()}"
        case {
            "msg_type": "tank_update_extended",
            "tank_id": int(tid),
            "flags": int(f),
            "status_data": bytes(sd),
        }:
            return f"tank={tid} flags=0x{f:02X} data={sd.hex()}"
        case {
            "msg_type": "tank_update_full",
            "tank_id": int(tid),
            "flags": int(f),
            "status_data": bytes(sd),
        }:
            return f"tank={tid} flags=0x{f:02X} data={sd.hex()}"
        case {"msg_type": "unknown_container", "length": int(length), "data": bytes(data)}:
            return f"len={length} data={data.hex()[:40]}"
        case _:
            return ""


def _format_message_details(d: protocol.BinaryMessage) -> str:
    """Get formatted details for a decoded message using msg_type discriminant.

    Args:
        d: Decoded binary protocol message.

    Returns:
        Formatted details string, or empty string for simple types.
    """
    # Handle container messages (string msg_type from container_decoder)
    if isinstance(d["msg_type"], str):
        return _format_container_details(d)
    mt = d["msg_type"]
    # Handle int msg_types from protocol module
    if mt in _COMBAT_MSG_TYPES:
        return _format_combat_details(d)
    if mt in _TANK_MSG_TYPES:
        return _format_tank_details(d)
    if mt in _RESOURCE_MSG_TYPES:
        return _format_resource_details(d)
    if mt in _POSITION_MSG_TYPES:
        return _format_position_details(d)
    if mt in _RADAR_MSG_TYPES:
        return _format_radar_details(d)
    if mt in _MISC_MSG_TYPES:
        return _format_misc_details(d)
    return ""


# Text message type bytes (ASCII chars that indicate text, not binary)
_TEXT_MESSAGE_TYPES: frozenset[int] = frozenset({0x3D, 0x2B, 0x24, 0x2A, 0x25, 0x2D})


def _decode_received_text_message(payload: str) -> None:
    """Decode and log received text messages (JOIN_CONFIRM, ROOM_LIST, etc.).

    Args:
        payload: Base64-encoded message payload.
    """
    # Validate base64 - must be valid characters and proper length
    if not payload or len(payload) % 4 != 0:
        return
    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    if not all(c in valid_chars for c in payload):
        return

    data = base64.b64decode(payload)
    if len(data) < 2:
        return

    body = data[2:]
    if len(body) == 0:
        return

    # Only process text message types
    if body[0] not in _TEXT_MESSAGE_TYPES:
        return

    text = body.decode("utf-8", errors="replace")
    decoded = _decode_text_message(text, len(body), "RECEIVED", body)
    log.info(decoded)


def _process_received_message(payload: str) -> None:
    """Decode and log ALL received messages using protocol module."""
    data = _decode_base64_safe(payload)
    if data is None or len(data) < 3:
        return

    body = data[2:]
    if len(body) == 0:
        return

    msg_type = body[0]

    # Text messages (not XOR encoded)
    if msg_type in _TEXT_MESSAGE_TYPES:
        text = body.decode("utf-8", errors="replace")
        decoded_str = _decode_text_message(text, len(body), "RECEIVED", body)
        log.info(decoded_str)
        return

    # Binary messages - XOR decode and use protocol module
    decoded_data = _xor_decode(body)
    if len(decoded_data) == 0:
        log.info(f"[RECEIVED] EMPTY: type=0x{msg_type:02X}")
        return

    # All messages go through protocol.decode_message (handles 0x2E routing internally)
    msg_char = chr(msg_type) if 32 <= msg_type < 127 else "?"
    _decode_and_log_binary(msg_type, decoded_data, msg_char, body)


# Message type -> minimum data length (from protocol.py _require_* calls)
_MSG_MIN_LENGTHS: dict[int, int] = {
    ord("S"): 12,  # ShootEvent
    ord("A"): 5,  # Deactivation
    ord("K"): 4,  # MinePlacement
    ord("E"): 0,  # MineDetonation (no minimum)
    ord("D"): 3,  # FuelGain
    ord("d"): 2,  # FuelDeposit
    ord("I"): 6,  # Inventory
    ord("g"): 6,  # EquipmentGain
    ord("t"): 5,  # EquipmentToggle
    ord("F"): 2,  # RadarResult
    ord("H"): 6,  # EnemyDetection
    ord("O"): 2,  # RadarScanResult
    ord("("): 10,  # TankEntry
    ord("X"): 2,  # TankExit
    ord("."): 1,  # 0x2E container (TankStatusSync or tunneled message)
    ord(">"): 13,  # TankStatus
    ord("!"): 10,  # TankInfo
    ord("G"): 9,  # Movement
    ord("="): 11,  # MovementResponse
    ord("Z"): 2,  # ViewportUpdate
    ord("J"): 0,  # TerrainUpdate (no minimum)
    ord("?"): 0,  # Sync (no minimum)
    ord("C"): 4,  # Container
    ord("M"): 3,  # ChatMessage
    ord("V"): 16,  # Statistics
    ord("*"): 4,  # ActiveForces
    ord("R"): 3,  # Supervisor
    ord("T"): 0,  # ActionDone (no minimum)
}


def _decode_and_log_binary(msg_type: int, data: bytes, type_label: str, raw_body: bytes) -> None:
    """Decode binary message using protocol module and log it.

    Args:
        msg_type: Message type byte.
        data: XOR-decoded message data (without msg_type byte).
        type_label: Label for logging.
        raw_body: Original raw body for length reporting.
    """
    msg_char = chr(msg_type) if 32 <= msg_type < 127 else "?"
    hex_preview = data[:20].hex() + "..." if len(data) > 20 else data.hex()

    # Check if type is known and data meets minimum length
    min_len = _MSG_MIN_LENGTHS.get(msg_type)
    if min_len is None:
        # Unknown type - show debug info
        log.info(
            "[RECEIVED] UNKNOWN 0x%02X '%s' len=%d data=%s",
            msg_type,
            msg_char,
            len(raw_body),
            hex_preview,
        )
        return

    if len(data) < min_len:
        # Data too short for this type
        log.info(
            "[RECEIVED] SHORT 0x%02X '%s' need=%d got=%d data=%s",
            msg_type,
            msg_char,
            min_len,
            len(data),
            hex_preview,
        )
        return

    # Decode using protocol module - for binary messages only
    # Text messages are handled separately in _decode_text_message
    binary_decoded = protocol.try_decode_binary_message(msg_type, data)
    if binary_decoded is None:
        log.info(
            "[RECEIVED] UNIMPL 0x%02X '%s' len=%d data=%s",
            msg_type,
            msg_char,
            len(raw_body),
            hex_preview,
        )
        return
    formatted = _format_decoded_message(msg_type, binary_decoded)
    log.info("[RECEIVED] %s", formatted)


# Default configuration constants
DEFAULT_TARGET_URL = "https://tankpit.com"
DEFAULT_OUTPUT_PATH = "capture_session.json"
DEFAULT_CAPTURE_DURATION_MS = 0  # 0 = indefinite (wait until browser closed)


def _decode_8byte_state(body: bytes, tag: str) -> str:
    """Decode 8-byte state message by subtype."""
    subtype = body[1]
    if subtype == 0x49:
        return f"[{tag}] ITEM_PICKUP: {body.hex()}"
    if subtype == 0x67:
        return f"[{tag}] GAME_STATE: {body.hex()}"
    return f"[{tag}] MSG_8B: sub=0x{subtype:02x} {body.hex()}"


def _decode_state_message(body: bytes, tag: str) -> str:
    """Decode a '.' prefixed state message based on length/pattern.

    State message types by length:
    - 2-3 bytes: Heartbeat/sync
    - 4-8 bytes: Entity position refs
    - 12 bytes: Hit confirmation (after shots)
    - 14 bytes: Tank status with fuel
    - 17-30 bytes: Entity updates
    - 673+ bytes: Map data

    Args:
        body: Raw body bytes starting with '.'.
        tag: Direction tag (SENT/RECEIVED).

    Returns:
        Human-readable decoded state string.
    """
    length = len(body)

    if length <= 3:
        return f"[{tag}] SYNC: {body.hex()}"

    if length > 500:
        return f"[{tag}] MAP_DATA: len={length}"

    if length == 12:
        return f"[{tag}] HIT: {body.hex()}"

    if length == 8:
        return _decode_8byte_state(body, tag)

    if 14 <= length <= 16:
        return f"[{tag}] STATE: sub=0x{body[1]:02x} len={length} hex={body.hex()}"

    if length == 17 and body[1] == 0x10:
        raw_p15 = int.from_bytes(body[15:17], "little")
        return f"[{tag}] FUEL_RAW: p15={raw_p15} hex={body.hex()}"

    if 17 <= length <= 30:
        return f"[{tag}] ENTITY: sub=0x{body[1]:02x} len={length} hex={body.hex()}"

    if 4 <= length <= 11:
        return f"[{tag}] POS: len={length} hex={body.hex()}"

    return f"[{tag}] UPDATE: len={length} hex={body[:20].hex()}..."


def _decode_text_message(text: str, body_len: int, tag: str, body: bytes | None = None) -> str:
    """Decode a text-based protocol message.

    Args:
        text: Decoded text body.
        body_len: Original body length in bytes.
        tag: Direction tag (SENT/RECEIVED).
        body: Raw body bytes for binary state messages.

    Returns:
        Human-readable decoded message string.
    """
    if text == "-":
        return f"[{tag}] QUIT: -"
    if text.startswith("%AUTH"):
        return f"[{tag}] AUTH: {text[:60]}..."
    if text.startswith("+") and "|" in text:
        return _decode_plus_message(text, tag)
    if text.startswith("*"):
        return f"[{tag}] SELECT: room={text[1:]}"
    if text.startswith("="):
        return _decode_join_confirm(text, tag)
    if text.startswith("$"):
        return f"[{tag}] RESPONSE: {text}"
    if text.startswith(".") and body is not None:
        return _decode_state_message(body, tag)
    if text.startswith("."):
        return f"[{tag}] STATE: len={body_len} bytes"
    # Unknown - show first 40 chars
    preview = text[:40].replace("\n", " ")
    return f"[{tag}] ???: {preview}..."


def _decode_message(payload: str, direction: str, magic: str | None = None) -> str:
    """Decode a WebSocket message payload for display.

    Args:
        payload: Base64-encoded message payload.
        direction: 'sent' or 'received'.
        magic: Captured XOR magic key.

    Returns:
        Human-readable decoded message string.
    """
    tag = direction.upper()
    data = _decode_base64_safe(payload)
    if data is None:
        return f"[{tag}] (invalid base64)"

    if len(data) < 2:
        return f"[{tag}] (too short: {data.hex()})"

    # Header is 2-byte little-endian length, body follows
    body = data[2:]

    # Handle XOR commands (starting with '!')
    if len(body) > 0 and body[0] == 0x21:  # 0x21 is '!'
        return _decode_command(body, tag, magic)

    text = body.decode("utf-8", errors="replace")
    return _decode_text_message(text, len(body), tag, body)


def _decode_plus_message(text: str, tag: str) -> str:
    """Decode a '+' prefixed message (ROOM_LIST or ACTION)."""
    parts = text.split("|")
    if len(parts) >= 3 and len(parts[0]) > 1 and parts[0][1:].isdigit():
        room_id = parts[0][1:]
        name = parts[1] if len(parts) > 1 else "?"
        return f"[{tag}] ROOM_LIST: room={room_id} name={name}"
    # Action message with coords
    room_id = parts[0][1:] if len(parts) > 0 else "?"
    coords = f"{parts[2]},{parts[3]}" if len(parts) >= 4 else "?"
    return f"[{tag}] ACTION: room={room_id} coords={coords}"


def _decode_join_confirm(text: str, tag: str) -> str:
    """Decode a '=' prefixed JOIN_CONFIRM message.

    Format: =room|date|name|rank|eq1|eq2|eq3|eq4
    Example: =2|Sep. 25, 2012|Yuppler|4|9|9|9|10

    Rank values: 0=recruit, 1=private, 2=corporal, 3=sergeant,
                 4=lieutenant, 5=captain, 6=major, 7=general
    """
    rank_names = [
        "recruit",
        "private",
        "corporal",
        "sergeant",
        "lieutenant",
        "captain",
        "major",
        "general",
    ]
    parts = text.split("|")
    room_id = parts[0][1:] if len(parts) > 0 else "?"
    tank_name = parts[2] if len(parts) > 2 else "?"
    rank_num = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else -1
    rank_str = rank_names[rank_num] if 0 <= rank_num < 8 else f"rank{rank_num}"
    return f"[{tag}] JOIN_CONFIRM: room={room_id} tank={tank_name} {rank_str}"


def _decode_command(body: bytes, tag: str, magic: str | None = None) -> str:
    """Decode a '!' prefixed command message."""
    if len(body) < 3:
        return f"[{tag}] CMD: ! (too short: {body.hex()})"

    # XOR decrypt if magic is available
    if magic:
        # Load static key (assuming same directory as this file)
        static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            static_key = _test_hooks.read_text(static_key_path).strip()
            # Build table
            table = bytearray(len(static_key))
            for i in range(len(static_key)):
                table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])

            # Decrypt
            decrypted = bytearray(len(body))
            decrypted[0] = body[0]  # '!'
            for i in range(1, len(body)):
                decrypted[i] = body[i] ^ table[i - 1]

            cmd_type = decrypted[1]
            cmd_id = decrypted[2]

            # Decode movement commands (type=4) with coordinates
            if cmd_type == 4 and len(decrypted) >= 5:
                x = decrypted[3]
                y = decrypted[4]
                cmd_name = {112: "MOVE", 106: "PICKUP", 116: "TELEPORT"}.get(cmd_id, "?")
                return f"[{tag}] {cmd_name}: ({x}, {y})"

            # Decode shoot commands (type=6) with target
            if cmd_type == 6 and len(decrypted) >= 5:
                x = decrypted[3]
                y = decrypted[4]
                return f"[{tag}] SHOOT: ({x}, {y})"

            return f"[{tag}] CMD: ! type={cmd_type} id={cmd_id}"

    # Fallback to hex if no magic or decrypt failed
    return f"[{tag}] CMD: ! {body.hex()}"


class SnifferError(Exception):
    """Raised when sniffer encounters an error."""


class WebSocketSniffer(BrowserSession):
    """Captures WebSocket traffic from a browser session.

    Extends BrowserSession with live decoding and script URL logging.
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        live_decode: bool = False,
        prefer_account: bool = False,
    ) -> None:
        """Initialize the sniffer.

        Args:
            target_url: URL to navigate to and capture WebSocket traffic from.
            headless: Whether to run the browser in headless mode.
            live_decode: Whether to print decoded messages in real-time.
            prefer_account: Skip guest login and use account credentials directly.
        """
        super().__init__(target_url, headless=headless, prefer_account=prefer_account)
        self._live_decode = live_decode
        self._game_log_entries: list[dict[str, str | int]] = []

    def _process_game_log_entry(self, entry: GameLogEntry) -> None:
        """Process a single game log entry and store it.

        Overrides BrowserSession._process_game_log_entry to also save entries.

        Args:
            entry: The game log entry to process.
        """
        # Store with timestamp
        from tankpit_bot.browser import get_current_time_ms

        self._game_log_entries.append(
            {
                "timestamp_ms": get_current_time_ms(),
                "text": entry["text"],
                "category": entry["category"],
            }
        )
        # Call parent to log and process combat
        super()._process_game_log_entry(entry)

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Log decoded message if live decode is enabled.

        Also polls game log and inventory for correlation with WebSocket events.

        Args:
            message: The captured message.
        """
        if self._magic:
            _init_trackers_with_magic(self._magic)

        if not self._live_decode:
            return

        payload = message["payload"]
        direction = message["direction"]

        if direction == "received":
            # Use unified decoder for received messages
            _process_received_message(payload)
        else:
            # Use simple decoder for sent messages
            mine_status = _mine_tracker.process_message(payload, "sent")
            if mine_status:
                log.info(mine_status)
            decoded = _decode_message(payload, direction, self._magic)
            log.info(decoded)

        self._poll_game_log()
        self._poll_inventory()

    def _probe_js_fuel(self, cdp: _test_hooks.CDPSessionProtocol) -> None:
        """Probe JavaScript for fuel/HP variables.

        Args:
            cdp: CDP session for JS execution.
        """
        # Search ALL numeric properties in fuel range (800-1600)
        js_probe = """
        (function() {
            const results = [];

            function search(obj, path, depth) {
                if (depth > 4 || !obj) return;
                try {
                    for (const key in obj) {
                        try {
                            const val = obj[key];
                            const fullPath = path + '.' + key;
                            if (typeof val === 'number' && val >= 800 && val <= 1600 &&
                                Number.isInteger(val)) {
                                results.push({path: fullPath, value: val});
                            }
                            if (typeof val === 'object' && val !== null &&
                                !(val instanceof HTMLElement) &&
                                !(val instanceof Window) &&
                                !Array.isArray(val) && depth < 3) {
                                search(val, fullPath, depth + 1);
                            }
                        } catch(e) {}
                    }
                } catch(e) {}
            }

            // Search common game objects
            const names = ['game', 'Game', 'g', 'G', 'player', 'Player', 'p', 'P',
                          'tank', 'Tank', 't', 'T', 'state', 's', 'S', 'me', 'my',
                          'ui', 'UI', 'hud', 'HUD', 'data', 'd', 'D', 'app', 'App'];
            for (const name of names) {
                if (window[name]) search(window[name], name, 0);
            }

            return results.slice(0, 50);  // Limit results
        })()
        """
        result = cdp.send("Runtime.evaluate", {"expression": js_probe, "returnByValue": True})
        result_obj = result.get("result")
        if not isinstance(result_obj, dict):
            log.info("No JS variables found in fuel range 800-1600")
            return
        findings = result_obj.get("value")
        if not isinstance(findings, list):
            log.info("No JS variables found in fuel range 800-1600")
            return
        if not findings:
            log.info("No JS variables found in fuel range 800-1600")
            return
        log.info("JS variables in fuel range (800-1600):")
        for item in findings:
            if isinstance(item, dict):
                path = item.get("path", "?")
                value = item.get("value", "?")
                log.info("  %s = %s", path, value)

    def run(self, capture_duration_ms: int) -> CaptureSession:
        """Run the sniffer and capture WebSocket traffic.

        Args:
            capture_duration_ms: How long to capture traffic in milliseconds.
                                 0 = wait until browser closed.

        Returns:
            CaptureSession containing all captured messages.

        Raises:
            PlaywrightNotInstalledError: If Playwright hook is not installed.
        """
        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError("Playwright is not installed.")

        self._start_timestamp_ms = get_current_time_ms()
        self._messages = []
        self._ws_urls = {}
        self._magic = None

        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self._headless)
            context = browser.new_context()
            page = context.new_page()
            cdp = context.new_cdp_session(page)

            # Reset CDP time offset for new session
            reset_cdp_time_offset()

            # Set up console listener and CDP handlers
            self._setup_console_listener(cdp)
            self._setup_cdp_handlers(cdp)

            # Navigate to target URL
            page.goto(self._target_url, wait_until="domcontentloaded")
            log.info("Landed on %s", page.url)

            # Handle login
            self._navigate_and_login(page, cdp, tank_name_prefix="B", auto_join_room=False)

            # Gather all available intel
            self._gather_intel(page, cdp)

            # Initialize DOM scrapers for log and inventory capture
            self._init_game_log_scraper(cdp)
            self._init_inventory_scraper(cdp)
            self._init_combat_tracker()

            # Probe for JS fuel variables
            self._probe_js_fuel(cdp)

            # Wait for specified capture duration (0 = wait until browser closed)
            if capture_duration_ms <= 0:
                log.info("Waiting indefinitely for browser close...")
                # We can't easily run capture_loop in background with sync playwright
                # without blocking. But _capture_magic_key is called in _navigate_and_login.
                # Let's just ensure it's captured once we are in game.
                page.wait_for_event("close", timeout=86_400_000)
            else:
                log.info("Waiting for %d ms...", capture_duration_ms)
                page.wait_for_timeout(float(capture_duration_ms))
                self._cleanup(cdp, page, context, browser)

        # Get tank names from tracker
        tank_names = {str(k): v for k, v in _tank_tracker.get_all_names().items()}

        # Convert game log entries to proper type
        from tankpit_bot.types import GameLogEntryWithTimestamp

        game_log: list[GameLogEntryWithTimestamp] = [
            GameLogEntryWithTimestamp(
                timestamp_ms=int(e["timestamp_ms"]),
                text=str(e["text"]),
                category=str(e["category"]),
            )
            for e in self._game_log_entries
        ]

        return CaptureSession(
            session_id=self._session_id,
            start_timestamp_ms=self._start_timestamp_ms,
            end_timestamp_ms=get_current_time_ms(),
            base_url=self._target_url,
            messages=self._messages,
            magic=self._magic,
            game_log=game_log,
            tank_names=tank_names,
        )


# Known decoded signatures with understanding level
# FULL = complete decoder, PARTIAL = key fields known, IDENTIFIED = type known
DECODED_SIGS: dict[int, tuple[str, str]] = {
    # Binary control messages (0x00-0x1F)
    0x00: ("sync_state", "IDENTIFIED"),
    0x01: ("heartbeat", "IDENTIFIED"),
    0x04: ("position_update", "IDENTIFIED"),
    0x08: ("entity_state", "IDENTIFIED"),
    0x0E: ("tick_update", "IDENTIFIED"),
    0x14: ("world_state", "IDENTIFIED"),
    0x15: ("spawn_state", "IDENTIFIED"),
    0x1D: ("combat_state", "IDENTIFIED"),
    # ASCII message types (0x20+)
    0x21: ("tank_info", "FULL"),
    0x22: ("entity_position", "IDENTIFIED"),  # '"' 13-byte position update
    0x28: ("tank_join", "IDENTIFIED"),
    0x29: ("tank_leave", "IDENTIFIED"),
    0x2B: ("promotion", "FULL"),
    0x2D: ("world_entity", "IDENTIFIED"),  # '-' 16-byte entity state
    0x2E: ("tank_status_sync", "PARTIAL"),
    0x2F: ("player_update", "IDENTIFIED"),
    0x31: ("top10_list", "IDENTIFIED"),  # '1' MSG_TOP10
    0x32: ("top10_extended", "IDENTIFIED"),  # '2' similar to top10
    0x33: ("score_update", "IDENTIFIED"),  # '3'
    0x3D: ("movement", "FULL"),
    0x3E: ("tank_status", "PARTIAL"),
    0x3F: ("position", "FULL"),
    0x40: ("mine_status", "IDENTIFIED"),  # '@' MSG_MINE_STATUS
    0x41: ("kill", "FULL"),
    0x43: ("container", "FULL"),
    0x45: ("mine_detonate", "FULL"),
    0x46: ("radar_ack", "FULL"),
    0x47: ("shooting", "FULL"),
    0x49: ("item_pickup", "FULL"),
    0x4A: ("terrain_update", "IDENTIFIED"),  # 'J' MSG_TERRAIN_UPDATE
    0x4B: ("mine_place", "FULL"),
    0x4C: ("tank_entry", "PARTIAL"),
    0x4D: ("player_list", "IDENTIFIED"),
    0x4F: ("deactivation", "FULL"),
    0x52: ("supervisor", "PARTIAL"),
    0x53: ("tank_move", "FULL"),
    0x54: ("tank_shoot", "FULL"),
    0x56: ("statistics", "FULL"),
    0x58: ("tank_exit", "FULL"),
    0x5A: ("viewport_update", "PARTIAL"),
    0x5F: ("action_event", "IDENTIFIED"),  # '_' 11-byte action/event
    0x64: ("fuel_deposit", "FULL"),
    0x66: ("fuel_state", "IDENTIFIED"),  # 'f' - lowercase variant
    0x67: ("equip_gain", "FULL"),
    0x69: ("inventory_state", "IDENTIFIED"),  # 'i' - lowercase variant
    0x74: ("equip_toggle", "FULL"),
    0x78: ("tank_disconnect", "IDENTIFIED"),  # 'x' - lowercase variant
    0x79: ("entity_spawn", "IDENTIFIED"),  # 'y'
    0x7A: ("zone_update", "IDENTIFIED"),  # 'z' - lowercase variant
}


def _empty_message_stats() -> MessageStats:
    """Return empty MessageStats."""
    return MessageStats(decoded={}, unknown={}, total_received=0, decode_coverage="0%")


def _is_valid_base64(s: str) -> bool:
    """Check if string is valid base64.

    Args:
        s: String to check.

    Returns:
        True if valid base64, False otherwise.
    """
    import re

    if not s:
        return False
    # Base64 characters plus padding
    pattern = r"^[A-Za-z0-9+/]*={0,2}$"
    if not re.match(pattern, s):
        return False
    # Length must be multiple of 4 (with padding)
    return len(s) % 4 == 0


def _decode_base64_safe(payload: str) -> bytes | None:
    """Validate and decode base64 payload without exceptions.

    Args:
        payload: Base64 encoded string.

    Returns:
        Decoded bytes or None if invalid.
    """
    if not _is_valid_base64(payload):
        return None
    return base64.b64decode(payload)


def _extract_message_signature(payload_b64: str, xor_table: bytes) -> bytes | None:
    """Extract and decode message signature from base64 payload.

    Args:
        payload_b64: Base64 encoded payload.
        xor_table: XOR decryption table.

    Returns:
        Decoded bytes or None if invalid format.
    """
    if not _is_valid_base64(payload_b64):
        return None

    payload = base64.b64decode(payload_b64)

    if b"." not in payload[:3]:
        return None

    dot_pos = payload.find(b".")
    if dot_pos < 0 or dot_pos >= 3:
        return None

    start = dot_pos + 1
    if len(payload) <= start:
        return None

    decode_len = min(len(payload) - start, len(xor_table))
    decoded = bytes(payload[start + j] ^ xor_table[j] for j in range(decode_len))
    return decoded if decoded else None


def _format_sig_key(sig: int) -> str:
    """Format signature as display key."""
    char = chr(sig) if 32 <= sig < 127 else "?"
    return f"0x{sig:02X} '{char}'"


# Exact length -> (name, level) mappings for message identification
_LENGTH_EXACT: dict[int, tuple[str, str]] = {
    1: ("heartbeat", "IDENTIFIED"),
    2: ("tank_status_sync", "FULL"),
    3: ("tank_status_sync", "FULL"),
    4: ("player_ack", "IDENTIFIED"),
    5: ("entity_sync", "IDENTIFIED"),
    6: ("control_msg", "IDENTIFIED"),
    7: ("action_ack", "IDENTIFIED"),
    9: ("tank_status_short", "FULL"),
    10: ("tank_update_compact", "FULL"),
    11: ("combat_hit", "FULL"),
    13: ("position_update", "FULL"),
    14: ("tank_update_extended", "FULL"),
    15: ("tank_update_full", "FULL"),
    16: ("tank_registry", "FULL"),
    17: ("tank_registry", "FULL"),
    18: ("tank_registry", "FULL"),
    19: ("tank_registry", "FULL"),
    20: ("tank_registry", "FULL"),
}

# Range-based length patterns: (min, max, name, level)
_LENGTH_RANGES: tuple[tuple[int, int, str, str], ...] = (
    (21, 28, "entity_extended", "IDENTIFIED"),
    (29, 60, "tip_notification", "IDENTIFIED"),
    (80, 130, "chunk_data", "IDENTIFIED"),
    (500, 100000, "world_state", "IDENTIFIED"),
)


def _identify_by_length(data: bytes) -> tuple[str, str] | None:
    """Identify message type by length (session-independent).

    Uses the same length-based matching as container_decoder for consistency.
    Length patterns are derived from captured session analysis.

    Args:
        data: XOR-decoded message bytes.

    Returns:
        Tuple of (name, level) if identified, None otherwise.
    """
    length = len(data)

    # Check exact length matches first
    if length in _LENGTH_EXACT:
        return _LENGTH_EXACT[length]

    # Check range-based patterns
    for min_len, max_len, name, level in _LENGTH_RANGES:
        if min_len <= length <= max_len:
            return (name, level)

    return None


def _build_message_stats(session: CaptureSession) -> MessageStats:
    """Build message statistics from captured session.

    Uses LENGTH-BASED identification for XOR-decoded messages, which is
    session-independent (unlike byte-based matching that varies per session).

    Args:
        session: The capture session to analyze.

    Returns:
        MessageStats with decoded vs unknown breakdown.
    """
    from collections import Counter

    magic = session.get("magic")
    if not magic:
        return _empty_message_stats()

    static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
    if not _test_hooks.path_exists(static_key_path):
        return _empty_message_stats()

    static_key = _test_hooks.read_text(static_key_path).strip()
    magic_bytes = magic.encode("utf-8")
    xor_table = bytes(
        ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
    )

    decoded_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    unknown_samples: dict[str, list[str]] = {}
    level_counts: Counter[str] = Counter()

    for msg in session["messages"]:
        if msg["direction"] != "received":
            continue

        decoded = _extract_message_signature(msg["payload"], xor_table)
        if decoded is None:
            continue

        # Use length-based identification (session-independent)
        result = _identify_by_length(decoded)
        if result is not None:
            name, level = result
            decoded_counts[f"len={len(decoded):02d} {name}"] += 1
            level_counts[level] += 1
        else:
            len_key = f"len={len(decoded):02d}"
            unknown_counts[len_key] += 1
            if len_key not in unknown_samples:
                unknown_samples[len_key] = []
            if len(unknown_samples[len_key]) < 3:
                unknown_samples[len_key].append(decoded[:20].hex())

    total = sum(decoded_counts.values()) + sum(unknown_counts.values())
    decoded_total = sum(decoded_counts.values())

    if total > 0:
        sig_coverage = 100 * decoded_total // total
        weighted = level_counts["FULL"] * 100 + level_counts["PARTIAL"] * 50
        weighted += level_counts["IDENTIFIED"] * 25
        understanding = weighted // total
        coverage = f"{sig_coverage}% sig, {understanding}% understood"
    else:
        coverage = "0%"

    unknown_dict: dict[str, dict[str, int | list[str]]] = {
        k: {"count": v, "samples": unknown_samples.get(k, [])} for k, v in unknown_counts.items()
    }

    return MessageStats(
        decoded=dict(decoded_counts),
        unknown=unknown_dict,
        total_received=total,
        decode_coverage=coverage,
    )


def _build_session_summary(session: CaptureSession) -> SessionSummary:
    """Build session summary from capture session.

    Args:
        session: The raw capture session.

    Returns:
        Processed SessionSummary.
    """
    # Extract combat events from game log
    combat: list[CombatEvent] = []
    combat_log = [e for e in session["game_log"] if e["category"] == "combat"]

    for entry in combat_log:
        text = entry["text"]
        event_type = "unknown"
        target = ""

        if text.startswith("You hit "):
            event_type = "hit"
            target = text[8:]
        elif text.startswith("You killed "):
            event_type = "kill"
            target = text[11:]
        elif " hit you" in text:
            event_type = "hit_by"
            target = text.split(" hit you")[0]
        elif " killed you" in text:
            event_type = "killed_by"
            target = text.split(" killed you")[0]

        if event_type != "unknown":
            combat.append(
                CombatEvent(
                    timestamp_ms=entry["timestamp_ms"],
                    event_type=event_type,
                    target=target,
                    tank_id=None,
                )
            )

    return SessionSummary(
        session_id=session["session_id"],
        start_timestamp_ms=session["start_timestamp_ms"],
        end_timestamp_ms=session["end_timestamp_ms"],
        magic=session["magic"],
        tanks=session["tank_names"],
        combat=combat,
        equipment_gains=[],
        game_log=combat_log,
        message_stats=_build_message_stats(session),
    )


def run_sniffer(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    capture_duration_ms: int = 30000,
    live_decode: bool = False,
    prefer_account: bool = False,
) -> CaptureSession:
    """Run the WebSocket sniffer and save results.

    Args:
        target_url: URL to navigate to and capture WebSocket traffic from.
        output_path: Path to save the capture session JSON.
        headless: Whether to run the browser in headless mode.
        capture_duration_ms: How long to capture traffic in milliseconds.
        live_decode: Whether to print decoded messages in real-time.
        prefer_account: Skip guest login and use account credentials directly.

    Returns:
        The completed CaptureSession.

    Raises:
        PlaywrightNotInstalledError: If Playwright is not installed.
    """
    sniffer = WebSocketSniffer(
        target_url, headless=headless, live_decode=live_decode, prefer_account=prefer_account
    )
    session = sniffer.run(capture_duration_ms)

    # Save raw capture (all messages for protocol research)
    output_dir = Path(output_path).parent
    raw_path = output_dir / "raw_capture.json"
    encoded = encode_capture_session(session)
    json_str = dump_json_str(encoded, compact=False, indent=2)
    _test_hooks.write_text(raw_path, json_str)
    log.info("Saved raw capture to %s", raw_path)

    # Save session summary (processed data)
    summary_path = output_dir / "session_summary.json"
    summary = _build_session_summary(session)
    summary_json = dump_json_str(encode_session_summary(summary), compact=False, indent=2)
    _test_hooks.write_text(summary_path, summary_json)
    log.info("Saved session summary to %s", summary_path)

    # Also save to legacy path for backwards compatibility
    _test_hooks.write_text(Path(output_path), json_str)

    return session


def main() -> None:
    """Entry point for tankpit-sniff command."""
    from dotenv import load_dotenv

    load_dotenv()
    setup_rich_logging(level="INFO")

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    target_url = _test_hooks.get_env("TANKPIT_URL") or DEFAULT_TARGET_URL
    output_path = _test_hooks.get_env("TANKPIT_OUTPUT") or DEFAULT_OUTPUT_PATH

    headless_str = _test_hooks.get_env("TANKPIT_HEADLESS")
    headless = headless_str is not None and headless_str.lower() in ("true", "1", "yes")

    duration_str = _test_hooks.get_env("TANKPIT_DURATION_MS")
    capture_duration_ms = int(duration_str) if duration_str else DEFAULT_CAPTURE_DURATION_MS
    log.info("Duration config: env=%s, using=%d ms", duration_str, capture_duration_ms)

    live_decode_str = _test_hooks.get_env("TANKPIT_LIVE_DECODE")
    live_decode = live_decode_str is None or live_decode_str.lower() not in ("false", "0", "no")

    prefer_account_str = _test_hooks.get_env("TANKPIT_PREFER_ACCOUNT")
    prefer_account = prefer_account_str is not None and prefer_account_str.lower() in (
        "true",
        "1",
        "yes",
    )

    session = run_sniffer(
        target_url,
        output_path,
        headless=headless,
        capture_duration_ms=capture_duration_ms,
        live_decode=live_decode,
        prefer_account=prefer_account,
    )

    msg_count = len(session["messages"])
    duration_sec = ((session["end_timestamp_ms"] or 0) - session["start_timestamp_ms"]) / 1000
    log.info("Captured %d WebSocket messages in %.1fs", msg_count, duration_sec)
    log.info("Saved to: %s", output_path)

    unique_urls: set[str] = set()
    for msg in session["messages"]:
        unique_urls.add(msg["ws_url"])

    if len(unique_urls) > 0:
        log.info("Discovered WebSocket URLs (%d):", len(unique_urls))
        for url in sorted(unique_urls):
            log.info("  - %s", url)


__all__ = [
    "PlaywrightNotInstalledError",
    "SnifferError",
    "WebSocketSniffer",
    "_decode_command",
    "_decode_join_confirm",
    "_decode_message",
    "_decode_plus_message",
    "_decode_state_message",
    "_decode_text_message",
    "main",
    "run_sniffer",
]
