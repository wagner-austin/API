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
from pathlib import Path

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.browser import (
    BrowserSession,
    PlaywrightNotInstalledError,
    get_current_time_ms,
)
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
        if len(body) < 6 or body[0] != 0x2e:
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
        return len(body) == 5 and body[0] == 0x2e

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
        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Check for blocked movement (5-byte response)
        if self.is_blocked_response(body):
            return "[POS:BLOCKED]"

        # Check for movement response (17-21 bytes)
        if len(body) < 17 or len(body) > 21 or body[0] != 0x2e:
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

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
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

    def process_message(self, payload: str) -> str | None:
        """Process a message and return item pickup status if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Item pickup status string, or None if not an item pickup message.
        """
        if self._xor_table is None:
            return None

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Item pickup messages are 8 bytes: 0x2E + variable subtype + 6 data bytes
        if len(body) != 8 or body[0] != 0x2E:
            return None

        # XOR decode from byte 1 (7 bytes total)
        decoded = bytearray(7)
        for i in range(7):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        # Item pickups decode to: 67 01 [armor] [dual] [missile] [homing] [radar]
        # Equipment indices from client JS gc array:
        # 0=armor shield, 1=dual shot, 2=missile shot, 3=homing shot, 4=extra radar
        # The signature 0x67 0x01 identifies this as an item pickup
        if decoded[0] != 0x67 or decoded[1] != 0x01:
            return None

        armor = decoded[2]    # Index 0: armor shield
        dual = decoded[3]     # Index 1: dual shot
        missile = decoded[4]  # Index 2: missile shot
        homing = decoded[5]   # Index 3: homing shot
        radar = decoded[6] if len(decoded) > 6 else 0  # Index 4: extra radar

        if armor == 0 and dual == 0 and missile == 0 and homing == 0 and radar == 0:
            return None

        self._total_armor += armor
        self._total_missile += missile
        self._total_homing += homing

        items = []
        if armor:
            items.append(f"{armor} armor")
        if dual:
            items.append(f"{dual} dual")
        if missile:
            items.append(f"{missile} missile")
        if homing:
            items.append(f"{homing} homing")
        if radar:
            items.append(f"{radar} radar")

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

    def process_message(self, payload: str) -> str | None:
        """Process a message and return radar results if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Radar results string, or None if not a radar result message.
        """
        if self._xor_table is None:
            return None

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Radar result messages start with 0x2E 0x70
        if len(body) < 4 or body[0] != 0x2E or body[1] != 0x70:
            return None

        # XOR decode from byte 1
        decoded = bytearray(len(body) - 1)
        for i in range(len(decoded)):
            if i < len(self._xor_table):
                decoded[i] = body[i + 1] ^ self._xor_table[i]
            else:
                decoded[i] = body[i + 1]

        # Format: 0x4F [count] 0x00 [records...]
        if decoded[0] != 0x4F:
            return None

        count = decoded[1]
        if count == 0:
            return "[RADAR] No entities found"

        # Parse entity records (4 bytes each)
        # Cache value interpretation (from client JS):
        # - Positive (0x0000-0x7FFF): fuel containers with amount
        # - Negative (0x8000-0xFFFE): equipment (abs value = type?)
        # - 0xFFFF: tank/entity
        records = decoded[3:]  # Skip marker, count, unknown byte
        fuel_containers = []
        equipment = []
        tanks = []

        for i in range(0, min(len(records) - 3, count * 4), 4):
            x = records[i]
            y = records[i + 1]
            val_lo = records[i + 2]
            val_hi = records[i + 3]
            val_unsigned = val_lo | (val_hi << 8)

            if val_unsigned == 0xFFFF:
                tanks.append(f"({x},{y})")
            elif val_unsigned >= 0x8000:
                # Negative value = equipment
                # Convert to signed: val_signed = val_unsigned - 0x10000
                val_signed = val_unsigned - 0x10000
                equipment.append(f"({x},{y})={abs(val_signed)}")
            else:
                fuel_containers.append(f"({x},{y})={val_unsigned}")

        parts = []
        if fuel_containers:
            parts.append(f"fuel: {' '.join(fuel_containers)}")
        if equipment:
            parts.append(f"equip: {' '.join(equipment)}")
        if tanks:
            parts.append(f"tanks: {' '.join(tanks)}")

        return f"[RADAR] {count} found - {'; '.join(parts)}"


# Team color names from client JS
TEAM_COLORS = ["red", "purple", "blue", "orange"]
# Rank names from client JS
RANK_NAMES = ["recruit", "private", "corporal", "sergeant", "lieutenant", "captain", "major", "general"]


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
        self._tanks: dict[int, dict] = {}  # tank_id -> info

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
        """Process a message and return tank info if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Tank info string, or None if not a tank-related message.
        """
        if self._xor_table is None:
            return None

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
            return None

        if len(data) < 4:
            return None

        body = data[2:]
        if len(body) < 2:
            return None

        msg_type = body[0]

        # XOR decode from byte 1
        max_decode = min(len(body) - 1, len(self._xor_table))
        decoded = bytearray(len(body) - 1)
        for i in range(len(decoded)):
            if i < max_decode:
                decoded[i] = body[i + 1] ^ self._xor_table[i]
            else:
                decoded[i] = body[i + 1]

        # Check decoded signature byte
        if len(decoded) < 1:
            return None

        sig = decoded[0]

        # Tank Entry: decoded starts with 0x28 '('
        if sig == 0x28 and len(decoded) >= 4:
            return self._parse_tank_join(decoded)

        # Tank Leave: decoded starts with 0x29 ')'
        if sig == 0x29 and len(decoded) >= 4:
            return self._parse_tank_leave(decoded)

        # Tank Status: decoded starts with 0x3E '>'
        if sig == 0x3E and len(decoded) >= 13:
            return self._parse_tank_status(decoded)

        # Movement Response: decoded starts with 0x3D '='
        # Verified format from Arterial (lieutenant) and Artax (major)
        if sig == 0x3D and len(decoded) >= 9:
            return self._parse_movement_response(decoded)

        # Movement: decoded starts with 0x47 'G'
        if sig == 0x47 and len(decoded) >= 5:
            return self._parse_movement(decoded)

        # Shooting: decoded starts with 0x53 'S'
        if sig == 0x53 and len(decoded) >= 5:
            return self._parse_shooting(decoded)

        # Tank Info: decoded starts with 0x21 '!' - contains tank_id -> name mapping
        if sig == 0x21 and len(decoded) >= 12:
            return self._parse_tank_info(decoded)

        # Player List: decoded starts with 0x4D 'M' - active players with ranks
        if sig == 0x4D and len(decoded) >= 6:
            return self._parse_player_list(decoded)

        # Player Update: decoded starts with 0x2F '/' - active players update
        if sig == 0x2F and len(decoded) >= 4:
            return self._parse_player_update(decoded)

        # Statistics: decoded starts with 0x56 'V' - player stats response
        if sig == 0x56 and len(decoded) >= 14:
            return self._parse_statistics(decoded)

        # Promotion: decoded starts with 0x2B '+' - rank promotion
        if sig == 0x2B and len(decoded) >= 3:
            return self._parse_promotion(decoded)

        # Supervisor message: 0x52 'R' - purpose unknown
        if sig == 0x52 and len(decoded) >= 4:
            return self._parse_supervisor_msg(decoded)

        # Tank status sync: 0x2E '.' - periodic state update
        if sig == 0x2E and len(decoded) >= 9:
            return self._parse_status_sync(decoded)

        return None

    def _parse_tank_join(self, decoded: bytearray) -> str:
        """Parse tank join message (0x28 '(').

        Format from sample 280056027c0009770000:
        - [0] = 0x28 signature
        - [1] = subtype/flags
        - [2-3] = tank_id (u16 LE)
        - [4+] = additional data (position, rank, etc.)
        """
        if len(decoded) < 4:
            return None
        tank_id = decoded[2] | (decoded[3] << 8)
        name = self.get_name(tank_id)
        name_str = f"'{name}'" if name else f"id={tank_id}"
        extra = decoded[4:].hex() if len(decoded) > 4 else ""
        return f"[JOIN] {name_str} data={extra}"

    def _parse_tank_leave(self, decoded: bytearray) -> str:
        """Parse tank leave message (0x29 ')').

        Format from sample 290056020000:
        - [0] = 0x29 signature
        - [1] = subtype/flags
        - [2-3] = tank_id (u16 LE)
        - [4+] = additional data
        """
        if len(decoded) < 4:
            return None
        tank_id = decoded[2] | (decoded[3] << 8)
        name = self.get_name(tank_id)
        name_str = f"'{name}'" if name else f"id={tank_id}"
        extra = decoded[4:].hex() if len(decoded) > 4 else ""
        return f"[LEAVE] {name_str} data={extra}"

    def _parse_tank_status(self, decoded: bytearray) -> str:
        """Parse tank status message (0x3E).

        Verified format from capture analysis:
        - Byte 0: 0x3E signature
        - Byte 1: info byte (rank/team encoded)
        - Bytes 2-3: tank_id (u16 LE)
        - Bytes 4-13: equipment/stats data
        - Bytes 14+: tank name (null-terminated string)
        """
        if len(decoded) < 14:
            return None

        info_byte = decoded[1]
        # Team in lower 2 bits, rank in bits 4-6
        team = info_byte & 0x03
        rank = (info_byte >> 4) & 0x07

        tank_id = decoded[2] | (decoded[3] << 8)

        # Name starts at byte 14
        name = ""
        if len(decoded) > 14:
            try:
                name = bytes(decoded[14:]).decode("utf-8", errors="ignore").rstrip("\x00")
            except Exception:
                pass

        team_name = TEAM_COLORS[team] if team < len(TEAM_COLORS) else f"team{team}"
        rank_name = RANK_NAMES[rank] if rank < len(RANK_NAMES) else f"rank{rank}"

        self._tanks[tank_id] = {"team": team_name, "rank": rank_name, "name": name}

        name_str = f" '{name}'" if name else ""
        return f"[TANK:STATUS] id={tank_id}{name_str} {team_name} {rank_name}"

    def _parse_movement_response(self, decoded: bytearray) -> str:
        """Parse movement response message (0x3D).

        Verified format from real players (Arterial, Artax, Yuppler):
        - [0] = 0x3D signature
        - [1] = team
        - [2-3] = tank_id (u16 LE)
        - [4] = x
        - [5] = y
        - [6] = direction
        - [7] = unknown (values: 1, 2, 3)
        - [8] = rank (0-7)
        - [9-11] = score (u24 BE) - verified: Artax 4586, Yuppler 12733
        """
        if len(decoded) < 12:
            return None

        team = decoded[1]
        tank_id = decoded[2] | (decoded[3] << 8)
        x = decoded[4]
        y = decoded[5]
        direction = decoded[6]
        rank = decoded[8]
        lb_pos = (decoded[9] << 16) | (decoded[10] << 8) | decoded[11]

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

    def _parse_movement(self, decoded: bytearray) -> str:
        """Parse movement message (0x47)."""
        # Based on Lg.h: bytes 0-1 tank_id, byte 2 x, byte 3 y, byte 4 direction
        if len(decoded) < 5:
            return None
        tank_id = decoded[1] | (decoded[2] << 8)
        x = decoded[3]
        y = decoded[4]
        direction = decoded[5] if len(decoded) > 5 else 0

        # Update tank position
        if tank_id in self._tanks:
            self._tanks[tank_id]["x"] = x
            self._tanks[tank_id]["y"] = y
            name = self._tanks[tank_id].get("name", "")
            if name:
                return f"[TANK:MOVE] {name} (id={tank_id}) to ({x},{y}) dir={direction}"

        return f"[TANK:MOVE] id={tank_id} to ({x},{y}) dir={direction}"

    def _parse_shooting(self, decoded: bytearray) -> str:
        """Parse shooting message (0x53)."""
        # Based on Gg.h: byte 0 shooter_team, bytes 1-2 shooter_id, byte 3 x, byte 4 y
        if len(decoded) < 5:
            return None
        shooter_team = decoded[1]
        shooter_id = decoded[2] | (decoded[3] << 8)
        shot_x = decoded[4]
        shot_y = decoded[5] if len(decoded) > 5 else 0

        team_name = TEAM_COLORS[shooter_team] if shooter_team < len(TEAM_COLORS) else f"team{shooter_team}"

        # Get shooter name if known
        if shooter_id in self._tanks:
            name = self._tanks[shooter_id].get("name", "")
            if name:
                return f"[TANK:SHOT] {name} ({team_name}) fired from ({shot_x},{shot_y})"

        return f"[TANK:SHOT] id={shooter_id} ({team_name}) fired from ({shot_x},{shot_y})"

    def _parse_tank_info(self, decoded: bytearray) -> str | None:
        """Parse tank info message (0x21) - contains tank_id -> name mapping.

        Format (from JS client analysis):
        - [0] = 0x21 signature
        - [1] = team (0=red, 1=purple, 2=blue, 3=orange)
        - [2-3] = tank_id (u16 LE)
        - [4-7] = decoration_state (4 bytes = 9 2-bit values)
        - [8-10] = score (24-bit BE)
        - [11+] = name (UTF-8)

        NOTE: This message does NOT contain the tank's current rank!
        """
        if len(decoded) < 12:
            return None

        tank_id = decoded[2] | (decoded[3] << 8)

        # Extract name from byte 11 onwards
        name = ""
        for b in decoded[11:]:
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

        Format: 6 bytes per entry
        - [0] = 0x4D signature
        - [1-2] = tank_id (u16 LE)
        - [3-5] = position or rank data
        """
        tank_id = decoded[1] | (decoded[2] << 8)
        name = self.get_name(tank_id)
        b3 = decoded[3]
        b4 = decoded[4]
        b5 = decoded[5]
        name_str = f"'{name}'" if name else f"id={tank_id}"
        return f"[PLAYERS] {name_str} data={b3:02x} {b4:02x} {b5:02x}"

    def _parse_player_update(self, decoded: bytearray) -> str:
        """Parse player update message (0x2F '/').

        Format: variable length, 3 bytes per entry after signature
        - [0] = 0x2F signature
        - Then repeating: tank_id_lo, tank_id_hi, data_byte
        """
        entries = []
        i = 1
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

        Format verified against DOM scrape - 15 bytes:
        - [0] = 0x56 signature
        - [1-2] = hours (u16 LE)
        - [3] = minutes
        - [4] = seconds
        - [5-7] = padding
        - [8] = destroyed (single byte)
        - [9] = deactivated (single byte)
        - [10-12] = padding
        - [13-14] = promotion points (u16 BE)
        """
        hours = decoded[1] | (decoded[2] << 8)
        mins = decoded[3]
        secs = decoded[4]
        destroyed = decoded[8]
        deactivated = decoded[9]
        promo_pts = (decoded[13] << 8) | decoded[14] if len(decoded) > 14 else 0
        return f"[STATS] {hours}h{mins}m{secs}s destroyed={destroyed} deactivated={deactivated} promo={promo_pts}"

    def _parse_promotion(self, decoded: bytearray) -> str:
        """Parse promotion message (0x2B '+').

        Format verified from JS (Rf class) - 3 bytes:
        - [0] = 0x2B signature
        - [1] = new rank level (0-8)
        - [2] = promoted flag (1 = promoted, 0 = rank set)

        Ranks: 0=recruit, 1=private, 2=corporal, 3=sergeant,
               4=lieutenant, 5=captain, 6=major, 7=colonel, 8=general
        """
        ranks = ["recruit", "private", "corporal", "sergeant", "lieutenant",
                 "captain", "major", "colonel", "general"]
        rank_idx = decoded[1]
        promoted = decoded[2] == 1
        rank_name = ranks[rank_idx] if rank_idx < len(ranks) else f"rank{rank_idx}"
        if promoted:
            return f"[PROMOTED] to {rank_name}!"
        return f"[DEMOTED] to {rank_name}"

    def _parse_supervisor_msg(self, decoded: bytearray) -> str:
        """Parse supervisor message (0x52 'R').

        Format: 4 bytes
        - [0] = 0x52 signature
        - [1] = always 0x01
        - [2] = always 0x00
        - [3] = status value (seen: 4, 7, 8)

        JS class: xg (supervisor message)

        Testing observations:
        - NOT a timer/heartbeat (5 min idle = zero messages)
        - NOT triggered by movement, radar, equipment, or teamchat
        - Inconsistent with combat (some sessions with combat have it, some don't)
        - Sometimes appears near other tanks, sometimes doesn't
        - Value changes: 8->4 after deactivation, 4->7 seen occasionally

        Trigger is UNPREDICTABLE - may be server-side state we can't control.
        Possibly: server load balancing, anti-cheat sampling, or random sync.

        Status values (4, 7, 8) meaning unknown.
        """
        status = decoded[3]
        return f"[SUPERVISOR] status={status}"

    def _parse_status_sync(self, decoded: bytearray) -> str:
        """Parse tank status sync message (0x2E '.').

        Periodic state update from server (Og class in JS).

        Format 13 bytes (subtype 0x03, about self):
        - [0] = 0x2E signature
        - [1] = 0x03 subtype
        - [2-3] = tank_id (u16 LE)
        - [4-6] = status flags
        - [7-8] = leaderboard_rank (u16 BE) - position on leaderboard (1 = top)
        - [9-10] = unknown
        - [11-12] = fuel (u16 LE)

        Note: Promotion points (toward next rank) come from 0x56 STATS message,
        not from this message. The rank here is leaderboard position.

        Format 9 bytes (subtype 0x01, about others):
        - [0] = 0x2E signature
        - [1] = 0x01 subtype
        - [2-3] = tank_id (u16 LE)
        - [4] = damage_state (0=full HP, 1=light, 2=medium, 3=critical)
        - [5] = rank (0-7: recruit to general)
        - [6] = flag (0 or 1)
        - [7-8] = leaderboard_position (u16 LE)

        The damage_state controls how dark the tank name appears in the UI.
        """
        rank_names = ["recruit", "private", "corporal", "sergeant",
                      "lieutenant", "captain", "major", "general"]
        damage_names = ["full", "light", "medium", "critical"]
        subtype = decoded[1]
        tank_id = decoded[2] | (decoded[3] << 8)
        name = self.get_name(tank_id)
        name_str = f"'{name}'" if name else f"id={tank_id}"

        if len(decoded) >= 13 and subtype == 0x03:
            rank_pos = (decoded[7] << 8) | decoded[8]  # BE - leaderboard position
            fuel = decoded[11] | (decoded[12] << 8)  # LE
            return f"[STATUS] {name_str} rank=#{rank_pos} fuel={fuel}"

        # Short format (9 bytes, subtype 0x01) - used for other tanks in viewport
        if len(decoded) == 9 and subtype == 0x01:
            damage_state = decoded[4]  # HP level: 0=full, 3=critical
            rank = decoded[5]
            rank_str = rank_names[rank] if 0 <= rank < 8 else f"rank{rank}"
            damage_str = damage_names[damage_state] if 0 <= damage_state < 4 else f"dmg{damage_state}"
            return f"[STATUS] {name_str} {rank_str} HP={damage_str}"

        # Unknown format - dump raw
        return f"[STATUS] {name_str} data={decoded[4:].hex()}"

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
        if tank_id in self._tanks:
            return self._tanks[tank_id].get("name")
        return None

    def get_all_names(self) -> dict[int, str]:
        """Get all known tank ID to name mappings.

        Returns:
            Dictionary of tank_id -> name.
        """
        return {
            tid: info.get("name", "")
            for tid, info in self._tanks.items()
            if info.get("name")
        }


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

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
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
        # Decrypt command
        decrypted = bytearray(len(body))
        decrypted[0] = body[0]  # '!'
        for i in range(1, len(body)):
            if i - 1 < len(self._xor_table):
                decrypted[i] = body[i] ^ self._xor_table[i - 1]
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

    EQUIPMENT_NAMES = ["armor", "dual", "missile", "homing", "radar"]

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

    def process_message(self, payload: str) -> str | None:
        """Process a message and return equipment toggle status if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Equipment toggle status string, or None if not a toggle message.
        """
        if self._xor_table is None:
            return None

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Equipment toggle messages are 7 bytes: 0x2E + subtype + 5 flag bytes
        if len(body) != 7 or body[0] != 0x2E:
            return None

        # XOR decode from byte 1
        decoded = bytearray(6)
        for i in range(6):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        # Check for 0x74 't' signature
        if decoded[0] != 0x74:
            return None

        # Parse equipment flags
        new_state = [bool(decoded[i + 1]) for i in range(5)]

        # Find what changed
        changes = []
        if self._prev_state is not None:
            for i, (old, new) in enumerate(zip(self._prev_state, new_state)):
                if old != new:
                    status = "ON" if new else "OFF"
                    changes.append(f"{self.EQUIPMENT_NAMES[i]}={status}")

        self._prev_state = self._state
        self._state = new_state

        if changes:
            return f"[EQUIP:TOGGLE] {', '.join(changes)}"

        # First message or no changes - show full state
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

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
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

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
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

    EQUIPMENT_NAMES = ["armor", "dual", "missile", "homing", "radar"]

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

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
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

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
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

        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError):
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


class FuelTracker:
    """Tracks fuel values from 14-byte state messages with XOR decoding.

    Fuel Encoding (verified):
    - 14-byte state messages (0x2e prefix) contain fuel as u16 at bytes 12-13
    - XOR encoded: decoded = (body[12] ^ xor_table[12]) | ((body[13] ^ xor_table[13]) << 8)
    - Subtype byte (body[1]) varies per session due to XOR encoding
    - Entity ID in bytes 2-6 identifies tank/container

    Verified Fuel Costs:
    - Radar (S key): -10 fuel
    - Movement: -1 fuel per tile
    - Fuel deposit: -100 fuel
    - Fuel pickup: +100 fuel
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self._xor_table: bytes | None = None
        self._last_fuel: int | None = None
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
        """Set magic key and build XOR table.

        Args:
            magic: Session magic string (typically 20 chars).
        """
        static_key = self._load_static_key()
        if static_key is None:
            log.warning("Fuel tracker: Static key not found")
            return

        # Build XOR table: table[i] = static_key[i] ^ magic[i % len(magic)]
        table = bytearray(len(static_key))
        for i in range(len(static_key)):
            table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
        self._xor_table = bytes(table)
        log.info("Fuel tracker: XOR table built (%d bytes)", len(self._xor_table))

    def decode_fuel(self, body: bytes) -> int | None:
        """Decode fuel from a 14-byte state message.

        Args:
            body: Raw message body (after frame header, starts with 0x2e).

        Returns:
            Decoded fuel value (u16), or None if can't decode.
        """
        if len(body) != 14 or body[0] != 0x2e:
            return None
        if self._xor_table is None or len(self._xor_table) < 14:
            return None

        # XOR decode bytes 12-13 as u16 little-endian
        low = body[12] ^ self._xor_table[12]
        high = body[13] ^ self._xor_table[13]
        return low | (high << 8)

    def process_message(self, payload: str) -> str | None:
        """Process a message and return fuel status if relevant.

        Args:
            payload: Base64 encoded message payload.

        Returns:
            Fuel status string, or None if not a fuel message.
        """
        try:
            data = base64.b64decode(payload)
        except (ValueError, TypeError) as e:
            log.debug("Invalid base64 in fuel message: %s", e)
            return None

        if len(data) < 4:
            return None

        body = data[2:]

        # Check for 14-byte state message (any subtype)
        if len(body) != 14 or body[0] != 0x2e:
            return None

        fuel = self.decode_fuel(body)
        if fuel is None:
            return None

        subtype = body[1]

        # Calculate delta
        delta_str = ""
        status = ""
        if self._last_fuel is not None:
            diff = fuel - self._last_fuel
            if diff != 0:
                delta_str = f" ({diff:+d})"
            # Detect specific events
            if diff == -10:
                status = " [radar]"
            elif diff == -100:
                status = " [deposit]"
            elif diff == 100:
                status = " [pickup]"
            elif diff == -1:
                status = " [move]"

        self._last_fuel = fuel
        return f"[FUEL:0x{subtype:02x}] {fuel}{delta_str}{status}"


# Global tracker instances
_fuel_tracker = FuelTracker()
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


# Default configuration constants
DEFAULT_TARGET_URL = "https://tankpit.com"
DEFAULT_OUTPUT_PATH = "capture_session.json"
DEFAULT_CAPTURE_DURATION_MS = 0  # 0 = indefinite (wait until browser closed)


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

    # Very short - heartbeat/sync
    if length <= 3:
        return f"[{tag}] SYNC: {body.hex()}"

    # Map data (very large)
    if length > 500:
        return f"[{tag}] MAP_DATA: len={length}"

    # 12-byte messages - appear after shots (hit confirmation?)
    if length == 12:
        return f"[{tag}] HIT: {body.hex()}"

    # 8-byte messages - multiple types based on subtype byte
    if length == 8:
        subtype = body[1]
        if subtype == 0x49:  # 'I' = Item pickup confirmation
            return f"[{tag}] ITEM_PICKUP: {body.hex()}"
        elif subtype == 0x67:  # 'g' = Game state
            return f"[{tag}] GAME_STATE: {body.hex()}"
        else:
            return f"[{tag}] MSG_8B: sub=0x{subtype:02x} {body.hex()}"

    # 14-16 byte messages - various state updates
    if 14 <= length <= 16:
        subtype = body[1]
        return f"[{tag}] STATE: sub=0x{subtype:02x} len={length} hex={body.hex()}"

    # 17-byte 0x10 messages - these contain fuel at position 15
    if length == 17 and body[1] == 0x10:
        # Raw value at position 15-16 (XOR encoded)
        raw_p15 = int.from_bytes(body[15:17], "little")
        return f"[{tag}] FUEL_RAW: p15={raw_p15} hex={body.hex()}"

    # Medium messages (17-30 bytes) - entity updates
    if 17 <= length <= 30:
        subtype = body[1]
        return f"[{tag}] ENTITY: sub=0x{subtype:02x} len={length} hex={body.hex()}"

    # Short position refs (4-11 bytes)
    if 4 <= length <= 11:
        return f"[{tag}] POS: len={length} hex={body.hex()}"

    # Longer entity updates (31-500 bytes)
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
    try:
        data = base64.b64decode(payload)
    except (ValueError, TypeError):
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
    """Decode a '=' prefixed JOIN_CONFIRM message."""
    parts = text.split("|")
    room_id = parts[0][1:] if len(parts) > 0 else "?"
    tank_name = parts[2] if len(parts) > 2 else "?"
    return f"[{tag}] JOIN_CONFIRM: room={room_id} tank={tank_name}"


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

    def _process_game_log_entry(self, entry: _test_hooks.GameLogEntry) -> None:
        """Process a single game log entry and store it.

        Overrides BrowserSession._process_game_log_entry to also save entries.

        Args:
            entry: The game log entry to process.
        """
        # Store with timestamp
        from tankpit_bot.browser import get_current_time_ms

        self._game_log_entries.append({
            "timestamp_ms": get_current_time_ms(),
            "text": entry["text"],
            "category": entry["category"],
        })
        # Call parent to log and process combat
        super()._process_game_log_entry(entry)

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Log decoded message if live decode is enabled.

        Also polls game log and inventory for correlation with WebSocket events.

        Args:
            message: The captured message.
        """
        # Update trackers with magic key when available
        if self._magic and _fuel_tracker._xor_table is None:
            _fuel_tracker.set_magic(self._magic)
        if self._magic and _position_tracker._xor_table is None:
            _position_tracker.set_magic(self._magic)
        if self._magic and _deactivation_tracker._xor_table is None:
            _deactivation_tracker.set_magic(self._magic)
        if self._magic and _item_tracker._xor_table is None:
            _item_tracker.set_magic(self._magic)
        if self._magic and _radar_tracker._xor_table is None:
            _radar_tracker.set_magic(self._magic)
        if self._magic and _tank_tracker._xor_table is None:
            _tank_tracker.set_magic(self._magic)
        if self._magic and _mine_tracker._xor_table is None:
            _mine_tracker.set_magic(self._magic)
        if self._magic and _equip_tracker._xor_table is None:
            _equip_tracker.set_magic(self._magic)
        if self._magic and _container_tracker._xor_table is None:
            _container_tracker.set_magic(self._magic)
        if self._magic and _exit_tracker._xor_table is None:
            _exit_tracker.set_magic(self._magic)
        if self._magic and _equip_gain_tracker._xor_table is None:
            _equip_gain_tracker.set_magic(self._magic)
        if self._magic and _deposit_tracker._xor_table is None:
            _deposit_tracker.set_magic(self._magic)
        if self._magic and _radar_ack_tracker._xor_table is None:
            _radar_ack_tracker.set_magic(self._magic)

        if self._live_decode:
            # Check for fuel and position updates first
            if message["direction"] == "received":
                fuel_status = _fuel_tracker.process_message(message["payload"])
                if fuel_status:
                    log.info(fuel_status)
                pos_status = _position_tracker.process_message(message["payload"])
                if pos_status:
                    log.info(pos_status)
                deact_status = _deactivation_tracker.process_message(message["payload"])
                if deact_status:
                    log.info(deact_status)
                item_status = _item_tracker.process_message(message["payload"])
                if item_status:
                    log.info(item_status)
                radar_status = _radar_tracker.process_message(message["payload"])
                if radar_status:
                    log.info(radar_status)
                tank_status = _tank_tracker.process_message(message["payload"])
                if tank_status:
                    log.info(tank_status)
                mine_status = _mine_tracker.process_message(message["payload"], "received")
                if mine_status:
                    log.info(mine_status)
                equip_status = _equip_tracker.process_message(message["payload"])
                if equip_status:
                    log.info(equip_status)
                container_status = _container_tracker.process_message(message["payload"])
                if container_status:
                    log.info(container_status)
                exit_status = _exit_tracker.process_message(message["payload"])
                if exit_status:
                    log.info(exit_status)
                equip_gain_status = _equip_gain_tracker.process_message(message["payload"])
                if equip_gain_status:
                    log.info(equip_gain_status)
                deposit_status = _deposit_tracker.process_message(message["payload"])
                if deposit_status:
                    log.info(deposit_status)
                radar_ack_status = _radar_ack_tracker.process_message(message["payload"])
                if radar_ack_status:
                    log.info(radar_ack_status)

            # Check for sent mine commands
            if message["direction"] == "sent":
                mine_status = _mine_tracker.process_message(message["payload"], "sent")
                if mine_status:
                    log.info(mine_status)

            decoded = _decode_message(message["payload"], message["direction"], self._magic)
            log.info(decoded)
            # Poll game log and inventory after each message for correlation
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
        try:
            result = cdp.send("Runtime.evaluate", {
                "expression": js_probe,
                "returnByValue": True
            })
            if "result" in result and "value" in result["result"]:
                findings = result["result"]["value"]
                if findings:
                    log.info("JS variables in fuel range (800-1600):")
                    for f in findings:
                        log.info("  %s = %s", f["path"], f["value"])
                else:
                    log.info("No JS variables found in fuel range 800-1600")
        except Exception as e:
            log.warning("Failed to probe JS: %s", e)

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


def _build_message_stats(session: CaptureSession) -> MessageStats:
    """Build message statistics from captured session.

    Args:
        session: The capture session to analyze.

    Returns:
        MessageStats with decoded vs unknown breakdown.
    """
    from collections import Counter

    magic = session.get("magic")
    if not magic:
        return MessageStats(
            decoded={},
            unknown={},
            total_received=0,
            decode_coverage="0%",
        )

    # Load static key
    static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
    if not _test_hooks.path_exists(static_key_path):
        return MessageStats(decoded={}, unknown={}, total_received=0, decode_coverage="0%")

    static_key = _test_hooks.read_text(static_key_path).strip()
    magic_bytes = magic.encode("utf-8")
    xor_table = bytes(ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key)))

    # Known decoded signatures with understanding level:
    # - FULL: All fields understood and verified
    # - PARTIAL: Signature known, some fields unknown or unverified
    # - IDENTIFIED: Signature recognized, format not documented
    DECODED_SIGS: dict[int, tuple[str, str]] = {
        0x21: ("tank_info", "FULL"),
        0x28: ("tank_join", "IDENTIFIED"),
        0x29: ("tank_leave", "IDENTIFIED"),
        0x2B: ("promotion", "FULL"),
        0x2E: ("tank_status_sync", "PARTIAL"),  # bytes 9-10 unknown, flags unclear
        0x2F: ("player_update", "IDENTIFIED"),
        0x3D: ("movement", "FULL"),
        0x3E: ("tank_status", "PARTIAL"),
        0x3F: ("position", "FULL"),
        0x41: ("kill", "FULL"),
        0x43: ("container", "FULL"),
        0x45: ("mine_detonate", "FULL"),
        0x46: ("radar_ack", "FULL"),
        0x47: ("shooting", "FULL"),
        0x49: ("item_pickup", "FULL"),
        0x4B: ("mine_place", "FULL"),
        0x4C: ("tank_entry", "PARTIAL"),
        0x4D: ("player_list", "IDENTIFIED"),
        0x4F: ("deactivation", "FULL"),
        0x52: ("supervisor", "PARTIAL"),  # trigger unknown, status values unclear
        0x53: ("tank_move", "FULL"),
        0x54: ("tank_shoot", "FULL"),
        0x56: ("statistics", "FULL"),
        0x58: ("tank_exit", "FULL"),
        0x5A: ("viewport_update", "PARTIAL"),  # After spawn/teleport, shows zone entities
        0x64: ("fuel_deposit", "FULL"),
        0x67: ("equip_gain", "FULL"),  # 67 01 [armor][dual][missile][homing][radar]
        0x74: ("equip_toggle", "FULL"),
    }

    decoded_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    unknown_samples: dict[str, list[str]] = {}
    level_counts: Counter[str] = Counter()  # Track FULL/PARTIAL/IDENTIFIED counts

    for msg in session["messages"]:
        if msg["direction"] != "received":
            continue

        try:
            payload = base64.b64decode(msg["payload"])
        except (ValueError, TypeError):
            continue

        # Find 0x2E prefix
        if b"." not in payload[:3]:
            continue
        dot_pos = payload.find(b".")
        if dot_pos < 0 or dot_pos >= 3:
            continue

        start = dot_pos + 1
        if len(payload) <= start:
            continue

        decoded = bytes(payload[start + j] ^ xor_table[j] for j in range(min(len(payload) - start, len(xor_table))))
        if not decoded:
            continue

        sig = decoded[0]
        if sig in DECODED_SIGS:
            name, level = DECODED_SIGS[sig]
            decoded_counts[f"0x{sig:02X} '{chr(sig) if 32<=sig<127 else '?'}' {name}"] += 1
            level_counts[level] += 1
        else:
            sig_key = f"0x{sig:02X} '{chr(sig) if 32<=sig<127 else '?'}'"
            unknown_counts[sig_key] += 1
            if sig_key not in unknown_samples:
                unknown_samples[sig_key] = []
            if len(unknown_samples[sig_key]) < 3:
                unknown_samples[sig_key].append(decoded[:20].hex())

    total = sum(decoded_counts.values()) + sum(unknown_counts.values())
    decoded_total = sum(decoded_counts.values())

    # Calculate weighted understanding score:
    # FULL=100%, PARTIAL=50%, IDENTIFIED=25%, UNKNOWN=0%
    full_count = level_counts["FULL"]
    partial_count = level_counts["PARTIAL"]
    identified_count = level_counts["IDENTIFIED"]
    unknown_total = sum(unknown_counts.values())

    if total > 0:
        # Signature recognition (old metric)
        sig_coverage = 100 * decoded_total // total
        # Weighted understanding
        understanding = (full_count * 100 + partial_count * 50 + identified_count * 25) // total
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
        equipment_gains=[],  # TODO: populate from tracker
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
