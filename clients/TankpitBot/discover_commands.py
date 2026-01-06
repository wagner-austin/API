"""Discover command IDs for game hotkeys.

Uses CDP Input.dispatchKeyEvent to send trusted key events,
captures WebSocket messages, and decodes XOR to get command IDs.
"""

import base64
from pathlib import Path

from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.browser import BrowserSession, get_current_time_ms

log = get_logger(__name__)


# Complete hotkey list - test everything
HOTKEYS: list[tuple[str, str, str]] = [
    # (key, code, description)
    # Leaderboards
    ("t", "KeyT", "Top 10 All"),
    ("r", "KeyR", "Top 10 Red"),
    ("p", "KeyP", "Top 10 Purple"),
    ("b", "KeyB", "Top 10 Blue"),
    ("o", "KeyO", "Top 10 Orange"),
    # Info commands
    ("c", "KeyC", "Statistics"),
    ("i", "KeyI", "Inventory"),
    ("x", "KeyX", "Active Forces"),
    ("/", "Slash", "Active Players"),
    ("e", "KeyE", "Nearest Enemy"),
    ("h", "KeyH", "Help"),
    # Map/radar
    ("f", "KeyF", "Open Map"),
    ("s", "KeyS", "Radar"),
    # Equipment toggles
    ("1", "Digit1", "Toggle Armor Shields"),
    ("2", "Digit2", "Toggle Dual Shots"),
    ("3", "Digit3", "Toggle Missile Shots"),
    ("4", "Digit4", "Toggle Homing Shots"),
    ("5", "Digit5", "Toggle Extra Radars"),
    # Other
    ("l", "KeyL", "Toggle Sound"),
    ("d", "KeyD", "Drop Mine"),
    ("a", "KeyA", "Toggle Autoscroll"),
    ("m", "KeyM", "Toggle Tips"),
    ("n", "KeyN", "Next Tip"),
    # Scope/view commands (arrow keys + corners)
    ("ArrowUp", "ArrowUp", "Scope N"),
    ("ArrowDown", "ArrowDown", "Scope S"),
    ("ArrowLeft", "ArrowLeft", "Scope W"),
    ("ArrowRight", "ArrowRight", "Scope E"),
    ("PageUp", "PageUp", "Scope NE"),
    ("PageDown", "PageDown", "Scope SE"),
    ("End", "End", "Scope SW"),
    ("Home", "Home", "Scope NW"),
    # Special keys
    ("F6", "F6", "Ping"),
]


def load_static_key() -> str:
    """Load the static XOR key."""
    path = Path(__file__).parent / "xor_static_key.txt"
    return path.read_text().strip()


def build_xor_table(static_key: str, magic: str) -> bytes:
    """Build XOR table from static key and magic."""
    table = bytearray(len(static_key))
    for i in range(len(static_key)):
        table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
    return bytes(table)


class CommandDiscoverer(BrowserSession):
    """Discovers command IDs by sending key events and capturing responses."""

    def __init__(self, target_url: str) -> None:
        super().__init__(target_url, headless=False, prefer_account=True)
        self._static_key = load_static_key()
        self._xor_table: bytes | None = None
        self._open_toggles: set[str] = set()  # Track open UI toggles

    def _send_cdp_key(self, cdp: object, key: str, code: str) -> None:
        """Send a key event via CDP Input.dispatchKeyEvent."""
        # Special key codes (Windows virtual key codes)
        special_keys = {
            "/": 191,
            "ArrowUp": 38,
            "ArrowDown": 40,
            "ArrowLeft": 37,
            "ArrowRight": 39,
            "PageUp": 33,
            "PageDown": 34,
            "Home": 36,
            "End": 35,
            "F6": 117,
        }

        if key in special_keys:
            key_code = special_keys[key]
        elif len(key) == 1:
            key_code = ord(key.upper())
        else:
            key_code = 0

        cdp.send(
            "Input.dispatchKeyEvent",
            {
                "type": "keyDown",
                "key": key,
                "code": code,
                "windowsVirtualKeyCode": key_code,
                "nativeVirtualKeyCode": key_code,
            },
        )
        cdp.send(
            "Input.dispatchKeyEvent",
            {
                "type": "keyUp",
                "key": key,
                "code": code,
                "windowsVirtualKeyCode": key_code,
                "nativeVirtualKeyCode": key_code,
            },
        )

    def _send_js_keypress(self, cdp: object, key: str) -> str:
        """Send a JavaScript keypress event to close a toggle UI.

        Used for closing UI elements (like map) that were opened via WebSocket.
        Sends both keydown and keyup events to multiple targets.
        """
        key_code = ord(key.upper()) if len(key) == 1 else 0
        js_code = f"""
        (() => {{
            const targets = [document, window, document.body,
                             document.querySelector('canvas'),
                             document.querySelector('#field-image')];
            let dispatched = 0;
            for (let target of targets) {{
                if (!target) continue;
                // Send keydown
                const downEvent = new KeyboardEvent('keydown', {{
                    key: '{key}', code: 'Key{key.upper()}', keyCode: {key_code},
                    which: {key_code}, bubbles: true, cancelable: true
                }});
                target.dispatchEvent(downEvent);
                // Send keyup
                const upEvent = new KeyboardEvent('keyup', {{
                    key: '{key}', code: 'Key{key.upper()}', keyCode: {key_code},
                    which: {key_code}, bubbles: true, cancelable: true
                }});
                target.dispatchEvent(upEvent);
                dispatched++;
            }}
            return 'JS_KEYPRESS_{key.upper()}_' + dispatched + '_targets';
        }})()
        """
        result = cdp.send("Runtime.evaluate", {"expression": js_code, "returnByValue": True})
        result_obj = result.get("result", {})
        if isinstance(result_obj, dict):
            val = result_obj.get("value", "?")
            return str(val) if val is not None else "?"
        return "?"

    def _decode_sent_message(self, payload_b64: str) -> str:
        """Decode a sent message and return description."""
        try:
            data = base64.b64decode(payload_b64)
        except (ValueError, TypeError):
            return f"invalid base64: {payload_b64}"

        if len(data) < 3:
            return f"too short: {data.hex()}"

        body = data[2:]  # Skip 2-byte length header

        # Check for XOR command (starts with '!')
        if len(body) >= 3 and body[0] == 0x21 and self._xor_table:
            # Decode all bytes after '!'
            decoded = bytearray(len(body) - 1)
            for i in range(len(body) - 1):
                decoded[i] = body[i + 1] ^ self._xor_table[i]
            type_byte = decoded[0]
            cmd_id = decoded[1]
            extra = decoded[2:].hex() if len(decoded) > 2 else ""
            return f"XOR type={type_byte} cmd_id={cmd_id} (0x{cmd_id:02x}) extra={extra}"

        # Plain text command
        try:
            text = body.decode("utf-8")
            return f"PLAIN: {text!r}"
        except UnicodeDecodeError:
            return f"BINARY: {body.hex()}"

    def _decode_received_message(self, payload_b64: str) -> str:
        """Decode a received message and return description."""
        try:
            data = base64.b64decode(payload_b64)
        except (ValueError, TypeError):
            return "invalid base64"

        if len(data) < 2:
            return f"too short: {data.hex()}"

        body = data[2:]  # Skip 2-byte length header

        # Check for '.' state message - use detailed state parser
        if len(body) > 1 and body[0] == 0x2E:
            return self._decode_dot_message(body)

        # Try to decode as text
        return self._decode_text_message(body)

    def _decode_dot_message(self, body: bytes) -> str:
        """Decode '.' prefixed message (XOR-encoded response or state)."""
        # If we have XOR table, check for known response types first
        if self._xor_table:
            decoded = bytearray(len(body) - 1)
            for i in range(len(body) - 1):
                decoded[i] = body[i + 1] ^ self._xor_table[i]

            # Check if this looks like a leaderboard response (starts with '1')
            if len(decoded) > 5 and decoded[0] == ord("1"):
                return self._parse_leaderboard(decoded)

            # Check for other known response prefixes
            if len(decoded) > 0:
                known_prefixes = (
                    ord("*"),
                    ord("I"),
                    ord("V"),
                    ord("/"),
                    ord("H"),
                    ord("t"),
                    ord("L"),
                )
                if decoded[0] in known_prefixes:
                    return self._format_xor_message(decoded, len(body))

        # Fall back to detailed state parsing
        return self._parse_state_message(body)

    def _decode_text_message(self, body: bytes) -> str:
        """Try to decode body as text, fall back to hex."""
        try:
            text = body.decode("utf-8")
            if len(text) > 100:
                return f"{text[:100]}... ({len(text)} chars)"
            return text
        except UnicodeDecodeError:
            if len(body) > 50:
                return f"[binary {len(body)}b] {body[:50].hex()}..."
            return f"[binary {len(body)}b] {body.hex()}"

    def _format_xor_message(self, decoded: bytearray, body_len: int) -> str:
        """Format decoded XOR message for display with hex dump."""
        if len(decoded) == 0:
            return f"[XOR {body_len}b] (empty)"

        # Route to specific parsers based on response type byte
        response_type = decoded[0]

        if response_type == ord("*") and len(decoded) >= 5:
            return self._parse_active_forces(decoded)

        if response_type == ord("I") and len(decoded) >= 7:
            return self._parse_inventory(decoded)

        if response_type == ord("V") and len(decoded) >= 15:
            return self._parse_statistics(decoded)

        if response_type == ord("/") and len(decoded) >= 4:
            return self._parse_active_players(decoded)

        if response_type == ord("H") and len(decoded) >= 7:
            return self._parse_nearest_enemy(decoded)

        if response_type == ord("t") and len(decoded) >= 6:
            return self._parse_toggle_state(decoded)

        if response_type == ord("L") and len(decoded) >= 100:
            return self._parse_map_data(decoded)

        # Fallback: show hex dump for debugging
        hex_dump = decoded[:32].hex()
        text = decoded.decode("utf-8", errors="replace")
        clean = text.replace("\x00", " ").replace("\x01", " ").replace("\x02", " ")
        clean = clean.replace("\x03", " ").replace("\x04", " ").replace("\x08", " ")
        if len(clean) > 100:
            clean = clean[:100] + "..."
        return f"[XOR {body_len}b] hex={hex_dump} text={clean}"

    def _parse_active_forces(self, data: bytearray) -> str:
        """Parse active forces response: * + red(1) + purple(1) + blue(1) + orange(1)."""
        # Full hex dump for verification
        hex_dump = data.hex()
        red = data[1]
        purple = data[2]
        blue = data[3]
        orange = data[4]
        total = red + purple + blue + orange
        return (
            f"[FORCES] red={red} purple={purple} blue={blue} orange={orange} "
            f"(total={total}) | raw: {hex_dump}"
        )

    def _parse_inventory(self, data: bytearray) -> str:
        """Parse inventory response: I + version(1) + 6 equipment slots.

        Slots: armor_shield, dual_shot, missile_shot, homing_shot, extra_radar, slot6
        Encoding: 0x80 bit = disabled flag, lower 7 bits = count
        - 0x94 = 0x80 | 0x14 = 20 items, disabled
        - 0x14 = 20 items, enabled
        - 0x00 = 0 items
        """
        hex_dump = data.hex()
        version = data[1] if len(data) > 1 else 0
        slot_names = ["armor", "dual", "missile", "homing", "radar", "slot6"]
        items = []
        for i, name in enumerate(slot_names):
            val = data[2 + i] if 2 + i < len(data) else 0
            disabled = (val & 0x80) != 0
            count = val & 0x7F  # Lower 7 bits = count
            state = "OFF" if disabled else "ON"
            items.append(f"{name}={count}({state})")
        return f"[INVENTORY] v={version} {' '.join(items)} | raw: {hex_dump}"

    def _parse_statistics(self, data: bytearray) -> str:
        """Parse statistics response - decode every byte.

        Based on game display:
        - Play time: H:M:S
        - Destroyed enemies: count
        - Deactivated: count
        - Promotion points: count
        """
        hex_dump = data.hex()
        # Byte-by-byte analysis
        parts = []
        for i, b in enumerate(data):
            parts.append(f"[{i}]=0x{b:02x}({b})")

        # Known mappings from hex: 56 00 00 08 08 00 00 00 00 00 00 00 00 00 1a
        # Byte 0: 'V' prefix
        # Bytes 1-2: hours (LE) = 0
        # Byte 3: minutes = 8
        # Byte 4: seconds = 8 (but user saw 47, so maybe different)
        # Actually let's show raw for now and figure it out
        hours = data[1] | (data[2] << 8) if len(data) > 2 else 0
        mins = data[3] if len(data) > 3 else 0
        secs = data[4] if len(data) > 4 else 0
        destroyed = data[5] | (data[6] << 8) if len(data) > 6 else 0
        deactivated = data[7] | (data[8] << 8) if len(data) > 8 else 0
        # Bytes 9-13: unknown/padding
        # Byte 14: promotion points (1 byte)
        promo_pts = data[14] if len(data) > 14 else 0

        return (
            f"[STATS] time={hours}h{mins}m{secs}s destroyed={destroyed} "
            f"deactivated={deactivated} promo_pts={promo_pts} | raw: {hex_dump}"
        )

    def _parse_active_players(self, data: bytearray) -> str:
        """Parse active players response - decode every byte.

        Format seems to be: / + total_capacity(1) + count(2 LE) + player_entries
        """
        hex_dump = data.hex()
        capacity = data[1] if len(data) > 1 else 0
        count = data[2] | (data[3] << 8) if len(data) > 3 else 0

        # Parse player entries if present
        players = []
        pos = 4
        while pos < len(data):
            # Try to parse each player entry
            # Format might be: team(1) + rank(1) + name_len(1) + name
            if pos + 2 >= len(data):
                # Show remaining bytes
                players.append(f"remaining: {data[pos:].hex()}")
                break
            team = data[pos]
            rank = data[pos + 1]
            if pos + 2 < len(data):
                name_len = data[pos + 2]
                if name_len > 0 and name_len < 30 and pos + 3 + name_len <= len(data):
                    name = data[pos + 3 : pos + 3 + name_len].decode("utf-8", errors="replace")
                    players.append(f"team={team} rank={rank} name={name}")
                    pos += 3 + name_len
                    continue
            # Fallback: show raw bytes
            players.append(f"@{pos}: {data[pos : pos + 3].hex()}")
            pos += 1

        player_str = ", ".join(players) if players else "(no player data)"
        return f"[PLAYERS] capacity={capacity} count={count} {player_str} | raw: {hex_dump}"

    def _parse_nearest_enemy(self, data: bytearray) -> str:
        """Parse nearest enemy response - target coordinates and info.

        Format: H + x(1) + y(1) + team(1) + player_num(1) + rank_id?(1) + ?(1)
        Based on: 483f8400011b02 => "red-4 (private) detected at [63,132]"
          - 48 = 'H' prefix
          - 3f = 63 (x)
          - 84 = 132 (y)
          - 00 = team 0 = red
          - 01 = player number? (but game shows "4")
          - 1b = 27 = rank code?
          - 02 = ?
        """
        hex_dump = data.hex()
        x = data[1] if len(data) > 1 else 0
        y = data[2] if len(data) > 2 else 0
        team_byte = data[3] if len(data) > 3 else 0
        player_num = data[4] if len(data) > 4 else 0
        rank_byte = data[5] if len(data) > 5 else 0
        byte6 = data[6] if len(data) > 6 else 0

        # Team: 0=red, 1=purple, 2=blue, 3=orange
        team_names = {0: "red", 1: "purple", 2: "blue", 3: "orange"}
        team = team_names.get(team_byte, f"team{team_byte}")

        # Rank encoding - discovered codes
        # Ranks: recruit, private, corporal, sergeant, lieutenant, captain, major, colonel, general
        rank_names = {
            0x18: "corporal",  # From red-2 sample
            0x1B: "private",  # From red-4 sample
        }
        rank = rank_names.get(rank_byte, f"rank0x{rank_byte:02x}")

        return (
            f"[ENEMY] coords=[{x},{y}] {team}-{player_num} ({rank}) "
            f"b6=0x{byte6:02x} | raw: {hex_dump}"
        )

    def _parse_toggle_state(self, data: bytearray) -> str:
        """Parse toggle state response: t + armor + dual + missile + homing + radar.

        Each byte: 0=off, 1=on
        Example: 740101010101 = all 5 equipment types active
        """
        hex_dump = data.hex()
        armor = data[1] if len(data) > 1 else 0
        dual = data[2] if len(data) > 2 else 0
        missile = data[3] if len(data) > 3 else 0
        homing = data[4] if len(data) > 4 else 0
        radar = data[5] if len(data) > 5 else 0

        active = []
        if armor:
            active.append("armor")
        if dual:
            active.append("dual")
        if missile:
            active.append("missile")
        if homing:
            active.append("homing")
        if radar:
            active.append("radar")

        state = ", ".join(active) if active else "none"
        return f"[TOGGLE] active=[{state}] | raw: {hex_dump}"

    def _parse_map_data(self, data: bytearray) -> str:
        """Parse map data response: L + map_bytes.

        Large binary blob containing map tile/entity data.
        """
        hex_dump = data[:64].hex()  # First 64 bytes
        return f"[MAP] {len(data)} bytes | first64: {hex_dump}..."

    def _parse_scope_response(self, data: bytearray) -> str:
        """Parse scope/view response - location coordinates.

        Format appears to be: prefix + x(1-2) + y(1-2) + other_data
        Look for the "LOCATION: x,y" data.
        """
        hex_dump = data.hex()

        # Try different interpretations
        if len(data) >= 3:
            # Single byte coords (0-255 range)
            x1 = data[1] if len(data) > 1 else 0
            y1 = data[2] if len(data) > 2 else 0
            # LE 16-bit coords
            x2 = data[1] | (data[2] << 8) if len(data) > 2 else 0
            y2 = data[3] | (data[4] << 8) if len(data) > 4 else 0

            return f"[SCOPE] 1b=[{x1},{y1}] 2bLE=[{x2},{y2}] | raw: {hex_dump}"
        return f"[SCOPE] raw: {hex_dump}"

    def _parse_autoscroll_response(self, data: bytearray) -> str:
        """Parse autoscroll toggle response."""
        hex_dump = data.hex()
        # Likely just a state byte: 0=off, 1=on
        state = data[1] if len(data) > 1 else 0
        return f"[AUTOSCROLL] state={state} | raw: {hex_dump}"

    def _parse_ping_response(self, data: bytearray) -> str:
        """Parse ping response - server latency."""
        hex_dump = data.hex()
        # Could be timestamp or latency value
        if len(data) >= 5:
            # Try as 32-bit value
            value = data[1] | (data[2] << 8) | (data[3] << 16) | (data[4] << 24)
            return f"[PING] value={value} | raw: {hex_dump}"
        return f"[PING] raw: {hex_dump}"

    def _parse_state_message(self, body: bytes) -> str:
        """Parse '.' prefixed state message for maximum intel.

        State messages contain game state updates:
        - Player positions (our location and others)
        - Projectile data
        - Game objects
        - Team assignments
        """
        hex_dump = body.hex()

        # Body starts with '.' (0x2e), then XOR-encoded data
        if len(body) < 3:
            return f"[STATE] too short: {hex_dump}"

        # Skip the '.' and decode with XOR
        if not self._xor_table:
            return f"[STATE raw] {len(body)}b: {hex_dump[:64]}..."

        decoded = bytearray(len(body) - 1)
        for i in range(len(body) - 1):
            decoded[i] = body[i + 1] ^ self._xor_table[i]

        # Look for patterns - try multiple interpretations
        results = []

        # Check first byte for message subtype
        subtype = decoded[0] if len(decoded) > 0 else 0
        results.append(f"subtype=0x{subtype:02x}('{chr(subtype) if 32 <= subtype < 127 else '?'}')")

        # Look for coordinate pairs (values in reasonable range 0-255)
        # The game map is roughly 200x200 based on "LOCATION: 107,135"
        coord_pairs = []
        for i in range(len(decoded) - 1):
            x, y = decoded[i], decoded[i + 1]
            # Filter for reasonable coordinate values
            if 50 <= x <= 200 and 50 <= y <= 200:
                coord_pairs.append(f"@{i}:[{x},{y}]")

        if coord_pairs:
            results.append(f"possible_coords={coord_pairs[:5]}")

        # Decode as text (printable chars)
        text_preview = decoded[:40].decode("utf-8", errors="replace")
        clean = "".join(c if c.isprintable() else "." for c in text_preview)
        results.append(f"text='{clean}'")

        # Full hex dump for debugging
        decoded_hex = decoded[:32].hex()
        results.append(f"hex={decoded_hex}")

        return f"[STATE {len(body)}b] " + " | ".join(results)

    def _get_all_state_messages(self, start_idx: int = 0) -> list[tuple[str, str]]:
        """Get ALL state messages from start_idx as (subtype, hex) tuples."""
        state_msgs = []
        for msg in self._messages[start_idx:]:
            if msg["direction"] != "received":
                continue
            try:
                raw = base64.b64decode(msg["payload"])
                if len(raw) < 4:
                    continue
                body = raw[2:]  # Skip length header
                if body[0] != 0x2E:  # Not a '.' message
                    continue
                # Decode with XOR
                if self._xor_table:
                    decoded = bytearray(len(body) - 1)
                    for i in range(len(body) - 1):
                        decoded[i] = body[i + 1] ^ self._xor_table[i]
                    if len(decoded) > 0:
                        byte0 = decoded[0]
                        subtype = chr(byte0) if 32 <= byte0 < 127 else f"0x{byte0:02x}"
                        state_msgs.append((subtype, decoded.hex()))
            except (ValueError, TypeError):
                continue
        return state_msgs

    def _analyze_tracked_values(self, tracked_values: list[dict[int, int]]) -> None:
        """Analyze tracked values to find fields that change (likely fuel/HP)."""
        log.info("=" * 50)
        log.info("VALUE TRACKING ANALYSIS")
        log.info("=" * 50)
        if len(tracked_values) < 2:
            log.info("Not enough data to analyze")
            return

        # Show each position's values across all captures
        for pos in range(14):
            vals = [v.get(pos, -1) for v in tracked_values]
            changes = []
            for i in range(1, len(vals)):
                if vals[i] != vals[i - 1] and vals[i] != -1 and vals[i - 1] != -1:
                    changes.append(f"{vals[i - 1]}->{vals[i]}")
            if changes:
                log.info("  Pos[%d]: %s | CHANGES: %s", pos, vals, changes)

    def _radar_fuel_test(self, page: object, cdp: object) -> None:
        """Send radar multiple times to find the fuel/HP field by diffing state."""
        log.info("=" * 50)
        log.info("RADAR FUEL TEST - Maximum Intel Mode")
        log.info("=" * 50)

        # Track values across radar uses to find decreasing field
        tracked_values: list[dict[int, int]] = []

        # Get initial state
        page.wait_for_timeout(500.0)
        msg_idx_start = len(self._messages)
        initial_states = self._get_all_state_messages(msg_idx_start - 20)
        log.info("Initial state - %d state messages in last 20", len(initial_states))
        for subtype, hexdata in initial_states[-5:]:
            log.info("  [%s] %s", subtype, hexdata[:60])

        # Send radar 10 times for better data
        for r in range(10):
            msg_idx_before = len(self._messages)
            self._send_cdp_key(cdp, "s", "KeyS")
            page.wait_for_timeout(600.0)

            # Get ALL new state messages after this radar
            new_states = self._get_all_state_messages(msg_idx_before)
            log.info("Radar #%d - %d new state messages:", r + 1, len(new_states))

            # Show every new message with full hex and track '.' subtype
            for subtype, hexdata in new_states:
                log.info("  [%s] %s", subtype, hexdata)
                if subtype == ".":
                    max_len = min(len(hexdata), 28)
                    values = {i // 2: int(hexdata[i : i + 2], 16) for i in range(0, max_len, 2)}
                    tracked_values.append(values)

        # Analyze the tracked values
        self._analyze_tracked_values(tracked_values)

    def _analyze_initial_messages(self) -> None:
        """Dump all initial messages to find LOCATION data."""
        log.info("=" * 50)
        log.info("ANALYZING %d INITIAL MESSAGES FOR LOCATION DATA", len(self._messages))
        log.info("=" * 50)

        for i, msg in enumerate(self._messages):
            decoded = self._decode_received_message(msg["payload"])
            direction = msg["direction"].upper()
            # Also show raw hex for debugging
            try:
                raw = base64.b64decode(msg["payload"])
                raw_hex = raw.hex()[:64]  # First 64 hex chars
            except (ValueError, TypeError):
                raw_hex = "?"

            # Look for coordinate patterns (107, 135 = 0x6b, 0x87)
            if "6b" in raw_hex or "87" in raw_hex:
                log.info("[%d] %s: %s | POTENTIAL COORDS: %s", i, direction, decoded[:80], raw_hex)
            else:
                # Still show for full intel
                safe_decoded = decoded.encode("ascii", errors="replace").decode("ascii")
                print(f"[{i}] {direction}: {safe_decoded[:100]} | raw: {raw_hex}")

    def _probe_all_hotkeys(self, page: object, cdp: object) -> None:
        """Probe all hotkeys and decode responses."""
        log.info("=" * 50)
        log.info("PROBING %d HOTKEYS", len(HOTKEYS))
        log.info("=" * 50)

        # Equipment toggles - both presses send WS commands
        equipment_toggles = {"1", "2", "3", "4", "5"}
        # UI toggles - open with WS, close with JS keypress
        ui_toggles = {"f"}
        # Local toggles - press twice (plain commands or client-side)
        local_toggles = {"l", "a"}  # Sound and autoscroll

        for key, code, description in HOTKEYS:
            self._probe_single_key(
                page, cdp, key, code, description, equipment_toggles, ui_toggles, local_toggles
            )

        log.info("=" * 50)
        log.info("DISCOVERY COMPLETE - waiting 20 seconds...")
        log.info("=" * 50)
        page.wait_for_timeout(20000.0)

    def _probe_single_key(
        self,
        page: object,
        cdp: object,
        key: str,
        code: str,
        description: str,
        equipment_toggles: set[str],
        ui_toggles: set[str],
        local_toggles: set[str],
    ) -> None:
        """Probe a single hotkey."""
        is_equipment = key in equipment_toggles
        is_ui_toggle = key in ui_toggles
        is_local_toggle = key in local_toggles
        presses = 2 if (is_equipment or is_ui_toggle or is_local_toggle) else 1

        for press_num in range(presses):
            msg_count_before = len(self._messages)

            # For UI toggles, second press uses JS keypress
            if is_ui_toggle and press_num == 1:
                result = self._send_js_keypress(cdp, key)
                log.info("%s [CLOSE] (%s): %s", description, key, result)
                page.wait_for_timeout(500.0)
                continue

            self._send_cdp_key(cdp, key, code)
            page.wait_for_timeout(1500.0)  # Wait longer for response

            new_msgs = self._messages[msg_count_before:]
            sent_after = [m for m in new_msgs if m["direction"] == "sent"]
            recv_after = [m for m in new_msgs if m["direction"] == "received"]

            # Label for toggle keys
            if presses == 2 and not is_ui_toggle:
                state = "ON" if press_num == 0 else "OFF"
                label = f"{description} [{state}]"
            elif is_ui_toggle:
                label = f"{description} [OPEN]"
            else:
                label = description

            if sent_after:
                decoded = self._decode_sent_message(sent_after[0]["payload"])
                log.info("%s (%s): %s", label, key, decoded)
            else:
                log.info("%s (%s): NO MESSAGE SENT", label, key)

            # Show received responses
            for msg in recv_after:
                resp = self._decode_received_message(msg["payload"])
                safe_resp = resp.encode("ascii", errors="replace").decode("ascii")
                print(f"  -> RECV: {safe_resp}")

    def _parse_leaderboard(self, data: bytearray) -> str:
        """Parse leaderboard response data."""
        # Header: '1' + team(1) + 4 bytes padding = 6 bytes
        # Record: rank(1) + mystery(1) + score(2 LE) + flag1(1) + flag2(1) + namelen(1) + name

        team_names = {0x00: "Red", 0x01: "Purple", 0x02: "Blue", 0x03: "Orange", 0xFF: "All"}
        team = team_names.get(data[1], f"0x{data[1]:02x}")

        entries = []
        pos = 6  # Skip 6-byte header

        while pos + 7 < len(data):
            try:
                rank = data[pos]
                mystery = data[pos + 1]
                score = data[pos + 2] | (data[pos + 3] << 8)
                flag1 = data[pos + 4]
                flag2 = data[pos + 5]
                name_len = data[pos + 6]
                pos += 7
                if pos + name_len > len(data) or name_len == 0 or name_len > 30:
                    break
                name = data[pos : pos + name_len].decode("utf-8", errors="replace")
                pos += name_len
                entries.append(f"#{rank}: {name} ({score}pts) m={mystery} f={flag1},{flag2}")
            except (IndexError, UnicodeDecodeError):
                break

        if entries:
            return f"[TOP10 {team}] " + ", ".join(entries)
        return f"[TOP10 {team}] (no entries parsed)"

    def run(self) -> None:
        """Run discovery."""
        if _test_hooks.sync_playwright is None:
            raise RuntimeError("Playwright not installed")

        self._start_timestamp_ms = get_current_time_ms()
        self._messages = []
        self._ws_urls = {}

        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self._headless)
            context = browser.new_context()
            page = context.new_page()
            cdp = context.new_cdp_session(page)

            self._setup_console_listener(cdp)
            self._setup_cdp_handlers(cdp)
            self._navigate_and_login(page, cdp, tank_name_prefix="D", auto_join_room=True)
            self._gather_intel(page, cdp)
            self._wait_for_game_ready(page)

            if not self._magic:
                log.error("No magic key captured!")
                self._cleanup(cdp, page, context, browser)
                return

            self._xor_table = build_xor_table(self._static_key, self._magic)
            log.info("XOR table ready")

            # First, dump all initial messages to find LOCATION data
            self._analyze_initial_messages()

            # Radar fuel test - find HP/fuel field by diffing state
            self._radar_fuel_test(page, cdp)

            # Skip hotkey probing - already discovered
            # self._probe_all_hotkeys(page, cdp)

            log.info("=" * 50)
            log.info("DISCOVERY COMPLETE - waiting 20 seconds...")
            log.info("=" * 50)
            page.wait_for_timeout(20000.0)

            self._cleanup(cdp, page, context, browser)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    setup_rich_logging(level="INFO")

    _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    discoverer = CommandDiscoverer("https://tankpit.com/play")
    discoverer.run()


if __name__ == "__main__":
    main()
