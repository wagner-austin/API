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


# Hotkeys to discover
HOTKEYS: list[tuple[str, str, str]] = [
    # (key, code, description)
    ("t", "KeyT", "Top 10"),
    ("r", "KeyR", "Top 10 Red"),
    ("p", "KeyP", "Top 10 Purple"),
    ("b", "KeyB", "Top 10 Blue"),
    ("o", "KeyO", "Top 10 Orange"),
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

    def _send_cdp_key(self, cdp: object, key: str, code: str) -> None:
        """Send a key event via CDP Input.dispatchKeyEvent."""
        key_code = ord(key.upper()) if len(key) == 1 else 0

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

        # Check for XOR-encoded state message (starts with '.')
        if len(body) > 1 and body[0] == 0x2e and self._xor_table:
            # Decode bytes after '.'
            decoded = bytearray(len(body) - 1)
            for i in range(len(body) - 1):
                decoded[i] = body[i + 1] ^ self._xor_table[i]

            # Check if this looks like a leaderboard response (starts with '1')
            if len(decoded) > 5 and decoded[0] == ord('1'):
                return self._parse_leaderboard(decoded)

            # Try to decode as text
            try:
                text = decoded.decode("utf-8", errors="replace")
                if len(text) > 150:
                    return f"[XOR {len(body)}b] {text[:150]}..."
                return f"[XOR {len(body)}b] {text}"
            except Exception:
                return f"[XOR {len(body)}b] {decoded[:50].hex()}..."

        # Try to decode as text
        try:
            text = body.decode("utf-8")
            if len(text) > 100:
                return f"{text[:100]}... ({len(text)} chars)"
            return text
        except UnicodeDecodeError:
            if len(body) > 50:
                return f"[binary {len(body)}b] {body[:50].hex()}..."
            return f"[binary {len(body)}b] {body.hex()}"

    def _parse_leaderboard(self, data: bytearray) -> str:
        """Parse leaderboard response data."""
        # Header: '1' + team(1) + 4 bytes padding = 6 bytes
        # Record: rank(1) + page(1) + score(2 LE) + flags(2) + namelen(1) + name

        team_names = {0x00: "Red", 0x01: "Purple", 0x02: "Blue", 0x03: "Orange", 0xff: "All"}
        team = team_names.get(data[1], f"0x{data[1]:02x}")

        entries = []
        pos = 6  # Skip 6-byte header

        while pos + 7 < len(data):
            try:
                rank = data[pos]  # 1 byte rank
                # page = data[pos + 1]  # 1 byte page indicator (ignored)
                score = data[pos + 2] | (data[pos + 3] << 8)
                # flags at pos+4, pos+5
                name_len = data[pos + 6]
                pos += 7
                if pos + name_len > len(data) or name_len == 0 or name_len > 30:
                    break
                name = data[pos:pos + name_len].decode("utf-8", errors="replace")
                pos += name_len
                entries.append(f"#{rank}: {name} ({score:,})")
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

            # Probe each hotkey
            log.info("=" * 50)
            log.info("PROBING %d HOTKEYS", len(HOTKEYS))
            log.info("=" * 50)

            for key, code, description in HOTKEYS:
                msg_count_before = len(self._messages)

                self._send_cdp_key(cdp, key, code)
                page.wait_for_timeout(800.0)

                new_msgs = self._messages[msg_count_before:]
                sent_after = [m for m in new_msgs if m["direction"] == "sent"]
                recv_after = [m for m in new_msgs if m["direction"] == "received"]

                if sent_after:
                    decoded = self._decode_sent_message(sent_after[0]["payload"])
                    log.info("%s (%s): %s", description, key, decoded)
                else:
                    log.info("%s (%s): NO MESSAGE SENT", description, key)

                # Show received responses
                for msg in recv_after:
                    resp = self._decode_received_message(msg["payload"])
                    # Use print to avoid Windows console encoding issues
                    safe_resp = resp.encode("ascii", errors="replace").decode("ascii")
                    print(f"  -> RECV: {safe_resp}")

            page.wait_for_timeout(3000.0)
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
