"""Protocol probe using Playwright and WebSocket injection.

Probes game protocol by:
1. Launching a browser and joining the game
2. Sending known commands via WebSocket injection
3. Capturing WebSocket responses from the server
4. Correlating inputs with protocol messages

Note: This probe sends KNOWN commands via WebSocket. It does not discover
new commands - use the sniffer for that (play manually and observe traffic).
Synthetic JavaScript KeyboardEvents don't work because isTrusted: false.
"""

from __future__ import annotations

import threading
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
)
from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol, PageProtocol
from tankpit_bot.browser import (
    BrowserSession,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
    get_current_time_ms,
)
from tankpit_bot.protocol.codec import (
    DEFAULT_STATIC_KEY_PATH,
    build_xor_table,
    extract_magic_from_auth_payload,
    load_static_key,
    xor_bytes,
)
from tankpit_bot.protocol.commands import (
    CMD_MAP_OPEN,
    CMD_MINE,
    CMD_RADAR,
    PLAIN_QUIT,
)
from tankpit_bot.protocol.framing import encode_frame
from tankpit_bot.sniffer import decode_message
from tankpit_bot.types import (
    CapturedMessage,
    KeyInput,
    MouseInput,
    ProbeInput,
    ProbeResult,
    ProbeSession,
    encode_probe_session,
)

log = get_logger(__name__)


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
    # Must be multiple of 4 in length (with padding)
    return len(s) % 4 == 0


def _extract_magic_from_payload(payload_b64: str) -> str | None:
    """Extract magic key from AUTH message payload.

    Args:
        payload_b64: Base64-encoded AUTH message payload.

    Returns:
        Magic key string, or None if not an AUTH message or extraction fails.
    """
    import base64

    if not _is_valid_base64(payload_b64):
        return None

    data = base64.b64decode(payload_b64)
    return extract_magic_from_auth_payload(data)


class ProbeError(Exception):
    """Raised when probe encounters an error."""


def extract_cdp_evaluate_value(result: JSONObject) -> str:
    """Extract value from CDP Runtime.evaluate result.

    Args:
        result: CDP result dictionary from Runtime.evaluate.

    Returns:
        The string value from the result.

    Raises:
        ProbeError: If result structure is invalid or value is missing.
    """
    result_obj = result.get("result")
    if not isinstance(result_obj, dict):
        raise ProbeError(f"CDP Runtime.evaluate returned invalid result: {result}")
    val = result_obj.get("value")
    if val is None:
        raise ProbeError(f"CDP Runtime.evaluate result missing value: {result_obj}")
    return str(val)


# =============================================================================
# Key to Command Mapping
# =============================================================================

# Maps keyboard keys to their XOR-encoded command IDs
# Only keys with known command IDs can be probed via WebSocket injection
KEY_TO_COMMAND: dict[str, int] = {
    "s": CMD_RADAR,  # 's' key -> radar (command ID 102)
    "d": CMD_MINE,  # 'd' key -> mine (command ID 107)
    "f": CMD_MAP_OPEN,  # 'f' key -> map open (command ID 108)
}

# Keys that use plain text commands (no XOR encoding)
KEY_TO_PLAIN_COMMAND: dict[str, bytes] = {
    "q": PLAIN_QUIT,  # 'q' key -> quit game ('-')
}

# Toggle keys that open/close UI elements
# First press: WebSocket command to open
# Second press: JavaScript keypress to close (matches test_map_command.py)
TOGGLE_KEYS: set[str] = {"f"}

# Session type byte for XOR commands (discovered from protocol analysis)
COMMAND_TYPE_BYTE = 2

# =============================================================================
# Default probe configuration
# =============================================================================

# Standard game action keys to probe (matches test_map_command.py order)
DEFAULT_PROBE_KEYS: list[str] = [
    "f",  # Map open (WebSocket)
    "f",  # Map close (JS keypress toggle)
    "s",  # Radar
    "d",  # Mine
    "q",  # Quit
]

# Mouse positions for aim testing (empty by default)
DEFAULT_MOUSE_POSITIONS: list[tuple[float, float]] = []


class ProtocolProbe(BrowserSession):
    """Probes game protocol by sending commands via WebSocket.

    Extends BrowserSession with command sending and result capture.
    Uses WebSocket injection instead of synthetic JavaScript events
    (which don't work because isTrusted: false).
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        static_key_path: Path | None = None,
    ) -> None:
        """Initialize the probe.

        Args:
            target_url: URL to navigate to (e.g., https://tankpit.com/play).
            headless: Whether to run the browser in headless mode.
            static_key_path: Path to static XOR key file. Uses default if None.

        Raises:
            FileNotFoundError: If static key file does not exist.
            InvalidKeyError: If static key is empty.
        """
        super().__init__(target_url, headless=headless, prefer_account=True)
        self._results: list[ProbeResult] = []
        self._received_event = threading.Event()

        # Load static key for XOR encoding
        key_path = static_key_path if static_key_path is not None else DEFAULT_STATIC_KEY_PATH
        static_key = load_static_key(key_path)
        self._static_key: str = static_key
        self._xor_table: bytes | None = None

        # Track open toggle keys (e.g., map opened with 'f')
        self._open_toggles: set[str] = set()

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Signal when a received message arrives and extract magic from AUTH.

        Args:
            message: The captured message.
        """
        # Extract magic from AUTH message (sent messages)
        if message["direction"] == "sent" and self._magic is None:
            magic = _extract_magic_from_payload(message["payload"])
            if magic is not None:
                self._magic = magic
                log.info("Captured magic from AUTH: %s...", magic[:20])

        if message["direction"] == "received":
            self._received_event.set()

    def _wait_for_response(
        self,
        page: PageProtocol,
        msg_count_before: int,
    ) -> tuple[bool, bool]:
        """Wait for SENT then RECEIVED after an action.

        Args:
            page: Playwright page for wait_for_timeout.
            msg_count_before: Message count before action.

        Returns:
            Tuple of (got_sent, got_received).
        """
        got_sent = False
        got_received = False

        for _ in range(30):  # Poll for up to 3 seconds
            page.wait_for_timeout(100.0)
            new_msgs = self._messages[msg_count_before:]
            sent_msgs = [m for m in new_msgs if m["direction"] == "sent"]
            recv_msgs = [m for m in new_msgs if m["direction"] == "received"]

            if sent_msgs:
                got_sent = True
            if recv_msgs:
                got_received = True

            if got_sent and got_received:
                page.wait_for_timeout(500.0)  # Give game time to render
                return (True, True)

        return (got_sent, got_received)

    def _build_xor_table(self) -> None:
        """Build XOR table from static key and session magic.

        Must be called after magic key is captured.

        Raises:
            ProbeError: If magic key not captured.
        """
        if self._magic is None:
            raise ProbeError("Cannot build XOR table: magic key not captured")
        self._xor_table = build_xor_table(self._static_key, self._magic)
        log.info("Built XOR table, first 10 bytes: %s", self._xor_table[:10].hex())

    def _encode_xor_command(self, cmd_id: int) -> bytes:
        """Encode an XOR command with length header.

        Creates a 3-byte command (! + type + cmd_id), XOR encodes
        the type and cmd_id bytes, and adds the 2-byte length header.

        Args:
            cmd_id: Command ID byte (e.g., CMD_RADAR, CMD_MINE).

        Returns:
            Framed command ready to send via WebSocket.

        Raises:
            ProbeError: If XOR table not initialized.
        """
        if self._xor_table is None:
            raise ProbeError("XOR table not initialized")

        # Raw command: ! + type + cmd_id
        raw = bytes([0x21, COMMAND_TYPE_BYTE, cmd_id])

        # XOR encode type and cmd_id bytes (skip the '!' prefix)
        encoded = bytearray(len(raw))
        encoded[0] = raw[0]  # '!' stays as-is
        encoded_part = xor_bytes(self._xor_table, raw[1:], offset=0)
        encoded[1:] = encoded_part

        # Add 2-byte length header
        return encode_frame(bytes(encoded))

    def _encode_plain_command(self, body: bytes) -> bytes:
        """Encode a plain text command with length header.

        Args:
            body: Command body (no XOR encoding needed).

        Returns:
            Framed command ready to send via WebSocket.
        """
        return encode_frame(body)

    def _send_key_command(self, cdp: CDPSessionProtocol, key: str) -> str:
        """Send a game command for the given key via WebSocket.

        Args:
            cdp: CDP session.
            key: Key to send command for (e.g., 's', 'd', 'f', 'q').

        Returns:
            Status string from WebSocket send, or 'UNKNOWN_KEY' if no mapping.

        Raises:
            ProbeError: If XOR table not initialized (for XOR commands).
        """
        # Check for XOR-encoded command
        if key in KEY_TO_COMMAND:
            cmd_id = KEY_TO_COMMAND[key]
            encoded = self._encode_xor_command(cmd_id)
            result = self._send_websocket_bytes(cdp, encoded)
            log.info("    Sent XOR command: key=%s cmd_id=%d -> %s", key, cmd_id, result)
            return result

        # Check for plain text command
        if key in KEY_TO_PLAIN_COMMAND:
            body = KEY_TO_PLAIN_COMMAND[key]
            encoded = self._encode_plain_command(body)
            result = self._send_websocket_bytes(cdp, encoded)
            log.info("    Sent plain command: key=%s body=%r -> %s", key, body, result)
            return result

        log.warning("    Unknown key: %s (no command mapping)", key)
        return "UNKNOWN_KEY"

    def _send_js_keypress(self, cdp: CDPSessionProtocol, key: str) -> str:
        """Send a JavaScript keypress event to close a toggle UI.

        Used for closing UI elements (like map) that were opened via WebSocket.
        Matches test_map_command.py behavior: dispatches KeyboardEvent to multiple targets.

        Args:
            cdp: CDP session.
            key: Key to send (e.g., 'f').

        Returns:
            Status string from JavaScript execution.
        """
        key_code = ord(key.upper()) if len(key) == 1 else 0
        js_code = f"""
        (() => {{
            const targets = [document, window, document.body,
                             document.querySelector('canvas')];
            for (let target of targets) {{
                if (!target) continue;
                const event = new KeyboardEvent('keydown', {{
                    key: '{key}', code: 'Key{key.upper()}', keyCode: {key_code}, which: {key_code},
                    bubbles: true, cancelable: true
                }});
                target.dispatchEvent(event);
            }}
            return 'JS_KEYPRESS_{key.upper()}';
        }})()
        """
        result = cdp.send("Runtime.evaluate", {"expression": js_code, "returnByValue": True})
        result_obj = result.get("result")
        if isinstance(result_obj, dict):
            val = result_obj.get("value", "?")
            return str(val) if val is not None else "?"
        return "?"

    def _inject_mouse_click(self, cdp: CDPSessionProtocol, x: int, y: int) -> None:
        """Inject a mouse click via CDP.

        Args:
            cdp: CDP session.
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
        """
        cdp.send(
            "Input.dispatchMouseEvent",
            {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1},
        )
        cdp.send(
            "Input.dispatchMouseEvent",
            {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1},
        )

    def _probe_single_key(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        key: str,
    ) -> None:
        """Probe a single key by sending its command via WebSocket or JS keypress.

        For toggle keys (like 'f' for map):
        - First press: WebSocket command to open
        - Second press: JavaScript keypress to close (matches test_map_command.py)

        Args:
            page: Playwright page.
            cdp: CDP session.
            key: Key to probe (e.g., 's', 'd', 'f', 'q').
        """
        msg_count_before = len(self._messages)
        timestamp = get_current_time_ms()

        # Handle toggle keys: alternate between WS open and JS close
        if key in TOGGLE_KEYS and key in self._open_toggles:
            log.info("Closing toggle key: %s (msg_count_before=%d)", key, msg_count_before)
            result = self._send_js_keypress(cdp, key)
            log.info("    Sent JS keypress: key=%s -> %s", key, result)
            self._open_toggles.remove(key)
            # JS keypress may trigger responses
            if result.startswith("JS_KEYPRESS_"):
                self._wait_for_response(page, msg_count_before)
        else:
            log.info("Probing key: %s (msg_count_before=%d)", key, msg_count_before)
            result = self._send_key_command(cdp, key)
            if result.startswith("SENT_"):
                self._wait_for_response(page, msg_count_before)
                # Mark toggle key as open
                if key in TOGGLE_KEYS:
                    self._open_toggles.add(key)

        all_after = self._messages[msg_count_before:]
        sent_after = [m for m in all_after if m["direction"] == "sent"]
        recv_after = [m for m in all_after if m["direction"] == "received"]

        log.info("  -> sent %d, recv %d", len(sent_after), len(recv_after))
        for msg in all_after:
            decoded = decode_message(msg["payload"], msg["direction"], self._magic)
            log.info("    %s", decoded)

        key_input = KeyInput(key=key)
        probe_input = ProbeInput(input_type="key", key_input=key_input, mouse_input=None)
        probe_result = ProbeResult(
            input=probe_input,
            timestamp_ms=timestamp,
            messages_before_count=msg_count_before,
            messages_after=sent_after,
        )
        self._results.append(probe_result)

    def _probe_single_mouse(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        x: int,
        y: int,
    ) -> None:
        """Probe a single mouse click.

        Args:
            page: Playwright page.
            cdp: CDP session.
            x: X coordinate.
            y: Y coordinate.
        """
        msg_count_before = len(self._messages)
        timestamp = get_current_time_ms()

        log.info("Probing mouse click at (%d, %d)", x, y)
        self._inject_mouse_click(cdp, x, y)

        self._wait_for_response(page, msg_count_before)

        all_after = self._messages[msg_count_before:]
        sent_after = [m for m in all_after if m["direction"] == "sent"]
        recv_after = [m for m in all_after if m["direction"] == "received"]

        log.info("  -> sent %d, recv %d", len(sent_after), len(recv_after))
        for msg in all_after:
            decoded = decode_message(msg["payload"], msg["direction"], self._magic)
            log.info("    %s", decoded)

        mouse_input = MouseInput(x=x, y=y, button="left")
        probe_input = ProbeInput(input_type="mouse", key_input=None, mouse_input=mouse_input)
        result = ProbeResult(
            input=probe_input,
            timestamp_ms=timestamp,
            messages_before_count=msg_count_before,
            messages_after=sent_after,
        )
        self._results.append(result)

    def _get_viewport_size(self, cdp: CDPSessionProtocol) -> tuple[int, int]:
        """Get browser viewport size.

        Args:
            cdp: CDP session.

        Returns:
            Tuple of (width, height).
        """
        expr = "JSON.stringify({w: window.innerWidth, h: window.innerHeight})"
        viewport_result = cdp.send(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True},
        )
        viewport_raw = viewport_result.get("result")
        viewport_str = '{"w":800,"h":600}'
        if type(viewport_raw) is dict:
            viewport_val = viewport_raw.get("value")
            if type(viewport_val) is str:
                viewport_str = viewport_val

        viewport_parsed = load_json_str(viewport_str)
        viewport_data = narrow_json_to_dict(viewport_parsed)
        return require_int(viewport_data, "w"), require_int(viewport_data, "h")

    def _probe_keys(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        keys: list[str],
    ) -> None:
        """Probe a list of keys by sending their commands via WebSocket.

        Args:
            page: Playwright page.
            cdp: CDP session.
            keys: Keys to probe.
        """
        for key in keys:
            # Wait between commands
            page.wait_for_timeout(500.0)
            self._probe_single_key(page, cdp, key)

    def _probe_mouse_positions(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        positions: list[tuple[float, float]],
        viewport_w: int,
        viewport_h: int,
    ) -> None:
        """Probe mouse positions.

        Args:
            page: Playwright page.
            cdp: CDP session.
            positions: List of (x_frac, y_frac) positions.
            viewport_w: Viewport width.
            viewport_h: Viewport height.
        """
        for x_frac, y_frac in positions:
            x = int(x_frac * viewport_w)
            y = int(y_frac * viewport_h)
            self._probe_single_mouse(page, cdp, x, y)

    def _dump_messages(self) -> None:
        """Dump all messages to file for analysis."""
        log.info("Writing %d messages to messages_dump.txt", len(self._messages))
        with open("messages_dump.txt", "w") as f:
            for i, msg in enumerate(self._messages):
                ts = msg["timestamp_ms"]
                direction = msg["direction"].upper()
                payload = msg["payload"]
                f.write(f"[{i}] {ts} {direction}: {payload}\n")

    def run(
        self,
        probe_keys: list[str],
        probe_mouse_positions: list[tuple[float, float]],
        wait_after_join_ms: int,
        wait_after_input_ms: int,
    ) -> ProbeSession:
        """Run the probe and return captured results.

        Sends known commands via WebSocket injection and captures responses.

        Args:
            probe_keys: List of keys to test (must have known command mappings).
            probe_mouse_positions: List of (x_fraction, y_fraction) positions.
            wait_after_join_ms: Time to wait after joining game (unused, kept for API compat).
            wait_after_input_ms: Time to wait after each input (unused, kept for API compat).

        Returns:
            ProbeSession with all captured results.

        Raises:
            PlaywrightNotInstalledError: If Playwright hook is not installed.
            GameNotJoinedError: If failed to join game.
            ProbeError: If magic key not captured (needed for XOR encoding).
        """
        _ = wait_after_join_ms  # Unused - wait logic is in _wait_for_game_ready
        _ = wait_after_input_ms  # Unused - wait logic is in _wait_for_response

        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError("Playwright is not installed.")

        self._start_timestamp_ms = get_current_time_ms()
        self._messages = []
        self._ws_urls = {}
        self._results = []

        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self._headless)
            context = browser.new_context()
            page = context.new_page()
            cdp = context.new_cdp_session(page)

            # Set up console listener and CDP handlers
            self._setup_console_listener(cdp)
            self._setup_cdp_handlers(cdp)
            self._navigate_and_login(page, cdp, tank_name_prefix="P", auto_join_room=True)

            # Gather all available intel
            self._gather_intel(page, cdp)

            self._wait_for_game_ready(page)

            # Build XOR table now that magic key is captured
            self._build_xor_table()

            viewport_w, viewport_h = self._get_viewport_size(cdp)
            log.info("Viewport size: %dx%d", viewport_w, viewport_h)

            # Send commands via WebSocket injection
            self._probe_keys(page, cdp, probe_keys)
            self._probe_mouse_positions(page, cdp, probe_mouse_positions, viewport_w, viewport_h)

            log.info("Waiting 10 seconds to observe...")
            page.wait_for_timeout(10000.0)

            self._cleanup(cdp, page, context, browser)

        self._dump_messages()

        return ProbeSession(
            session_id=self._session_id,
            start_timestamp_ms=self._start_timestamp_ms,
            end_timestamp_ms=get_current_time_ms(),
            base_url=self._target_url,
            results=self._results,
        )


def run_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    probe_keys: list[str] | None = None,
    probe_mouse_positions: list[tuple[float, float]] | None = None,
    wait_after_join_ms: int = 5000,
    wait_after_input_ms: int = 200,
) -> ProbeSession:
    """Run the protocol probe and save results.

    Args:
        target_url: URL to navigate to.
        output_path: Path to save probe results JSON.
        headless: Whether to run browser headlessly.
        probe_keys: Keys to test (defaults to common game keys).
        probe_mouse_positions: Mouse positions as (x_frac, y_frac) tuples.
        wait_after_join_ms: Time to wait after joining game.
        wait_after_input_ms: Time to wait after each input.

    Returns:
        The completed ProbeSession.

    Raises:
        PlaywrightNotInstalledError: If Playwright is not installed.
        GameNotJoinedError: If failed to join game.
    """
    keys = probe_keys if probe_keys is not None else DEFAULT_PROBE_KEYS
    positions = (
        probe_mouse_positions if probe_mouse_positions is not None else DEFAULT_MOUSE_POSITIONS
    )

    probe = ProtocolProbe(target_url, headless=headless)
    session = probe.run(
        probe_keys=keys,
        probe_mouse_positions=positions,
        wait_after_join_ms=wait_after_join_ms,
        wait_after_input_ms=wait_after_input_ms,
    )

    encoded = encode_probe_session(session)
    json_str = dump_json_str(encoded, compact=False, indent=2)
    _test_hooks.write_text(Path(output_path), json_str)

    return session


def _log_discovered_commands(results: list[ProbeResult]) -> None:
    """Log discovered commands from probe results.

    Args:
        results: List of probe results that generated messages.
    """
    for r in results:
        inp = r["input"]
        count = len(r["messages_after"])
        key_input = inp["key_input"]
        mouse_input = inp["mouse_input"]
        if inp["input_type"] == "key" and key_input is not None:
            log.info("Discovered: Key '%s' -> %d msg(s)", key_input["key"], count)
        elif inp["input_type"] == "mouse" and mouse_input is not None:
            x, y = mouse_input["x"], mouse_input["y"]
            log.info("Discovered: Mouse (%d,%d) -> %d msg(s)", x, y, count)


def main() -> None:
    """Entry point for tankpit-probe command."""
    from dotenv import load_dotenv

    load_dotenv()
    setup_rich_logging(level="INFO")

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/play"
    output_path = _test_hooks.get_env("TANKPIT_PROBE_OUTPUT") or "probe_session.json"

    headless_str = _test_hooks.get_env("TANKPIT_HEADLESS")
    headless = headless_str is not None and headless_str.lower() in ("true", "1", "yes")

    wait_join_str = _test_hooks.get_env("TANKPIT_WAIT_JOIN_MS")
    wait_after_join_ms = int(wait_join_str) if wait_join_str else 5000

    wait_input_str = _test_hooks.get_env("TANKPIT_WAIT_INPUT_MS")
    wait_after_input_ms = int(wait_input_str) if wait_input_str else 1000

    # Basic CLI arg parsing for keys
    argv = _test_hooks.get_argv()

    probe_keys = None
    if "--keys" in argv:
        idx = argv.index("--keys")
        if idx + 1 < len(argv):
            keys_str = argv[idx + 1]
            probe_keys = [k.strip() for k in keys_str.split(",")]
            log.info("Overriding probe keys: %s", probe_keys)

    session = run_probe(
        target_url,
        output_path,
        headless=headless,
        probe_keys=probe_keys,
        wait_after_join_ms=wait_after_join_ms,
        wait_after_input_ms=wait_after_input_ms,
    )

    results_with_messages = [r for r in session["results"] if len(r["messages_after"]) > 0]
    log.info(
        "Probe complete: %d inputs tested, %d generated messages",
        len(session["results"]),
        len(results_with_messages),
    )
    log.info("Saved to: %s", output_path)

    _log_discovered_commands(results_with_messages)


__all__ = [
    "COMMAND_TYPE_BYTE",
    "DEFAULT_MOUSE_POSITIONS",
    "DEFAULT_PROBE_KEYS",
    "KEY_TO_COMMAND",
    "KEY_TO_PLAIN_COMMAND",
    "GameNotJoinedError",
    "PlaywrightNotInstalledError",
    "ProbeError",
    "ProtocolProbe",
    "_log_discovered_commands",
    "extract_cdp_evaluate_value",
    "main",
    "run_probe",
]
