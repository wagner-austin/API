"""Shared browser session management for WebSocket capture.

Provides a base class that handles:
- Playwright browser launch and CDP setup
- WebSocket event handlers and message capture
- Magic key capture for XOR decoding
- Login flow integration

Both sniffer.py and probe.py inherit from this to avoid code duplication.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.combat import CombatEvent, CombatTracker
from tankpit_bot.dom_scraper import (
    GameLogEntry,
    GameLogScraper,
)
from tankpit_bot.fuel_probe import (
    FuelProber,
    FuelProbeResult,
)
from tankpit_bot.inventory import (
    InventoryChange,
    InventoryScraper,
)
from tankpit_bot.login import handle_login_flow
from tankpit_bot.types import (
    CapturedMessage,
    MessageDirection,
    decode_cdp_websocket_created_event,
    decode_cdp_websocket_frame_event,
)

log = get_logger(__name__)

# Known binary message signatures from TankPit protocol.
# These are the first decoded byte of binary messages after XOR decoding.
KNOWN_PROTOCOL_SIGNATURES: frozenset[int] = frozenset(
    {
        0x21,
        0x28,
        0x29,
        0x2B,
        0x2E,
        0x2F,
        0x3D,
        0x3E,
        0x3F,
        0x41,
        0x43,
        0x45,
        0x46,
        0x47,
        0x49,
        0x4A,
        0x4B,
        0x4C,
        0x4D,
        0x4F,
        0x52,
        0x53,
        0x54,
        0x56,
        0x58,
        0x5A,
        0x64,
        0x67,
        0x74,
    }
)

# Text message type bytes that should be skipped during XOR analysis.
TEXT_MESSAGE_TYPES: frozenset[int] = frozenset({0x2B, 0x2D, 0x3D, 0x25, 0x2A, 0x7E})

# Path to the static XOR key file.
STATIC_KEY_PATH: Path = Path(__file__).parent.parent.parent / "xor_static_key.txt"

# Expected length of the static XOR key.
STATIC_KEY_LENGTH: int = 1000


def extract_xor_first_bytes(messages: list[CapturedMessage]) -> list[int]:
    """Extract first XOR-encoded bytes from binary messages.

    Parses received messages, skips text messages, and extracts the first
    XOR-encoded data byte from each binary message.

    Args:
        messages: List of captured WebSocket messages.

    Returns:
        List of first XOR-encoded bytes from binary messages.
    """
    raw_first_bytes: list[int] = []

    for msg in messages:
        if msg["direction"] != "received":
            continue

        payload_b64 = msg["payload"]
        payload = base64.b64decode(payload_b64)

        if len(payload) < 4:
            continue

        # First 2 bytes are length header, byte[2] is message type
        msg_type = payload[2]

        # Skip text messages
        if msg_type in TEXT_MESSAGE_TYPES:
            continue

        # Binary messages have XOR-encoded data starting at byte[3]
        raw_first_bytes.append(payload[3])

    return raw_first_bytes


def find_best_static_byte(raw_first_bytes: list[int], magic_first_byte: int) -> tuple[int, int]:
    """Find the static key's first byte that maximizes known signature matches.

    Brute-forces all 256 possible values to find which static[0] produces
    the most known protocol signatures when XOR'd with captured data.

    Args:
        raw_first_bytes: First XOR-encoded bytes from binary messages.
        magic_first_byte: ASCII value of magic key's first character.

    Returns:
        Tuple of (best_static_byte, match_count).
    """
    best_static_0 = 0
    best_coverage = 0

    for static_0 in range(256):
        table_0 = static_0 ^ magic_first_byte
        known_count = sum(
            1 for raw_0 in raw_first_bytes if (raw_0 ^ table_0) in KNOWN_PROTOCOL_SIGNATURES
        )
        if known_count > best_coverage:
            best_coverage = known_count
            best_static_0 = static_0

    return best_static_0, best_coverage


def load_static_key() -> str:
    """Load the static XOR key from file.

    Returns:
        The 1000-character static key.

    Raises:
        FileNotFoundError: If key file does not exist.
        ValueError: If key is not exactly 1000 characters.
    """
    content = _test_hooks.read_text(STATIC_KEY_PATH)
    key = content.strip()
    if len(key) != STATIC_KEY_LENGTH:
        raise ValueError(f"Static key has {len(key)} chars, expected {STATIC_KEY_LENGTH}")
    return key


def save_static_key(key: str) -> None:
    """Save the static XOR key to file.

    Args:
        key: The 1000-character static key.

    Raises:
        ValueError: If key is not exactly 1000 characters.
    """
    if len(key) != STATIC_KEY_LENGTH:
        raise ValueError(f"Static key has {len(key)} chars, expected {STATIC_KEY_LENGTH}")
    _test_hooks.write_text(STATIC_KEY_PATH, key + "\n")


class BrowserError(Exception):
    """Base error for browser operations."""


class PlaywrightNotInstalledError(BrowserError):
    """Raised when Playwright hook is not installed."""


class GameNotJoinedError(BrowserError):
    """Raised when game doesn't load properly."""


def get_current_time_ms() -> int:
    """Get current time in milliseconds.

    Returns:
        Current Unix timestamp in milliseconds.
    """
    return int(time.time() * 1000)


# Global offset for CDP timestamp conversion (set on first message)
_cdp_time_offset_ms: int | None = None


def cdp_timestamp_to_ms(timestamp: float) -> int:
    """Convert CDP monotonic timestamp to Unix milliseconds.

    CDP timestamps are monotonic seconds since browser start.
    We calculate an offset on first call to convert to Unix time.

    Args:
        timestamp: CDP monotonic timestamp in seconds.

    Returns:
        Unix timestamp in milliseconds.
    """
    global _cdp_time_offset_ms
    cdp_ms = int(timestamp * 1000)
    if _cdp_time_offset_ms is None:
        # First call - calculate offset from current time
        _cdp_time_offset_ms = get_current_time_ms() - cdp_ms
    return cdp_ms + _cdp_time_offset_ms


def reset_cdp_time_offset() -> None:
    """Reset CDP time offset for new browser session."""
    global _cdp_time_offset_ms
    _cdp_time_offset_ms = None


class BrowserSession:
    """Base class for browser-based WebSocket capture.

    Handles common functionality:
    - Browser/CDP setup
    - WebSocket event handlers
    - Message capture and storage
    - Magic key capture

    Subclasses implement specific behavior (passive sniffing vs active probing).
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
    ) -> None:
        """Initialize browser session.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Skip guest login and use account credentials.
        """
        self._target_url = target_url
        self._headless = headless
        self._prefer_account = prefer_account
        self._session_id = str(uuid.uuid4())
        self._start_timestamp_ms = 0
        self._messages: list[CapturedMessage] = []
        self._ws_urls: dict[str, str] = {}  # requestId -> url mapping
        self._magic: str | None = None
        self._static_key: str | None = None
        self._game_log_scraper: GameLogScraper | None = None
        self._inventory_scraper: InventoryScraper | None = None
        self._combat_tracker: CombatTracker | None = None
        self._fuel_prober: FuelProber | None = None
        self._last_fuel_result: FuelProbeResult | None = None

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    @property
    def messages(self) -> list[CapturedMessage]:
        """Get captured messages."""
        return self._messages

    @property
    def magic(self) -> str | None:
        """Get captured magic key for XOR decoding."""
        return self._magic

    @property
    def static_key(self) -> str | None:
        """Get captured static XOR key from game JS."""
        return self._static_key

    def _init_game_log_scraper(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the game log scraper.

        Args:
            cdp: CDP session for DOM access.
        """
        self._game_log_scraper = GameLogScraper(cdp)
        log.info("Game log scraper initialized")

    def _poll_game_log(self) -> list[GameLogEntry]:
        """Poll for new game log entries, log them, and process combat.

        Returns:
            List of new entries found since last poll.
        """
        if self._game_log_scraper is None:
            return []
        new_entries = self._game_log_scraper.get_new_entries()
        for entry in new_entries:
            self._process_game_log_entry(entry)
        return new_entries

    def _process_game_log_entry(self, entry: GameLogEntry) -> None:
        """Process a single game log entry.

        Args:
            entry: The game log entry to process.
        """
        prefix = f"[GAME:{entry['category'].upper()}]"
        log.info("%s %s", prefix, entry["text"])
        # Process combat events
        if entry["category"] != "combat" or self._combat_tracker is None:
            return
        event = self._combat_tracker.process_log_line(entry["text"])
        if event is None:
            return
        self._combat_tracker.log_event(event)

    def _init_combat_tracker(self) -> None:
        """Initialize the combat tracker."""
        self._combat_tracker = CombatTracker()
        log.info("Combat tracker initialized")

    def _get_combat_events(self) -> list[CombatEvent]:
        """Get all recorded combat events.

        Returns:
            List of CombatEvents, or empty list if tracker not initialized.
        """
        if self._combat_tracker is None:
            return []
        return self._combat_tracker.get_events()

    def _init_inventory_scraper(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the inventory scraper.

        Args:
            cdp: CDP session for DOM access.
        """
        self._inventory_scraper = InventoryScraper(cdp)
        log.info("Inventory scraper initialized")

    def _poll_inventory(self) -> list[InventoryChange]:
        """Poll for inventory changes and log them.

        Returns:
            List of changes found since last poll.
        """
        if self._inventory_scraper is None:
            return []
        return self._inventory_scraper.log_changes()

    def _init_fuel_prober(self, cdp: CDPSessionProtocol) -> None:
        """Initialize the fuel prober.

        Args:
            cdp: CDP session for JavaScript execution.
        """
        self._fuel_prober = FuelProber(cdp)
        log.info("Fuel prober initialized")

    def _poll_fuel(self) -> FuelProbeResult | None:
        """Poll for fuel values and log findings.

        Returns:
            FuelProbeResult if prober initialized, None otherwise.
        """
        if self._fuel_prober is None:
            return None

        result = self._fuel_prober.probe()

        # Log any interesting findings
        if result["js_variables"]:
            for var in result["js_variables"]:
                log.info("[FUEL:JS] %s = %s", var["path"], var["value"])

        if result["dom_bars"]:
            for bar in result["dom_bars"]:
                if bar["width"]:
                    log.info(
                        "[FUEL:DOM] %s width=%s class=%s",
                        bar["tag"],
                        bar["width"],
                        bar["class_name"][:20] if bar["class_name"] else "",
                    )

        self._last_fuel_result = result
        return result

    def _on_websocket_created(self, params: JSONObject) -> None:
        """Handle Network.webSocketCreated CDP event.

        Args:
            params: CDP event parameters.
        """
        event = decode_cdp_websocket_created_event(params)
        self._ws_urls[event["requestId"]] = event["url"]

    def _on_websocket_frame_received(self, params: JSONObject) -> None:
        """Handle Network.webSocketFrameReceived CDP event.

        Args:
            params: CDP event parameters.
        """
        self._record_frame(params, "received")

    def _on_websocket_frame_sent(self, params: JSONObject) -> None:
        """Handle Network.webSocketFrameSent CDP event.

        Args:
            params: CDP event parameters.
        """
        self._record_frame(params, "sent")

    def _record_frame(self, params: JSONObject, direction: MessageDirection) -> None:
        """Record a WebSocket frame.

        Args:
            params: CDP event parameters.
            direction: Whether the frame was sent or received.
        """
        event = decode_cdp_websocket_frame_event(params)
        request_id = event["requestId"]
        ws_url = self._ws_urls.get(request_id, "unknown")
        payload = event["response"]["payloadData"]

        message = CapturedMessage(
            timestamp_ms=cdp_timestamp_to_ms(event["timestamp"]),
            direction=direction,
            payload=payload,
            ws_url=ws_url,
        )
        self._messages.append(message)
        self._on_message_captured(message)

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Called when a message is captured. Override in subclasses.

        Args:
            message: The captured message.
        """

    def _setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        """Set up CDP event handlers for WebSocket capture.

        Also installs a WebSocket prototype hook to capture the game's
        WebSocket instance for later command injection.

        Args:
            cdp: CDP session.
        """
        # Enable Page domain first - required for script injection to work
        cdp.send("Page.enable")

        # Install WebSocket prototype hook BEFORE any page loads
        # This captures the WebSocket instance when the game first sends data
        cdp.send(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
            (function() {
                window.__capturedWS = null;
                const origSend = WebSocket.prototype.send;
                WebSocket.prototype.send = function(data) {
                    if (!window.__capturedWS && this.readyState === 1) {
                        window.__capturedWS = this;
                    }
                    return origSend.call(this, data);
                };
            })();
            """
            },
        )

        # Enable Network domain for WebSocket frame capture
        cdp.send("Network.enable")
        cdp.on("Network.webSocketCreated", self._on_websocket_created)
        cdp.on("Network.webSocketFrameReceived", self._on_websocket_frame_received)
        cdp.on("Network.webSocketFrameSent", self._on_websocket_frame_sent)

    def _setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        """Set up console message listener for WebSocket debug info.

        Logs console messages containing 'WS', 'Hook', or 'WebSocket'.

        Args:
            cdp: CDP session.
        """
        cdp.send("Runtime.enable")

        def on_console(params: JSONObject) -> None:
            msg_type = params.get("type", "?")
            args = params.get("args", [])
            if isinstance(args, list):
                texts = []
                for a in args:
                    if isinstance(a, dict):
                        val = a.get("value", a.get("description", "?"))
                        texts.append(str(val) if val is not None else "?")
                text = " ".join(texts)
                if "WS" in text or "Hook" in text or "WebSocket" in text:
                    log.info("[Console %s] %s", msg_type, text)

        cdp.on("Runtime.consoleAPICalled", on_console)

    def _log_websocket_urls(self) -> None:
        """Log all captured WebSocket URLs."""
        ws_urls = list(self._ws_urls.values())
        log.info("Captured WebSocket URLs: %s", ws_urls)

    def _debug_js_websocket(self, cdp: CDPSessionProtocol) -> None:
        """Check for WebSocket instances in JavaScript and log findings.

        Args:
            cdp: CDP session.
        """
        debug_js = """
        (() => {
            let found = [];
            for (let key in window) {
                try {
                    if (window[key] instanceof WebSocket) {
                        found.push('window.' + key + ' (state=' + window[key].readyState + ')');
                    }
                } catch(e) {}
            }
            if (typeof tankpit !== 'undefined') {
                for (let key in tankpit) {
                    try {
                        if (tankpit[key] instanceof WebSocket) {
                            let s = tankpit[key].readyState;
                            found.push('tankpit.' + key + ' (state=' + s + ')');
                        }
                    } catch(e) {}
                }
            }
            if (window.__capturedWS) {
                found.push('__capturedWS (state=' + window.__capturedWS.readyState + ')');
            }
            return found.length > 0 ? found.join(', ') : 'NO WebSocket found';
        })()
        """
        debug_result = cdp.send("Runtime.evaluate", {"expression": debug_js, "returnByValue": True})
        debug_obj = debug_result.get("result")
        if isinstance(debug_obj, dict):
            debug_val = debug_obj.get("value", "?")
            log.info("JS WebSocket check: %s", debug_val)

    def _log_script_urls(self, page: PageProtocol) -> None:
        """Log all loaded script URLs for protocol analysis.

        Args:
            page: Playwright page.
        """
        script_urls = page.evaluate(
            "Array.from(document.querySelectorAll('script[src]')).map(s => s.src)"
        )
        if script_urls and isinstance(script_urls, list):
            log.info("Loaded scripts (%d):", len(script_urls))
            for url in script_urls:
                if isinstance(url, str):
                    log.info("  - %s", url)

    def _gather_intel(self, page: PageProtocol, cdp: CDPSessionProtocol) -> None:
        """Gather and log all available intel after login.

        Logs WebSocket URLs, JS WebSocket instances, and script URLs.

        Args:
            page: Playwright page.
            cdp: CDP session.
        """
        self._log_websocket_urls()
        self._debug_js_websocket(cdp)
        self._log_script_urls(page)
        self._capture_static_key(page)

    def _capture_magic_key(self, page: PageProtocol) -> None:
        """Capture tankpit.magic XOR key from page.

        Args:
            page: Playwright page.
        """
        magic_value = page.evaluate("tankpit.magic")
        if isinstance(magic_value, str) and len(magic_value) > 0:
            self._magic = magic_value
            log.info("Captured magic key: %s...", magic_value[:20])

    def _capture_static_key(self, page: PageProtocol) -> None:
        """Extract static XOR key from tpclient JS source.

        Args:
            page: Playwright page.
        """
        import re

        # Wait for tpclient script to be loaded
        js_check_loaded = (
            "Array.from(document.querySelectorAll('script[src]'))"
            ".some(s => s.src.includes('tpclient'))"
        )
        page.wait_for_function(js_check_loaded, timeout=10000)

        # Get the URL
        js_get_url = (
            "Array.from(document.querySelectorAll('script[src]'))"
            ".find(s => s.src.includes('tpclient'))?.src"
        )
        tpclient_url = page.evaluate(js_get_url)
        if not isinstance(tpclient_url, str):
            log.warning("Could not find tpclient script URL")
            return

        # Fetch the JS content
        js_content = page.evaluate(f"fetch('{tpclient_url}').then(r => r.text())")
        if not isinstance(js_content, str):
            log.warning("Could not fetch tpclient JS content")
            return

        # Extract 1000-char static key: any 1000-char quoted string
        match = re.search(r'"([^"]{1000})"', js_content)
        if not match:
            log.warning("Could not find static key in tpclient JS")
            return

        static_key: str = match.group(1)
        self._static_key = static_key
        save_static_key(static_key)
        log.info("Captured static key: %s...", static_key[:20])

    def _derive_static_key_from_messages(self) -> None:
        """Derive static key by analyzing captured messages.

        Uses brute-force to find which static[0] value makes captured messages
        decode to known protocol signatures. This is robust against game updates.

        Raises:
            FileNotFoundError: If static key file does not exist.
            ValueError: If static key file has invalid content.
        """
        if not self._magic or not self._messages:
            return

        raw_first_bytes = extract_xor_first_bytes(self._messages)
        if not raw_first_bytes:
            log.warning("No binary messages to derive static key from")
            return

        magic_0 = ord(self._magic[0])
        best_static_0, best_coverage = find_best_static_byte(raw_first_bytes, magic_0)

        if best_coverage == 0:
            log.warning("Could not derive static key - no known signatures matched")
            return

        pct = 100 * best_coverage / len(raw_first_bytes)
        log.info(
            "Derived static[0]=0x%02x (%r) with %.1f%% coverage (%d/%d)",
            best_static_0,
            chr(best_static_0),
            pct,
            best_coverage,
            len(raw_first_bytes),
        )

        current_key = load_static_key()
        new_key = chr(best_static_0) + current_key[1:]

        if new_key != current_key:
            self._static_key = new_key
            save_static_key(new_key)
            log.info("Updated static key file: %s", STATIC_KEY_PATH)

    def _send_websocket_bytes(self, cdp: CDPSessionProtocol, data: bytes) -> str:
        """Send raw bytes via the captured WebSocket.

        Uses the WebSocket instance captured by the prototype hook installed
        in _setup_cdp_handlers, with fallbacks to tankpit.ws and window.ws.

        Args:
            cdp: CDP session.
            data: Raw bytes to send.

        Returns:
            Status string: 'SENT_N_BYTES via URL' on success, error message otherwise.
        """
        import base64

        b64 = base64.b64encode(data).decode()

        send_js = f"""
        (() => {{
            // Use captured WebSocket from prototype hook
            let ws = window.__capturedWS;

            // Fallback: try common locations
            if (!ws && typeof tankpit !== 'undefined' && tankpit.ws) {{
                ws = tankpit.ws;
            }}
            if (!ws && typeof window.ws !== 'undefined') {{
                ws = window.ws;
            }}

            if (!ws) {{
                let status = window.__capturedWS ? 'exists' : 'null';
                return 'NO_WEBSOCKET_FOUND (__capturedWS=' + status + ')';
            }}

            if (ws.readyState !== 1) {{
                return 'WEBSOCKET_NOT_OPEN: ' + ws.readyState;
            }}

            // Decode base64 to binary
            const binary = atob('{b64}');
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {{
                bytes[i] = binary.charCodeAt(i);
            }}

            // Send as binary
            ws.send(bytes.buffer);
            return 'SENT_' + bytes.length + '_BYTES via ' + ws.url;
        }})()
        """
        result = cdp.send("Runtime.evaluate", {"expression": send_js, "returnByValue": True})
        result_obj = result.get("result")
        if isinstance(result_obj, dict):
            val = result_obj.get("value", "?")
            return str(val) if val is not None else "?"
        return "?"

    def _wait_for_game_ready(self, page: PageProtocol) -> None:
        """Wait for game to fully load (message flow stabilizes).

        Args:
            page: Playwright page.

        Raises:
            GameNotJoinedError: If no messages captured.
        """
        log.info("Waiting for game to initialize...")
        page.wait_for_timeout(2000.0)  # Initial wait

        # Wait until no new messages for 1 second
        last_count = len(self._messages)
        stable_checks = 0
        while stable_checks < 3:  # Need 3 consecutive stable checks
            page.wait_for_timeout(500.0)
            current_count = len(self._messages)
            if current_count == last_count:
                stable_checks += 1
            else:
                stable_checks = 0
                last_count = current_count

        if len(self._messages) == 0:
            raise GameNotJoinedError("No WebSocket messages captured - game may not have loaded")

        log.info("Game ready, captured %d initial messages", len(self._messages))

    def _launch_browser(
        self,
    ) -> tuple[BrowserProtocol, BrowserContextProtocol, PageProtocol, CDPSessionProtocol]:
        """Launch browser and set up CDP session.

        Returns:
            Tuple of (browser, context, page, cdp).

        Raises:
            PlaywrightNotInstalledError: If Playwright not installed.
        """
        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError(
                "Playwright is not installed. Call _test_hooks._install_real_playwright() first."
            )

        # This returns a context manager, we need to handle it properly
        # The caller should use this within a with block or manage cleanup
        playwright = _test_hooks.sync_playwright()
        pw = playwright.__enter__()
        browser = pw.chromium.launch(headless=self._headless)
        context = browser.new_context()
        page = context.new_page()
        cdp = context.new_cdp_session(page)

        # Reset CDP time offset for new session
        reset_cdp_time_offset()
        self._setup_cdp_handlers(cdp)

        return browser, context, page, cdp

    def _navigate_and_login(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        *,
        tank_name_prefix: str = "T",
        auto_join_room: bool = False,
    ) -> None:
        """Navigate to target URL and handle login.

        Args:
            page: Playwright page.
            cdp: CDP session.
            tank_name_prefix: Prefix for tank name.
            auto_join_room: Whether to automatically join a room.
        """
        page.goto(self._target_url, wait_until="domcontentloaded")
        log.info("Navigated to %s", page.url)

        handle_login_flow(
            page,
            cdp,
            tank_name_prefix=tank_name_prefix,
            prefer_account=self._prefer_account,
            auto_join_room=auto_join_room,
        )

        self._capture_magic_key(page)

    def _cleanup(
        self,
        cdp: CDPSessionProtocol,
        page: PageProtocol,
        context: BrowserContextProtocol,
        browser: BrowserProtocol,
    ) -> None:
        """Clean up browser resources.

        Args:
            cdp: CDP session.
            page: Playwright page.
            context: Browser context.
            browser: Browser instance.
        """
        cdp.detach()
        page.close()
        context.close()
        browser.close()


__all__ = [
    "BrowserError",
    "BrowserSession",
    "GameNotJoinedError",
    "PlaywrightNotInstalledError",
    "cdp_timestamp_to_ms",
    "extract_xor_first_bytes",
    "find_best_static_byte",
    "get_current_time_ms",
    "load_static_key",
    "reset_cdp_time_offset",
    "save_static_key",
]
