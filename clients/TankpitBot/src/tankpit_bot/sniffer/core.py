"""Core WebSocket sniffer class and entry points.

This module provides the main WebSocketSniffer class that captures
WebSocket traffic from TankPit using Playwright and CDP.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.browser import (
    BrowserSession,
    GameLogEntry,
    PlaywrightNotInstalledError,
    get_current_time_ms,
    reset_cdp_time_offset,
)
from tankpit_bot.capture import build_session_summary
from tankpit_bot.sniffer.constants import (
    DEFAULT_CAPTURE_DURATION_MS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TARGET_URL,
)
from tankpit_bot.sniffer.decoders import decode_message, process_received_message
from tankpit_bot.sniffer.trackers import (
    init_trackers_with_magic,
    mine_tracker,
    tank_tracker,
)
from tankpit_bot.sniffer.viewport import reset_viewport_tracking
from tankpit_bot.types import (
    CapturedMessage,
    CaptureSession,
    encode_capture_session,
    encode_session_summary,
)

log = get_logger(__name__)


def _log_js_fuel_findings(result: JSONObject) -> None:
    """Log JavaScript fuel probe findings from CDP Runtime.evaluate result.

    CDP Runtime.evaluate with returnByValue returns:
    {"result": {"type": "array", "value": [...]}} for JS arrays.

    Args:
        result: CDP Runtime.evaluate result.
    """
    result_wrapper = result.get("result")
    if not isinstance(result_wrapper, dict):
        return
    findings = result_wrapper.get("value")
    if not isinstance(findings, list):
        return
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        path = finding.get("path")
        val = finding.get("value")
        if path is not None and val is not None:
            log.info("[JS:FUEL] %s = %s", path, val)


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
        self._game_log_entries.append(
            {
                "timestamp_ms": get_current_time_ms(),
                "text": entry["text"],
                "category": entry["category"],
            }
        )
        # Call parent to log and process combat
        super()._process_game_log_entry(entry)

    def _on_magic_captured(self, magic: str) -> None:
        """Initialize trackers when magic key is captured.

        Args:
            magic: The session magic string.
        """
        init_trackers_with_magic(magic)

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Log decoded message if live decode is enabled.

        Also polls game log and inventory for correlation with WebSocket events.

        Args:
            message: The captured message.
        """
        super()._on_message_captured(message)

        payload = message["payload"]
        direction = message["direction"]

        if not self._live_decode:
            return

        if direction == "received":
            # Use unified decoder for received messages
            process_received_message(payload)
        else:
            # Use simple decoder for sent messages
            mine_status = mine_tracker.process_message(payload, "sent")
            if mine_status:
                log.info(mine_status)
            decoded = decode_message(payload, direction, self._magic)
            log.info(decoded)

        self._poll_game_log()

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
        _log_js_fuel_findings(result)

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

            # Reset session state for new session
            reset_cdp_time_offset()
            reset_viewport_tracking()

            # Set up console listener and CDP handlers
            self._setup_console_listener(cdp)
            self._setup_cdp_handlers(cdp)

            # Navigate to target URL
            page.goto(self._target_url, wait_until="domcontentloaded")
            log.info("Landed on %s", page.url)

            # Handle login
            self._navigate_and_login(page, cdp, tank_name_prefix="B", auto_join_room=True)

            # Gather all available intel
            self._gather_intel(page, cdp)

            # Initialize DOM scraper for game log and combat tracker
            self._init_game_log_scraper(cdp)
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
        tank_names = {str(k): v for k, v in tank_tracker.get_all_names().items()}

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
    summary = build_session_summary(session)
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
    "SnifferError",
    "WebSocketSniffer",
    "main",
    "run_sniffer",
]
