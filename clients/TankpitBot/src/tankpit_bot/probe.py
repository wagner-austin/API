"""Protocol probe using Playwright and CDP input injection.

Automatically discovers game commands by:
1. Launching a browser and joining the game
2. Injecting keyboard and mouse inputs via CDP
3. Capturing WebSocket messages sent after each input
4. Correlating inputs with protocol messages
"""

from __future__ import annotations

import threading
from pathlib import Path

from platform_core.json_utils import (
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


class ProbeError(Exception):
    """Raised when probe encounters an error."""


# =============================================================================
# Default probe configuration
# =============================================================================

# Standard game action keys to probe
DEFAULT_PROBE_KEYS: list[str] = [
    "w",  # Forward
    "s",  # Brake
    "d",  # Right turn
    " ",  # Fire
    "r",  # Radar
    "x",  # Use item
    "f",  # Map open
    "f",  # Map close
]

# Mouse positions for aim testing (empty by default)
DEFAULT_MOUSE_POSITIONS: list[tuple[float, float]] = []


class ProtocolProbe(BrowserSession):
    """Probes game protocol by injecting inputs and capturing responses.

    Extends BrowserSession with input injection and result capture.
    """

    def __init__(self, target_url: str, *, headless: bool = False) -> None:
        """Initialize the probe.

        Args:
            target_url: URL to navigate to (e.g., https://tankpit.com/play).
            headless: Whether to run the browser in headless mode.
        """
        super().__init__(target_url, headless=headless, prefer_account=True)
        self._results: list[ProbeResult] = []
        self._received_event = threading.Event()

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Signal when a received message arrives.

        Args:
            message: The captured message.
        """
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

    def _inject_key(self, cdp: CDPSessionProtocol, key: str) -> None:
        """Inject a key press via CDP.

        Args:
            cdp: CDP session.
            key: Key to press (e.g., 'w', 'ArrowUp', ' ').
        """
        key_map: dict[str, str] = {" ": "Space"}
        dom_key = key_map.get(key, key)
        key_code = f"Key{key.upper()}" if len(key) == 1 and key.isalpha() else dom_key

        log.info("    CDP keyDown: key=%s code=%s", dom_key, key_code)

        text = key if len(key) == 1 else ""
        cdp.send(
            "Input.dispatchKeyEvent",
            {"type": "keyDown", "key": dom_key, "code": key_code, "text": text},
        )
        cdp.send(
            "Input.dispatchKeyEvent",
            {"type": "keyUp", "key": dom_key, "code": key_code},
        )

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
        """Probe a single key input.

        Args:
            page: Playwright page.
            cdp: CDP session.
            key: Key to press.
        """
        msg_count_before = len(self._messages)
        timestamp = get_current_time_ms()

        log.info("Probing key: %s (msg_count_before=%d)", key, msg_count_before)
        self._inject_key(cdp, key)

        self._wait_for_response(page, msg_count_before)

        all_after = self._messages[msg_count_before:]
        sent_after = [m for m in all_after if m["direction"] == "sent"]
        recv_after = [m for m in all_after if m["direction"] == "received"]

        log.info("  -> sent %d, recv %d", len(sent_after), len(recv_after))
        for msg in all_after:
            preview = msg["payload"][:60] if len(msg["payload"]) > 60 else msg["payload"]
            log.info("    %s: %s", msg["direction"].upper(), preview)

        key_input = KeyInput(key=key)
        probe_input = ProbeInput(input_type="key", key_input=key_input, mouse_input=None)
        result = ProbeResult(
            input=probe_input,
            timestamp_ms=timestamp,
            messages_before_count=msg_count_before,
            messages_after=sent_after,
        )
        self._results.append(result)

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
        viewport_w: int,
        viewport_h: int,
    ) -> None:
        """Probe a list of keys.

        Args:
            page: Playwright page.
            cdp: CDP session.
            keys: Keys to probe.
            viewport_w: Viewport width.
            viewport_h: Viewport height.
        """
        for key in keys:
            log.info("Pressing Escape to close chat/menus")
            self._inject_key(cdp, "Escape")
            page.wait_for_timeout(100.0)

            focus_x = viewport_w // 4
            focus_y = viewport_h // 4
            log.info("Clicking (%d, %d) to focus game canvas", focus_x, focus_y)
            self._inject_mouse_click(cdp, focus_x, focus_y)
            page.wait_for_timeout(200.0)

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

        Args:
            probe_keys: List of keys to test.
            probe_mouse_positions: List of (x_fraction, y_fraction) positions.
            wait_after_join_ms: Time to wait after joining game (unused, kept for API compat).
            wait_after_input_ms: Time to wait after each input (unused, kept for API compat).

        Returns:
            ProbeSession with all captured results.

        Raises:
            PlaywrightNotInstalledError: If Playwright hook is not installed.
            GameNotJoinedError: If failed to join game.
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

            self._setup_cdp_handlers(cdp)
            self._navigate_and_login(page, cdp, tank_name_prefix="P", auto_join_room=True)
            self._wait_for_game_ready(page)

            viewport_w, viewport_h = self._get_viewport_size(cdp)
            log.info("Viewport size: %dx%d", viewport_w, viewport_h)

            self._probe_keys(page, cdp, probe_keys, viewport_w, viewport_h)
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

    session = run_probe(
        target_url,
        output_path,
        headless=headless,
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
    "DEFAULT_MOUSE_POSITIONS",
    "DEFAULT_PROBE_KEYS",
    "GameNotJoinedError",
    "PlaywrightNotInstalledError",
    "ProbeError",
    "ProtocolProbe",
    "_log_discovered_commands",
    "main",
    "run_probe",
]
