"""Protocol probe using Playwright and CDP input injection.

Automatically discovers game commands by:
1. Launching a browser and joining the game
2. Injecting keyboard and mouse inputs via CDP
3. Capturing WebSocket messages sent after each input
4. Correlating inputs with protocol messages
"""

from __future__ import annotations

import time
import uuid
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
from tankpit_bot.login import handle_login_flow
from tankpit_bot.types import (
    CapturedMessage,
    KeyInput,
    MessageDirection,
    MouseInput,
    ProbeInput,
    ProbeResult,
    ProbeSession,
    decode_cdp_websocket_created_event,
    decode_cdp_websocket_frame_event,
    encode_probe_session,
)

log = get_logger(__name__)


class ProbeError(Exception):
    """Raised when probe encounters an error."""


class PlaywrightNotInstalledError(ProbeError):
    """Raised when Playwright hook is not installed."""


class GameNotJoinedError(ProbeError):
    """Raised when failed to join the game for probing."""


def _get_current_time_ms() -> int:
    """Get current time in milliseconds.

    Returns:
        Current Unix timestamp in milliseconds.
    """
    return int(time.time() * 1000)


def _cdp_timestamp_to_ms(timestamp: float) -> int:
    """Convert CDP monotonic timestamp to milliseconds.

    Args:
        timestamp: CDP monotonic timestamp in seconds.

    Returns:
        Timestamp in milliseconds.
    """
    return int(timestamp * 1000)


# Default keys to probe
DEFAULT_PROBE_KEYS: list[str] = [
    "w",
    "a",
    "s",
    "d",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    " ",
    "1",
    "2",
    "3",
    "4",
    "5",
    "q",
    "e",
    "r",
    "f",
]

# Default mouse positions to probe (as fractions of viewport)
DEFAULT_MOUSE_POSITIONS: list[tuple[float, float]] = [
    (0.5, 0.5),  # Center
    (0.25, 0.25),  # Top-left quadrant
    (0.75, 0.25),  # Top-right quadrant
    (0.25, 0.75),  # Bottom-left quadrant
    (0.75, 0.75),  # Bottom-right quadrant
]


class ProtocolProbe:
    """Probes game protocol by injecting inputs and capturing responses.

    Uses Playwright to launch a browser, joins the game, then systematically
    injects keyboard and mouse inputs while capturing WebSocket messages.
    """

    def __init__(self, target_url: str, *, headless: bool = False) -> None:
        """Initialize the probe.

        Args:
            target_url: URL to navigate to (e.g., https://tankpit.com/play).
            headless: Whether to run the browser in headless mode.
        """
        self._target_url = target_url
        self._headless = headless
        self._session_id = str(uuid.uuid4())
        self._start_timestamp_ms = 0
        self._messages: list[CapturedMessage] = []
        self._ws_urls: dict[str, str] = {}
        self._results: list[ProbeResult] = []

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

        message = CapturedMessage(
            timestamp_ms=_cdp_timestamp_to_ms(event["timestamp"]),
            direction=direction,
            payload=event["response"]["payloadData"],
            ws_url=ws_url,
        )
        self._messages.append(message)

    def _inject_key(
        self,
        cdp: _test_hooks.CDPSessionProtocol,
        key: str,
    ) -> None:
        """Inject a key press via CDP.

        Args:
            cdp: CDP session.
            key: Key to press (e.g., 'w', 'ArrowUp', ' ').
        """
        # Map special keys to their DOM key values
        key_map: dict[str, str] = {
            " ": "Space",
        }
        dom_key = key_map.get(key, key)

        # keyDown
        cdp.send(
            "Input.dispatchKeyEvent",
            {
                "type": "keyDown",
                "key": dom_key,
                "code": f"Key{key.upper()}" if len(key) == 1 and key.isalpha() else dom_key,
                "text": key if len(key) == 1 else "",
            },
        )
        # keyUp
        cdp.send(
            "Input.dispatchKeyEvent",
            {
                "type": "keyUp",
                "key": dom_key,
                "code": f"Key{key.upper()}" if len(key) == 1 and key.isalpha() else dom_key,
            },
        )

    def _inject_mouse_click(
        self,
        cdp: _test_hooks.CDPSessionProtocol,
        x: int,
        y: int,
    ) -> None:
        """Inject a mouse click via CDP.

        Args:
            cdp: CDP session.
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
        """
        # mousePressed
        cdp.send(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": 1,
            },
        )
        # mouseReleased
        cdp.send(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": 1,
            },
        )

    def _get_sent_messages_since(self, count: int) -> list[CapturedMessage]:
        """Get sent messages since a given message count.

        Args:
            count: Number of messages at the start.

        Returns:
            List of sent messages captured after that point.
        """
        new_messages = self._messages[count:]
        return [m for m in new_messages if m["direction"] == "sent"]

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
            wait_after_join_ms: Time to wait after joining game.
            wait_after_input_ms: Time to wait after each input for response.

        Returns:
            ProbeSession with all captured results.

        Raises:
            PlaywrightNotInstalledError: If Playwright hook is not installed.
            GameNotJoinedError: If failed to join game.
        """
        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError(
                "Playwright is not installed. Run probe from main() entry point."
            )

        self._start_timestamp_ms = _get_current_time_ms()
        self._messages = []
        self._ws_urls = {}
        self._results = []

        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self._headless)
            context = browser.new_context()
            page = context.new_page()

            # Create CDP session
            cdp = context.new_cdp_session(page)
            cdp.send("Network.enable")

            # Register WebSocket handlers
            cdp.on("Network.webSocketCreated", self._on_websocket_created)
            cdp.on("Network.webSocketFrameReceived", self._on_websocket_frame_received)
            cdp.on("Network.webSocketFrameSent", self._on_websocket_frame_sent)

            # Navigate to game
            page.goto(self._target_url, wait_until="domcontentloaded")
            log.info("Navigated to %s", page.url)

            # Handle login if needed
            handle_login_flow(page, cdp, tank_name_prefix="P")

            # Wait for game to fully load and join
            page.wait_for_timeout(float(wait_after_join_ms))

            # Check if we have WebSocket messages (indicates game joined)
            if len(self._messages) == 0:
                cdp.detach()
                page.close()
                context.close()
                browser.close()
                raise GameNotJoinedError(
                    "No WebSocket messages captured - game may not have loaded"
                )

            log.info("Game joined, captured %d initial messages", len(self._messages))

            # Get viewport size for mouse coordinate calculation
            viewport_result = cdp.send(
                "Runtime.evaluate",
                {
                    "expression": "JSON.stringify({w: window.innerWidth, h: window.innerHeight})",
                    "returnByValue": True,
                },
            )
            viewport_raw = viewport_result.get("result")
            default_viewport = '{"w":800,"h":600}'
            viewport_str = default_viewport
            if type(viewport_raw) is dict:
                viewport_val = viewport_raw.get("value")
                if type(viewport_val) is str:
                    viewport_str = viewport_val

            # Parse viewport using platform_core JSON utils
            viewport_parsed = load_json_str(viewport_str)
            viewport_data = narrow_json_to_dict(viewport_parsed)
            viewport_w = require_int(viewport_data, "w")
            viewport_h = require_int(viewport_data, "h")
            log.info("Viewport size: %dx%d", viewport_w, viewport_h)

            # Probe keyboard inputs
            for key in probe_keys:
                msg_count_before = len(self._messages)
                timestamp = _get_current_time_ms()

                log.info("Probing key: %s", key)
                self._inject_key(cdp, key)

                # Wait for response
                page.wait_for_timeout(float(wait_after_input_ms))

                # Capture result
                sent_after = self._get_sent_messages_since(msg_count_before)
                key_input = KeyInput(key=key)
                probe_input = ProbeInput(
                    input_type="key",
                    key_input=key_input,
                    mouse_input=None,
                )
                result = ProbeResult(
                    input=probe_input,
                    timestamp_ms=timestamp,
                    messages_before_count=msg_count_before,
                    messages_after=sent_after,
                )
                self._results.append(result)

                if len(sent_after) > 0:
                    log.info("  -> %d message(s) sent", len(sent_after))

            # Probe mouse inputs
            for x_frac, y_frac in probe_mouse_positions:
                x = int(x_frac * viewport_w)
                y = int(y_frac * viewport_h)
                msg_count_before = len(self._messages)
                timestamp = _get_current_time_ms()

                log.info("Probing mouse click at (%d, %d)", x, y)
                self._inject_mouse_click(cdp, x, y)

                # Wait for response
                page.wait_for_timeout(float(wait_after_input_ms))

                # Capture result
                sent_after = self._get_sent_messages_since(msg_count_before)
                mouse_input = MouseInput(x=x, y=y, button="left")
                probe_input = ProbeInput(
                    input_type="mouse",
                    key_input=None,
                    mouse_input=mouse_input,
                )
                result = ProbeResult(
                    input=probe_input,
                    timestamp_ms=timestamp,
                    messages_before_count=msg_count_before,
                    messages_after=sent_after,
                )
                self._results.append(result)

                if len(sent_after) > 0:
                    log.info("  -> %d message(s) sent", len(sent_after))

            # Cleanup
            cdp.detach()
            page.close()
            context.close()
            browser.close()

        end_timestamp_ms = _get_current_time_ms()

        return ProbeSession(
            session_id=self._session_id,
            start_timestamp_ms=self._start_timestamp_ms,
            end_timestamp_ms=end_timestamp_ms,
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

    # Save to file
    encoded = encode_probe_session(session)
    json_str = dump_json_str(encoded, compact=False, indent=2)
    _test_hooks.write_text(Path(output_path), json_str)

    return session


def main() -> None:
    """Entry point for tankpit-probe command.

    Reads configuration from environment variables:
    - TANKPIT_URL: Target URL (default: https://tankpit.com/play)
    - TANKPIT_PROBE_OUTPUT: Output file (default: probe_session.json)
    - TANKPIT_HEADLESS: Run headless (default: false)
    - TANKPIT_WAIT_JOIN_MS: Wait after join (default: 5000)
    - TANKPIT_WAIT_INPUT_MS: Wait after each input (default: 200)
    """
    from dotenv import load_dotenv

    load_dotenv()
    setup_rich_logging(level="INFO")

    # Install real Playwright hook
    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    # Read config
    target_url = _test_hooks.get_env("TANKPIT_URL")
    if target_url is None:
        target_url = "https://tankpit.com/play"

    output_path = _test_hooks.get_env("TANKPIT_PROBE_OUTPUT")
    if output_path is None:
        output_path = "probe_session.json"

    headless_str = _test_hooks.get_env("TANKPIT_HEADLESS")
    headless = headless_str is not None and headless_str.lower() in ("true", "1", "yes")

    wait_join_str = _test_hooks.get_env("TANKPIT_WAIT_JOIN_MS")
    wait_after_join_ms = 5000
    if wait_join_str is not None:
        wait_after_join_ms = int(wait_join_str)

    wait_input_str = _test_hooks.get_env("TANKPIT_WAIT_INPUT_MS")
    wait_after_input_ms = 200
    if wait_input_str is not None:
        wait_after_input_ms = int(wait_input_str)

    session = run_probe(
        target_url,
        output_path,
        headless=headless,
        wait_after_join_ms=wait_after_join_ms,
        wait_after_input_ms=wait_after_input_ms,
    )

    # Log summary
    results_with_messages = [r for r in session["results"] if len(r["messages_after"]) > 0]
    log.info(
        "Probe complete: %d inputs tested, %d generated messages",
        len(session["results"]),
        len(results_with_messages),
    )
    log.info("Saved to: %s", output_path)

    # Print discovered commands
    _log_discovered_commands(results_with_messages)


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


__all__ = [
    "DEFAULT_MOUSE_POSITIONS",
    "DEFAULT_PROBE_KEYS",
    "GameNotJoinedError",
    "PlaywrightNotInstalledError",
    "ProbeError",
    "ProtocolProbe",
    "main",
    "run_probe",
]
