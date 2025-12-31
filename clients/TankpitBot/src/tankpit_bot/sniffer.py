"""WebSocket traffic sniffer using Playwright and CDP.

Captures WebSocket messages from tankpit.com by:
1. Launching a Chromium browser via Playwright
2. Creating a CDP session to intercept Network events
3. Navigating to the target URL
4. Recording all WebSocket frames (sent and received)
5. Saving the capture session to a JSON file
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.types import (
    CapturedMessage,
    CaptureSession,
    MessageDirection,
    decode_cdp_websocket_created_event,
    decode_cdp_websocket_frame_event,
    encode_capture_session,
)

log = get_logger(__name__)


class SnifferError(Exception):
    """Raised when sniffer encounters an error."""


class PlaywrightNotInstalledError(SnifferError):
    """Raised when Playwright hook is not installed."""


def _get_current_time_ms() -> int:
    """Get current time in milliseconds.

    Returns:
        Current Unix timestamp in milliseconds.
    """
    return int(time.time() * 1000)


def _cdp_timestamp_to_ms(timestamp: float) -> int:
    """Convert CDP monotonic timestamp to approximate Unix milliseconds.

    CDP timestamps are monotonic and relative to some unspecified epoch.
    We convert to approximate wall-clock time by using the current time
    as a reference point.

    Args:
        timestamp: CDP monotonic timestamp in seconds.

    Returns:
        Approximate Unix timestamp in milliseconds.
    """
    # CDP timestamps are relative, so we use current time as approximation
    # This is imprecise but sufficient for ordering and debugging
    return int(timestamp * 1000)


class WebSocketSniffer:
    """Captures WebSocket traffic from a browser session.

    Uses Playwright to launch a browser and CDP to intercept WebSocket frames.
    All captured messages are stored in a CaptureSession structure.
    """

    def __init__(self, target_url: str, *, headless: bool = False) -> None:
        """Initialize the sniffer.

        Args:
            target_url: URL to navigate to and capture WebSocket traffic from.
            headless: Whether to run the browser in headless mode.
        """
        self._target_url = target_url
        self._headless = headless
        self._session_id = str(uuid.uuid4())
        self._start_timestamp_ms = 0
        self._messages: list[CapturedMessage] = []
        self._ws_urls: dict[str, str] = {}  # requestId -> url mapping

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

    def run(self, capture_duration_ms: int) -> CaptureSession:
        """Run the sniffer and capture WebSocket traffic.

        Args:
            capture_duration_ms: How long to capture traffic in milliseconds.

        Returns:
            CaptureSession containing all captured messages.

        Raises:
            PlaywrightNotInstalledError: If Playwright hook is not installed.
        """
        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError(
                "Playwright is not installed. Call _test_hooks._install_real_playwright() first."
            )

        self._start_timestamp_ms = _get_current_time_ms()
        self._messages = []
        self._ws_urls = {}

        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self._headless)
            context = browser.new_context()
            page = context.new_page()

            # Create CDP session for network interception
            cdp = context.new_cdp_session(page)
            cdp.send("Network.enable")

            # Register WebSocket event handlers
            cdp.on("Network.webSocketCreated", self._on_websocket_created)
            cdp.on("Network.webSocketFrameReceived", self._on_websocket_frame_received)
            cdp.on("Network.webSocketFrameSent", self._on_websocket_frame_sent)

            # Navigate to target URL
            page.goto(self._target_url, wait_until="domcontentloaded")

            # Log actual URL after navigation
            log.info("Landed on %s", page.url)

            # If on before-playing page, try to create guest tank
            if "before-playing" in page.url:
                log.info("Attempting guest login...")
                # Wait for page to fully load
                page.wait_for_timeout(2000.0)
                tank_name = f"B{uuid.uuid4().hex[:8]}"
                fill_js = f"""
                (() => {{
                    const input = document.querySelector('input[name="tank_name"]');
                    if (input) {{
                        input.value = '{tank_name}';
                        return 'filled';
                    }}
                    return 'input not found';
                }})()
                """
                fill_result = cdp.send(
                    "Runtime.evaluate",
                    {"expression": fill_js, "returnByValue": True},
                )
                fill_obj = fill_result.get("result")
                fill_val = fill_obj.get("value", "?") if isinstance(fill_obj, dict) else "?"
                log.info("Fill result: %s", fill_val)

                submit_js = """
                (() => {
                    const btn = document.querySelector('input[value="Play Now"]');
                    if (btn) {
                        btn.click();
                        return 'clicked Play Now';
                    }
                    const form = document.querySelector('form[action="/guest/create-tank"]');
                    if (form) {
                        form.submit();
                        return 'submitted form';
                    }
                    return 'nothing found';
                })()
                """
                submit_result = cdp.send(
                    "Runtime.evaluate",
                    {"expression": submit_js, "returnByValue": True},
                )
                sub_obj = submit_result.get("result")
                sub_val = sub_obj.get("value", "?") if isinstance(sub_obj, dict) else "?"
                log.info("Submit result: %s", sub_val)

                # Wait for navigation
                page.wait_for_timeout(3000.0)
                log.info("After submit, URL: %s", page.url)

                # Check for any error messages on page
                error_js = """
                (() => {
                    const errors = document.querySelectorAll('.error, .alert, [class*=error]');
                    return Array.from(errors).map(e => e.textContent.trim()).join(' | ');
                })()
                """
                err_result = cdp.send(
                    "Runtime.evaluate",
                    {"expression": error_js, "returnByValue": True},
                )
                err_obj = err_result.get("result")
                err_raw = err_obj.get("value", "") if isinstance(err_obj, dict) else ""
                err_val = str(err_raw) if err_raw else ""
                log.info("Page errors: %s", err_val if err_val else "(none)")

                # If rate-limited, try logging in with credentials
                if "too many tanks" in err_val.lower():
                    username = _test_hooks.get_env("TANKPIT_USERNAME")
                    password = _test_hooks.get_env("TANKPIT_PASSWORD")

                    if username is None or password is None:
                        log.warning(
                            "Rate limited. Set TANKPIT_USERNAME and TANKPIT_PASSWORD "
                            "in .env to login."
                        )
                    else:
                        log.info("Rate limited - logging in as %s...", username)

                        # Step 1: Open login overlay
                        open_js = """
                        (() => {
                            const loginLink = document.querySelector('a[href="#login"]');
                            if (loginLink) {
                                loginLink.click();
                                return 'opened login';
                            }
                            return 'login link not found';
                        })()
                        """
                        cdp.send(
                            "Runtime.evaluate",
                            {"expression": open_js, "returnByValue": True},
                        )
                        page.wait_for_timeout(500.0)

                        # Step 2: Fill credentials
                        fill_login_js = f"""
                        (() => {{
                            const userInput = document.querySelector('#login-username');
                            const passInput = document.querySelector(
                                'form[action="/guest/sign-in"] input[name="password"]'
                            );
                            if (userInput) userInput.value = '{username}';
                            if (passInput) passInput.value = '{password}';
                            return userInput && passInput ? 'filled' : 'inputs not found';
                        }})()
                        """
                        cdp.send(
                            "Runtime.evaluate",
                            {"expression": fill_login_js, "returnByValue": True},
                        )

                        # Step 3: Submit login
                        submit_login_js = """
                        (() => {
                            const submit = document.querySelector(
                                'form[action="/guest/sign-in"] input[type="submit"]'
                            );
                            if (submit) {
                                submit.click();
                                return 'clicked login';
                            }
                            return 'submit not found';
                        })()
                        """
                        login_result = cdp.send(
                            "Runtime.evaluate",
                            {"expression": submit_login_js, "returnByValue": True},
                        )
                        login_obj = login_result.get("result")
                        login_val = (
                            login_obj.get("value", "?") if isinstance(login_obj, dict) else "?"
                        )
                        log.info("Login: %s", login_val)

                        # Wait for login to complete
                        page.wait_for_timeout(3000.0)
                        log.info("After login, URL: %s", page.url)

                        # Check for login errors
                        login_err_js = """
                        (() => {
                            const errors = document.querySelectorAll(
                                '.error, .alert, [class*=error], #login .message'
                            );
                            const texts = Array.from(errors).map(e => e.textContent.trim());
                            return texts.filter(t => t.length > 0).join(' | ');
                        })()
                        """
                        login_err_result = cdp.send(
                            "Runtime.evaluate",
                            {"expression": login_err_js, "returnByValue": True},
                        )
                        login_err_obj = login_err_result.get("result")
                        login_err_raw = (
                            login_err_obj.get("value", "")
                            if isinstance(login_err_obj, dict)
                            else ""
                        )
                        login_err = str(login_err_raw) if login_err_raw else ""
                        if login_err:
                            log.warning("Login errors: %s", login_err)
                        else:
                            log.info("Login successful, navigating to game...")
                            page.goto("https://tankpit.com/play", wait_until="domcontentloaded")
                            page.wait_for_timeout(2000.0)
                            log.info("Game URL: %s", page.url)

            # Wait for specified capture duration
            page.wait_for_timeout(float(capture_duration_ms))

            # Clean up
            cdp.detach()
            page.close()
            context.close()
            browser.close()

        end_timestamp_ms = _get_current_time_ms()

        return CaptureSession(
            session_id=self._session_id,
            start_timestamp_ms=self._start_timestamp_ms,
            end_timestamp_ms=end_timestamp_ms,
            base_url=self._target_url,
            messages=self._messages,
        )


def run_sniffer(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    capture_duration_ms: int = 30000,
) -> CaptureSession:
    """Run the WebSocket sniffer and save results.

    Args:
        target_url: URL to navigate to and capture WebSocket traffic from.
        output_path: Path to save the capture session JSON.
        headless: Whether to run the browser in headless mode.
        capture_duration_ms: How long to capture traffic in milliseconds.

    Returns:
        The completed CaptureSession.

    Raises:
        PlaywrightNotInstalledError: If Playwright is not installed.
    """
    sniffer = WebSocketSniffer(target_url, headless=headless)
    session = sniffer.run(capture_duration_ms)

    # Save to file
    encoded = encode_capture_session(session)
    json_str = dump_json_str(encoded, compact=False, indent=2)
    _test_hooks.write_text(Path(output_path), json_str)

    return session


def main() -> None:
    """Entry point for tankpit-sniff command.

    Reads configuration from environment variables:
    - TANKPIT_URL: Target URL (default: https://tankpit.com)
    - TANKPIT_OUTPUT: Output file path (default: capture_session.json)
    - TANKPIT_HEADLESS: Run headless (default: false)
    - TANKPIT_DURATION_MS: Capture duration in ms (default: 60000)
    """
    from dotenv import load_dotenv

    load_dotenv()
    setup_rich_logging(level="INFO")

    # Install real Playwright hook if not already set (allows test overrides)
    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    # Read config from environment
    target_url = _test_hooks.get_env("TANKPIT_URL")
    if target_url is None:
        target_url = "https://tankpit.com"

    output_path = _test_hooks.get_env("TANKPIT_OUTPUT")
    if output_path is None:
        output_path = "capture_session.json"

    headless_str = _test_hooks.get_env("TANKPIT_HEADLESS")
    headless = headless_str is not None and headless_str.lower() in ("true", "1", "yes")

    duration_str = _test_hooks.get_env("TANKPIT_DURATION_MS")
    capture_duration_ms = 60000
    if duration_str is not None:
        capture_duration_ms = int(duration_str)

    session = run_sniffer(
        target_url,
        output_path,
        headless=headless,
        capture_duration_ms=capture_duration_ms,
    )

    msg_count = len(session["messages"])
    duration_sec = ((session["end_timestamp_ms"] or 0) - session["start_timestamp_ms"]) / 1000
    log.info("Captured %d WebSocket messages in %.1fs", msg_count, duration_sec)
    log.info("Saved to: %s", output_path)

    # Print unique WebSocket URLs discovered
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
    "main",
    "run_sniffer",
]
