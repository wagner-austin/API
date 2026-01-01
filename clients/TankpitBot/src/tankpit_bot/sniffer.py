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
    encode_capture_session,
)

log = get_logger(__name__)

# Default configuration constants
DEFAULT_TARGET_URL = "https://tankpit.com"
DEFAULT_OUTPUT_PATH = "capture_session.json"
DEFAULT_CAPTURE_DURATION_MS = 0  # 0 = indefinite (wait until browser closed)


def _decode_text_message(text: str, body_len: int, tag: str) -> str:
    """Decode a text-based protocol message.

    Args:
        text: Decoded text body.
        body_len: Original body length in bytes.
        tag: Direction tag (SENT/RECEIVED).

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
    return _decode_text_message(text, len(body), tag)


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

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Log decoded message if live decode is enabled.

        Args:
            message: The captured message.
        """
        if self._live_decode:
            decoded = _decode_message(message["payload"], message["direction"], self._magic)
            log.info(decoded)

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

            self._setup_cdp_handlers(cdp)

            # Navigate to target URL
            page.goto(self._target_url, wait_until="domcontentloaded")
            log.info("Landed on %s", page.url)

            # Handle login
            self._navigate_and_login(page, cdp, tank_name_prefix="B", auto_join_room=False)

            # Log loaded script URLs for protocol analysis
            script_urls = page.evaluate(
                "Array.from(document.querySelectorAll('script[src]')).map(s => s.src)"
            )
            if script_urls and isinstance(script_urls, list):
                for url in script_urls:
                    if isinstance(url, str):
                        log.info("Script: %s", url)

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

        return CaptureSession(
            session_id=self._session_id,
            start_timestamp_ms=self._start_timestamp_ms,
            end_timestamp_ms=get_current_time_ms(),
            base_url=self._target_url,
            messages=self._messages,
            magic=self._magic,
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

    # Save to file
    encoded = encode_capture_session(session)
    json_str = dump_json_str(encoded, compact=False, indent=2)
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
    "main",
    "run_sniffer",
]
