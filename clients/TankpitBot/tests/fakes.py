"""Fake Playwright classes for testing.

Provides fake implementations of Playwright protocols that don't require
real browser installation. All fakes match the protocol signatures in
tankpit_bot._test_hooks.
"""

from __future__ import annotations

import base64
import types
from collections.abc import Callable

from platform_core.json_utils import JSONObject, JSONValue

from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeProtocol,
    CDPSessionProtocol,
    KeyboardProtocol,
    PageProtocol,
    PlaywrightProtocol,
    ResponseProtocol,
    SyncPlaywrightContextManagerProtocol,
)


def _make_auth_payload(magic: str) -> str:
    """Create a base64-encoded AUTH payload containing the magic key.

    The AUTH message format is:
    - 2-byte length prefix (little-endian)
    - Text body: %AUTH !be <session>|<hash>|<ts> <magic>

    Args:
        magic: The magic key to include in the AUTH payload.

    Returns:
        Base64-encoded AUTH payload string.
    """
    body = f"%AUTH !be test_session|test_hash|12345 {magic}"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    return base64.b64encode(length_prefix + body_bytes).decode("ascii")


class FakeResponse:
    """Fake Playwright Response."""

    def __init__(self, status: int = 200, url: str = "https://example.com") -> None:
        """Initialize fake response.

        Args:
            status: HTTP status code.
            url: Response URL.
        """
        self._status = status
        self._url = url

    @property
    def status(self) -> int:
        """Get status code."""
        return self._status

    @property
    def url(self) -> str:
        """Get URL."""
        return self._url


class FakeKeyboard:
    """Fake Playwright Keyboard for testing.

    In tests, keyboard input is typically handled via CDP synthetic events,
    but this class satisfies the KeyboardProtocol for type checking.
    """

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Press a keyboard key (no-op in tests).

        Args:
            key: Key name.
            delay: Time between keydown and keyup.
        """
        _ = (key, delay)

    def type(self, text: str, *, delay: float | None = None) -> None:
        """Type text (no-op in tests).

        Args:
            text: Text to type.
            delay: Time between key presses.
        """
        _ = (text, delay)


class FakeTerrainMap:
    """Fake TerrainMap for testing sniffer world state integration.

    Returns ground for all coordinates by default.
    """

    ROCK: str = "#"
    GROUND: str = "."
    WATER: str = "W"

    def __init__(self, terrain_data: dict[tuple[int, int], str] | None = None) -> None:
        """Initialize fake terrain map.

        Args:
            terrain_data: Optional dict mapping (x, y) to terrain character.
        """
        self._terrain_data = terrain_data or {}

    def get_terrain(self, x: int, y: int) -> str:
        """Get terrain at coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Terrain character.
        """
        return self._terrain_data.get((x, y), self.GROUND)

    def is_passable(self, x: int, y: int) -> bool:
        """Check if terrain is passable.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if passable.
        """
        terrain = self.get_terrain(x, y)
        return terrain not in (self.ROCK, self.WATER)

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        """Render viewport grid.

        Args:
            center_x: Center X.
            center_y: Center Y.
            width: Viewport width.
            height: Viewport height.

        Returns:
            2D list of terrain characters.
        """
        left = center_x - width // 2
        top = center_y - height // 2
        grid: list[list[str]] = []
        for row in range(height):
            row_data: list[str] = []
            for col in range(width):
                x = left + col
                y = top + row
                row_data.append(self.get_terrain(x, y))
            grid.append(row_data)
        return grid


class FakeCDPSession:
    """Fake Playwright CDPSession."""

    def __init__(self) -> None:
        """Initialize fake CDP session."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}
        self._sent_methods: list[str] = []
        self._detached = False

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command."""
        _ = params
        self._sent_methods.append(method)
        result: JSONObject = {}
        return result

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""
        self._detached = True

    def emit_event(self, event: str, params: JSONObject) -> None:
        """Emit a CDP event for testing."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                handler(params)


class FakeCDPSessionRateLimited:
    """Fake CDP session that simulates rate-limiting error then successful login."""

    def __init__(self, *, login_fails: bool = False) -> None:
        """Initialize fake CDP session."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}
        self._sent_methods: list[str] = []
        self._detached = False
        self._eval_count = 0
        self._login_fails = login_fails

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command, returning rate limit error on 3rd Runtime.evaluate."""
        self._sent_methods.append(method)
        if method == "Runtime.evaluate":
            self._eval_count += 1
            # 3rd evaluate is the error check, return rate limit error
            if self._eval_count == 3:
                return {"result": {"value": "There are too many tanks"}}
            # 7th evaluate is login error check
            if self._eval_count == 7:
                if self._login_fails:
                    return {"result": {"value": "Invalid username or password"}}
                return {"result": {"value": ""}}
            # Other evaluates return success
            return {"result": {"value": "success"}}
        result: JSONObject = {}
        return result

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""
        self._detached = True

    def emit_event(self, event: str, params: JSONObject) -> None:
        """Emit a CDP event for testing."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                handler(params)


class FakePage:
    """Fake Playwright Page that emits WebSocket events."""

    def __init__(
        self,
        cdp_session: FakeCDPSession | FakeCDPSessionRateLimited,
        *,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake page.

        Args:
            cdp_session: CDP session to use for events.
            script_urls: Optional list of script URLs to return from evaluate.
            magic: Optional magic key to embed in AUTH messages.
        """
        self._cdp_session = cdp_session
        self._goto_url: str | None = None
        self._wait_timeout: float | None = None
        self._closed = False
        self._url = ""
        self._script_urls: list[JSONValue] = script_urls if script_urls is not None else []
        self._magic = magic

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface.

        Returns:
            FakeKeyboard instance.
        """
        return FakeKeyboard()

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._goto_url = url
        self._url = url
        return FakeResponse(url=url)

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait and emit WebSocket events."""
        self._wait_timeout = timeout
        self._cdp_session.emit_event(
            "Network.webSocketCreated",
            {"requestId": "1.1", "url": "wss://example.com/ws"},
        )
        # Emit AUTH message with magic if configured
        sent_payload = "sent message"
        if self._magic:
            sent_payload = _make_auth_payload(self._magic)
        self._cdp_session.emit_event(
            "Network.webSocketFrameSent",
            {
                "requestId": "1.1",
                "timestamp": 100.0,
                "response": {"opcode": 1, "mask": True, "payloadData": sent_payload},
            },
        )
        self._cdp_session.emit_event(
            "Network.webSocketFrameReceived",
            {
                "requestId": "1.1",
                "timestamp": 100.5,
                "response": {"opcode": 1, "mask": False, "payloadData": "received message"},
            },
        )

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event - also emit WebSocket events like wait_for_timeout."""
        _ = (event, timeout)
        # Emit the same events as wait_for_timeout for test compatibility
        self._cdp_session.emit_event(
            "Network.webSocketCreated",
            {"requestId": "1.1", "url": "wss://example.com/ws"},
        )
        # Emit AUTH message with magic if configured
        sent_payload = "sent message"
        if self._magic:
            sent_payload = _make_auth_payload(self._magic)
        self._cdp_session.emit_event(
            "Network.webSocketFrameSent",
            {
                "requestId": "1.1",
                "timestamp": 100.0,
                "response": {"opcode": 1, "mask": True, "payloadData": sent_payload},
            },
        )
        self._cdp_session.emit_event(
            "Network.webSocketFrameReceived",
            {
                "requestId": "1.1",
                "timestamp": 100.5,
                "response": {"opcode": 1, "mask": False, "payloadData": "received message"},
            },
        )

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for JavaScript function to return truthy.

        Args:
            expression: JavaScript expression to evaluate.
            timeout: Maximum wait time in milliseconds.
        """
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression.

        Returns:
            Script URLs list for script queries, empty list otherwise.
        """
        _ = expression
        return self._script_urls


class FakePageNoMessages:
    """Fake Playwright Page that doesn't emit WebSocket messages."""

    def __init__(self, cdp_session: FakeCDPSession | FakeCDPSessionRateLimited) -> None:
        """Initialize fake page."""
        self._cdp_session = cdp_session
        self._closed = False
        self._url = ""

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface.

        Returns:
            FakeKeyboard instance.
        """
        return FakeKeyboard()

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url
        return FakeResponse()

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait without emitting any WebSocket events."""
        _ = timeout

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event - returns immediately in tests."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for JavaScript function to return truthy.

        Args:
            expression: JavaScript expression to evaluate.
            timeout: Maximum wait time in milliseconds.
        """
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression - returns empty list in tests."""
        _ = expression
        return []


class FakeBrowserContext:
    """Fake Playwright BrowserContext."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake browser context.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        cdp: FakeCDPSession | FakeCDPSessionRateLimited = (
            FakeCDPSessionRateLimited(login_fails=login_fails) if rate_limited else FakeCDPSession()
        )
        self._cdp_session = cdp
        self._pages: list[FakePage | FakePageNoMessages] = []
        self._closed = False
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._script_urls = script_urls
        self._magic = magic

    def new_page(self) -> PageProtocol:
        """Create new page."""
        page: FakePage | FakePageNoMessages
        if self._emit_messages:
            page = FakePage(self._cdp_session, script_urls=self._script_urls, magic=self._magic)
        else:
            page = FakePageNoMessages(self._cdp_session)
        self._pages.append(page)
        return page

    def new_cdp_session(self, page: PageProtocol) -> CDPSessionProtocol:
        """Create CDP session for page."""
        _ = page
        return self._cdp_session

    def close(self, *, reason: str | None = None) -> None:
        """Close context."""
        _ = reason
        self._closed = True


class FakeBrowser:
    """Fake Playwright Browser."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake browser.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        self._contexts: list[FakeBrowserContext] = []
        self._closed = False
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._login_fails = login_fails
        self._script_urls = script_urls
        self._magic = magic

    def new_context(self) -> BrowserContextProtocol:
        """Create new context."""
        ctx = FakeBrowserContext(
            emit_messages=self._emit_messages,
            rate_limited=self._rate_limited,
            login_fails=self._login_fails,
            script_urls=self._script_urls,
            magic=self._magic,
        )
        self._contexts.append(ctx)
        return ctx

    def close(self, *, reason: str | None = None) -> None:
        """Close browser."""
        _ = reason
        self._closed = True


class FakeBrowserType:
    """Fake Playwright BrowserType."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake browser type.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        self._browsers: list[FakeBrowser] = []
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._login_fails = login_fails
        self._script_urls = script_urls
        self._magic = magic

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
    ) -> BrowserProtocol:
        """Launch browser."""
        _ = (headless, slow_mo, timeout)
        browser = FakeBrowser(
            emit_messages=self._emit_messages,
            rate_limited=self._rate_limited,
            login_fails=self._login_fails,
            script_urls=self._script_urls,
            magic=self._magic,
        )
        self._browsers.append(browser)
        return browser


class FakePlaywright:
    """Fake Playwright instance."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake Playwright.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        self._chromium = FakeBrowserType(
            emit_messages=emit_messages,
            rate_limited=rate_limited,
            login_fails=login_fails,
            script_urls=script_urls,
            magic=magic,
        )
        self._stopped = False

    @property
    def chromium(self) -> BrowserTypeProtocol:
        """Get chromium browser type."""
        return self._chromium

    def stop(self) -> None:
        """Stop Playwright."""
        self._stopped = True


class FakeSyncPlaywrightContextManager:
    """Fake sync_playwright() context manager."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
        script_urls: list[JSONValue] | None = None,
        magic: str | None = None,
    ) -> None:
        """Initialize fake context manager.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            rate_limited: Whether to simulate rate limiting.
            login_fails: Whether login should fail.
            script_urls: Script URLs to return from page.evaluate().
            magic: Magic key to embed in AUTH messages.
        """
        self._playwright: FakePlaywright | None = None
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._login_fails = login_fails
        self._script_urls = script_urls
        self._magic = magic

    def start(self) -> PlaywrightProtocol:
        """Start Playwright."""
        self._playwright = FakePlaywright(
            emit_messages=self._emit_messages,
            rate_limited=self._rate_limited,
            login_fails=self._login_fails,
            script_urls=self._script_urls,
            magic=self._magic,
        )
        return self._playwright

    def __enter__(self) -> PlaywrightProtocol:
        """Enter context."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context."""
        _ = (exc_type, exc_val, exc_tb)
        if self._playwright is not None:
            self._playwright.stop()


def fake_sync_playwright() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that emits WebSocket messages."""
    return FakeSyncPlaywrightContextManager(emit_messages=True)


def fake_sync_playwright_no_messages() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that doesn't emit messages."""
    return FakeSyncPlaywrightContextManager(emit_messages=False)


def fake_sync_playwright_rate_limited() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that simulates rate-limiting with successful login."""
    return FakeSyncPlaywrightContextManager(emit_messages=True, rate_limited=True)


def fake_sync_playwright_login_fails() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that simulates rate-limiting with failed login."""
    return FakeSyncPlaywrightContextManager(emit_messages=True, rate_limited=True, login_fails=True)


def fake_sync_playwright_with_scripts() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that returns script URLs from page.evaluate().

    Returns:
        Context manager that produces pages returning script URLs.
    """
    return FakeSyncPlaywrightContextManager(
        emit_messages=True,
        script_urls=[
            "https://tankpit.com/js/game.js",
            "https://tankpit.com/js/protocol.js",
        ],
    )


def fake_sync_playwright_with_mixed_scripts() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright with mixed types in script_urls list.

    This tests the isinstance(url, str) check by including non-string values.

    Returns:
        Context manager that produces pages returning mixed script URL types.
    """
    return FakeSyncPlaywrightContextManager(
        emit_messages=True,
        script_urls=[
            "https://tankpit.com/js/valid.js",
            123,  # Non-string value to test isinstance check
            None,  # Another non-string value
            "https://tankpit.com/js/another.js",
        ],
    )


def fake_sync_playwright_with_magic() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright that emits AUTH messages with magic key.

    The magic key is embedded in AUTH messages emitted during WebSocket events,
    allowing tests to verify magic extraction from AUTH message payloads.

    Returns:
        Context manager that produces pages emitting AUTH messages with magic.
    """
    return FakeSyncPlaywrightContextManager(
        emit_messages=True,
        magic="test_magic_xor_key_value",
    )


# =============================================================================
# Probe-specific fakes
# =============================================================================


class FakeCDPSessionProbe:
    """Fake CDP session for probe testing that responds to WebSocket sends.

    Now detects WebSocket injection (ws.send) instead of JavaScript KeyboardEvents,
    since the probe uses WebSocket injection for sending commands.
    """

    def __init__(
        self,
        *,
        emit_on_key: bool = True,
        emit_on_mouse: bool = False,
        viewport_result: JSONObject | None = None,
        return_invalid_result: bool = False,
        return_missing_value: bool = False,
        js_keypress_fails: bool = False,
    ) -> None:
        """Initialize fake CDP session for probing.

        Args:
            emit_on_key: Whether to emit messages when WebSocket commands are sent.
            emit_on_mouse: Whether to emit messages when mouse inputs are injected.
            viewport_result: Custom viewport result to return, None uses default.
            return_invalid_result: Return non-dict result for Runtime.evaluate.
            return_missing_value: Return dict without value for Runtime.evaluate.
            js_keypress_fails: If True, JS keypress returns ERROR instead of JS_KEYPRESS_X.
        """
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}
        self._sent_methods: list[str] = []
        self._detached = False
        self._emit_on_key = emit_on_key
        self._emit_on_mouse = emit_on_mouse
        self._viewport_result = viewport_result
        self._return_invalid_result = return_invalid_result
        self._return_missing_value = return_missing_value
        self._js_keypress_fails = js_keypress_fails
        self._input_count = 0
        self._ws_url = "wss://tankpit.com/ws/"

    def _handle_runtime_evaluate(self, params: JSONObject) -> JSONObject:
        """Handle Runtime.evaluate CDP command."""
        expression = params.get("expression", "")
        expr_str = str(expression)

        if "innerWidth" in expr_str:
            if self._viewport_result is not None:
                return self._viewport_result
            return {"result": {"value": '{"w":800,"h":600}'}}

        if self._return_invalid_result:
            return {"error": "simulated error"}

        if self._return_missing_value:
            return {"result": {}}

        # Detect WebSocket send via _send_websocket_bytes and emit messages
        if "ws.send" in expr_str and "__capturedWS" in expr_str and self._emit_on_key:
            self._input_count += 1
            self._emit_ws_sent(f"key_input_{self._input_count}")
            self._emit_ws_received(f"key_response_{self._input_count}")
            return {"result": {"value": f"SENT_5_BYTES via {self._ws_url}"}}

        # Detect JS keypress for toggle close
        if "KeyboardEvent" in expr_str and "dispatchEvent" in expr_str:
            if self._js_keypress_fails:
                return {"result": {"value": "ERROR"}}
            if "'f'" in expr_str or '"f"' in expr_str:
                return {"result": {"value": "JS_KEYPRESS_F"}}
            return {"result": {"value": "JS_KEYPRESS_?"}}

        return {"result": {"value": "success"}}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command and optionally emit WebSocket response."""
        self._sent_methods.append(method)

        if method == "Runtime.evaluate" and params is not None:
            return self._handle_runtime_evaluate(params)

        if method == "Input.dispatchMouseEvent" and self._emit_on_mouse:
            event_type = params.get("type", "") if params else ""
            if event_type == "mousePressed":
                self._input_count += 1
                self._emit_ws_sent(f"mouse_input_{self._input_count}")
                self._emit_ws_received(f"mouse_response_{self._input_count}")

        result: JSONObject = {}
        return result

    def _emit_ws_sent(self, payload: str) -> None:
        """Emit a WebSocket sent event."""
        if "Network.webSocketFrameSent" in self._handlers:
            for handler in self._handlers["Network.webSocketFrameSent"]:
                handler(
                    {
                        "requestId": "1.1",
                        "timestamp": 100.0 + self._input_count,
                        "response": {"opcode": 1, "mask": True, "payloadData": payload},
                    }
                )

    def _emit_ws_received(self, payload: str) -> None:
        """Emit a WebSocket received event."""
        if "Network.webSocketFrameReceived" in self._handlers:
            for handler in self._handlers["Network.webSocketFrameReceived"]:
                handler(
                    {
                        "requestId": "1.1",
                        "timestamp": 100.0 + self._input_count + 0.001,
                        "response": {"opcode": 1, "mask": False, "payloadData": payload},
                    }
                )

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach CDP session."""
        self._detached = True

    def emit_event(self, event: str, params: JSONObject) -> None:
        """Emit a CDP event for testing."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                handler(params)


class FakePageProbe:
    """Fake Playwright Page for probe testing."""

    # Default magic value for XOR table construction
    DEFAULT_MAGIC = "test_magic_12345678"

    def __init__(
        self,
        cdp_session: FakeCDPSessionProbe,
        *,
        before_playing: bool = False,
        login_redirects_to_play: bool = False,
        emit_during_stabilization: bool = False,
    ) -> None:
        """Initialize fake page for probing.

        Args:
            cdp_session: CDP session to use.
            before_playing: Whether to simulate being on before-playing page.
            login_redirects_to_play: If True, simulates login redirecting to /play.
            emit_during_stabilization: If True, emit messages during stabilization loop.
        """
        self._cdp_session = cdp_session
        self._closed = False
        self._url = ""
        self._before_playing = before_playing
        self._login_redirects_to_play = login_redirects_to_play
        self._first_wait = True
        self._wait_count = 0
        self._emit_during_stabilization = emit_during_stabilization

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface.

        Returns:
            FakeKeyboard instance (no-op for tests using CDP synthetic events).
        """
        return FakeKeyboard()

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        if self._before_playing:
            self._url = url.replace("/play", "/before-playing")
        else:
            self._url = url
        return FakeResponse(url=self._url)

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait and emit initial WebSocket events on first call."""
        _ = timeout
        self._wait_count += 1

        # On second wait (after login click), simulate URL redirect if configured
        if self._wait_count == 2 and self._login_redirects_to_play:
            self._url = "https://tankpit.com/play"

        if self._first_wait:
            self._first_wait = False
            # Emit initial connection and auth messages
            self._cdp_session.emit_event(
                "Network.webSocketCreated",
                {"requestId": "1.1", "url": "wss://tankpit.com/ws/"},
            )
            # Emit AUTH message with magic for XOR table construction
            auth_payload = _make_auth_payload(self.DEFAULT_MAGIC)
            self._cdp_session.emit_event(
                "Network.webSocketFrameSent",
                {
                    "requestId": "1.1",
                    "timestamp": 1.0,
                    "response": {"opcode": 1, "mask": True, "payloadData": auth_payload},
                },
            )
            self._cdp_session.emit_event(
                "Network.webSocketFrameReceived",
                {
                    "requestId": "1.1",
                    "timestamp": 2.0,
                    "response": {"opcode": 1, "mask": False, "payloadData": "room_list"},
                },
            )
        elif self._emit_during_stabilization and self._wait_count == 5:
            # Emit extra message during stabilization loop (iteration 2)
            # Calls: 1=join_room, 2=join_room, 3=pre-stabilization, 4=loop iter 1, 5=loop iter 2
            self._cdp_session.emit_event(
                "Network.webSocketFrameReceived",
                {
                    "requestId": "1.1",
                    "timestamp": 3.0,
                    "response": {"opcode": 1, "mask": False, "payloadData": "extra_msg"},
                },
            )

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event - returns immediately in tests."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for JavaScript function to return truthy.

        Args:
            expression: JavaScript expression to evaluate.
            timeout: Maximum wait time in milliseconds.
        """
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression.

        Returns:
            Empty list for all expressions (magic comes from AUTH messages).
        """
        _ = expression
        return []


class FakePageProbeNoMessages:
    """Fake Page for probe testing that doesn't emit any messages."""

    def __init__(self, cdp_session: FakeCDPSessionProbe) -> None:
        """Initialize fake page."""
        self._cdp_session = cdp_session
        self._closed = False
        self._url = ""

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface.

        Returns:
            FakeKeyboard instance (no-op for tests using CDP synthetic events).
        """
        return FakeKeyboard()

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url
        return FakeResponse(url=url)

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait without emitting any WebSocket events."""
        _ = timeout

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event - returns immediately in tests."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for JavaScript function to return truthy.

        Args:
            expression: JavaScript expression to evaluate.
            timeout: Maximum wait time in milliseconds.
        """
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression.

        Returns:
            Empty list for all expressions (magic comes from AUTH messages).
        """
        _ = expression
        return []


class FakeBrowserContextProbe:
    """Fake BrowserContext for probe testing."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        before_playing: bool = False,
        login_redirects_to_play: bool = False,
        emit_on_key: bool = True,
        emit_on_mouse: bool = False,
        viewport_result: JSONObject | None = None,
        emit_during_stabilization: bool = False,
    ) -> None:
        """Initialize fake browser context for probing.

        Args:
            emit_messages: Whether to emit initial WebSocket messages.
            before_playing: Whether to simulate before-playing page.
            login_redirects_to_play: If True, login redirects to /play.
            emit_on_key: Whether to emit messages on key input.
            emit_on_mouse: Whether to emit messages on mouse input.
            viewport_result: Custom viewport result to return.
            emit_during_stabilization: If True, emit during stabilization loop.
        """
        self._cdp_session = FakeCDPSessionProbe(
            emit_on_key=emit_on_key if emit_messages else False,
            emit_on_mouse=emit_on_mouse,
            viewport_result=viewport_result,
        )
        self._pages: list[FakePageProbe | FakePageProbeNoMessages] = []
        self._closed = False
        self._emit_messages = emit_messages
        self._before_playing = before_playing
        self._login_redirects_to_play = login_redirects_to_play
        self._emit_during_stabilization = emit_during_stabilization

    def new_page(self) -> PageProtocol:
        """Create new page."""
        page: FakePageProbe | FakePageProbeNoMessages
        if self._emit_messages:
            page = FakePageProbe(
                self._cdp_session,
                before_playing=self._before_playing,
                emit_during_stabilization=self._emit_during_stabilization,
            )
        else:
            page = FakePageProbeNoMessages(self._cdp_session)
        self._pages.append(page)
        return page

    def new_cdp_session(self, page: PageProtocol) -> CDPSessionProtocol:
        """Create CDP session for page."""
        _ = page
        return self._cdp_session

    def close(self, *, reason: str | None = None) -> None:
        """Close context."""
        _ = reason
        self._closed = True


class FakeBrowserProbe:
    """Fake Browser for probe testing."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        before_playing: bool = False,
        emit_on_key: bool = True,
        emit_on_mouse: bool = False,
        viewport_result: JSONObject | None = None,
        emit_during_stabilization: bool = False,
    ) -> None:
        """Initialize fake browser for probing.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            before_playing: Whether to simulate before-playing page.
            emit_on_key: Whether to emit messages on key input.
            emit_on_mouse: Whether to emit messages on mouse input.
            viewport_result: Custom viewport result to return.
            emit_during_stabilization: If True, emit during stabilization loop.
        """
        self._contexts: list[FakeBrowserContextProbe] = []
        self._closed = False
        self._emit_messages = emit_messages
        self._before_playing = before_playing
        self._emit_on_key = emit_on_key
        self._emit_on_mouse = emit_on_mouse
        self._viewport_result = viewport_result
        self._emit_during_stabilization = emit_during_stabilization

    def new_context(self) -> BrowserContextProtocol:
        """Create new context."""
        ctx = FakeBrowserContextProbe(
            emit_messages=self._emit_messages,
            before_playing=self._before_playing,
            emit_on_key=self._emit_on_key,
            emit_on_mouse=self._emit_on_mouse,
            viewport_result=self._viewport_result,
            emit_during_stabilization=self._emit_during_stabilization,
        )
        self._contexts.append(ctx)
        return ctx

    def close(self, *, reason: str | None = None) -> None:
        """Close browser."""
        _ = reason
        self._closed = True


class FakeBrowserTypeProbe:
    """Fake BrowserType for probe testing."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        before_playing: bool = False,
        emit_on_key: bool = True,
        emit_on_mouse: bool = False,
        viewport_result: JSONObject | None = None,
        emit_during_stabilization: bool = False,
    ) -> None:
        """Initialize fake browser type for probing.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            before_playing: Whether to simulate before-playing page.
            emit_on_key: Whether to emit messages on key input.
            emit_on_mouse: Whether to emit messages on mouse input.
            viewport_result: Custom viewport result to return.
            emit_during_stabilization: If True, emit during stabilization loop.
        """
        self._browsers: list[FakeBrowserProbe] = []
        self._emit_messages = emit_messages
        self._before_playing = before_playing
        self._emit_on_key = emit_on_key
        self._emit_on_mouse = emit_on_mouse
        self._viewport_result = viewport_result
        self._emit_during_stabilization = emit_during_stabilization

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
    ) -> BrowserProtocol:
        """Launch browser."""
        _ = (headless, slow_mo, timeout)
        browser = FakeBrowserProbe(
            emit_messages=self._emit_messages,
            before_playing=self._before_playing,
            emit_on_key=self._emit_on_key,
            emit_on_mouse=self._emit_on_mouse,
            viewport_result=self._viewport_result,
            emit_during_stabilization=self._emit_during_stabilization,
        )
        self._browsers.append(browser)
        return browser


class FakePlaywrightProbe:
    """Fake Playwright instance for probe testing."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        before_playing: bool = False,
        emit_on_key: bool = True,
        emit_on_mouse: bool = False,
        viewport_result: JSONObject | None = None,
        emit_during_stabilization: bool = False,
    ) -> None:
        """Initialize fake Playwright for probing.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            before_playing: Whether to simulate before-playing page.
            emit_on_key: Whether to emit messages on key input.
            emit_on_mouse: Whether to emit messages on mouse input.
            viewport_result: Custom viewport result to return.
            emit_during_stabilization: If True, emit during stabilization loop.
        """
        self._chromium = FakeBrowserTypeProbe(
            emit_messages=emit_messages,
            before_playing=before_playing,
            emit_on_key=emit_on_key,
            emit_on_mouse=emit_on_mouse,
            viewport_result=viewport_result,
            emit_during_stabilization=emit_during_stabilization,
        )
        self._stopped = False

    @property
    def chromium(self) -> BrowserTypeProtocol:
        """Get chromium browser type."""
        return self._chromium

    def stop(self) -> None:
        """Stop Playwright."""
        self._stopped = True


class FakeSyncPlaywrightContextManagerProbe:
    """Fake sync_playwright() context manager for probe testing."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        before_playing: bool = False,
        emit_on_key: bool = True,
        emit_on_mouse: bool = False,
        viewport_result: JSONObject | None = None,
        emit_during_stabilization: bool = False,
    ) -> None:
        """Initialize fake context manager for probing.

        Args:
            emit_messages: Whether to emit WebSocket messages.
            before_playing: Whether to simulate before-playing page.
            emit_on_key: Whether to emit messages on key input.
            emit_on_mouse: Whether to emit messages on mouse input.
            viewport_result: Custom viewport result to return.
            emit_during_stabilization: If True, emit during stabilization loop.
        """
        self._playwright: FakePlaywrightProbe | None = None
        self._emit_messages = emit_messages
        self._before_playing = before_playing
        self._emit_on_key = emit_on_key
        self._emit_on_mouse = emit_on_mouse
        self._viewport_result = viewport_result
        self._emit_during_stabilization = emit_during_stabilization

    def start(self) -> PlaywrightProtocol:
        """Start Playwright."""
        self._playwright = FakePlaywrightProbe(
            emit_messages=self._emit_messages,
            before_playing=self._before_playing,
            emit_on_key=self._emit_on_key,
            emit_on_mouse=self._emit_on_mouse,
            viewport_result=self._viewport_result,
            emit_during_stabilization=self._emit_during_stabilization,
        )
        return self._playwright

    def __enter__(self) -> PlaywrightProtocol:
        """Enter context."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context."""
        _ = (exc_type, exc_val, exc_tb)
        if self._playwright is not None:
            self._playwright.stop()


def fake_sync_playwright_probe() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for probe testing with message emission."""
    return FakeSyncPlaywrightContextManagerProbe(emit_messages=True)


def fake_sync_playwright_probe_no_messages() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for probe testing without messages."""
    return FakeSyncPlaywrightContextManagerProbe(emit_messages=False)


def fake_sync_playwright_probe_before_playing() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for probe that simulates before-playing page."""
    return FakeSyncPlaywrightContextManagerProbe(emit_messages=True, before_playing=True)


def fake_sync_playwright_probe_mouse_emits() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for probe that emits messages on mouse input."""
    return FakeSyncPlaywrightContextManagerProbe(
        emit_messages=True,
        emit_on_key=False,
        emit_on_mouse=True,
    )


def fake_sync_playwright_probe_no_key_emits() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for probe that does not emit messages on key input."""
    return FakeSyncPlaywrightContextManagerProbe(
        emit_messages=True,
        emit_on_key=False,
        emit_on_mouse=False,
    )


def fake_sync_playwright_probe_invalid_viewport() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for probe with invalid viewport result.

    The viewport_raw is a dict, but value is an int not str.
    This covers the branch at line 368->372 in probe.py.
    """
    return FakeSyncPlaywrightContextManagerProbe(
        emit_messages=True,
        viewport_result={"result": {"value": 12345}},  # value is int, not str
    )


def fake_sync_playwright_probe_non_dict_viewport() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for probe where viewport result is not a dict."""
    return FakeSyncPlaywrightContextManagerProbe(
        emit_messages=True,
        viewport_result={"result": ["not", "a", "dict"]},  # result is list not dict
    )


def fake_sync_playwright_probe_both_emit() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for probe that emits on both key and mouse."""
    return FakeSyncPlaywrightContextManagerProbe(
        emit_messages=True,
        emit_on_key=True,
        emit_on_mouse=True,
    )


def fake_sync_playwright_probe_delayed_messages() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for probe that emits during stabilization.

    This tests the branch where message count changes during the stabilization
    wait loop, triggering the stable_checks reset.
    """
    return FakeSyncPlaywrightContextManagerProbe(
        emit_messages=True,
        emit_during_stabilization=True,
    )


# =============================================================================
# Bot Test Fakes - for testing Bot class with KeyboardInterrupt exit
# =============================================================================


class FakePageInterrupting:
    """Page that implements PageProtocol and raises KeyboardInterrupt after wait.

    Used for testing Bot._game_loop which needs a page that will interrupt
    the game loop to exit cleanly.
    """

    def __init__(self, interrupt_after: int = 1) -> None:
        """Initialize fake page that interrupts.

        Args:
            interrupt_after: Number of wait_for_timeout calls before interrupt.
        """
        self._wait_count = 0
        self._url = ""
        self._interrupt_after = interrupt_after

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get the keyboard interface."""
        return FakeKeyboard()

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url
        return FakeResponse(url=url)

    def wait_for_timeout(self, timeout: float) -> None:
        """Raise KeyboardInterrupt after configured number of calls."""
        _ = timeout
        self._wait_count += 1
        if self._wait_count > self._interrupt_after:
            raise KeyboardInterrupt

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for JavaScript function."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression."""
        _ = expression
        return []


class FakeCDPSessionBot:
    """CDP fake for bot testing that tracks handlers and emits events."""

    def __init__(self) -> None:
        """Initialize fake CDP session."""
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}
        self._sent_methods: list[str] = []
        self._detached = False

    def send(
        self,
        method: str,
        params: JSONObject | None = None,
    ) -> JSONObject:
        """Send CDP command."""
        _ = params
        self._sent_methods.append(method)
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def detach(self) -> None:
        """Detach session."""
        self._detached = True

    def emit_event(self, event: str, params: JSONObject) -> None:
        """Emit a CDP event to registered handlers."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                handler(params)


class FakePageBot:
    """Page fake for bot testing that emits WebSocket events and exits after waits.

    This fake simulates a successful login by setting the URL to /play after goto(),
    and emits WebSocket events through the CDP session when wait_for_timeout is called.
    This lets Bot.run() reach the game loop code by passing _wait_for_game_ready checks.
    """

    def __init__(self, cdp_session: FakeCDPSessionBot, *, interrupt_after: int = 15) -> None:
        """Initialize fake page.

        Args:
            cdp_session: CDP session to emit events to.
            interrupt_after: Number of wait_for_timeout calls before interrupt.
                Default is 15 to allow login flow and game loop entry.
        """
        self._cdp_session = cdp_session
        self._url = ""
        self._wait_count = 0
        self._interrupt_after = interrupt_after

    @property
    def url(self) -> str:
        """Return URL."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Return keyboard."""
        return FakeKeyboard()

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol:
        """Navigate and set URL to /play to simulate successful login."""
        _ = (referer, timeout, wait_until)
        # Always end up on /play (simulates successful login)
        if "tankpit" in url:
            self._url = "https://tankpit.com/play"
        else:
            self._url = url
        return FakeResponse()

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait, emit WebSocket events, and raise KeyboardInterrupt after configured calls.

        Only emits events on the first 2 calls so that _wait_for_game_ready's
        stability check (3 consecutive checks with same message count) can pass.
        """
        _ = timeout
        self._wait_count += 1

        # Only emit events on first 2 waits - allows stability check to pass
        # _wait_for_game_ready does: 1 initial wait + loop waiting for 3 stable checks
        # By only emitting on waits 1-2, waits 3-5 have stable message count
        if self._wait_count <= 2:
            self._cdp_session.emit_event(
                "Network.webSocketCreated",
                {"requestId": "1.1", "url": "wss://tankpit.com/ws"},
            )
            self._cdp_session.emit_event(
                "Network.webSocketFrameSent",
                {
                    "requestId": "1.1",
                    "timestamp": 100.0 + self._wait_count,
                    "response": {"opcode": 1, "mask": True, "payloadData": "sent"},
                },
            )
            self._cdp_session.emit_event(
                "Network.webSocketFrameReceived",
                {
                    "requestId": "1.1",
                    "timestamp": 100.5 + self._wait_count,
                    "response": {"opcode": 1, "mask": False, "payloadData": "received"},
                },
            )

        if self._wait_count > self._interrupt_after:
            raise KeyboardInterrupt

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for JavaScript function."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Return empty list."""
        _ = expression
        return []


class FakeBrowserContextBot:
    """Browser context fake for bot testing that coordinates page and CDP session."""

    def __init__(self, interrupt_after: int = 15) -> None:
        """Initialize fake browser context.

        Args:
            interrupt_after: Number of wait_for_timeout calls before interrupt.
        """
        self._interrupt_after = interrupt_after
        self._cdp_session: FakeCDPSessionBot | None = None
        self._page: FakePageBot | None = None

    def new_page(self) -> PageProtocol:
        """Create new page. Must call new_cdp_session first or use default CDP."""
        # Create CDP session first if needed
        if self._cdp_session is None:
            self._cdp_session = FakeCDPSessionBot()
        self._page = FakePageBot(self._cdp_session, interrupt_after=self._interrupt_after)
        return self._page

    def new_cdp_session(self, page: PageProtocol) -> CDPSessionProtocol:
        """Create new CDP session linked to the page."""
        _ = page
        if self._cdp_session is None:
            self._cdp_session = FakeCDPSessionBot()
        # Link the CDP session to the page if it exists
        if self._page is not None:
            self._page._cdp_session = self._cdp_session
        return self._cdp_session

    def close(self, *, reason: str | None = None) -> None:
        """Close context."""
        _ = reason


class FakeBrowserBot:
    """Minimal browser fake for bot testing."""

    def __init__(self, interrupt_after: int = 2) -> None:
        """Initialize fake browser.

        Args:
            interrupt_after: Number of wait_for_timeout calls before interrupt.
        """
        self._interrupt_after = interrupt_after

    def new_context(self) -> BrowserContextProtocol:
        """Create new context."""
        return FakeBrowserContextBot(interrupt_after=self._interrupt_after)

    def close(self, *, reason: str | None = None) -> None:
        """Close browser."""
        _ = reason


class FakeBrowserTypeBot:
    """Minimal browser type fake for bot testing."""

    def __init__(self, interrupt_after: int = 2) -> None:
        """Initialize fake browser type.

        Args:
            interrupt_after: Number of wait_for_timeout calls before interrupt.
        """
        self._interrupt_after = interrupt_after

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
    ) -> BrowserProtocol:
        """Launch browser."""
        _ = (headless, slow_mo, timeout)
        return FakeBrowserBot(interrupt_after=self._interrupt_after)


class FakePlaywrightBot:
    """Minimal Playwright fake for bot testing."""

    def __init__(self, interrupt_after: int = 2) -> None:
        """Initialize fake Playwright.

        Args:
            interrupt_after: Number of wait_for_timeout calls before interrupt.
        """
        self._chromium = FakeBrowserTypeBot(interrupt_after=interrupt_after)

    @property
    def chromium(self) -> BrowserTypeProtocol:
        """Get chromium browser type."""
        return self._chromium

    def stop(self) -> None:
        """Stop Playwright."""


class FakeSyncPlaywrightContextManagerBot:
    """Fake sync_playwright() context manager for bot testing.

    This creates a full Playwright hierarchy that raises KeyboardInterrupt
    after a configured number of wait_for_timeout calls. Useful for testing
    Bot.main() and Bot.run() which need the full Playwright stack.
    """

    def __init__(self, interrupt_after: int = 2) -> None:
        """Initialize fake context manager.

        Args:
            interrupt_after: Number of wait_for_timeout calls before interrupt.
        """
        self._playwright: FakePlaywrightBot | None = None
        self._interrupt_after = interrupt_after

    def start(self) -> PlaywrightProtocol:
        """Start Playwright."""
        self._playwright = FakePlaywrightBot(interrupt_after=self._interrupt_after)
        return self._playwright

    def __enter__(self) -> PlaywrightProtocol:
        """Enter context."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context."""
        _ = (exc_type, exc_val, exc_tb)
        if self._playwright is not None:
            self._playwright.stop()


def fake_sync_playwright_bot() -> SyncPlaywrightContextManagerProtocol:
    """Create fake sync_playwright for bot testing that exits via KeyboardInterrupt.

    Uses interrupt_after=15 to allow the login flow, wait_for_game_ready (4+ waits),
    and game loop entry before raising KeyboardInterrupt.
    """
    return FakeSyncPlaywrightContextManagerBot(interrupt_after=15)


__all__ = [
    "FakeBrowser",
    "FakeBrowserBot",
    "FakeBrowserContext",
    "FakeBrowserContextBot",
    "FakeBrowserContextProbe",
    "FakeBrowserProbe",
    "FakeBrowserType",
    "FakeBrowserTypeBot",
    "FakeBrowserTypeProbe",
    "FakeCDPSession",
    "FakeCDPSessionBot",
    "FakeCDPSessionProbe",
    "FakeCDPSessionRateLimited",
    "FakePage",
    "FakePageBot",
    "FakePageInterrupting",
    "FakePageNoMessages",
    "FakePageProbe",
    "FakePageProbeNoMessages",
    "FakePlaywright",
    "FakePlaywrightBot",
    "FakePlaywrightProbe",
    "FakeResponse",
    "FakeSyncPlaywrightContextManager",
    "FakeSyncPlaywrightContextManagerBot",
    "FakeSyncPlaywrightContextManagerProbe",
    "FakeTerrainMap",
    "fake_sync_playwright",
    "fake_sync_playwright_bot",
    "fake_sync_playwright_login_fails",
    "fake_sync_playwright_no_messages",
    "fake_sync_playwright_probe",
    "fake_sync_playwright_probe_before_playing",
    "fake_sync_playwright_probe_both_emit",
    "fake_sync_playwright_probe_delayed_messages",
    "fake_sync_playwright_probe_invalid_viewport",
    "fake_sync_playwright_probe_mouse_emits",
    "fake_sync_playwright_probe_no_key_emits",
    "fake_sync_playwright_probe_no_messages",
    "fake_sync_playwright_probe_non_dict_viewport",
    "fake_sync_playwright_rate_limited",
    "fake_sync_playwright_with_magic",
    "fake_sync_playwright_with_mixed_scripts",
    "fake_sync_playwright_with_scripts",
]
