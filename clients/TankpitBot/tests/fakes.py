"""Fake Playwright classes for testing.

Provides fake implementations of Playwright protocols that don't require
real browser installation. All fakes match the protocol signatures in
tankpit_bot._test_hooks.
"""

from __future__ import annotations

import types
from collections.abc import Callable

from platform_core.json_utils import JSONObject, JSONValue

from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeProtocol,
    CDPSessionProtocol,
    PageProtocol,
    PlaywrightProtocol,
    ResponseProtocol,
    SyncPlaywrightContextManagerProtocol,
)


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
            magic: Optional magic value to return for tankpit.magic evaluate.
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
        self._cdp_session.emit_event(
            "Network.webSocketFrameSent",
            {
                "requestId": "1.1",
                "timestamp": 100.0,
                "response": {"opcode": 1, "mask": True, "payloadData": "sent message"},
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
        self._cdp_session.emit_event(
            "Network.webSocketFrameSent",
            {
                "requestId": "1.1",
                "timestamp": 100.0,
                "response": {"opcode": 1, "mask": True, "payloadData": "sent message"},
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

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression.

        Returns:
            Magic value if expression is 'tankpit.magic', otherwise script_urls.
        """
        if expression == "tankpit.magic":
            return self._magic
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
            magic: Magic value to return for tankpit.magic evaluate.
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
            magic: Magic value to return for tankpit.magic evaluate.
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
            magic: Magic value to return for tankpit.magic evaluate.
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
            magic: Magic value to return for tankpit.magic evaluate.
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
            magic: Magic value to return for tankpit.magic evaluate.
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
    """Create fake sync_playwright that returns a magic value for tankpit.magic.

    Returns:
        Context manager that produces pages returning magic value.
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
    ) -> None:
        """Initialize fake CDP session for probing.

        Args:
            emit_on_key: Whether to emit messages when WebSocket commands are sent.
            emit_on_mouse: Whether to emit messages when mouse inputs are injected.
            viewport_result: Custom viewport result to return, None uses default.
            return_invalid_result: Return non-dict result for Runtime.evaluate.
            return_missing_value: Return dict without value for Runtime.evaluate.
        """
        self._handlers: dict[str, list[Callable[[JSONObject], None]]] = {}
        self._sent_methods: list[str] = []
        self._detached = False
        self._emit_on_key = emit_on_key
        self._emit_on_mouse = emit_on_mouse
        self._viewport_result = viewport_result
        self._return_invalid_result = return_invalid_result
        self._return_missing_value = return_missing_value
        self._input_count = 0
        self._ws_url = "wss://tankpit.com/ws/"

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command and optionally emit WebSocket response."""
        self._sent_methods.append(method)

        # Return viewport size for Runtime.evaluate
        if method == "Runtime.evaluate" and params is not None:
            expression = params.get("expression", "")
            expr_str = str(expression)
            if "innerWidth" in expr_str:
                if self._viewport_result is not None:
                    return self._viewport_result
                return {"result": {"value": '{"w":800,"h":600}'}}
            # Return invalid result if configured
            if self._return_invalid_result:
                return {"error": "simulated error"}
            # Return missing value if configured
            if self._return_missing_value:
                return {"result": {}}
            # Detect WebSocket send via _send_websocket_bytes and emit messages
            if "ws.send" in expr_str and "__capturedWS" in expr_str and self._emit_on_key:
                self._input_count += 1
                self._emit_ws_sent(f"key_input_{self._input_count}")
                self._emit_ws_received(f"key_response_{self._input_count}")
                return {"result": {"value": f"SENT_5_BYTES via {self._ws_url}"}}
            # Other evaluates return success
            return {"result": {"value": "success"}}

        # When mouse input is dispatched, emit a WebSocket message
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
            self._cdp_session.emit_event(
                "Network.webSocketFrameSent",
                {
                    "requestId": "1.1",
                    "timestamp": 1.0,
                    "response": {"opcode": 1, "mask": True, "payloadData": "auth"},
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

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression.

        Returns magic value for tankpit.magic, empty list otherwise.
        """
        if expression == "tankpit.magic":
            return self.DEFAULT_MAGIC
        return []


class FakePageProbeNoMessages:
    """Fake Page for probe testing that doesn't emit any messages."""

    # Default magic value for XOR table construction
    DEFAULT_MAGIC = "test_magic_12345678"

    def __init__(self, cdp_session: FakeCDPSessionProbe) -> None:
        """Initialize fake page."""
        self._cdp_session = cdp_session
        self._closed = False
        self._url = ""

    @property
    def url(self) -> str:
        """Get the current URL of the page."""
        return self._url

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

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression.

        Returns magic value for tankpit.magic, empty list otherwise.
        """
        if expression == "tankpit.magic":
            return self.DEFAULT_MAGIC
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


__all__ = [
    "FakeBrowser",
    "FakeBrowserContext",
    "FakeBrowserContextProbe",
    "FakeBrowserProbe",
    "FakeBrowserType",
    "FakeBrowserTypeProbe",
    "FakeCDPSession",
    "FakeCDPSessionProbe",
    "FakeCDPSessionRateLimited",
    "FakePage",
    "FakePageNoMessages",
    "FakePageProbe",
    "FakePageProbeNoMessages",
    "FakePlaywright",
    "FakePlaywrightProbe",
    "FakeResponse",
    "FakeSyncPlaywrightContextManager",
    "FakeSyncPlaywrightContextManagerProbe",
    "fake_sync_playwright",
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
