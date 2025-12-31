"""Fake Playwright classes for testing.

Provides fake implementations of Playwright protocols that don't require
real browser installation. All fakes match the protocol signatures in
tankpit_bot._test_hooks.
"""

from __future__ import annotations

import types
from collections.abc import Callable

from platform_core.json_utils import JSONObject

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

    def __init__(self, cdp_session: FakeCDPSession | FakeCDPSessionRateLimited) -> None:
        """Initialize fake page."""
        self._cdp_session = cdp_session
        self._goto_url: str | None = None
        self._wait_timeout: float | None = None
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

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True


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

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)
        self._closed = True


class FakeBrowserContext:
    """Fake Playwright BrowserContext."""

    def __init__(
        self,
        *,
        emit_messages: bool = True,
        rate_limited: bool = False,
        login_fails: bool = False,
    ) -> None:
        """Initialize fake browser context."""
        cdp: FakeCDPSession | FakeCDPSessionRateLimited = (
            FakeCDPSessionRateLimited(login_fails=login_fails) if rate_limited else FakeCDPSession()
        )
        self._cdp_session = cdp
        self._pages: list[FakePage | FakePageNoMessages] = []
        self._closed = False
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited

    def new_page(self) -> PageProtocol:
        """Create new page."""
        page: FakePage | FakePageNoMessages
        if self._emit_messages:
            page = FakePage(self._cdp_session)
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
    ) -> None:
        """Initialize fake browser."""
        self._contexts: list[FakeBrowserContext] = []
        self._closed = False
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._login_fails = login_fails

    def new_context(self) -> BrowserContextProtocol:
        """Create new context."""
        ctx = FakeBrowserContext(
            emit_messages=self._emit_messages,
            rate_limited=self._rate_limited,
            login_fails=self._login_fails,
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
    ) -> None:
        """Initialize fake browser type."""
        self._browsers: list[FakeBrowser] = []
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._login_fails = login_fails

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
    ) -> None:
        """Initialize fake Playwright."""
        self._chromium = FakeBrowserType(
            emit_messages=emit_messages,
            rate_limited=rate_limited,
            login_fails=login_fails,
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
    ) -> None:
        """Initialize fake context manager."""
        self._playwright: FakePlaywright | None = None
        self._emit_messages = emit_messages
        self._rate_limited = rate_limited
        self._login_fails = login_fails

    def start(self) -> PlaywrightProtocol:
        """Start Playwright."""
        self._playwright = FakePlaywright(
            emit_messages=self._emit_messages,
            rate_limited=self._rate_limited,
            login_fails=self._login_fails,
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


__all__ = [
    "FakeBrowser",
    "FakeBrowserContext",
    "FakeBrowserType",
    "FakeCDPSession",
    "FakeCDPSessionRateLimited",
    "FakePage",
    "FakePageNoMessages",
    "FakePlaywright",
    "FakeResponse",
    "FakeSyncPlaywrightContextManager",
    "fake_sync_playwright",
    "fake_sync_playwright_login_fails",
    "fake_sync_playwright_no_messages",
    "fake_sync_playwright_rate_limited",
]
