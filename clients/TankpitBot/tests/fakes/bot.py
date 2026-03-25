"""Bot-specific fake Playwright classes for testing.

Provides fake implementations for testing Bot class with KeyboardInterrupt exit.
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
    KeyboardProtocol,
    PageProtocol,
    PlaywrightProtocol,
    ResponseProtocol,
    SyncPlaywrightContextManagerProtocol,
)
from tests.fakes.base import FakeKeyboard, FakeResponse


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
        """Send CDP command.

        Returns a valid CDP response with ``{"result": {"value": ...}}``,
        matching the real Chrome DevTools Protocol contract.
        """
        _ = params
        self._sent_methods.append(method)
        return {"result": {"value": "ok"}}

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
    "FakeBrowserBot",
    "FakeBrowserContextBot",
    "FakeBrowserTypeBot",
    "FakeCDPSessionBot",
    "FakePageBot",
    "FakePageInterrupting",
    "FakePlaywrightBot",
    "FakeSyncPlaywrightContextManagerBot",
    "fake_sync_playwright_bot",
]
