"""Fake Playwright browser stack and the sync-context factories.

``base.py`` was 1,301 lines; the payloads, terrain map, CDP sessions,
and page fakes are now siblings, re-exported through
:mod:`tests.fakes`.
"""

from __future__ import annotations

import types

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
)

from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeProtocol,
    CDPSessionProtocol,
    PageProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
)
from tests.fakes.cdp import (
    FakeCDPSession,
    FakeCDPSessionRateLimited,
)
from tests.fakes.page import (
    FakePage,
    FakePageNoMessages,
)


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
            FakeCDPSessionRateLimited(
                login_fails=login_fails,
                emit_runtime_frames=emit_messages,
            )
            if rate_limited
            else FakeCDPSession(emit_runtime_frames=emit_messages)
        )
        self._cdp_session = cdp
        if not emit_messages:
            self._cdp_session._raw_messages_ready = True
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

    def storage_state(self) -> JSONObject:
        """Return an empty Playwright storage-state snapshot.

        The fake browser has no real cookies or origins to serialise;
        returning the canonical empty shape lets
        :func:`save_storage_state` write a valid file through the fake
        filesystem without touching real Playwright.
        """
        empty_cookies: list[JSONValue] = []
        empty_origins: list[JSONValue] = []
        return {"cookies": empty_cookies, "origins": empty_origins}

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

    def new_context(
        self,
        *,
        no_viewport: bool | None = None,
        storage_state: str | None = None,
    ) -> BrowserContextProtocol:
        """Create new context."""
        _ = (no_viewport, storage_state)
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
        args: list[str] | None = None,
    ) -> BrowserProtocol:
        """Launch browser."""
        _ = (headless, slow_mo, timeout, args)
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
