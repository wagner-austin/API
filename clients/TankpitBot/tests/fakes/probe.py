"""Fake Playwright browser stack and sync-context factories for probes.

``probe.py`` was 815 lines; the CDP session and page fakes are now
siblings, re-exported through :mod:`tests.fakes`.
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
from tests.fakes.probe_cdp import FakeCDPSessionProbe
from tests.fakes.probe_page import (
    FakePageProbe,
    FakePageProbeNoMessages,
)


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
        if not emit_messages:
            self._cdp_session._raw_messages_ready = True
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

    def storage_state(self) -> JSONObject:
        """Return an empty Playwright storage-state snapshot for the probe fake."""
        empty_cookies: list[JSONValue] = []
        empty_origins: list[JSONValue] = []
        return {"cookies": empty_cookies, "origins": empty_origins}

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

    def new_context(
        self,
        *,
        no_viewport: bool | None = None,
        storage_state: str | None = None,
    ) -> BrowserContextProtocol:
        """Create new context."""
        _ = (no_viewport, storage_state)
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
        args: list[str] | None = None,
    ) -> BrowserProtocol:
        """Launch browser."""
        _ = (headless, slow_mo, timeout, args)
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
