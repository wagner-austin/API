"""Playwright Browser/BrowserType/BrowserContext/Playwright protocols.

The browser bootstrap path (launch -> new_context -> new_page) is fully
expressed through these protocols so the rest of the bot never depends
on Playwright concrete types. The companion ``playwright_loader`` module
hosts the factory hook that turns ``sync_playwright`` into one of these
objects.
"""

from __future__ import annotations

import types
from typing import Protocol

from platform_core.json_utils import JSONObject

from tankpit_bot._test_hooks.cdp import CDPSessionProtocol, PageProtocol


class BrowserContextProtocol(Protocol):
    """Protocol for Playwright BrowserContext.

    Matches playwright.sync_api.BrowserContext interface for methods we use.
    """

    def new_page(self) -> PageProtocol:
        """Create a new page in the browser context.

        Returns:
            New page instance.
        """
        ...

    def new_cdp_session(self, page: PageProtocol) -> CDPSessionProtocol:
        """Create a new CDP session attached to the page.

        Args:
            page: Page to attach CDP session to.

        Returns:
            CDP session instance.
        """
        ...

    def storage_state(self) -> JSONObject:
        """Snapshot the context's cookies + localStorage as a JSON object.

        The bot's session-storage layer dumps this snapshot to disk
        after ``wait_for_game_ready`` so the next launch can restore
        auth without repeating the login flow.

        Returns:
            A dict shaped like ``{"cookies": [...], "origins": [...]}``
            per Playwright's storage-state format. The bot serialises
            it via ``platform_core.json_utils.dump_json_str``.
        """
        ...

    def close(self, *, reason: str | None = None) -> None:
        """Close the browser context.

        Args:
            reason: Reason to be reported to operations interrupted by context closure.
        """
        ...


class BrowserProtocol(Protocol):
    """Protocol for Playwright Browser.

    Matches playwright.sync_api.Browser interface for methods we use.
    """

    def new_context(
        self,
        *,
        no_viewport: bool | None = None,
        storage_state: str | None = None,
    ) -> BrowserContextProtocol:
        """Create a new browser context.

        Args:
            no_viewport: When ``True``, Playwright will not clamp the
                page to its default 1280x720 viewport. Used by the
                streamed-display path so the browser fills the
                display-sized window (see
                ``_chrome_stream_no_viewport``); other callers omit it
                and get Playwright's stable default viewport for tests.
            storage_state: Optional filesystem path Playwright will
                seed the context's cookies + localStorage from. The bot
                passes the path returned by
                :func:`tankpit_bot.browser.session_storage.load_storage_state`
                so subsequent launches skip the tankpit login flow.
                ``None`` starts fresh.

        Returns:
            New browser context instance.
        """
        ...

    def close(self, *, reason: str | None = None) -> None:
        """Close the browser.

        Args:
            reason: Reason to be reported to operations interrupted by browser closure.
        """
        ...


class BrowserTypeLaunchProtocol(Protocol):
    """Protocol for BrowserType.launch method."""

    def __call__(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> BrowserProtocol:
        """Launch a browser instance.

        Args:
            headless: Whether to run browser in headless mode. Defaults to True.
            slow_mo: Slows down operations by the specified milliseconds.
            timeout: Maximum time to wait for browser to start in milliseconds.
            args: Extra command-line flags forwarded to Chromium. The
                sniffer + bot use this to pin the browser to the streamed
                display (see ``_chrome_stream_display_args``).
            env: Environment for the browser process. ``None`` inherits
                this process's. Playwright REPLACES rather than merges,
                so the display-capture path passes a full copy of the
                parent environment with ``DISPLAY`` overlaid.

        Returns:
            Browser instance.
        """
        ...


class BrowserTypeProtocol(Protocol):
    """Protocol for Playwright BrowserType (e.g., playwright.chromium)."""

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> BrowserProtocol:
        """Launch a browser instance.

        Args:
            headless: Whether to run browser in headless mode. Defaults to True.
            slow_mo: Slows down operations by the specified milliseconds.
            timeout: Maximum time to wait for browser to start in milliseconds.
            args: Extra command-line flags forwarded to Chromium. The
                sniffer + bot use this to pin the browser to the streamed
                display (see ``_chrome_stream_display_args``).
            env: Environment for the browser process. ``None`` inherits
                this process's. Playwright REPLACES rather than merges,
                so the display-capture path passes a full copy of the
                parent environment with ``DISPLAY`` overlaid.

        Returns:
            Browser instance.
        """
        ...


class PlaywrightProtocol(Protocol):
    """Protocol for Playwright instance from sync_playwright().start()."""

    @property
    def chromium(self) -> BrowserTypeProtocol:
        """Chromium browser type.

        Returns:
            BrowserType for Chromium.
        """
        ...

    def stop(self) -> None:
        """Stop the Playwright instance."""
        ...


class SyncPlaywrightContextManagerProtocol(Protocol):
    """Protocol for sync_playwright() context manager."""

    def start(self) -> PlaywrightProtocol:
        """Start Playwright and return the instance.

        Returns:
            Playwright instance.
        """
        ...

    def __enter__(self) -> PlaywrightProtocol:
        """Enter context manager.

        Returns:
            Playwright instance.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context manager.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception instance if an exception was raised.
            exc_tb: Traceback if an exception was raised.
        """
        ...


class SyncPlaywrightFactoryProtocol(Protocol):
    """Protocol for sync_playwright() function."""

    def __call__(self) -> SyncPlaywrightContextManagerProtocol:
        """Create a Playwright context manager.

        Returns:
            Context manager that yields Playwright instance.
        """
        ...


class PageKeyboardProtocol(Protocol):
    """The single keyboard member the key probe presses.

    Named for its one remaining consumer: the autoscroll enforcement
    stopped pressing keys 2026-08-13 (hotkey maps are per-account, so
    a keypress is not a reliable command instrument — see
    ``browser/autoscroll.py``); ``action_lab/key_probe.py`` still
    presses physical keys deliberately, because capturing what a key
    EMITS is its whole purpose.
    """

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Press a keyboard key."""
        ...


class PageWaitProtocol(Protocol):
    """The one page member every poll-and-read flow needs.

    Three flows wait on the wire — the autoscroll dance, the lobby
    join, and the account-stats panel read — and each used to name the
    whole :class:`PageProtocol` for it. Narrowing to what is actually
    called is what lets the SIMULATOR stand in for a page it has no
    browser for; the real Playwright page satisfies every one of these
    structurally ([[session-state-deglobalisation]]).
    """

    def wait_for_timeout(self, timeout: float) -> None:
        """Pump the event loop while the wire answers."""
        ...


class GamePageProtocol(PageWaitProtocol, Protocol):
    """The page surface a live game session holds.

    The bot's ``_page`` type: event-loop pumping for every
    poll-and-read flow (the inherited wait) plus the keyboard the key
    probe presses.
    """

    @property
    def keyboard(self) -> PageKeyboardProtocol:
        """Keyboard interface for deliberate key presses (key probe)."""
        ...


class RoomJoinPageProtocol(PageWaitProtocol, Protocol):
    """The slice of the page surface the lobby join flow needs."""

    @property
    def url(self) -> str:
        """Current page URL — one field of the room-entry metadata."""
        ...


__all__ = [
    "BrowserContextProtocol",
    "BrowserProtocol",
    "BrowserTypeLaunchProtocol",
    "BrowserTypeProtocol",
    "GamePageProtocol",
    "PageKeyboardProtocol",
    "PageWaitProtocol",
    "PlaywrightProtocol",
    "RoomJoinPageProtocol",
    "SyncPlaywrightContextManagerProtocol",
    "SyncPlaywrightFactoryProtocol",
]
