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
    ) -> BrowserProtocol:
        """Launch a browser instance.

        Args:
            headless: Whether to run browser in headless mode. Defaults to True.
            slow_mo: Slows down operations by the specified milliseconds.
            timeout: Maximum time to wait for browser to start in milliseconds.
            args: Extra command-line flags forwarded to Chromium. The
                sniffer + bot use this to pin the browser to the streamed
                display (see ``_chrome_stream_display_args``).

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
    ) -> BrowserProtocol:
        """Launch a browser instance.

        Args:
            headless: Whether to run browser in headless mode. Defaults to True.
            slow_mo: Slows down operations by the specified milliseconds.
            timeout: Maximum time to wait for browser to start in milliseconds.
            args: Extra command-line flags forwarded to Chromium. The
                sniffer + bot use this to pin the browser to the streamed
                display (see ``_chrome_stream_display_args``).

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


__all__ = [
    "BrowserContextProtocol",
    "BrowserProtocol",
    "BrowserTypeLaunchProtocol",
    "BrowserTypeProtocol",
    "PlaywrightProtocol",
    "SyncPlaywrightContextManagerProtocol",
    "SyncPlaywrightFactoryProtocol",
]
