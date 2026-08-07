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
from tankpit_bot.types.message import CapturedMessage


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


class AutoscrollKeyProtocol(Protocol):
    """The single keyboard member the autoscroll toggle dance uses."""

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Press a keyboard key."""
        ...


class AutoscrollPageProtocol(Protocol):
    """The slice of the page surface the autoscroll dance needs.

    Narrower than :class:`PageProtocol` so tests fake exactly the two
    members involved; the real Playwright page satisfies it
    structurally.
    """

    @property
    def keyboard(self) -> AutoscrollKeyProtocol:
        """Keyboard interface for the ``a`` toggle press."""
        ...

    def wait_for_timeout(self, timeout: float) -> None:
        """Pump the event loop while the server ack lands."""
        ...


class RoomJoinPageProtocol(Protocol):
    """The slice of the page surface the lobby join flow needs.

    Narrower than :class:`PageProtocol` for the same reason
    :class:`AutoscrollPageProtocol` is: the real Playwright page
    satisfies it structurally, and it is what lets the SIMULATOR drive
    the production ``join_room`` — the sim has no browser, so a lobby
    that demanded the full page surface would have forced a second
    copy of the join flow ([[session-state-deglobalisation]]).
    """

    @property
    def url(self) -> str:
        """Current page URL — one field of the room-entry metadata."""
        ...

    def wait_for_timeout(self, timeout: float) -> None:
        """Pump the event loop between lobby-response polls."""
        ...


class AutoscrollEnforcerProtocol(Protocol):
    """Session-start autoscroll-off enforcement seam.

    The real implementation presses the ``a`` toggle and verifies the
    server's plaintext ack (``tankpit_bot.browser.autoscroll``); the
    run-path tests replace it because their fake pages have no wire to
    ack from. Save-and-restore on this attribute per the DI testing
    contract -- never monkey-patch the autoscroll module.
    """

    def __call__(self, page: AutoscrollPageProtocol, messages: list[CapturedMessage]) -> None:
        """Leave the session with autoscroll wire-verified OFF."""
        ...


def _real_ensure_autoscroll_off(
    page: AutoscrollPageProtocol,
    messages: list[CapturedMessage],
) -> None:
    """Real implementation -- delegate to the autoscroll module.

    The import is deferred so the hooks package (imported by nearly
    everything) never drags the browser layer in at import time.

    Args:
        page: Live game page.
        messages: Capture buffer shared with the CDP service.
    """
    from tankpit_bot.browser.autoscroll import ensure_autoscroll_off as _impl

    _impl(page, messages)


ensure_autoscroll_off: AutoscrollEnforcerProtocol = _real_ensure_autoscroll_off

__all__ = [
    "AutoscrollEnforcerProtocol",
    "AutoscrollKeyProtocol",
    "AutoscrollPageProtocol",
    "BrowserContextProtocol",
    "BrowserProtocol",
    "BrowserTypeLaunchProtocol",
    "BrowserTypeProtocol",
    "PlaywrightProtocol",
    "RoomJoinPageProtocol",
    "SyncPlaywrightContextManagerProtocol",
    "SyncPlaywrightFactoryProtocol",
    "_real_ensure_autoscroll_off",
    "ensure_autoscroll_off",
]
