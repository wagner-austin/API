"""A visitor's browser: the installed Microsoft Edge, driven through Playwright.

Board task 46934cd6. ``test_live_page.py`` opens austinwagner.org/tankpit the
way a visitor does and watches a tile play, so it needs a browser that
decodes what the demo streams, H.264 video and AAC audio. Playwright's own
Chromium build carries neither codec, and its video element would sit at
``currentTime`` 0 forever; the Edge every Windows node ships with carries
both, and Playwright drives it through its ``msedge`` channel without a
download of its own.

The bot's protocols in ``tankpit_bot._test_hooks.browser`` declare
``launch`` without ``channel``, because no bot path picks a browser build;
this module declares the one launcher shape a visitor needs and binds the
real ``sync_playwright`` to it, as ``playwright_loader`` does for the bot.
The browser, context and page it hands back are the bot's own protocols.
"""

from __future__ import annotations

import types
from typing import Final, Protocol

from tankpit_bot._test_hooks import BrowserProtocol

#: Playwright's channel name for the installed Microsoft Edge.
EDGE_CHANNEL: Final[str] = "msedge"


class EdgeLauncherProtocol(Protocol):
    """The ``launch`` of Playwright's ``BrowserType``, with the channel it takes."""

    def launch(
        self,
        *,
        headless: bool | None = None,
        channel: str | None = None,
    ) -> BrowserProtocol:
        """Launch an installed browser build.

        Args:
            headless: Whether to run without a window.
            channel: The installed build to run, e.g. ``msedge``.

        Returns:
            The browser.
        """
        ...


class VisitorPlaywrightProtocol(Protocol):
    """A started Playwright, as far as a visitor's browser needs it."""

    @property
    def chromium(self) -> EdgeLauncherProtocol:
        """The Chromium browser type, which launches Edge by channel."""
        ...


class VisitorPlaywrightManagerProtocol(Protocol):
    """The context manager ``sync_playwright()`` returns."""

    def __enter__(self) -> VisitorPlaywrightProtocol:
        """Start Playwright.

        Returns:
            The started Playwright.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Stop Playwright and every browser it started.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception instance if an exception was raised.
            exc_tb: Traceback if an exception was raised.
        """
        ...


class VisitorPlaywrightFactoryProtocol(Protocol):
    """``playwright.sync_api.sync_playwright`` itself."""

    def __call__(self) -> VisitorPlaywrightManagerProtocol:
        """Create the context manager that starts Playwright.

        Returns:
            The context manager.
        """
        ...


def visitor_playwright() -> VisitorPlaywrightFactoryProtocol:
    """Bind the real ``sync_playwright`` to the visitor's launcher shape.

    Returns:
        ``playwright.sync_api.sync_playwright``.
    """
    pw_module = __import__("playwright.sync_api", fromlist=["sync_playwright"])
    factory: VisitorPlaywrightFactoryProtocol = pw_module.sync_playwright
    return factory
