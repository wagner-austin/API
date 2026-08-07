"""Fake page and keyboard doubles for the tick-loop test modules.

``test_tick_loop_coverage.py`` was 1,605 lines; it is now four modules
over these doubles.
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue

from tankpit_bot._test_hooks import (
    KeyboardProtocol,
    ResponseProtocol,
)
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.bot.base import Bot


class _NoOpKeyboard:
    """Minimal keyboard stub."""

    def press(self, key: str, *, delay: float | None = None) -> None:
        """No-op."""

    def type(self, text: str, *, delay: float | None = None) -> None:
        """No-op."""


class _FakePage:
    """Minimal page stub for tick-loop testing."""

    def __init__(self) -> None:
        """Initialize."""
        self._url = "https://test.tankpit.com/play"
        self._keyboard = _NoOpKeyboard()

    @property
    def url(self) -> str:
        """Return URL."""
        return self._url

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Return keyboard."""
        return self._keyboard

    def wait_for_timeout(self, timeout: float) -> None:
        """No-op wait."""

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """No-op wait for event."""

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """No-op wait for function."""

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """No-op goto."""
        self._url = url
        return None

    def evaluate(self, expression: str) -> JSONValue:
        """No-op evaluate."""
        return None

    def close(
        self,
        *,
        reason: str | None = None,
        run_before_unload: bool | None = None,
    ) -> None:
        """No-op close."""


class _BrowserClosedPage(_FakePage):
    """Page stub whose ``wait_for_timeout`` raises ``TargetClosedError``.

    Models the operator closing the browser while the bot is sleeping
    between ticks. The first tick runs to completion; the wait afterward
    is what fails. Lets the tick-loop graceful-shutdown path execute
    end-to-end without launching Playwright.
    """

    def wait_for_timeout(self, timeout: float) -> None:
        """Raise to model the browser being closed between ticks."""
        from playwright._impl._errors import TargetClosedError

        raise TargetClosedError("Page.wait_for_timeout: target closed")


class _TickRaisesBrowserClosedPage(_FakePage):
    """Page stub whose first ``set_content`` (used in tick_once) raises.

    Used to exercise the ``except TargetClosedError`` around
    ``_tick_once`` itself, the in-loop path. Modeled on a Playwright
    call failing because the browser shut mid-tick.
    """


def _fail_tick_once_with_browser_closed(bot: Bot) -> None:
    """Drop-in ``_tick_once`` that simulates the browser closing mid-tick."""
    _ = bot
    from playwright._impl._errors import TargetClosedError

    raise TargetClosedError("Page.goto: target closed mid-tick")


def _fail_tick_once_with_session_exit(bot: Bot) -> None:
    """Drop-in ``_tick_once`` that simulates a decision-owner exit request."""
    _ = bot
    from tankpit_bot.bot.session_exit import SessionExitError

    raise SessionExitError("no_viable_targets", "fresh map snapshot has no affordable enemy")
