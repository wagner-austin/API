"""Browser lifecycle functions — standalone navigation, login, and cleanup.

These functions operate on page/CDP session handles directly, without
requiring a BrowserSession instance. Used by probe_runtime to bootstrap
probes via composition instead of inheritance.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    BrowserProtocol,
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.browser.types import GameNotJoinedError
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)

_TEARDOWN_WATCHDOG_SECONDS = 30.0
_TEARDOWN_HANG_EXIT_CODE = 75


def _handle_teardown_hang() -> None:
    """Force the process to exit after a hung browser teardown."""
    log.error(
        "Teardown exceeded %.0fs; forcing process exit (artifacts were saved before cleanup)",
        _TEARDOWN_WATCHDOG_SECONDS,
    )
    _test_hooks.force_exit(_TEARDOWN_HANG_EXIT_CODE)


def navigate_and_login(
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    *,
    target_url: str,
    prefer_account: bool,
    tank_name_prefix: str = "TP",
    auto_join_room: bool = True,
) -> None:
    """Navigate to the game URL and handle login.

    Args:
        page: Playwright page.
        cdp: CDP session.
        target_url: URL to navigate to.
        prefer_account: Whether to prefer account login.
        tank_name_prefix: Prefix for tank name.
        auto_join_room: Whether to automatically join a room.

    Raises:
        GameNotJoinedError: If login or room join fails.
    """
    from tankpit_bot.browser.login import handle_login_flow

    page.goto(target_url, wait_until="domcontentloaded")
    log.info("Navigated to %s", page.url)

    success = handle_login_flow(
        page,
        cdp,
        tank_name_prefix=tank_name_prefix,
        prefer_account=prefer_account,
        auto_join_room=auto_join_room,
    )
    if not success:
        raise GameNotJoinedError("login or room join did not complete successfully")


def wait_for_game_ready(
    page: PageProtocol,
    messages: list[CapturedMessage],
) -> None:
    """Wait for game to fully load (message flow stabilizes).

    Args:
        page: Playwright page.
        messages: Captured message list to monitor for stability.

    Raises:
        GameNotJoinedError: If no messages captured after stabilization.
    """
    log.info("Waiting for game to initialize...")
    page.wait_for_timeout(2000.0)

    last_count = len(messages)
    stable_checks = 0
    while stable_checks < 3:
        page.wait_for_timeout(500.0)
        current_count = len(messages)
        if current_count == last_count:
            stable_checks += 1
        else:
            stable_checks = 0
            last_count = current_count

    if len(messages) == 0:
        raise GameNotJoinedError("No WebSocket messages captured - game may not have loaded")

    log.info("Game ready, captured %d initial messages", len(messages))


def cleanup_browser(browser: BrowserProtocol) -> None:
    """Close the browser with a teardown watchdog.

    Args:
        browser: Browser instance to close.
    """
    _test_hooks.start_watchdog(_TEARDOWN_WATCHDOG_SECONDS, _handle_teardown_hang)
    log.info("Teardown: closing browser")
    try:
        browser.close()
    except (OSError, RuntimeError) as exc:
        log.debug("Browser close failed (already closed): %s", exc)
    log.info("Teardown: browser closed")


__all__ = [
    "cleanup_browser",
    "navigate_and_login",
    "wait_for_game_ready",
]
