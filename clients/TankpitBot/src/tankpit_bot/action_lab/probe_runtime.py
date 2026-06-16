"""Shared live-probe runtime bootstrap helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypedDict, TypeVar

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeProtocol,
    CDPSessionProtocol,
    PageProtocol,
    PlaywrightProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.browser import PlaywrightNotInstalledError, reset_cdp_time_offset
from tankpit_bot.sniffer import reset_all_trackers, reset_world_state
from tankpit_bot.sniffer.viewport import reset_viewport_tracking
from tankpit_bot.state import SelfStateDict
from tankpit_bot.types import CapturedMessage


class ProbeRuntimeStateProtocol(Protocol):
    """Mutable probe state needed for shared runtime bootstrap."""

    _start_timestamp_ms: int
    _messages: list[CapturedMessage]
    _ws_urls: dict[str, str]
    _magic: str | None
    _cdp_message_buffer: list[str]
    _cdp: CDPSessionProtocol | None
    _page: PageProtocol | None

    def _reset_action_cycle_tracker(self) -> None:
        """Reset probe-local action phase tracking."""

    def _setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        """Install runtime console listeners."""

    def _setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        """Install runtime CDP handlers."""

    def _navigate_and_login(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        *,
        tank_name_prefix: str = "TP",
        auto_join_room: bool = True,
    ) -> None:
        """Navigate and log in to the live game session."""

    def _wait_for_game_ready(self, page: PageProtocol) -> None:
        """Wait until the game client is ready."""

    def _gather_intel(self, page: PageProtocol, cdp: CDPSessionProtocol) -> None:
        """Capture runtime intel needed before issuing commands."""


class ProbeCommandReadyProtocol(
    ProbeRuntimeStateProtocol,
    action_session.BufferedWorldStateProviderProtocol,
    action_session.StartupStateDriverProtocol,
    Protocol,
):
    """Probe protocol needed for initial world sync and command readiness."""


class ProbeSessionRunnerProtocol(ProbeCommandReadyProtocol, Protocol):
    """Probe protocol required for the shared live-session bootstrap shell."""

    _headless: bool

    def _cleanup(
        self,
        cdp: CDPSessionProtocol,
        page: PageProtocol,
        context: BrowserContextProtocol,
        browser: BrowserProtocol,
    ) -> None:
        """Clean up browser resources after the session."""


class ProbeCommandReadyContextDict(TypedDict):
    """Typed command-ready context returned after shared runtime bootstrap."""

    game_ready_timestamp_ms: int
    intel_ready_timestamp_ms: int
    initial_sync_started_ms: int
    initial_world_timestamp_ms: int
    spawn: SelfStateDict
    command_ready_timestamp_ms: int


SessionT = TypeVar("SessionT")


def initialize_live_probe_session(probe: ProbeRuntimeStateProtocol) -> int:
    """Reset mutable probe state for a fresh live session.

    Args:
        probe: Probe instance whose session state should be cleared.

    Returns:
        Session start timestamp in milliseconds.
    """
    start_timestamp_ms = action_hooks.get_current_time_ms()
    probe._start_timestamp_ms = start_timestamp_ms
    probe._messages = []
    probe._ws_urls = {}
    probe._magic = None
    probe._cdp_message_buffer = []
    probe._reset_action_cycle_tracker()
    return start_timestamp_ms


def launch_probe_browser(
    playwright: PlaywrightProtocol,
    *,
    headless: bool,
) -> tuple[BrowserProtocol, BrowserContextProtocol, PageProtocol, CDPSessionProtocol]:
    """Launch a browser, context, page, and CDP session for a probe.

    Args:
        playwright: Active Playwright runtime.
        headless: Whether to launch the browser headlessly.

    Returns:
        Browser, context, page, and CDP session handles.
    """
    browser_type: BrowserTypeProtocol = playwright.chromium
    browser = browser_type.launch(headless=headless)
    context = browser.new_context()
    page = context.new_page()
    cdp = context.new_cdp_session(page)
    return browser, context, page, cdp


def prepare_live_probe_runtime(
    probe: ProbeRuntimeStateProtocol,
    *,
    page: PageProtocol,
    cdp: CDPSessionProtocol,
    tank_name_prefix: str = "TP",
    auto_join_room: bool = True,
) -> tuple[int, int]:
    """Prepare runtime listeners, navigation, and captured intel.

    Args:
        probe: Probe instance to initialize.
        page: Active Playwright page.
        cdp: Active CDP session.
        tank_name_prefix: Prefix to use for guest tank names when needed.
        auto_join_room: Whether login should auto-join a room.

    Returns:
        Game-ready and intel-ready timestamps in milliseconds.
    """
    probe._cdp = cdp
    probe._page = page
    reset_world_state()
    reset_all_trackers()
    reset_cdp_time_offset()
    reset_viewport_tracking()
    probe._setup_console_listener(cdp)
    probe._setup_cdp_handlers(cdp)
    probe._navigate_and_login(
        page,
        cdp,
        tank_name_prefix=tank_name_prefix,
        auto_join_room=auto_join_room,
    )
    probe._wait_for_game_ready(page)
    game_ready_timestamp_ms = action_hooks.get_current_time_ms()
    probe._gather_intel(page, cdp)
    intel_ready_timestamp_ms = action_hooks.get_current_time_ms()
    return game_ready_timestamp_ms, intel_ready_timestamp_ms


def wait_for_probe_command_ready(
    probe: ProbeCommandReadyProtocol,
    *,
    page: PageProtocol,
    initial_sync_timeout_ms: int,
) -> tuple[int, int, SelfStateDict, int]:
    """Wait for initial self-state sync and command readiness.

    Args:
        probe: Probe instance to advance.
        page: Active Playwright page.
        initial_sync_timeout_ms: Maximum wait for the initial sync.

    Returns:
        Initial-sync start time, initial world timestamp, spawn state, and
        command-ready timestamp.
    """
    initial_sync_started_ms = action_hooks.get_current_time_ms()
    initial_world_timestamp_ms, spawn = action_hooks.wait_for_initial_self_state(
        page,
        probe,
        initial_sync_started_ms,
        initial_sync_timeout_ms,
    )
    action_hooks.advance_startup_state(probe)
    command_ready_timestamp_ms = action_hooks.get_current_time_ms()
    return (
        initial_sync_started_ms,
        initial_world_timestamp_ms,
        spawn,
        command_ready_timestamp_ms,
    )


def execute_live_probe_bootstrap(
    probe: ProbeSessionRunnerProtocol,
    *,
    initial_sync_timeout_ms: int,
    run_ready_session: Callable[[ProbeCommandReadyContextDict], SessionT],
) -> SessionT:
    """Execute the shared browser/bootstrap shell for one live probe session.

    Args:
        probe: Probe instance that owns the runtime session.
        initial_sync_timeout_ms: Maximum wait for the initial sync.
        run_ready_session: Probe-specific callback invoked after command readiness.

    Returns:
        Probe-specific session payload returned by ``run_ready_session``.

    Raises:
        PlaywrightNotInstalledError: If Playwright is unavailable.
    """
    if _test_hooks.sync_playwright is None:
        raise PlaywrightNotInstalledError("Playwright is not installed.")

    probe._start_timestamp_ms = initialize_live_probe_session(probe)

    with _test_hooks.sync_playwright() as playwright:
        browser, context, page, cdp = launch_probe_browser(
            playwright,
            headless=probe._headless,
        )
        game_ready_timestamp_ms, intel_ready_timestamp_ms = prepare_live_probe_runtime(
            probe,
            page=page,
            cdp=cdp,
            tank_name_prefix="TP",
            auto_join_room=True,
        )

        try:
            (
                initial_sync_started_ms,
                initial_world_timestamp_ms,
                spawn,
                command_ready_timestamp_ms,
            ) = wait_for_probe_command_ready(
                probe,
                page=page,
                initial_sync_timeout_ms=initial_sync_timeout_ms,
            )
            return run_ready_session(
                ProbeCommandReadyContextDict(
                    game_ready_timestamp_ms=game_ready_timestamp_ms,
                    intel_ready_timestamp_ms=intel_ready_timestamp_ms,
                    initial_sync_started_ms=initial_sync_started_ms,
                    initial_world_timestamp_ms=initial_world_timestamp_ms,
                    spawn=spawn,
                    command_ready_timestamp_ms=command_ready_timestamp_ms,
                )
            )
        finally:
            clear_live_probe_runtime(probe)
            probe._cleanup(cdp, page, context, browser)


def build_probe_startup_timing(
    *,
    game_ready_timestamp_ms: int,
    intel_ready_timestamp_ms: int,
    initial_sync_started_ms: int,
    initial_world_timestamp_ms: int,
    command_ready_timestamp_ms: int,
    first_attempt_started_ms: int | None,
) -> TeleportStartupTimingDict:
    """Build shared startup timing metrics for a live probe session.

    Args:
        game_ready_timestamp_ms: When the game UI became ready.
        intel_ready_timestamp_ms: When probe intel capture completed.
        initial_sync_started_ms: When initial self-state sync started.
        initial_world_timestamp_ms: First authoritative world timestamp seen.
        command_ready_timestamp_ms: When the probe was ready to issue commands.
        first_attempt_started_ms: Timestamp of the first attempt, if any.

    Returns:
        Shared startup timing payload.
    """
    return TeleportStartupTimingDict(
        game_ready_timestamp_ms=game_ready_timestamp_ms,
        intel_ready_timestamp_ms=intel_ready_timestamp_ms,
        initial_sync_started_ms=initial_sync_started_ms,
        initial_world_timestamp_ms=initial_world_timestamp_ms,
        command_ready_timestamp_ms=command_ready_timestamp_ms,
        first_attempt_started_ms=first_attempt_started_ms,
        game_ready_to_intel_ready_ms=intel_ready_timestamp_ms - game_ready_timestamp_ms,
        intel_ready_to_initial_world_ms=initial_world_timestamp_ms - intel_ready_timestamp_ms,
        initial_world_to_command_ready_ms=command_ready_timestamp_ms - initial_world_timestamp_ms,
        command_ready_to_first_attempt_ms=(
            None
            if first_attempt_started_ms is None
            else first_attempt_started_ms - command_ready_timestamp_ms
        ),
    )


def clear_live_probe_runtime(probe: ProbeRuntimeStateProtocol) -> None:
    """Clear runtime browser handles after session shutdown.

    Args:
        probe: Probe instance whose runtime handles should be cleared.
    """
    probe._cdp = None
    probe._page = None


__all__ = [
    "ProbeCommandReadyContextDict",
    "ProbeCommandReadyProtocol",
    "ProbeRuntimeStateProtocol",
    "ProbeSessionRunnerProtocol",
    "build_probe_startup_timing",
    "clear_live_probe_runtime",
    "execute_live_probe_bootstrap",
    "initialize_live_probe_session",
    "launch_probe_browser",
    "prepare_live_probe_runtime",
    "wait_for_probe_command_ready",
]
