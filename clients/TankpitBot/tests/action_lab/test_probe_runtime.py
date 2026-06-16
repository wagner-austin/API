"""Tests for shared live-probe runtime bootstrap helpers."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import ClockAdvancingPage, ReplayClock, StubSnapshotCDPSession

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import probe_runtime
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state, make_self_state
from tankpit_bot.types import CapturedMessage

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


class _ProbeHarness:
    def __init__(self, clock: ReplayClock | None = None) -> None:
        self._start_timestamp_ms = 0
        self._messages: list[CapturedMessage] = [
            CapturedMessage(
                timestamp_ms=1,
                direction="received",
                payload="x",
                ws_url="wss://tankpit.com/ws/",
            )
        ]
        self._ws_urls = {"a": "b"}
        self._magic: str | None = "old"
        self._cdp_message_buffer = ["x"]
        self._cdp: CDPSessionProtocol | None = None
        self._page: PageProtocol | None = None
        self._world_state = make_empty_world_state()
        self._headless = False
        self.reset_calls = 0
        self.console_calls = 0
        self.handler_calls = 0
        self.navigate_calls = 0
        self.ready_calls = 0
        self.intel_calls = 0
        self.cleanup_calls = 0
        self._clock = clock
        self.ready_advance_ms = 0
        self.intel_advance_ms = 0

    def _reset_action_cycle_tracker(self) -> None:
        self.reset_calls += 1

    def _setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        assert cdp is self._cdp
        self.console_calls += 1

    def _setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        assert cdp is self._cdp
        self.handler_calls += 1

    def _navigate_and_login(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        *,
        tank_name_prefix: str = "TP",
        auto_join_room: bool = True,
    ) -> None:
        assert page is self._page
        assert cdp is self._cdp
        assert tank_name_prefix == "TP"
        assert auto_join_room is True
        self.navigate_calls += 1

    def _wait_for_game_ready(self, page: PageProtocol) -> None:
        assert page is self._page
        self.ready_calls += 1
        if self._clock is not None:
            self._clock.advance(self.ready_advance_ms)

    def _gather_intel(self, page: PageProtocol, cdp: CDPSessionProtocol) -> None:
        assert page is self._page
        assert cdp is self._cdp
        self.intel_calls += 1
        if self._clock is not None:
            self._clock.advance(self.intel_advance_ms)

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages

    @property
    def magic(self) -> str | None:
        return self._magic

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_state(self) -> str:
        return "WAITING_FOR_POSITION"

    def _update_state_from_world(self) -> None:
        return None

    def _cleanup(
        self,
        cdp: CDPSessionProtocol,
        page: PageProtocol,
        context: BrowserContextProtocol,
        browser: BrowserProtocol,
    ) -> None:
        _ = (cdp, page, context, browser)
        self.cleanup_calls += 1


def _spawn() -> SelfStateDict:
    return make_self_state(
        tank_id=1,
        x=100,
        y=120,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=5,
    )


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    original_get_time = action_hooks.get_current_time_ms
    original_wait_initial = action_hooks.wait_for_initial_self_state
    original_advance = action_hooks.advance_startup_state
    original_sync_playwright = core_hooks.sync_playwright
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.wait_for_initial_self_state = original_wait_initial
    action_hooks.advance_startup_state = original_advance
    core_hooks.sync_playwright = original_sync_playwright


def test_initialize_live_probe_session_resets_runtime_state() -> None:
    clock = ReplayClock(500)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()

    start_timestamp_ms = probe_runtime.initialize_live_probe_session(probe)

    assert start_timestamp_ms == 500
    assert probe._start_timestamp_ms == 500
    assert probe._messages == []
    assert probe._ws_urls == {}
    assert probe._magic is None
    assert probe._cdp_message_buffer == []
    assert probe.reset_calls == 1


def test_launch_probe_browser_returns_live_handles() -> None:
    probe = _ProbeHarness(ReplayClock())
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)

    (
        launched_browser,
        launched_context,
        launched_page,
        launched_cdp,
    ) = probe_runtime.launch_probe_browser(recorded.playwright, headless=True)

    assert launched_browser is recorded.browser
    assert launched_context is recorded.context
    assert launched_page is recorded.page
    assert launched_cdp is recorded.cdp
    assert recorded.browser_type.launches == [True]


def test_prepare_live_probe_runtime_sets_handles_and_records_timing() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    page = ClockAdvancingPage(ReplayClock())
    cdp = StubSnapshotCDPSession()
    probe.ready_advance_ms = 25
    probe.intel_advance_ms = 35

    game_ready_timestamp_ms, intel_ready_timestamp_ms = probe_runtime.prepare_live_probe_runtime(
        probe,
        page=page,
        cdp=cdp,
    )

    assert probe._page is page
    assert probe._cdp is cdp
    assert game_ready_timestamp_ms == 1025
    assert intel_ready_timestamp_ms == 1060
    assert probe.console_calls == 1
    assert probe.handler_calls == 1
    assert probe.navigate_calls == 1
    assert probe.ready_calls == 1
    assert probe.intel_calls == 1


def test_wait_for_probe_command_ready_advances_startup_state() -> None:
    clock = ReplayClock(2000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()
    page = ClockAdvancingPage(ReplayClock())
    spawn = _spawn()
    advance_calls = 0

    def _wait_initial(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        assert page_arg is page
        assert provider.get_world_state() == probe.get_world_state()
        assert started_ms == 2000
        assert timeout_ms == 9000
        clock.advance(45)
        return 2600, spawn

    def _advance(provider: action_session.StartupStateDriverProtocol) -> None:
        assert provider.get_state() == "WAITING_FOR_POSITION"
        nonlocal advance_calls
        advance_calls += 1
        clock.advance(10)

    wait_attr = "wait_for_initial_self_state"
    advance_attr = "advance_startup_state"
    setattr(action_hooks, wait_attr, _wait_initial)
    setattr(action_hooks, advance_attr, _advance)

    (
        initial_sync_started_ms,
        initial_world_timestamp_ms,
        returned_spawn,
        command_ready_timestamp_ms,
    ) = probe_runtime.wait_for_probe_command_ready(
        probe,
        page=page,
        initial_sync_timeout_ms=9000,
    )

    assert initial_sync_started_ms == 2000
    assert initial_world_timestamp_ms == 2600
    assert returned_spawn == spawn
    assert command_ready_timestamp_ms == 2055
    assert advance_calls == 1


def test_build_probe_startup_timing_computes_deltas() -> None:
    startup_timing = probe_runtime.build_probe_startup_timing(
        game_ready_timestamp_ms=100,
        intel_ready_timestamp_ms=160,
        initial_sync_started_ms=200,
        initial_world_timestamp_ms=450,
        command_ready_timestamp_ms=500,
        first_attempt_started_ms=575,
    )

    assert startup_timing["game_ready_to_intel_ready_ms"] == 60
    assert startup_timing["intel_ready_to_initial_world_ms"] == 290
    assert startup_timing["initial_world_to_command_ready_ms"] == 50
    assert startup_timing["command_ready_to_first_attempt_ms"] == 75


def test_clear_live_probe_runtime_clears_page_and_cdp() -> None:
    probe = _ProbeHarness()
    probe._page = ClockAdvancingPage(ReplayClock())
    probe._cdp = StubSnapshotCDPSession()

    probe_runtime.clear_live_probe_runtime(probe)

    assert probe._page is None
    assert probe._cdp is None


def test_execute_live_probe_bootstrap_runs_ready_callback_and_cleans_up() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    probe.ready_advance_ms = 25
    probe.intel_advance_ms = 35
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    spawn = _spawn()

    def _wait_initial(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        assert page_arg is recorded.page
        assert provider is probe
        assert started_ms == 1060
        assert timeout_ms == 9000
        clock.advance(45)
        return 2600, spawn

    advance_calls = 0

    def _advance(provider: action_session.StartupStateDriverProtocol) -> None:
        assert provider is probe
        nonlocal advance_calls
        advance_calls += 1
        clock.advance(10)

    wait_initial_name = "wait_for_initial_self_state"
    advance_name = "advance_startup_state"
    setattr(action_hooks, wait_initial_name, _wait_initial)
    setattr(action_hooks, advance_name, _advance)

    def _run_ready_session(context_dict: probe_runtime.ProbeCommandReadyContextDict) -> str:
        assert context_dict == probe_runtime.ProbeCommandReadyContextDict(
            game_ready_timestamp_ms=1025,
            intel_ready_timestamp_ms=1060,
            initial_sync_started_ms=1060,
            initial_world_timestamp_ms=2600,
            spawn=spawn,
            command_ready_timestamp_ms=1115,
        )
        assert probe._page is recorded.page
        assert probe._cdp is recorded.cdp
        return "session-result"

    result = probe_runtime.execute_live_probe_bootstrap(
        probe,
        initial_sync_timeout_ms=9000,
        run_ready_session=_run_ready_session,
    )

    assert result == "session-result"
    assert probe._start_timestamp_ms == 1000
    assert probe.cleanup_calls == 1
    assert probe._page is None
    assert probe._cdp is None
    assert probe.console_calls == 1
    assert probe.handler_calls == 1
    assert probe.navigate_calls == 1
    assert probe.ready_calls == 1
    assert probe.intel_calls == 1
    assert advance_calls == 1
    assert recorded.browser_type.launches == [False]


def test_execute_live_probe_bootstrap_raises_without_playwright() -> None:
    probe = _ProbeHarness()
    core_hooks.sync_playwright = None

    with pytest.raises(PlaywrightNotInstalledError, match=r"Playwright is not installed\."):
        probe_runtime.execute_live_probe_bootstrap(
            probe,
            initial_sync_timeout_ms=9000,
            run_ready_session=lambda context: "unused",
        )
