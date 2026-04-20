"""Tests for shared live-probe runtime bootstrap helpers."""

from __future__ import annotations

import types
from collections.abc import Callable, Generator

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeProtocol,
    CDPSessionProtocol,
    KeyboardProtocol,
    PageProtocol,
    PlaywrightProtocol,
    ResponseProtocol,
    SyncPlaywrightContextManagerProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import probe_runtime
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state, make_self_state
from tankpit_bot.types import CapturedMessage


class _Clock:
    def __init__(self, start_ms: int) -> None:
        self._now_ms = start_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class _FakeKeyboard:
    def press(self, key: str, *, delay: float | None = None) -> None:
        _ = (key, delay)

    def type(self, text: str, *, delay: float | None = None) -> None:
        _ = (text, delay)


class _FakePage:
    url = "https://tankpit.com/play"

    def __init__(self) -> None:
        self._keyboard = _FakeKeyboard()

    @property
    def keyboard(self) -> KeyboardProtocol:
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        _ = (url, referer, timeout, wait_until)
        return None

    def wait_for_timeout(self, timeout: float) -> None:
        _ = timeout

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        _ = (expression, timeout)

    def close(
        self,
        *,
        reason: str | None = None,
        run_before_unload: bool | None = None,
    ) -> None:
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> None:
        _ = expression
        return


class _FakeCDP:
    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        _ = (method, params)
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        _ = (event, handler)

    def detach(self) -> None:
        return None


class _FakeContext:
    def __init__(self, page: PageProtocol, cdp: CDPSessionProtocol) -> None:
        self._page = page
        self._cdp = cdp

    def new_page(self) -> PageProtocol:
        return self._page

    def new_cdp_session(self, page: PageProtocol) -> CDPSessionProtocol:
        assert page is self._page
        return self._cdp

    def close(self, *, reason: str | None = None) -> None:
        _ = reason


class _FakeBrowser:
    def __init__(self, context: BrowserContextProtocol) -> None:
        self._context = context

    def new_context(self) -> BrowserContextProtocol:
        return self._context

    def close(self, *, reason: str | None = None) -> None:
        _ = reason


class _FakeChromium:
    def __init__(self, browser: BrowserProtocol) -> None:
        self._browser = browser
        self.last_headless: bool | None = None

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
    ) -> BrowserProtocol:
        _ = (slow_mo, timeout)
        self.last_headless = headless
        return self._browser


class _FakePlaywright:
    def __init__(self, chromium: BrowserTypeProtocol) -> None:
        self.chromium = chromium

    def stop(self) -> None:
        return None


class _FakePlaywrightContextManager:
    def __init__(self, playwright: PlaywrightProtocol) -> None:
        self._playwright = playwright

    def __enter__(self) -> PlaywrightProtocol:
        return self._playwright

    def start(self) -> PlaywrightProtocol:
        return self._playwright

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


class _FakePlaywrightFactory:
    def __init__(self, manager: SyncPlaywrightContextManagerProtocol) -> None:
        self._manager = manager

    def __call__(self) -> SyncPlaywrightContextManagerProtocol:
        return self._manager


class _ProbeHarness:
    def __init__(self, clock: _Clock | None = None) -> None:
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
    original_wait_initial = action_session.wait_for_initial_self_state
    original_advance = action_session.advance_startup_state
    original_sync_playwright = core_hooks.sync_playwright
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_session.wait_for_initial_self_state = original_wait_initial
    action_session.advance_startup_state = original_advance
    core_hooks.sync_playwright = original_sync_playwright


def test_initialize_live_probe_session_resets_runtime_state() -> None:
    clock = _Clock(500)
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
    page = _FakePage()
    cdp = _FakeCDP()
    context = _FakeContext(page, cdp)
    browser = _FakeBrowser(context)
    chromium = _FakeChromium(browser)
    playwright: PlaywrightProtocol = _FakePlaywright(chromium)

    (
        launched_browser,
        launched_context,
        launched_page,
        launched_cdp,
    ) = probe_runtime.launch_probe_browser(playwright, headless=True)

    assert launched_browser is browser
    assert launched_context is context
    assert launched_page is page
    assert launched_cdp is cdp
    assert chromium.last_headless is True


def test_prepare_live_probe_runtime_sets_handles_and_records_timing() -> None:
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    page = _FakePage()
    cdp = _FakeCDP()
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
    clock = _Clock(2000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()
    page = _FakePage()
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
    setattr(action_session, wait_attr, _wait_initial)
    setattr(action_session, advance_attr, _advance)

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
    probe._page = _FakePage()
    probe._cdp = _FakeCDP()

    probe_runtime.clear_live_probe_runtime(probe)

    assert probe._page is None
    assert probe._cdp is None


def test_execute_live_probe_bootstrap_runs_ready_callback_and_cleans_up() -> None:
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    probe.ready_advance_ms = 25
    probe.intel_advance_ms = 35
    page = _FakePage()
    cdp = _FakeCDP()
    context = _FakeContext(page, cdp)
    browser = _FakeBrowser(context)
    chromium = _FakeChromium(browser)
    manager = _FakePlaywrightContextManager(_FakePlaywright(chromium))
    core_hooks.sync_playwright = _FakePlaywrightFactory(manager)
    spawn = _spawn()

    def _wait_initial(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        assert page_arg is page
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
    setattr(action_session, wait_initial_name, _wait_initial)
    setattr(action_session, advance_name, _advance)

    def _run_ready_session(context_dict: probe_runtime.ProbeCommandReadyContextDict) -> str:
        assert context_dict == probe_runtime.ProbeCommandReadyContextDict(
            game_ready_timestamp_ms=1025,
            intel_ready_timestamp_ms=1060,
            initial_sync_started_ms=1060,
            initial_world_timestamp_ms=2600,
            spawn=spawn,
            command_ready_timestamp_ms=1115,
        )
        assert probe._page is page
        assert probe._cdp is cdp
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
    assert chromium.last_headless is False


def test_execute_live_probe_bootstrap_raises_without_playwright() -> None:
    probe = _ProbeHarness()
    core_hooks.sync_playwright = None

    with pytest.raises(PlaywrightNotInstalledError, match=r"Playwright is not installed\."):
        probe_runtime.execute_live_probe_bootstrap(
            probe,
            initial_sync_timeout_ms=9000,
            run_ready_session=lambda context: "unused",
        )
