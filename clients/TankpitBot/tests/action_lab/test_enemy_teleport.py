"""Tests for live enemy-directed teleport probe helpers."""

from __future__ import annotations

import types
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.json_utils import JSONObject, JSONValue, load_json_str, narrow_json_to_dict
from tests.conftest import FakeFileSystem

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
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.enemy_teleport import (
    EnemyTeleportProbe,
    _enemy_by_id,
    _format_enemy_label,
    _make_terminal_result,
    _require_fresh_enemy_threat,
    format_enemy_teleport_probe_summary,
    run_enemy_teleport_probe,
)
from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportAttemptResultDict,
    EnemyTeleportProbeSessionDict,
    decode_enemy_teleport_probe_session,
)
from tankpit_bot.action_lab.teleport import TeleportProbeError
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.ai.types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.bot.states import make_in_flight_action
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import (
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.types import CapturedMessage, decode_capture_session


class _EnemyTeleportModuleProtocol(Protocol):
    analyze_threats: Callable[[WorldStateDict, SelfStateDict], list[EnemyThreatDict]]
    choose_combat_landing_tile: Callable[
        [WorldStateDict, SelfStateDict, EnemyThreatDict, TerrainMapProtocol | None],
        tuple[int, int],
    ]
    _wait_for_teleport_outcome: _WaitForTeleportOutcomeProtocol
    _require_fresh_enemy_threat: Callable[
        [EnemyTeleportProbe, int, frozenset[int]],
        EnemyThreatDict | None,
    ]
    _enemy_by_id: Callable[[EnemyTeleportProbe, int], EnemyThreatDict | None]


class _WaitForTeleportOutcomeProtocol(Protocol):
    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict: ...


class _EnemyTeleportProbeClassModuleProtocol(Protocol):
    EnemyTeleportProbe: type[EnemyTeleportProbe]


_enemy_module_import = __import__(
    "tankpit_bot.action_lab.enemy_teleport",
    fromlist=["enemy_teleport"],
)
enemy_module: _EnemyTeleportModuleProtocol = _enemy_module_import
enemy_probe_module: _EnemyTeleportProbeClassModuleProtocol = _enemy_module_import


def _enemy(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 130,
    distance: int = 4,
    timestamp_ms: int = 1000,
) -> EnemyThreatDict:
    return make_enemy_threat(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=distance,
        damage_state=0,
        rank=1,
        team=1,
        name=f"enemy-{tank_id}",
        is_bot=False,
        timestamp_ms=timestamp_ms,
    )


def _target() -> TeleportTargetDict:
    return TeleportTargetDict(label="enemy_50_120_130", x=119, y=130)


def _make_world(timestamp_ms: int, x: int, y: int, fuel: int) -> WorldStateDict:
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1,
            x=x,
            y=y,
            team=2,
            rank=1,
            fuel=fuel,
            leaderboard_position=5,
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=ViewportStateDict(left=0, top=0, width=16, height=16),
        scanned_viewports=world["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


class _Clock:
    def __init__(self, start_ms: int) -> None:
        self._now_ms = start_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class _SequencedProvider:
    def __init__(self, worlds: list[WorldStateDict]) -> None:
        self._worlds = worlds
        self._index = 0
        self._cdp_message_buffer: list[str] = []

    def get_world_state(self) -> WorldStateDict:
        return self._worlds[self._index]

    def advance(self) -> None:
        if self._index + 1 < len(self._worlds):
            self._index += 1


class _FakeKeyboard:
    def press(self, key: str, *, delay: float | None = None) -> None:
        _ = (key, delay)

    def type(self, text: str, *, delay: float | None = None) -> None:
        _ = (text, delay)


class _FakePage:
    url = "https://tankpit.com/play"

    def __init__(self, clock: _Clock, provider: _SequencedProvider) -> None:
        self._clock = clock
        self._provider = provider
        self.waits: list[float] = []
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
        self.waits.append(timeout)
        self._clock.advance(int(timeout))
        self._provider.advance()

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

    def evaluate(self, expression: str) -> JSONValue:
        _ = expression
        return None


class _ProbeHarness(EnemyTeleportProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=False)
        self._self_state: SelfStateDict | None = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        self._world_state = _make_world(1000, 100, 100, 900)
        self._fake_page = _FakePage(_Clock(1000), _SequencedProvider([self._world_state]))
        self._cdp = _FakeCDPSession()
        self.map_open_result = True
        self.request_enemy_result = True
        self.teleport_result = True
        self.open_map_calls = 0
        self.request_enemy_calls = 0
        self.teleport_calls: list[tuple[int, int]] = []

    def _require_page(self) -> PageProtocol:
        return self._fake_page

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._self_state

    def open_map(self) -> bool:
        self.open_map_calls += 1
        return self.map_open_result

    def request_nearest_enemy(self) -> bool:
        self.request_enemy_calls += 1
        return self.request_enemy_result

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleport_calls.append((x, y))
        return self.teleport_result


class _FakeCDPSession:
    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        _ = params
        if method == "Runtime.evaluate":
            return {
                "result": {
                    "value": {
                        "phase": "before_teleport",
                        "timestamp_ms": 1000,
                        "client_present": True,
                        "map_visible": True,
                        "client_state": 13,
                        "client_busy": False,
                        "pending_actions": 0,
                        "heartbeat_age_ms": 10,
                        "last_page_client_send_age_ms": 20,
                        "last_bot_send_age_ms": 5,
                        "ws_ready_state": 1,
                        "current_send_label": None,
                        "sent_frame_meta_queue_length": 0,
                    }
                }
            }
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        _ = (event, handler)

    def detach(self) -> None:
        return None


class _FakeContext:
    def __init__(self, page: _FakePage, cdp: _FakeCDPSession) -> None:
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
    def __init__(self, context: _FakeContext) -> None:
        self._context = context

    def new_context(self) -> BrowserContextProtocol:
        return self._context

    def close(self, *, reason: str | None = None) -> None:
        _ = reason


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser) -> None:
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
    def __init__(self, chromium: _FakeChromium) -> None:
        self.chromium: BrowserTypeProtocol = chromium

    def stop(self) -> None:
        return None


class _FakePlaywrightContextManager:
    def __init__(self, playwright: _FakePlaywright) -> None:
        self._playwright: PlaywrightProtocol = playwright

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


class _ExecuteHarness(EnemyTeleportProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=False, prefer_account=True)
        self._world_state = _make_world(900, 100, 100, 900)
        self.cleanup_calls = 0
        self.results: list[EnemyTeleportAttemptResultDict] = []
        self.acquisition_strategies: list[str] = []
        self.excluded_tank_ids: list[frozenset[int]] = []

    def _setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        _ = cdp

    def _setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        _ = cdp

    def _navigate_and_login(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        *,
        tank_name_prefix: str = "TP",
        auto_join_room: bool = True,
    ) -> None:
        _ = (page, cdp, tank_name_prefix, auto_join_room)

    def _wait_for_game_ready(self, page: PageProtocol) -> None:
        _ = page

    def _gather_intel(self, page: PageProtocol, cdp: CDPSessionProtocol) -> None:
        _ = (page, cdp)
        self._magic = "fake-magic"

    def _cleanup(
        self,
        cdp: CDPSessionProtocol,
        page: PageProtocol,
        context: BrowserContextProtocol,
        browser: BrowserProtocol,
    ) -> None:
        _ = (cdp, page, context, browser)
        self.cleanup_calls += 1

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._world_state["self_state"]

    def _probe_single_enemy_attempt(
        self,
        *,
        acquisition_strategy: Literal["map_open", "nearest_enemy"],
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyTeleportAttemptResultDict:
        _ = (acquisition_timeout_ms, teleport_timeout_ms, settle_delay_ms)
        self.acquisition_strategies.append(acquisition_strategy)
        self.excluded_tank_ids.append(excluded_tank_ids)
        return self.results[len(self.acquisition_strategies) - 1]


class _FakeEnemyTeleportProbe(EnemyTeleportProbe):
    def __init__(self, target_url: str, *, headless: bool, prefer_account: bool) -> None:
        super().__init__(target_url, headless=headless, prefer_account=prefer_account)

    def execute_probe(
        self,
        *,
        acquisition_strategy: Literal["map_open", "nearest_enemy"],
        max_attempts: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
    ) -> EnemyTeleportProbeSessionDict:
        return EnemyTeleportProbeSessionDict(
            session_id="enemy-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self._target_url,
            spawn_x=100,
            spawn_y=100,
            acquisition_strategy=acquisition_strategy,
            max_attempts=max_attempts,
            capture_session_path="",
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            startup_timing={
                "game_ready_timestamp_ms": 100,
                "intel_ready_timestamp_ms": 150,
                "initial_sync_started_ms": 200,
                "initial_world_timestamp_ms": 400,
                "command_ready_timestamp_ms": 450,
                "first_attempt_started_ms": 500,
                "game_ready_to_intel_ready_ms": 50,
                "intel_ready_to_initial_world_ms": 250,
                "initial_world_to_command_ready_ms": 50,
                "command_ready_to_first_attempt_ms": 50,
            },
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            attempts=[],
        )

    @property
    def messages(self) -> list[CapturedMessage]:
        return []

    @property
    def magic(self) -> str | None:
        return None


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    original_get_time = action_hooks.get_current_time_ms
    original_wait_sync = action_session.wait_for_world_sync
    original_wait_initial = action_session.wait_for_initial_self_state
    original_require_enemy = enemy_module._require_fresh_enemy_threat
    original_enemy_by_id = enemy_module._enemy_by_id
    original_choose_landing = enemy_module.choose_combat_landing_tile
    original_wait_outcome = enemy_module._wait_for_teleport_outcome
    original_probe_class = enemy_probe_module.EnemyTeleportProbe
    original_sync_playwright = core_hooks.sync_playwright
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_session.wait_for_world_sync = original_wait_sync
    action_session.wait_for_initial_self_state = original_wait_initial
    enemy_module._require_fresh_enemy_threat = original_require_enemy
    enemy_module._enemy_by_id = original_enemy_by_id
    enemy_module.choose_combat_landing_tile = original_choose_landing
    enemy_module._wait_for_teleport_outcome = original_wait_outcome
    enemy_probe_module.EnemyTeleportProbe = original_probe_class
    core_hooks.sync_playwright = original_sync_playwright


def test_require_fresh_enemy_threat_filters_old_entries() -> None:
    probe = _ProbeHarness()
    original_analyze = enemy_module.analyze_threats
    enemy_module.analyze_threats = lambda world, self_state: [
        _enemy(tank_id=1, distance=9, timestamp_ms=900),
        _enemy(tank_id=2, distance=2, timestamp_ms=1500),
    ]
    try:
        result = _require_fresh_enemy_threat(probe, 1000, frozenset())
    finally:
        enemy_module.analyze_threats = original_analyze

    assert result == _enemy(tank_id=2, distance=2, timestamp_ms=1500)


def test_require_fresh_enemy_threat_returns_none_without_self_state() -> None:
    probe = _ProbeHarness()
    probe._self_state = None

    assert _require_fresh_enemy_threat(probe, 1000, frozenset()) is None


def test_require_fresh_enemy_threat_excludes_previously_targeted_enemy_ids() -> None:
    probe = _ProbeHarness()
    original_analyze = enemy_module.analyze_threats
    enemy_module.analyze_threats = lambda world, self_state: [
        _enemy(tank_id=1, distance=1, timestamp_ms=1500),
        _enemy(tank_id=2, distance=2, timestamp_ms=1500),
    ]
    try:
        result = _require_fresh_enemy_threat(probe, 1000, frozenset({1}))
    finally:
        enemy_module.analyze_threats = original_analyze

    assert result == _enemy(tank_id=2, distance=2, timestamp_ms=1500)


def test_enemy_by_id_returns_matching_enemy_and_none_when_missing() -> None:
    probe = _ProbeHarness()
    original_analyze = enemy_module.analyze_threats
    enemy_module.analyze_threats = lambda world, self_state: [
        _enemy(tank_id=11),
        _enemy(tank_id=12),
    ]
    try:
        match = _enemy_by_id(probe, 12)
        missing = _enemy_by_id(probe, 99)
    finally:
        enemy_module.analyze_threats = original_analyze

    assert match == _enemy(tank_id=12)
    assert missing is None


def test_enemy_by_id_returns_none_without_self_state() -> None:
    probe = _ProbeHarness()
    probe._self_state = None

    assert _enemy_by_id(probe, 12) is None


def test_format_enemy_helpers_cover_terminal_result_and_summary() -> None:
    enemy = _enemy()
    target = _target()
    result = _make_terminal_result(
        acquisition_strategy="map_open",
        status="no_landing_tile",
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=None,
        fuel_before=900,
        world_timestamp_before=950,
        completion_timestamp_ms=1200,
        fuel_after=880,
        world_timestamp_after=1100,
        enemy=enemy,
        landing_target=target,
        landed_x=100,
        landed_y=101,
        message_start_index=5,
        message_end_index=9,
    )
    session = EnemyTeleportProbeSessionDict(
        session_id="enemy-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        acquisition_strategy="nearest_enemy",
        max_attempts=6,
        capture_session_path="enemy.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 100,
            "intel_ready_timestamp_ms": 150,
            "initial_sync_started_ms": 200,
            "initial_world_timestamp_ms": 250,
            "command_ready_timestamp_ms": 300,
            "first_attempt_started_ms": 325,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 50,
            "command_ready_to_first_attempt_ms": 25,
        },
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
        attempts=[
            EnemyTeleportAttemptResultDict(**{**result, "status": "landed_adjacent"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "landed_not_adjacent"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "no_enemy"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "no_landing_tile"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "acquisition_timeout"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "teleport_timeout"}),
        ],
    )

    assert _format_enemy_label(enemy) == "enemy_50_120_130"
    assert result["acquisition_elapsed_ms"] is None
    assert format_enemy_teleport_probe_summary(session) == (
        "Enemy teleport probe complete: strategy=nearest_enemy attempts=6 "
        "landed_adjacent=1 landed_not_adjacent=1 no_enemy=1 no_landing_tile=1 "
        "acquisition_timeout=1 teleport_timeout=1 session_to_initial_sync_ms=199 "
        "initial_sync_to_command_ready_ms=100"
    )


def test_send_enemy_acquisition_dispatches_by_strategy() -> None:
    probe = _ProbeHarness()

    assert probe._send_enemy_acquisition("map_open") is True
    assert probe._send_enemy_acquisition("nearest_enemy") is True
    assert probe.open_map_calls == 1
    assert probe.request_enemy_calls == 1


def test_finish_non_teleport_attempt_resets_state_and_settles() -> None:
    probe = _ProbeHarness()
    probe._state_data["state"] = "TELEPORTING"
    probe._state_data["in_flight_action"] = make_in_flight_action("teleport", 119, 130, 1000)
    result = probe._finish_non_teleport_attempt(
        page=probe._fake_page,
        acquisition_strategy="nearest_enemy",
        status="no_enemy",
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=1100,
        fuel_before=900,
        world_timestamp_before=950,
        enemy=None,
        landing_target=None,
        message_start_index=4,
        settle_delay_ms=250,
    )

    assert result["status"] == "no_enemy"
    assert result["message_start_index"] == 4
    assert result["message_end_index"] == 0
    assert probe.get_state() == "IDLE"
    assert probe.get_state_data()["in_flight_action"]["kind"] == "none"
    assert probe._fake_page.waits[-1] == 250.0


def test_probe_single_enemy_attempt_raises_when_acquisition_dispatch_fails() -> None:
    probe = _ProbeHarness()
    probe.request_enemy_result = False

    with pytest.raises(TeleportProbeError, match="enemy acquisition command dispatch failed"):
        probe._probe_single_enemy_attempt(
            acquisition_strategy="nearest_enemy",
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=0,
            excluded_tank_ids=frozenset(),
        )


def test_probe_single_enemy_attempt_returns_acquisition_timeout() -> None:
    probe = _ProbeHarness()
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: None

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "acquisition_timeout"
    assert result["teleport_started_ms"] is None
    assert probe.teleport_calls == []


def test_probe_single_enemy_attempt_returns_no_enemy() -> None:
    probe = _ProbeHarness()
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _missing_enemy(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return None

    enemy_module._require_fresh_enemy_threat = _missing_enemy

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "no_enemy"


def test_probe_single_enemy_attempt_returns_no_landing_tile() -> None:
    probe = _ProbeHarness()
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _no_landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain)
        return (-1, -1)

    enemy_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _no_landing

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "no_landing_tile"
    assert result["enemy"] == _enemy()


def test_probe_single_enemy_attempt_raises_when_teleport_dispatch_fails() -> None:
    probe = _ProbeHarness()
    probe.teleport_result = False
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain)
        return (119, 130)

    enemy_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _landing

    with pytest.raises(TeleportProbeError, match="teleport command dispatch failed"):
        probe._probe_single_enemy_attempt(
            acquisition_strategy="nearest_enemy",
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=0,
            excluded_tank_ids=frozenset(),
        )


def test_probe_single_enemy_attempt_records_teleport_timeout() -> None:
    probe = _ProbeHarness()
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain)
        return (119, 130)

    def _timeout_result(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            teleport_cycle_id,
            message_start_index,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="teleport_timeout",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=850,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=False,
            landed_x=100,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    def _enemy_after(
        probe: EnemyTeleportProbe,
        tank_id: int,
    ) -> EnemyThreatDict | None:
        _ = (probe, tank_id)
        return _enemy()

    enemy_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _landing
    enemy_module._wait_for_teleport_outcome = _timeout_result
    enemy_module._enemy_by_id = _enemy_after

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "teleport_timeout"
    assert result["enemy_still_visible"] is True


def test_probe_single_enemy_attempt_settles_after_landed_result() -> None:
    probe = _ProbeHarness()
    probe._self_state = make_self_state(
        tank_id=1,
        x=119,
        y=130,
        team=2,
        rank=1,
        fuel=820,
        leaderboard_position=1,
    )
    probe._world_state = _make_world(1450, 119, 130, 820)
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain)
        return (119, 130)

    def _landed_result(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            teleport_cycle_id,
            message_start_index,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="landed_exact",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=820,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=119,
            landed_y=130,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    def _enemy_after(
        probe: EnemyTeleportProbe,
        tank_id: int,
    ) -> EnemyThreatDict | None:
        _ = (probe, tank_id)
        return _enemy(x=120, y=130)

    enemy_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _landing
    enemy_module._wait_for_teleport_outcome = _landed_result
    enemy_module._enemy_by_id = _enemy_after

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=250,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "landed_adjacent"
    assert probe._fake_page.waits[-1] == 250.0


@pytest.mark.parametrize(
    ("enemy_after", "expected_status"),
    [
        (_enemy(x=120, y=130), "landed_adjacent"),
        (_enemy(x=123, y=130), "landed_not_adjacent"),
    ],
)
def test_probe_single_enemy_attempt_records_landed_outcome(
    enemy_after: EnemyThreatDict,
    expected_status: str,
) -> None:
    probe = _ProbeHarness()
    probe._self_state = make_self_state(
        tank_id=1,
        x=119,
        y=130,
        team=2,
        rank=1,
        fuel=820,
        leaderboard_position=1,
    )
    probe._world_state = _make_world(1450, 119, 130, 820)
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain)
        return (119, 130)

    def _landed_result(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            teleport_cycle_id,
            message_start_index,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="landed_exact",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=820,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=119,
            landed_y=130,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    def _enemy_after(
        probe: EnemyTeleportProbe,
        tank_id: int,
    ) -> EnemyThreatDict | None:
        _ = (probe, tank_id)
        return enemy_after

    enemy_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _landing
    enemy_module._wait_for_teleport_outcome = _landed_result
    enemy_module._enemy_by_id = _enemy_after

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == expected_status
    assert result["enemy_distance_after"] == (
        abs(119 - enemy_after["x"]) + abs(130 - enemy_after["y"])
    )


def test_execute_probe_raises_for_invalid_max_attempts() -> None:
    probe = _ProbeHarness()

    with pytest.raises(ValueError, match="max_attempts must be positive"):
        probe.execute_probe(
            acquisition_strategy="nearest_enemy",
            max_attempts=0,
            initial_sync_timeout_ms=10000,
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=500,
        )


def test_execute_probe_raises_when_playwright_is_missing() -> None:
    probe = _ProbeHarness()
    original_sync_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            probe.execute_probe(
                acquisition_strategy="nearest_enemy",
                max_attempts=1,
                initial_sync_timeout_ms=10000,
                acquisition_timeout_ms=3000,
                teleport_timeout_ms=10000,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_sync_playwright


def test_execute_probe_collects_attempts() -> None:
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    page = _FakePage(clock, _SequencedProvider([_make_world(900, 100, 100, 900)]))
    cdp = _FakeCDPSession()
    chromium = _FakeChromium(_FakeBrowser(_FakeContext(page, cdp)))
    manager = _FakePlaywrightContextManager(_FakePlaywright(chromium))
    core_hooks.sync_playwright = _FakePlaywrightFactory(manager)
    probe = _ExecuteHarness()
    probe.results = [
        EnemyTeleportAttemptResultDict(
            acquisition_strategy="nearest_enemy",
            status="landed_adjacent",
            acquisition_started_ms=1000,
            acquisition_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            completion_timestamp_ms=1400,
            acquisition_elapsed_ms=100,
            teleport_elapsed_ms=200,
            fuel_before=900,
            fuel_after=820,
            world_timestamp_before=950,
            world_timestamp_after=1450,
            enemy=_enemy(tank_id=50),
            landing_target=_target(),
            landed_signal_received=True,
            landed_x=119,
            landed_y=130,
            enemy_still_visible=True,
            enemy_distance_after=1,
            enemy_x_after=120,
            enemy_y_after=130,
            message_start_index=0,
            message_end_index=1,
        ),
        EnemyTeleportAttemptResultDict(
            acquisition_strategy="nearest_enemy",
            status="landed_adjacent",
            acquisition_started_ms=1100,
            acquisition_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            acquisition_elapsed_ms=100,
            teleport_elapsed_ms=200,
            fuel_before=820,
            fuel_after=760,
            world_timestamp_before=1450,
            world_timestamp_after=1550,
            enemy=_enemy(tank_id=51, x=121, y=130),
            landing_target=TeleportTargetDict(label="enemy_51_121_130", x=120, y=130),
            landed_signal_received=True,
            landed_x=120,
            landed_y=130,
            enemy_still_visible=True,
            enemy_distance_after=1,
            enemy_x_after=121,
            enemy_y_after=130,
            message_start_index=1,
            message_end_index=2,
        ),
    ]

    def _wait_initial(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page, provider, started_ms, timeout_ms)
        return (
            1200,
            make_self_state(
                tank_id=1,
                x=100,
                y=100,
                team=2,
                rank=1,
                fuel=900,
                leaderboard_position=1,
            ),
        )

    action_session.wait_for_initial_self_state = _wait_initial

    session = probe.execute_probe(
        acquisition_strategy="nearest_enemy",
        max_attempts=2,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 2
    assert probe.acquisition_strategies == ["nearest_enemy", "nearest_enemy"]
    assert probe.excluded_tank_ids == [frozenset(), frozenset({50})]
    assert session["startup_timing"]["initial_world_timestamp_ms"] == 1200
    assert session["startup_timing"]["first_attempt_started_ms"] == 1000
    assert probe.cleanup_calls == 1
    assert chromium.last_headless is False


def test_execute_probe_does_not_exclude_when_attempt_has_no_enemy() -> None:
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    page = _FakePage(clock, _SequencedProvider([_make_world(900, 100, 100, 900)]))
    cdp = _FakeCDPSession()
    chromium = _FakeChromium(_FakeBrowser(_FakeContext(page, cdp)))
    manager = _FakePlaywrightContextManager(_FakePlaywright(chromium))
    core_hooks.sync_playwright = _FakePlaywrightFactory(manager)
    probe = _ExecuteHarness()
    probe.results = [
        EnemyTeleportAttemptResultDict(
            acquisition_strategy="map_open",
            status="no_enemy",
            acquisition_started_ms=1000,
            acquisition_sync_timestamp_ms=1100,
            teleport_started_ms=None,
            completion_timestamp_ms=1200,
            acquisition_elapsed_ms=100,
            teleport_elapsed_ms=None,
            fuel_before=900,
            fuel_after=900,
            world_timestamp_before=950,
            world_timestamp_after=1150,
            enemy=None,
            landing_target=None,
            landed_signal_received=False,
            landed_x=100,
            landed_y=100,
            enemy_still_visible=False,
            enemy_distance_after=None,
            enemy_x_after=None,
            enemy_y_after=None,
            message_start_index=0,
            message_end_index=0,
        ),
        EnemyTeleportAttemptResultDict(
            acquisition_strategy="map_open",
            status="landed_adjacent",
            acquisition_started_ms=1300,
            acquisition_sync_timestamp_ms=1400,
            teleport_started_ms=1500,
            completion_timestamp_ms=1700,
            acquisition_elapsed_ms=100,
            teleport_elapsed_ms=200,
            fuel_before=900,
            fuel_after=840,
            world_timestamp_before=1150,
            world_timestamp_after=1650,
            enemy=_enemy(tank_id=60, x=121, y=130),
            landing_target=TeleportTargetDict(label="enemy_60_121_130", x=120, y=130),
            landed_signal_received=True,
            landed_x=120,
            landed_y=130,
            enemy_still_visible=True,
            enemy_distance_after=1,
            enemy_x_after=121,
            enemy_y_after=130,
            message_start_index=0,
            message_end_index=1,
        ),
    ]

    def _wait_initial(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page, provider, started_ms, timeout_ms)
        return (
            1200,
            make_self_state(
                tank_id=1,
                x=100,
                y=100,
                team=2,
                rank=1,
                fuel=900,
                leaderboard_position=1,
            ),
        )

    action_session.wait_for_initial_self_state = _wait_initial

    session = probe.execute_probe(
        acquisition_strategy="map_open",
        max_attempts=2,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 2
    assert probe.excluded_tank_ids == [frozenset(), frozenset()]


def test_run_enemy_teleport_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    enemy_probe_module.EnemyTeleportProbe = _FakeEnemyTeleportProbe
    session = run_enemy_teleport_probe(
        "https://tankpit.com/play",
        "enemy_teleport_probe.json",
        acquisition_strategy="map_open",
        max_attempts=3,
    )

    written = fake_fs.read_text(Path("enemy_teleport_probe.json"))
    decoded = decode_enemy_teleport_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("enemy_teleport_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))

    assert session == decoded
    assert session["capture_session_path"] == "enemy_teleport_probe.capture_session.json"
    assert session["acquisition_strategy"] == "map_open"
    assert capture_decoded["session_id"] == "enemy-session"
