"""Tests for the respawn-watch probe phases and runner."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, Protocol

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from tests.conftest import FakeFileSystem

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportProbeSessionDict,
    decode_enemy_teleport_probe_session,
)
from tankpit_bot.action_lab.respawn_watch import RespawnWatchProbe, run_respawn_watch_probe
from tankpit_bot.bot.ai.world_types import (
    EnemyThreatDict,
    make_enemy_threat,
)
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state
from tankpit_bot.state.types import make_self_state, make_tank_state, make_viewport_state


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
        viewport=make_viewport_state(left=0, top=0, width=16, height=16),
        scanned_tiles=world["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def _enemy(*, tank_id: int = 50, x: int = 120, y: int = 130) -> EnemyThreatDict:
    return make_enemy_threat(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=1,
        damage_state=1,
        rank=0,
        team=1,
        name=f"purple-{tank_id}",
        is_bot=False,
        timestamp_ms=1000,
    )


class _WatchHarness(RespawnWatchProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=False)
        self._self_state: SelfStateDict | None = make_self_state(
            tank_id=1,
            x=121,
            y=130,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        self._world_state = _make_world(1000, 121, 130, 900)
        self._clock = ReplayClock(1000)
        self.shoot_calls: list[tuple[int, int, int]] = []
        self.open_map_calls = 0
        self.drains = 0

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._self_state

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        self.shoot_calls.append((x, y, target_id))
        return True

    def open_map(self) -> bool:
        self.open_map_calls += 1
        return True

    def install_enemy(self, *, tank_id: int, x: int, y: int) -> None:
        self._world_state["tanks"][str(tank_id)] = make_tank_state(
            tank_id=tank_id,
            x=x,
            y=y,
            team=1,
            rank=0,
            damage_state=1,
            name=f"purple-{tank_id}",
            is_bot=False,
            is_self=False,
            timestamp_ms=self._clock.now_ms,
        )

    def remove_enemy(self, tank_id: int) -> None:
        self._world_state["tanks"].pop(str(tank_id), None)


def _install_counting_drain(harness: _WatchHarness) -> None:
    def _drain(provider: BufferedMessageSourceProtocol, ws: WorldService) -> int:
        del provider
        harness.drains += 1
        return 0

    action_hooks.drain_buffered_messages = _drain


def test_engage_fires_at_current_position_until_target_vanishes() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_counting_drain(probe)
    probe.engage_ms = 30000
    probe.shot_interval_ms = 2000
    probe.install_enemy(tank_id=50, x=120, y=130)
    waits = [0]

    def _on_wait() -> None:
        waits[0] += 1
        if waits[0] == 1:
            probe.install_enemy(tank_id=50, x=119, y=130)
        elif waits[0] == 2:
            probe.remove_enemy(50)

    page = ClockAdvancingPage(probe._clock, on_wait=_on_wait)

    vanished = probe._engage_phase(page, _enemy(tank_id=50))
    assert vanished is True
    assert probe.shoot_calls == [(120, 130, 50), (119, 130, 50)]
    assert probe.drains == 3


def test_engage_window_expires_when_target_survives() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_counting_drain(probe)
    probe.engage_ms = 4000
    probe.shot_interval_ms = 2000
    probe.install_enemy(tank_id=50, x=120, y=130)

    def _refresh() -> None:
        probe.install_enemy(tank_id=50, x=120, y=130)

    page = ClockAdvancingPage(probe._clock, on_wait=_refresh)

    vanished = probe._engage_phase(page, _enemy(tank_id=50))
    assert vanished is False
    assert probe.shoot_calls == [(120, 130, 50), (120, 130, 50)]


def test_engage_stops_immediately_without_registry_entry() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_counting_drain(probe)
    page = ClockAdvancingPage(probe._clock)

    vanished = probe._engage_phase(page, _enemy(tank_id=50))
    assert vanished is True
    assert probe.shoot_calls == []


def test_map_poll_phase_sends_periodic_map_opens() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_counting_drain(probe)
    probe.poll_ms = 6000
    probe.poll_interval_ms = 2000
    page = ClockAdvancingPage(probe._clock)

    probe._map_poll_phase(page)
    assert probe.open_map_calls == 3
    assert probe.drains == 3


def test_post_landing_phase_engages_then_polls() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_counting_drain(probe)
    probe.engage_ms = 2000
    probe.shot_interval_ms = 2000
    probe.poll_ms = 2000
    probe.poll_interval_ms = 2000
    probe.install_enemy(tank_id=50, x=120, y=130)
    page = ClockAdvancingPage(probe._clock)

    probe._post_landing_phase(page, _enemy(tank_id=50), 999, 999)
    assert probe.shoot_calls == [(120, 130, 50)]
    assert probe.open_map_calls == 1


class _RespawnModuleProtocol(Protocol):
    RespawnWatchProbe: type[RespawnWatchProbe]


_respawn_module_import = __import__(
    "tankpit_bot.action_lab.respawn_watch",
    fromlist=["respawn_watch"],
)
respawn_module: _RespawnModuleProtocol = _respawn_module_import


class _FakeRespawnWatchProbe(RespawnWatchProbe):
    knob_snapshots: ClassVar[list[tuple[int, int, int, int]]] = []

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
    ) -> None:
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            cdp_service=cdp_service,
            command_service=command_service,
        )

    def execute_probe(
        self,
        *,
        acquisition_strategy: Literal["map_open", "nearest_enemy"],
        max_attempts: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
        heartbeat_interval_ms: int,
    ) -> EnemyTeleportProbeSessionDict:
        _FakeRespawnWatchProbe.knob_snapshots.append(
            (self.engage_ms, self.shot_interval_ms, self.poll_ms, self.poll_interval_ms)
        )
        return EnemyTeleportProbeSessionDict(
            session_id="respawn-session",
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
            heartbeat_interval_ms=heartbeat_interval_ms,
            attempts=[],
        )


def test_run_respawn_watch_probe_sets_knobs_and_writes_json(fake_fs: FakeFileSystem) -> None:
    original_class = respawn_module.RespawnWatchProbe
    _FakeRespawnWatchProbe.knob_snapshots = []
    respawn_module.RespawnWatchProbe = _FakeRespawnWatchProbe
    try:
        session = run_respawn_watch_probe(
            "https://tankpit.com/play",
            "respawn_watch_probe.json",
            max_attempts=2,
            engage_ms=15000,
            shot_interval_ms=1000,
            poll_ms=45000,
            poll_interval_ms=3000,
        )
    finally:
        respawn_module.RespawnWatchProbe = original_class

    assert _FakeRespawnWatchProbe.knob_snapshots == [(15000, 1000, 45000, 3000)]
    written = fake_fs.read_text(Path("respawn_watch_probe.json"))
    decoded = decode_enemy_teleport_probe_session(narrow_json_to_dict(load_json_str(written)))
    assert session == decoded
    assert session["capture_session_path"] == "respawn_watch_probe.capture_session.json"
    assert session["acquisition_strategy"] == "map_open"
    assert session["settle_delay_ms"] == 0
    assert session["heartbeat_interval_ms"] == 0
