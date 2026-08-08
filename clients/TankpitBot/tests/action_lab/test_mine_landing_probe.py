"""Tests for the mine-landing (teleport-onto-mine) probe."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_list
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.mine_landing_probe import (
    MINE_HIT_COST,
    MineLandingAttemptDict,
    MineLandingProbe,
    MineLandingProbeSessionDict,
    encode_mine_landing_probe_session,
    format_mine_landing_probe_summary,
    run_mine_landing_probe,
)
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state
from tankpit_bot.state.types import (
    MineStateDict,
    make_mine_state,
    make_self_state,
    make_viewport_state,
)

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


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


def _enemy_mine(x: int, y: int, *, team: int = 3) -> MineStateDict:
    return make_mine_state(x=x, y=y, mine_type=0, tank_id=-1, team=team)


def _install_noop_drain() -> None:
    def _drain(provider: BufferedMessageSourceProtocol, ws: WorldService) -> int:
        del provider
        return 0

    action_hooks.drain_buffered_messages = _drain


class _MineHarness(MineLandingProbe):
    """Fake-wire harness: commands mutate local state, no browser."""

    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=True)
        self._clock = ReplayClock(1000)
        self._page = ClockAdvancingPage(self._clock)
        self.fuel = 1100
        self.position = (100, 100)
        self.visible_mines: dict[str, MineStateDict] = {}
        self.map_calls = 0
        self.radar_calls = 0
        self.teleports: list[tuple[int, int]] = []
        self.landing_offset = (1, 0)
        self.landing_detonates = False

    def open_map(self) -> bool:
        self.map_calls += 1
        return True

    def use_radar(self) -> bool:
        self.radar_calls += 1
        return True

    def request_inventory(self) -> bool:
        return True

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleports.append((x, y))
        cost_from = self.position
        landed = (x + self.landing_offset[0], y + self.landing_offset[1])
        self.position = landed
        from tankpit_bot.physics.costs import teleport_cost

        self.fuel -= teleport_cost(cost_from[0], cost_from[1], landed[0], landed[1])
        if self.landing_detonates:
            self.fuel -= MINE_HIT_COST
            self.visible_mines.pop(f"{x},{y}", None)
        return True

    def get_world_state(self) -> WorldStateDict:
        world = _make_world(1000, self.position[0], self.position[1], self.fuel)
        return WorldStateDict(
            self_state=world["self_state"],
            tanks=world["tanks"],
            containers=world["containers"],
            mines=self.visible_mines,
            terrain=world["terrain"],
            viewport=world["viewport"],
            scanned_tiles=world["scanned_tiles"],
            timestamp_ms=world["timestamp_ms"],
        )

    def get_self_state(self) -> SelfStateDict | None:
        return make_self_state(
            tank_id=1,
            x=self.position[0],
            y=self.position[1],
            team=2,
            rank=1,
            fuel=self.fuel,
            leaderboard_position=1,
        )


def _harness() -> _MineHarness:
    probe = _MineHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    return probe


def test_nearest_enemy_mine_skips_own_team_and_tried() -> None:
    """Own-team and already-attempted mines never anchor an attempt."""
    probe = _harness()
    probe.visible_mines = {
        "101,100": _enemy_mine(101, 100, team=2),
        "103,100": _enemy_mine(103, 100),
        "110,100": _enemy_mine(110, 100),
    }
    found = probe._nearest_enemy_mine({(103, 100)})
    assert found == _enemy_mine(110, 100)


def test_nearest_enemy_mine_none_when_field_is_clear() -> None:
    probe = _harness()
    probe.visible_mines = {"101,100": _enemy_mine(101, 100, team=2)}
    assert probe._nearest_enemy_mine(set()) is None


def test_nearest_enemy_mine_keeps_the_closer_of_two() -> None:
    """A farther candidate later in the registry never displaces the winner."""
    probe = _harness()
    probe.visible_mines = {
        "103,100": _enemy_mine(103, 100),
        "140,140": _enemy_mine(140, 140),
    }
    assert probe._nearest_enemy_mine(set()) == _enemy_mine(103, 100)


def test_own_team_requires_self_state() -> None:
    class _Blind(_MineHarness):
        def get_self_state(self) -> SelfStateDict | None:
            return None

    probe = _Blind()
    with pytest.raises(ProbeError, match="self state unavailable"):
        probe._own_team()


def test_search_returns_visible_mine_without_spending() -> None:
    probe = _harness()
    probe.visible_mines = {"103,100": _enemy_mine(103, 100)}
    found, scans, hops = probe._search_enemy_mine(set(), 6)
    assert found == _enemy_mine(103, 100)
    assert (scans, hops) == (0, 0)


def test_search_hops_and_scans_until_a_mine_shows() -> None:
    class _RevealingHarness(_MineHarness):
        def use_radar(self) -> bool:
            self.visible_mines = {"97,96": _enemy_mine(97, 96)}
            return super().use_radar()

    probe = _RevealingHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    found, scans, hops = probe._search_enemy_mine(set(), 6)
    assert found == _enemy_mine(97, 96)
    assert (scans, hops) == (1, 1)


def test_search_stops_at_the_scan_budget() -> None:
    probe = _harness()
    found, scans, hops = probe._search_enemy_mine(set(), 2)
    assert found is None
    assert scans == 2
    assert hops >= 2


def test_search_exhausts_every_site_under_a_big_budget() -> None:
    """A dry sweep with budget above the site count ends at the site list."""
    probe = _harness()
    found, scans, hops = probe._search_enemy_mine(set(), 20)
    assert found is None
    assert scans == 16
    assert hops == 16


def test_search_skips_unlanded_sites_without_scanning() -> None:
    """A rejected site teleport preserves its extra, density-sweep style."""

    class _StuckHarness(_MineHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            return True

    probe = _StuckHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    probe.world.map_fuel_dots = ()
    found, scans, hops = probe._search_enemy_mine(set(), 2)
    # (96, 96) is within landing tolerance of the (100, 100) start, so
    # exactly one site scans; every other rejected teleport is skipped.
    assert found is None
    assert scans == 1
    assert hops == 16


def test_attempt_records_a_displaced_landing_with_no_bill() -> None:
    """The live 2026-07-28 verdict shape: displaced off, zero extra loss."""
    probe = _harness()
    probe.visible_mines = {"110,100": _enemy_mine(110, 100)}
    attempt = probe._attempt_mine_landing(probe.visible_mines["110,100"])
    assert attempt["landed_on_mine"] is False
    assert (attempt["landed_x"], attempt["landed_y"]) == (111, 100)
    assert attempt["extra_loss"] == 0
    assert attempt["mine_survived"] is True
    assert attempt["landing_teleport_cost"] == attempt["fuel_before"] - attempt["fuel_after"]


def test_attempt_records_a_detonation_bill() -> None:
    """A hypothetical detonating landing books the 45 and the removal."""
    probe = _harness()
    probe.landing_offset = (0, 0)
    probe.landing_detonates = True
    probe.visible_mines = {"110,100": _enemy_mine(110, 100)}
    attempt = probe._attempt_mine_landing(probe.visible_mines["110,100"])
    assert attempt["landed_on_mine"] is True
    assert attempt["extra_loss"] == MINE_HIT_COST
    assert attempt["mine_survived"] is False


def test_execute_probe_rejects_bad_budgets() -> None:
    probe = _MineHarness()
    with pytest.raises(ProbeError, match="max_attempts must be positive"):
        probe.execute_mine_landing_probe(max_attempts=0, max_extras=6, initial_sync_timeout_ms=1000)
    with pytest.raises(ProbeError, match="max_extras must be positive"):
        probe.execute_mine_landing_probe(max_attempts=3, max_extras=0, initial_sync_timeout_ms=1000)


def _attempt(
    *,
    landed_on_mine: bool,
    extra_loss: int,
    mine_survived: bool,
) -> MineLandingAttemptDict:
    return MineLandingAttemptDict(
        mine_x=131,
        mine_y=124,
        mine_team=1,
        own_team=2,
        start_x=131,
        start_y=126,
        landed_x=132,
        landed_y=124,
        landed_on_mine=landed_on_mine,
        fuel_before=1100,
        fuel_after=1087,
        landing_teleport_cost=13,
        extra_loss=extra_loss,
        mine_survived=mine_survived,
    )


def _session() -> MineLandingProbeSessionDict:
    return MineLandingProbeSessionDict(
        session_id="mine-landing-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="mine_landing_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
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
        max_attempts=3,
        max_extras=6,
        search_scans=1,
        search_hops=1,
        attempts=[
            _attempt(landed_on_mine=False, extra_loss=0, mine_survived=True),
            _attempt(landed_on_mine=False, extra_loss=0, mine_survived=True),
        ],
        detonations=0,
        coexists=0,
        displaced_off=2,
        extras_before=24,
        extras_enabled_at_start=True,
        toggles_sent=0,
        extras_after=23,
        fuel_before=1100,
        fuel_after=824,
    )


def test_encode_and_summary() -> None:
    session = _session()
    encoded = encode_mine_landing_probe_session(session)
    assert encoded["displaced_off"] == 2
    attempts = narrow_json_to_list(encoded["attempts"])
    assert len(attempts) == 2
    first = narrow_json_to_dict(attempts[0])
    assert first["landed_on_mine"] is False
    assert first["extra_loss"] == 0
    assert format_mine_landing_probe_summary(session) == (
        "Mine-landing probe complete: attempts=2/3 detonations=0 coexists=0 "
        "displaced_off=2 scans=1 hops=1 extras 24->23 fuel 1100->824"
    )


class _MineModuleProtocol(Protocol):
    MineLandingProbe: type[MineLandingProbe]


_mine_module_import = __import__(
    "tankpit_bot.action_lab.mine_landing_probe",
    fromlist=["mine_landing_probe"],
)
mine_module: _MineModuleProtocol = _mine_module_import


class _FakeMineLandingProbe(MineLandingProbe):
    def execute_mine_landing_probe(
        self,
        *,
        max_attempts: int,
        max_extras: int,
        initial_sync_timeout_ms: int,
    ) -> MineLandingProbeSessionDict:
        session = _session()
        session["max_attempts"] = max_attempts
        session["max_extras"] = max_extras
        session["initial_sync_timeout_ms"] = initial_sync_timeout_ms
        session["capture_session_path"] = ""
        return session


def test_run_mine_landing_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    original_class = mine_module.MineLandingProbe
    mine_module.MineLandingProbe = _FakeMineLandingProbe
    try:
        session = run_mine_landing_probe(
            "https://tankpit.com/play",
            "mine_landing_probe.json",
            max_attempts=2,
            max_extras=4,
        )
    finally:
        mine_module.MineLandingProbe = original_class

    written = fake_fs.read_text(Path("mine_landing_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["capture_session_path"] == "mine_landing_probe.capture_session.json"
    assert decoded["max_attempts"] == 2
    assert session["displaced_off"] == 2


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, MineLandingProbe):
    def __init__(self, *, mines_to_serve: int) -> None:
        MineLandingProbe.__init__(
            self, "https://tankpit.com/play", headless=False, prefer_account=True
        )
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 1100)
        self.phases: list[str] = []
        self.mines_to_serve = mines_to_serve
        self.attempt_script: list[MineLandingAttemptDict] = []

    def _current_fuel(self) -> tuple[int, int, int]:
        self.phases.append("fuel")
        return 1100, 100, 100

    def _ensure_extras_enabled(self) -> tuple[int, bool, int]:
        self.phases.append("enable")
        return 24, True, 0

    def _search_enemy_mine(
        self,
        tried: set[tuple[int, int]],
        scans_left: int,
    ) -> tuple[MineStateDict | None, int, int]:
        self.phases.append(f"search:{scans_left}")
        if self.mines_to_serve <= 0:
            return None, 1, 2
        self.mines_to_serve -= 1
        return _enemy_mine(131 + len(tried), 124), 0, 1

    def _attempt_mine_landing(self, mine: MineStateDict) -> MineLandingAttemptDict:
        self.phases.append(f"attempt:{mine['x']}")
        if self.attempt_script:
            return self.attempt_script.pop(0)
        return _attempt(landed_on_mine=False, extra_loss=0, mine_survived=True)

    def _restore_extras_state(self, was_enabled: bool) -> int:
        self.phases.append(f"restore:{was_enabled}")
        return 0

    def _read_extras(self) -> tuple[int, bool]:
        self.phases.append("read")
        return 23, True

    def _quit_to_lobby(self) -> None:
        self.phases.append("quit")


def _run_execute_harness(
    probe: _ExecuteHarness,
    *,
    max_attempts: int,
) -> MineLandingProbeSessionDict:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    original_sync_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = recorded.sync_playwright_factory

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
                fuel=1100,
                leaderboard_position=1,
            ),
        )

    action_hooks.wait_for_initial_self_state = _wait_initial
    try:
        return probe.execute_mine_landing_probe(
            max_attempts=max_attempts,
            max_extras=6,
            initial_sync_timeout_ms=10000,
        )
    finally:
        core_hooks.sync_playwright = original_sync_playwright


def test_execute_probe_fills_the_attempt_budget_and_tallies() -> None:
    probe = _ExecuteHarness(mines_to_serve=5)
    session = _run_execute_harness(probe, max_attempts=2)

    assert probe.phases == [
        "fuel",
        "enable",
        "search:6",
        "attempt:131",
        "search:6",
        "attempt:132",
        "restore:True",
        "read",
        "fuel",
        "quit",
    ]
    assert len(session["attempts"]) == 2
    assert session["displaced_off"] == 2
    assert session["detonations"] == 0
    assert session["coexists"] == 0
    assert session["search_hops"] == 2
    assert session["capture_session_path"] == ""


def test_execute_probe_tallies_detonation_and_coexist_verdicts() -> None:
    """The two counter-verdict shapes book into their own tallies."""
    probe = _ExecuteHarness(mines_to_serve=5)
    probe.attempt_script = [
        _attempt(landed_on_mine=True, extra_loss=MINE_HIT_COST, mine_survived=False),
        _attempt(landed_on_mine=True, extra_loss=0, mine_survived=True),
    ]
    session = _run_execute_harness(probe, max_attempts=2)

    assert session["detonations"] == 1
    assert session["coexists"] == 1
    assert session["displaced_off"] == 0


def test_execute_probe_stops_when_no_mine_is_found() -> None:
    probe = _ExecuteHarness(mines_to_serve=1)
    session = _run_execute_harness(probe, max_attempts=3)

    assert probe.phases == [
        "fuel",
        "enable",
        "search:6",
        "attempt:131",
        "search:6",
        "restore:True",
        "read",
        "fuel",
        "quit",
    ]
    assert len(session["attempts"]) == 1
    assert session["search_scans"] == 1
    assert session["search_hops"] == 3
