"""Tests for the shoot+move weave probe.

Same discipline as the cadence probe tests: the burst engine runs
against a stubbed page/world with the ammo ledger played back through
the drain hook; the session loop scripts its two seams; the run
wiring is pinned through the standard save path.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Protocol

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab import _combat_probe_harness as harness
from tests.action_lab._combat_probe_harness import _make_world, stub_initial_sync
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.action_lab._replay_page import ClockAdvancingPage, ReplayClock
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import BufferedMessageSourceProtocol, PageProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.weave_probe import (
    WeaveProbe,
    format_weave_probe_summary,
    run_weave_probe,
)
from tankpit_bot.action_lab.weave_probe_types import (
    WeaveBeatDict,
    WeaveBurstDict,
    WeaveProbeSessionDict,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.state.types import make_tank_state

_ENEMY_ID = 900


class _WeaveModuleProtocol(Protocol):
    """The probe-module attribute the run-wiring test swaps."""

    WeaveProbe: type[WeaveProbe]


_weave_module_import = __import__(
    "tankpit_bot.action_lab.weave_probe",
    fromlist=["weave_probe"],
)


weave_swap: _WeaveModuleProtocol = _weave_module_import


@pytest.fixture(autouse=True)
def _restore_weave_hooks() -> Generator[None, None, None]:
    """Restore every hook and module attribute these tests swap."""
    original_drain = action_hooks.drain_buffered_messages
    original_get_time = action_hooks.get_current_time_ms
    original_wait_initial = action_hooks.wait_for_initial_self_state
    original_sync_playwright = core_hooks.sync_playwright
    original_probe_class = weave_swap.WeaveProbe
    yield
    action_hooks.drain_buffered_messages = original_drain
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.wait_for_initial_self_state = original_wait_initial
    core_hooks.sync_playwright = original_sync_playwright
    weave_swap.WeaveProbe = original_probe_class


def _install_clock() -> None:
    """Install a deterministic advancing millisecond clock."""
    ticks = {"now": 1000}

    def _now() -> int:
        ticks["now"] += 10
        return ticks["now"]

    action_hooks.get_current_time_ms = _now


class _WeaveHarness(WeaveProbe):
    """Weave probe with page, world, terrain, and dispatch stubbed."""

    def __init__(self, *, fuel: int = 1000) -> None:
        """Seed a spawned tank at (100,100) beside enemy 900."""
        ws = WorldService()
        ws.terrain_map = InMemoryTerrainMap()
        super().__init__(
            "https://tankpit.com/play",
            headless=True,
            prefer_account=False,
            world=ws,
        )
        self._world_state = _make_world(1000, 100, 100, fuel)
        self._world_state["tanks"][str(_ENEMY_ID)] = make_tank_state(
            tank_id=_ENEMY_ID,
            x=101,
            y=100,
            team=3,
            rank=1,
            name="orange-1",
            is_self=False,
            is_bot=True,
            damage_state=3,
            timestamp_ms=1000,
            last_wire_seen_ms=1000,
            last_position_update_ms=1000,
            last_viewport_observation_ms=1000,
        )
        self._fake_page = ClockAdvancingPage(ReplayClock(1000))
        self.shoot_calls: list[tuple[int, int, int]] = []
        self.move_calls: list[tuple[int, int]] = []
        self.inventory_requests = 0

    def _require_page(self) -> PageProtocol:
        """Return the clock-advancing fake page."""
        return self._fake_page

    def get_world_state(self) -> WorldStateDict:
        """Return the seeded world."""
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        """Return the seeded self state."""
        return self._world_state["self_state"]

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        """Record a shot instead of dispatching it."""
        self.shoot_calls.append((x, y, target_id))
        return True

    def move_to(self, x: int, y: int) -> bool:
        """Record a move instead of dispatching it."""
        self.move_calls.append((x, y))
        return True

    def request_inventory(self) -> bool:
        """Record the snapshot request instead of dispatching it."""
        self.inventory_requests += 1
        return True


def _seed_ammo(probe: WeaveProbe, dual: int, homing: int) -> None:
    """Write the ammo counts the next drained 0x49 would carry."""
    probe.world.inventory_state["dual_shots"]["count"] = dual
    probe.world.inventory_state["homing_shots"]["count"] = homing


def _enemy(tank_id: int = _ENEMY_ID) -> EnemyThreatDict:
    """Build the acquired-adjacent threat for the seeded enemy."""
    return make_enemy_threat(
        tank_id=tank_id,
        x=101,
        y=100,
        distance=1,
        damage_state=3,
        rank=1,
        team=3,
        name="orange-1",
        is_bot=True,
        timestamp_ms=1000,
        last_wire_seen_ms=1000,
        last_position_update_ms=1000,
    )


def test_weave_burst_alternates_and_books_the_ledger() -> None:
    """Even beats also move, weaving between home and the neighbor."""
    _install_clock()
    probe = _WeaveHarness()

    def _drain(source: BufferedMessageSourceProtocol, ws: WorldService, /) -> int:
        _ = (source, ws)
        _seed_ammo(probe, 40 - len(probe.shoot_calls), 20)
        return 1

    action_hooks.drain_buffered_messages = _drain

    burst = probe._weave_burst(_enemy(), 4)

    if burst is None:
        raise AssertionError("expected a completed burst")
    assert burst["shots_dispatched"] == 4
    assert burst["moves_dispatched"] == 2
    assert [b["moved"] for b in burst["beats"]] == [False, True, False, True]
    # The weave leaves home on the first move and returns on the second.
    assert probe.move_calls == [(100, 101), (100, 100)]
    assert [(b["move_x"], b["move_y"]) for b in burst["beats"]] == [
        (-1, -1),
        (100, 101),
        (-1, -1),
        (100, 100),
    ]
    assert burst["served_hits"] == 4
    assert burst["target_killed"] is False


def test_weave_burst_ends_when_the_target_dies() -> None:
    """A vanished registry entry ends the burst as a mid-burst kill."""
    _install_clock()
    probe = _WeaveHarness()

    def _drain(source: BufferedMessageSourceProtocol, ws: WorldService, /) -> int:
        _ = (source, ws)
        _seed_ammo(probe, 40 - len(probe.shoot_calls), 20)
        if len(probe.shoot_calls) >= 3:
            probe._world_state["tanks"].pop(str(_ENEMY_ID), None)
        return 1

    action_hooks.drain_buffered_messages = _drain

    burst = probe._weave_burst(_enemy(), 8)

    if burst is None:
        raise AssertionError("expected a completed burst")
    assert burst["shots_dispatched"] == 3
    assert burst["target_killed"] is True


def test_a_boxed_in_tank_declines_the_burst() -> None:
    """No walkable neighbor means no weave — the burst is refused."""
    _install_clock()
    probe = _WeaveHarness()
    probe.world.terrain_map = InMemoryTerrainMap.from_passable_set(set())

    def _drain(source: BufferedMessageSourceProtocol, ws: WorldService, /) -> int:
        _ = (source, ws)
        return 1

    action_hooks.drain_buffered_messages = _drain

    assert probe._weave_burst(_enemy(), 4) is None
    assert probe.shoot_calls == []


def test_a_missing_terrain_map_declines_the_burst() -> None:
    """Without terrain the weave cannot vet tiles, so it refuses."""
    _install_clock()
    probe = _WeaveHarness()
    probe.world.terrain_map = None

    def _drain(source: BufferedMessageSourceProtocol, ws: WorldService, /) -> int:
        _ = (source, ws)
        return 1

    action_hooks.drain_buffered_messages = _drain

    assert probe._weave_burst(_enemy(), 4) is None


def _burst(*, killed: bool = False) -> WeaveBurstDict:
    """Build a canned burst for the session-level script."""
    return WeaveBurstDict(
        target_id=_ENEMY_ID,
        target_name="orange-1",
        beats=[
            WeaveBeatDict(
                beat_number=1,
                dispatched_ms=1500,
                target_x=101,
                target_y=100,
                moved=False,
                move_x=-1,
                move_y=-1,
            )
        ],
        shots_dispatched=8,
        moves_dispatched=4,
        dual_before=40,
        dual_after=32,
        homing_before=20,
        homing_after=20,
        fuel_before=1000,
        fuel_after=900,
        served_hits=8,
        target_killed=killed,
    )


class _ExecuteWeaveHarness(
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
    WeaveProbe,
):
    """Probe whose acquisition and burst steps are scripted."""

    def __init__(self) -> None:
        """Install the bootstrap stubs and an empty script."""
        WeaveProbe.__init__(
            self,
            "https://tankpit.com/play",
            headless=False,
            prefer_account=True,
        )
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.acquisitions: list[EnemyThreatDict | None] = []
        self.burst_results: list[WeaveBurstDict | None] = []
        self.fuel_after_burst: int | None = None
        self._call_count = 0
        self._burst_count = 0

    def _acquire_adjacent_enemy(
        self,
        *,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        excluded_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        """Advance the acquisition script."""
        _ = (acquisition_timeout_ms, teleport_timeout_ms, excluded_ids)
        result = self.acquisitions[self._call_count]
        self._call_count += 1
        return result

    def _weave_burst(
        self,
        enemy: EnemyThreatDict,
        beats_per_burst: int,
    ) -> WeaveBurstDict | None:
        """Return the next scripted burst, optionally draining the tank."""
        _ = (enemy, beats_per_burst)
        result = self.burst_results[self._burst_count]
        self._burst_count += 1
        if self.fuel_after_burst is not None:
            self_state = self._world_state["self_state"]
            if self_state is not None:
                self_state["fuel"] = self.fuel_after_burst
        return result


def _execute(probe: _ExecuteWeaveHarness, *, bursts: int = 1) -> WeaveProbeSessionDict:
    """Run the session with standard bounds and the bootstrap stubbed."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    recorded = RecordedChromiumSession.from_capture_path(probe, harness._CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    stub_initial_sync()
    return probe.execute_weave_probe(
        beats_per_burst=8,
        burst_count=bursts,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=5000,
        teleport_timeout_ms=10000,
    )


def test_execute_rejects_non_positive_bounds() -> None:
    """Zero beats or bursts is a caller bug, not a quiet no-op."""
    probe = _ExecuteWeaveHarness()
    with pytest.raises(ValueError, match="must be positive"):
        probe.execute_weave_probe(
            beats_per_burst=0,
            burst_count=1,
            initial_sync_timeout_ms=10000,
            acquisition_timeout_ms=5000,
            teleport_timeout_ms=10000,
        )


def test_execute_collects_the_scripted_bursts() -> None:
    """Each burst acquires a target and lands in the session."""
    probe = _ExecuteWeaveHarness()
    probe.acquisitions = [_enemy(), _enemy()]
    probe.burst_results = [_burst(), _burst(killed=True)]

    session = _execute(probe, bursts=2)

    assert len(session["bursts"]) == 2
    assert session["beats_per_burst"] == 8
    assert session["capture_session_path"] == ""


def test_a_declined_burst_is_not_recorded() -> None:
    """A boxed-in ``None`` burst leaves no session entry."""
    probe = _ExecuteWeaveHarness()
    probe.acquisitions = [_enemy()]
    probe.burst_results = [None]

    session = _execute(probe)

    assert session["bursts"] == []


def test_failed_acquisitions_skip_the_burst() -> None:
    """Three misses exhaust the burst's acquisition budget."""
    probe = _ExecuteWeaveHarness()
    probe.acquisitions = [None, None, None]

    session = _execute(probe)

    assert session["bursts"] == []


def test_the_fuel_floor_stops_new_bursts() -> None:
    """Once fuel cannot absorb return fire, no new burst opens."""
    probe = _ExecuteWeaveHarness()
    probe.acquisitions = [_enemy()]
    probe.burst_results = [_burst()]
    probe.fuel_after_burst = 300

    session = _execute(probe, bursts=3)

    assert len(session["bursts"]) == 1


def test_summary_names_the_split_and_flags_kills() -> None:
    """The verdict line shows shoot-only vs shoot+move arithmetic."""
    session = WeaveProbeSessionDict(
        session_id="s",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        beats_per_burst=8,
        capture_session_path="",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 300,
            "intel_ready_timestamp_ms": 350,
            "initial_sync_started_ms": 400,
            "initial_world_timestamp_ms": 450,
            "command_ready_timestamp_ms": 460,
            "first_attempt_started_ms": 500,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 10,
            "command_ready_to_first_attempt_ms": 40,
        },
        bursts=[_burst(killed=True)],
    )

    text = format_weave_probe_summary(session)

    assert "orange-1: 8 served of 8 shots (4 shoot-only + 4 shoot+move) KILLED" in text


def test_summary_reports_an_empty_session_honestly() -> None:
    """No bursts renders as exactly that."""
    session = WeaveProbeSessionDict(
        session_id="s",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        beats_per_burst=8,
        capture_session_path="",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 300,
            "intel_ready_timestamp_ms": 350,
            "initial_sync_started_ms": 400,
            "initial_world_timestamp_ms": 450,
            "command_ready_timestamp_ms": 460,
            "first_attempt_started_ms": 500,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 10,
            "command_ready_to_first_attempt_ms": 40,
        },
        bursts=[],
    )

    assert "no bursts completed" in format_weave_probe_summary(session)


class _FakeWeaveProbe(WeaveProbe):
    """Probe whose whole session is canned, for the run/save wiring."""

    def execute_weave_probe(
        self,
        *,
        beats_per_burst: int,
        burst_count: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
    ) -> WeaveProbeSessionDict:
        """Echo the bounds it was handed back in a canned session."""
        _ = (burst_count, acquisition_timeout_ms, teleport_timeout_ms)
        return WeaveProbeSessionDict(
            session_id="weave-session",
            start_timestamp_ms=1,
            end_timestamp_ms=2,
            base_url="https://tankpit.com/play",
            spawn_x=100,
            spawn_y=100,
            beats_per_burst=beats_per_burst,
            capture_session_path="",
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            startup_timing={
                "game_ready_timestamp_ms": 300,
                "intel_ready_timestamp_ms": 350,
                "initial_sync_started_ms": 400,
                "initial_world_timestamp_ms": 450,
                "command_ready_timestamp_ms": 460,
                "first_attempt_started_ms": 500,
                "game_ready_to_intel_ready_ms": 50,
                "intel_ready_to_initial_world_ms": 100,
                "initial_world_to_command_ready_ms": 10,
                "command_ready_to_first_attempt_ms": 40,
            },
            bursts=[_burst()],
        )


def test_run_weave_probe_writes_the_session_json(fake_fs: FakeFileSystem) -> None:
    """The run wiring threads the bounds through and saves the payload."""
    weave_swap.WeaveProbe = _FakeWeaveProbe

    session = run_weave_probe(
        "https://tankpit.com/play",
        "weave_probe.json",
        beats_per_burst=6,
        burst_count=2,
    )

    written = fake_fs.read_text(Path("weave_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["beats_per_burst"] == 6
    assert decoded["capture_session_path"] == "weave_probe.capture_session.json"
    assert session["beats_per_burst"] == 6
