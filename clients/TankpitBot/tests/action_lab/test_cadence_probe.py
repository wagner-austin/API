"""Tests for the fire-cadence probe.

The burst engine is driven with the page, world, and dispatch stubbed
(the combat-probe harness discipline): the ammo ledger is played back
through the drain hook exactly as 0x49 snapshots would land, so the
served-shot arithmetic is exercised against the real read path. The
session-level tests script the two seams the production loop composes
(``_acquire_adjacent_enemy`` / ``_fire_burst``), and the run wiring is
pinned through the standard save path.
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

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import BufferedMessageSourceProtocol, PageProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.cadence_probe import (
    CadenceProbe,
    _read_fresh_ammo,
    format_cadence_probe_summary,
    run_cadence_probe,
)
from tankpit_bot.action_lab.cadence_probe_types import (
    CadenceBurstDict,
    CadenceProbeSessionDict,
    CadenceShotDict,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.state.types import make_tank_state

_ENEMY_ID = 900


class _CadenceModuleProtocol(Protocol):
    """The probe-module attribute the run-wiring test swaps.

    Typing the surface keeps the stub checked against the real class
    rather than reaching at the module untyped (the combat-probe
    suite's ``_CombatModuleProtocol`` discipline).
    """

    CadenceProbe: type[CadenceProbe]


_cadence_module_import = __import__(
    "tankpit_bot.action_lab.cadence_probe",
    fromlist=["cadence_probe"],
)


cadence_swap: _CadenceModuleProtocol = _cadence_module_import


@pytest.fixture(autouse=True)
def _restore_cadence_hooks() -> Generator[None, None, None]:
    """Restore every hook and module attribute these tests swap."""
    original_drain = action_hooks.drain_buffered_messages
    original_get_time = action_hooks.get_current_time_ms
    original_wait_initial = action_hooks.wait_for_initial_self_state
    original_sync_playwright = core_hooks.sync_playwright
    original_probe_class = cadence_swap.CadenceProbe
    yield
    action_hooks.drain_buffered_messages = original_drain
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.wait_for_initial_self_state = original_wait_initial
    core_hooks.sync_playwright = original_sync_playwright
    cadence_swap.CadenceProbe = original_probe_class


def _install_clock() -> None:
    """Install a deterministic advancing millisecond clock."""
    ticks = {"now": 1000}

    def _now() -> int:
        ticks["now"] += 10
        return ticks["now"]

    action_hooks.get_current_time_ms = _now


class _BurstHarness(CadenceProbe):
    """Cadence probe with page, world, and dispatch stubbed."""

    def __init__(self, *, fuel: int = 1000) -> None:
        """Seed a spawned tank at (100,100) beside enemy 900."""
        ws = WorldService()
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

    def request_inventory(self) -> bool:
        """Record the snapshot request instead of dispatching it."""
        self.inventory_requests += 1
        return True


def _seed_ammo(probe: CadenceProbe, dual: int, homing: int) -> None:
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


def test_read_fresh_ammo_takes_a_server_snapshot() -> None:
    """The read requests a 0x49, drains it in, and returns the counts."""
    _install_clock()
    probe = _BurstHarness()
    drains: list[int] = []

    def _drain(source: BufferedMessageSourceProtocol, ws: WorldService, /) -> int:
        _ = (source, ws)
        drains.append(1)
        _seed_ammo(probe, 40, 20)
        return 1

    action_hooks.drain_buffered_messages = _drain

    assert _read_fresh_ammo(probe) == (40, 20)
    assert probe.inventory_requests == 1
    assert drains == [1]


def test_fire_burst_books_the_ammo_ledger() -> None:
    """Every dispatched shot lands in the ledger with served arithmetic."""
    _install_clock()
    probe = _BurstHarness()

    def _drain(source: BufferedMessageSourceProtocol, ws: WorldService, /) -> int:
        _ = (source, ws)
        _seed_ammo(probe, 40 - len(probe.shoot_calls), 20)
        return 1

    action_hooks.drain_buffered_messages = _drain

    burst = probe._fire_burst(_enemy(), 500, 3)

    assert probe.shoot_calls == [(101, 100, _ENEMY_ID)] * 3
    assert [s["shot_number"] for s in burst["shots"]] == [1, 2, 3]
    assert burst["dispatched"] == 3
    assert (burst["dual_before"], burst["dual_after"]) == (40, 37)
    assert (burst["homing_before"], burst["homing_after"]) == (20, 20)
    assert burst["served_hits"] == 3
    assert burst["spacing_ms"] == 500
    assert burst["target_id"] == _ENEMY_ID
    assert burst["target_killed"] is False
    assert (burst["fuel_before"], burst["fuel_after"]) == (1000, 1000)


def test_fire_burst_ends_when_the_target_dies() -> None:
    """A vanished registry entry ends the burst as a mid-burst kill."""
    _install_clock()
    probe = _BurstHarness()

    def _drain(source: BufferedMessageSourceProtocol, ws: WorldService, /) -> int:
        _ = (source, ws)
        _seed_ammo(probe, 40 - len(probe.shoot_calls), 20)
        if len(probe.shoot_calls) >= 2:
            probe._world_state["tanks"].pop(str(_ENEMY_ID), None)
        return 1

    action_hooks.drain_buffered_messages = _drain

    burst = probe._fire_burst(_enemy(), 250, 6)

    assert burst["dispatched"] == 2
    assert burst["target_killed"] is True
    assert burst["served_hits"] == 2


def _burst(spacing_ms: int, *, killed: bool = False) -> CadenceBurstDict:
    """Build a canned burst for the session-level script."""
    return CadenceBurstDict(
        spacing_ms=spacing_ms,
        target_id=_ENEMY_ID,
        target_name="orange-1",
        shots=[
            CadenceShotDict(
                shot_number=1,
                dispatched_ms=1500,
                target_x=101,
                target_y=100,
            )
        ],
        dispatched=1,
        dual_before=40,
        dual_after=39,
        homing_before=20,
        homing_after=20,
        fuel_before=1000,
        fuel_after=990,
        served_hits=1,
        target_killed=killed,
    )


class _ExecuteCadenceHarness(
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
    CadenceProbe,
):
    """Probe whose acquisition and burst steps are scripted."""

    def __init__(self) -> None:
        """Install the bootstrap stubs and an empty script."""
        CadenceProbe.__init__(
            self,
            "https://tankpit.com/play",
            headless=False,
            prefer_account=True,
        )
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.acquisitions: list[EnemyThreatDict | None] = []
        self.excluded_ids_log: list[frozenset[int]] = []
        self.fuel_after_burst: int | None = None
        self._call_count = 0

    def _acquire_adjacent_enemy(
        self,
        *,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        excluded_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        """Advance the acquisition script, logging exclusions."""
        _ = (acquisition_timeout_ms, teleport_timeout_ms)
        self.excluded_ids_log.append(excluded_ids)
        result = self.acquisitions[self._call_count]
        self._call_count += 1
        return result

    def _fire_burst(
        self,
        enemy: EnemyThreatDict,
        spacing_ms: int,
        shots_per_burst: int,
    ) -> CadenceBurstDict:
        """Return a canned burst, optionally draining the tank after."""
        _ = (enemy, shots_per_burst)
        if self.fuel_after_burst is not None:
            self_state = self._world_state["self_state"]
            if self_state is not None:
                self_state["fuel"] = self.fuel_after_burst
        return _burst(spacing_ms)


def _execute(probe: _ExecuteCadenceHarness, spacings: tuple[int, ...]) -> CadenceProbeSessionDict:
    """Run the session with standard bounds and the bootstrap stubbed."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    recorded = RecordedChromiumSession.from_capture_path(probe, harness._CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    stub_initial_sync()
    return probe.execute_cadence_probe(
        spacings_ms=spacings,
        shots_per_burst=6,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=5000,
        teleport_timeout_ms=10000,
    )


def test_execute_rejects_an_empty_spacing_list() -> None:
    """No spacings is a caller bug, not a quiet no-op."""
    probe = _ExecuteCadenceHarness()
    with pytest.raises(ValueError, match="spacings_ms must be non-empty"):
        _execute(probe, ())


def test_execute_rejects_a_non_positive_burst_budget() -> None:
    """Zero shots per burst is a caller bug, not a quiet no-op."""
    probe = _ExecuteCadenceHarness()
    with pytest.raises(ValueError, match="shots_per_burst positive"):
        probe.execute_cadence_probe(
            spacings_ms=(2000,),
            shots_per_burst=0,
            initial_sync_timeout_ms=10000,
            acquisition_timeout_ms=5000,
            teleport_timeout_ms=10000,
        )


def test_execute_runs_one_burst_per_spacing_excluding_used_targets() -> None:
    """Each spacing bursts a fresh target; used ids are excluded."""
    probe = _ExecuteCadenceHarness()
    probe.acquisitions = [_enemy(), _enemy(901)]

    session = _execute(probe, (2000, 1000))

    assert [b["spacing_ms"] for b in session["bursts"]] == [2000, 1000]
    assert probe.excluded_ids_log == [frozenset(), frozenset({_ENEMY_ID})]
    assert session["shots_per_burst"] == 6
    assert session["capture_session_path"] == ""


def test_a_failed_acquisition_skips_its_spacing_only() -> None:
    """A no-target spacing is skipped; later spacings still run."""
    probe = _ExecuteCadenceHarness()
    probe.acquisitions = [None, _enemy()]

    session = _execute(probe, (1000, 500))

    assert [b["spacing_ms"] for b in session["bursts"]] == [500]
    assert probe.excluded_ids_log == [frozenset(), frozenset()]


def test_a_session_with_no_targets_completes_empty() -> None:
    """Every acquisition failing yields an honest empty session."""
    probe = _ExecuteCadenceHarness()
    probe.acquisitions = [None, None]

    session = _execute(probe, (2000, 1000))

    assert session["bursts"] == []


def test_the_fuel_floor_stops_new_bursts() -> None:
    """Once fuel cannot absorb return fire, no new burst opens.

    2026-07-25 contract ("never leave the tank exposed"): the probe
    stops volunteering for fire instead of bursting itself dead.
    """
    probe = _ExecuteCadenceHarness()
    probe.acquisitions = [_enemy()]
    probe.fuel_after_burst = 300

    session = _execute(probe, (2000, 1000, 500))

    assert [b["spacing_ms"] for b in session["bursts"]] == [2000]
    assert len(probe.excluded_ids_log) == 1


def test_summary_names_every_burst_and_flags_kills() -> None:
    """The summary is the serve-rate table, kill-flagged."""
    session = CadenceProbeSessionDict(
        session_id="s",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        shots_per_burst=6,
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
        bursts=[_burst(1000), _burst(500, killed=True)],
    )

    text = format_cadence_probe_summary(session)

    assert "1000ms: 1/1 served" in text
    assert "500ms: 1/1 served KILLED" in text


def test_summary_reports_an_empty_session_honestly() -> None:
    """No bursts renders as exactly that."""
    session = CadenceProbeSessionDict(
        session_id="s",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        shots_per_burst=6,
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

    assert "no bursts completed" in format_cadence_probe_summary(session)


class _FakeCadenceProbe(CadenceProbe):
    """Probe whose whole session is canned, for the run/save wiring."""

    def execute_cadence_probe(
        self,
        *,
        spacings_ms: tuple[int, ...],
        shots_per_burst: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
    ) -> CadenceProbeSessionDict:
        """Echo the bounds it was handed back in a canned session."""
        _ = (acquisition_timeout_ms, teleport_timeout_ms)
        return CadenceProbeSessionDict(
            session_id="cadence-session",
            start_timestamp_ms=1,
            end_timestamp_ms=2,
            base_url="https://tankpit.com/play",
            spawn_x=100,
            spawn_y=100,
            shots_per_burst=shots_per_burst,
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
            bursts=[_burst(spacings_ms[0])],
        )


def test_run_cadence_probe_writes_the_session_json(fake_fs: FakeFileSystem) -> None:
    """The run wiring threads the bounds through and saves the payload."""
    cadence_swap.CadenceProbe = _FakeCadenceProbe

    session = run_cadence_probe(
        "https://tankpit.com/play",
        "cadence_probe.json",
        spacings_ms=(750,),
        shots_per_burst=4,
    )

    written = fake_fs.read_text(Path("cadence_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["shots_per_burst"] == 4
    assert decoded["capture_session_path"] == "cadence_probe.capture_session.json"
    assert session["shots_per_burst"] == 4
    assert session["bursts"][0]["spacing_ms"] == 750
