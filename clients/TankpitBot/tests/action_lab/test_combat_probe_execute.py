"""Tests for combat-probe acquisition, session execution, and run wiring.

``_acquire_and_engage`` is driven with the acquisition and teleport
phases stubbed at module scope, so each of its five give-up paths and
both landing outcomes are reached without a browser. The session-level
tests cover ``execute_probe``'s engagement loop and the
``run_combat_probe`` factory-and-save wiring.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab import _combat_probe_harness as harness
from tests.action_lab._combat_probe_harness import (
    _engagement,
    _ExecuteHarness,
    _FakeCombatProbe,
    _make_world,
    _ProbeHarness,
    combat_module,
    require_engagement,
    require_threat,
)
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_page import ClockAdvancingPage, ReplayClock
from tests.action_lab._teleport_attempt_harness import _snapshot
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.combat_probe import (
    _create_combat_probe,
    _current_enemy_by_id,
    run_combat_probe,
)
from tankpit_bot.action_lab.combat_probe_types import CombatEngagementDict
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.teleport_helpers import TeleportProbeError
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterProtocol,
    TeleportPhaseProbeProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_self_state
from tankpit_bot.state.types import make_tank_state

_SnapshotPhase = Literal[
    "before_map_open",
    "before_teleport",
    "after_map_data",
    "landed",
    "timeout",
]

_TeleportStatus = Literal[
    "landed_exact",
    "landed_offset",
    "map_sync_timeout",
    "teleport_timeout",
]


@pytest.fixture(autouse=True)
def _restore_combat_probe_hooks() -> Generator[None, None, None]:
    """Restore every hook and module attribute these tests swap."""
    original_get_time = action_hooks.get_current_time_ms
    original_wait_initial = action_hooks.wait_for_initial_self_state
    original_sync_playwright = core_hooks.sync_playwright
    original_acquisition = combat_module.run_tracked_acquisition_phase
    original_teleport = combat_module.run_tracked_teleport_command
    original_landing = combat_module.choose_combat_landing_tile
    original_terrain = combat_module.get_terrain_map
    original_find_fresh = combat_module._find_fresh_enemy
    original_current = combat_module._current_enemy_by_id
    original_probe_class = combat_module.CombatProbe
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.wait_for_initial_self_state = original_wait_initial
    core_hooks.sync_playwright = original_sync_playwright
    combat_module.run_tracked_acquisition_phase = original_acquisition
    combat_module.run_tracked_teleport_command = original_teleport
    combat_module.choose_combat_landing_tile = original_landing
    combat_module.get_terrain_map = original_terrain
    combat_module._find_fresh_enemy = original_find_fresh
    combat_module._current_enemy_by_id = original_current
    combat_module.CombatProbe = original_probe_class
    reset_world_state()


def _enemy(tank_id: int = 7, *, x: int = 101, y: int = 100) -> EnemyThreatDict:
    """Build an enemy threat at the given tile."""
    return make_enemy_threat(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=abs(x - 100) + abs(y - 100),
        damage_state=3,
        rank=1,
        team=1,
        name=f"enemy-{tank_id}",
        is_bot=True,
        timestamp_ms=1000,
        last_wire_seen_ms=1000,
        last_position_update_ms=1000,
        last_aim_x=0,
        last_aim_y=0,
        last_aim_weapon=0,
        last_aim_ms=0,
    )


def _teleport_result(
    target: TeleportTargetDict,
    status: _TeleportStatus,
) -> TeleportAttemptResultDict:
    """Build a teleport outcome carrying the requested status."""
    return TeleportAttemptResultDict(
        target=target,
        teleport_cycle_id=1,
        status=status,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        completion_timestamp_ms=1300,
        map_sync_elapsed_ms=100,
        teleport_elapsed_ms=100,
        fuel_before=900,
        fuel_after=880,
        world_timestamp_before=900,
        world_timestamp_after=1300,
        landed_signal_received=status.startswith("landed"),
        landed_x=100,
        landed_y=100,
        message_start_index=0,
        message_end_index=0,
        page_snapshots=[],
    )


def _stub_acquisition(sync_ms: int | None) -> None:
    """Make the acquisition phase resolve immediately with ``sync_ms``."""

    def _phase(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        *,
        cdp: CDPSessionProtocol | None,
        send_command: Callable[[], bool],
        command_name: str,
        capture_before_map_open: bool,
        wait_for_sync: bool,
        sync_timeout_ms: int,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
    ) -> tuple[
        int,
        int | None,
        list[TeleportPageSnapshotDict],
        Callable[[_SnapshotPhase], TeleportPageSnapshotDict],
    ]:
        _ = (page, provider, cdp, send_command, command_name)
        _ = (capture_before_map_open, wait_for_sync, sync_timeout_ms)
        _ = (dispatch_failure_error, dispatch_failure_message)
        _ = (unavailable_error, unavailable_message)
        return (1000, sync_ms, [], _snapshot)

    combat_module.run_tracked_acquisition_phase = _phase


def _stub_teleport(status: _TeleportStatus) -> None:
    """Make the teleport phase resolve immediately with ``status``."""

    def _command(
        page: action_session.WaitPageProtocol,
        probe: TeleportPhaseProbeProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle: ActionPhaseCycleDict,
        message_start_index: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[[_SnapshotPhase], TeleportPageSnapshotDict],
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "teleport command dispatch failed",
    ) -> tuple[TeleportAttemptResultDict, int]:
        _ = (page, probe, teleport_cycle, message_start_index, map_open_started_ms)
        _ = (map_sync_timestamp_ms, fuel_before, world_timestamp_before, timeout_ms)
        _ = (page_snapshots, capture_page_snapshot, wait_for_outcome)
        _ = (dispatch_failure_error, dispatch_failure_message)
        return (_teleport_result(target, status), 1000)

    combat_module.run_tracked_teleport_command = _command


def _stub_enemy_lookup(enemy: EnemyThreatDict | None) -> None:
    """Make both enemy lookups resolve to ``enemy``."""

    def _fresh(
        probe: ProbeBase,
        started_ms: int,
        excluded_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_ids)
        return enemy

    def _current(probe: ProbeBase, tank_id: int) -> EnemyThreatDict | None:
        _ = (probe, tank_id)
        return enemy

    combat_module._find_fresh_enemy = _fresh
    combat_module._current_enemy_by_id = _current


def _stub_landing(x: int, y: int) -> None:
    """Make landing selection return a fixed tile over absent terrain."""

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
        now_ms: int,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain, now_ms)
        return (x, y)

    def _terrain() -> TerrainMapProtocol | None:
        return None

    combat_module.choose_combat_landing_tile = _landing
    combat_module.get_terrain_map = _terrain


def _stub_initial_sync() -> None:
    """Resolve the bootstrap's initial world sync at a fixed spawn."""

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
                leaderboard_position=5,
            ),
        )

    action_hooks.wait_for_initial_self_state = _wait_initial


class _ScriptedEngageHarness(_ProbeHarness):
    """Probe recording which targets reached the engagement stage."""

    def __init__(self) -> None:
        """Start with an empty engagement log."""
        super().__init__()
        self.engaged: list[EnemyThreatDict] = []

    def _engage_single_target(
        self,
        target: EnemyThreatDict,
        max_shots: int,
    ) -> CombatEngagementDict:
        """Record the target instead of firing at it."""
        _ = max_shots
        self.engaged.append(target)
        return _engagement(target["tank_id"])


def _acquire(probe: _ProbeHarness) -> CombatEngagementDict | None:
    """Run one acquisition-and-engage cycle with the standard bounds."""
    return probe._acquire_and_engage(
        acquisition_timeout_ms=5000,
        teleport_timeout_ms=10000,
        max_shots=2,
        excluded_ids=frozenset(),
    )


def test_acquire_raises_without_a_cdp_session() -> None:
    """No CDP session means the probe cannot capture, so it refuses."""
    probe = _ProbeHarness()
    probe._cdp = None
    with pytest.raises(TeleportProbeError, match="cdp session is unavailable"):
        _acquire(probe)


def test_acquire_gives_up_when_the_map_never_syncs() -> None:
    """A ``None`` sync timestamp is an acquisition timeout, not a crash."""
    probe = _ProbeHarness()
    _stub_acquisition(None)
    assert _acquire(probe) is None


def test_acquire_gives_up_when_no_enemy_is_fresh() -> None:
    """The map opened but showed nothing worth engaging."""
    probe = _ProbeHarness()
    _stub_acquisition(1100)
    _stub_enemy_lookup(None)
    assert _acquire(probe) is None


def test_acquire_gives_up_without_a_landing_tile() -> None:
    """``(-1, -1)`` is the no-landing sentinel and ends the attempt."""
    probe = _ProbeHarness()
    _stub_acquisition(1100)
    _stub_enemy_lookup(_enemy())
    _stub_landing(-1, -1)
    assert _acquire(probe) is None


def test_acquire_gives_up_on_a_teleport_timeout() -> None:
    """A teleport that never lands cannot be engaged from."""
    probe = _ProbeHarness()
    _stub_acquisition(1100)
    _stub_enemy_lookup(_enemy())
    _stub_landing(100, 100)
    _stub_teleport("teleport_timeout")
    assert _acquire(probe) is None


def test_acquire_gives_up_when_the_enemy_vanishes_after_landing() -> None:
    """Landing beside an enemy that has since left is not an engagement."""
    probe = _ProbeHarness()
    _stub_acquisition(1100)
    _stub_enemy_lookup(_enemy())
    _stub_landing(100, 100)
    _stub_teleport("landed_exact")

    def _gone(probe_arg: ProbeBase, tank_id: int) -> EnemyThreatDict | None:
        _ = (probe_arg, tank_id)
        return None

    combat_module._current_enemy_by_id = _gone
    assert _acquire(probe) is None


def test_acquire_engages_and_warns_when_the_landing_is_not_adjacent() -> None:
    """A non-adjacent landing still engages -- it only warns.

    The server displaces a teleport onto the nearest open ground, so
    ending up short of the target is normal; refusing to engage there
    would throw away the whole approach for nothing.
    """
    probe = _ScriptedEngageHarness()
    _stub_acquisition(1100)
    far = _enemy(x=110, y=110)
    _stub_enemy_lookup(far)
    _stub_landing(100, 100)
    _stub_teleport("landed_exact")

    assert require_engagement(_acquire(probe))["target_id"] == far["tank_id"]
    assert probe.engaged == [far]


def test_acquire_engages_without_warning_when_the_landing_is_adjacent() -> None:
    """The adjacent case reaches the same engage call, warning skipped."""
    probe = _ScriptedEngageHarness()
    _stub_acquisition(1100)
    near = _enemy(x=101, y=100)
    _stub_enemy_lookup(near)
    _stub_landing(100, 100)
    _stub_teleport("landed_exact")

    assert require_engagement(_acquire(probe))["target_id"] == near["tank_id"]
    assert probe.engaged == [near]


def test_current_enemy_by_id_scans_past_a_non_matching_threat() -> None:
    """The lookup walks the whole threat list, not just its first entry."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    probe._world_state = _make_world(1000, 100, 100, 900)
    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=101,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-50",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
        "51": make_tank_state(
            tank_id=51,
            x=102,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-51",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
    }

    found = require_threat(_current_enemy_by_id(probe, 51))

    assert (found["tank_id"], found["x"], found["y"], found["name"]) == (
        51,
        102,
        100,
        "red-51",
    )


def test_engage_records_misses_until_the_shot_budget_runs_out() -> None:
    """Every shot missing exhausts max_shots rather than ending early.

    The loop's ordinary exit is the budget, not a kill or a flee, and a
    miss is what the probe records when the server answers without a
    confirmed hit.
    """
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()
    world_service = get_world_service()
    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=101,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-50",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
    }

    def _on_wait() -> None:
        world_service.got_our_shot_response = True
        world_service.got_confirmed_hit = False

    probe._fake_page = ClockAdvancingPage(clock, on_wait=_on_wait)

    result = probe._engage_single_target(_enemy(50), max_shots=3)

    assert [shot["result"] for shot in result["shots"]] == ["miss", "miss", "miss"]
    assert result["total_misses"] == 3
    assert result["total_hits"] == 0
    assert result["kill_confirmed"] is False


def test_execute_probe_runs_every_engagement_and_excludes_hit_targets() -> None:
    """The loop runs ``max_engagements`` times, growing the exclusion set.

    A ``None`` result contributes no engagement and no exclusion, which
    is what lets a barren round be retried against the same enemies.
    """
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ExecuteHarness()
    probe.engagement_results = [
        _engagement(7, shot_ms=1500),
        None,
        _engagement(9),
    ]
    recorded = RecordedChromiumSession.from_capture_path(probe, harness._CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    _stub_initial_sync()

    session = probe.execute_probe(
        max_engagements=3,
        max_shots_per_engagement=10,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=5000,
        teleport_timeout_ms=10000,
    )

    assert [e["target_id"] for e in session["engagements"]] == [7, 9]
    assert probe.excluded_ids_log == [frozenset(), frozenset({7}), frozenset({7})]
    assert session["max_engagements"] == 3
    assert session["max_shots_per_engagement"] == 10
    assert session["capture_session_path"] == ""


def test_execute_probe_tolerates_engagements_without_shots() -> None:
    """First-attempt timing comes from the first shot; there may be none."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ExecuteHarness()
    probe.engagement_results = [_engagement(11)]
    recorded = RecordedChromiumSession.from_capture_path(probe, harness._CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    _stub_initial_sync()

    session = probe.execute_probe(
        max_engagements=1,
        max_shots_per_engagement=10,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=5000,
        teleport_timeout_ms=10000,
    )
    assert session["engagements"][0]["shots"] == []


def test_create_combat_probe_builds_a_combat_probe() -> None:
    """The factory returns a CombatProbe carrying the requested flags."""
    probe = _create_combat_probe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    )
    assert (probe._target_url, probe._headless, probe._prefer_account) == (
        "https://tankpit.com/play",
        True,
        False,
    )
    assert type(probe) is combat_module.CombatProbe


def test_run_combat_probe_writes_the_session_json(fake_fs: FakeFileSystem) -> None:
    """The run wiring threads both bounds through and saves the payload."""
    combat_module.CombatProbe = _FakeCombatProbe

    session = run_combat_probe(
        "https://tankpit.com/play",
        "combat_probe.json",
        max_engagements=4,
        max_shots_per_engagement=6,
    )

    written = fake_fs.read_text(Path("combat_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["max_engagements"] == 4
    assert decoded["max_shots_per_engagement"] == 6
    assert decoded["capture_session_path"] == "combat_probe.capture_session.json"
    assert session["max_engagements"] == 4
