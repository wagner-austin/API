"""Tests for the enemy-tracking probe's individual session stages.

Acquisition, the approach, the single shot and the sampling loop,
each driven on its own with the browser stubbed out. The session
that composes them is :mod:`tests.action_lab.test_enemy_tracking_execute`.
"""

from __future__ import annotations

from collections.abc import Callable, Generator

import pytest
from tests.action_lab._combat_probe_harness import _SnapshotPhase
from tests.action_lab._enemy_tracking_harness import (
    _make_snapshot,
    _make_tracked,
    tracking_module,
)
from tests.action_lab._replay_cdp import StubSnapshotCDPSession
from tests.action_lab._replay_page import ClockAdvancingPage, ReplayClock
from tests.action_lab._teleport_attempt_harness import _snapshot

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol, PageProtocol, TerrainMapProtocol
from tankpit_bot._test_hooks.bot import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.enemy_tracking import (
    EnemyTrackingProbe,
)
from tankpit_bot.action_lab.enemy_tracking_types import (
    TrackingObservationDict,
)
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
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import make_viewport_state


def _world(x: int = 100, y: int = 100) -> WorldStateDict:
    """Build a world with self at the given tile."""
    world = make_empty_world_state()
    world["self_state"] = make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=5,
    )
    world["viewport"] = make_viewport_state(left=x - 8, top=y - 8, width=16, height=16)
    world["timestamp_ms"] = 1000
    return world


def _enemy(tank_id: int = 77, *, x: int = 101, y: int = 100) -> EnemyThreatDict:
    """Build an enemy threat at a tile."""
    return make_enemy_threat(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=abs(x - 100) + abs(y - 100),
        damage_state=3,
        rank=2,
        team=1,
        name=f"red-{tank_id}",
        is_bot=False,
        timestamp_ms=1000,
        last_wire_seen_ms=1000,
        last_position_update_ms=1000,
    )


class _ProbeHarness(EnemyTrackingProbe):
    """Tracking probe with page, CDP, world and shooting stubbed."""

    def __init__(self) -> None:
        """Seed a spawned tank at (100, 100)."""
        ws = WorldService()
        super().__init__(
            "https://tankpit.com/play",
            headless=True,
            prefer_account=False,
            world=ws,
        )
        self._world_state = _world()
        self._self_state: SelfStateDict | None = self._world_state["self_state"]
        self._fake_page = ClockAdvancingPage(ReplayClock(1000))
        self.stub_cdp = StubSnapshotCDPSession()
        self._cdp = self.stub_cdp
        self.shots: list[tuple[int, int, int]] = []

    def _require_page(self) -> PageProtocol:
        """Return the clock-advancing fake page."""
        return self._fake_page

    def get_world_state(self) -> WorldStateDict:
        """Return the seeded world."""
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        """Return the seeded self state."""
        return self._self_state

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        """Record a shot instead of dispatching it."""
        self.shots.append((x, y, target_id))
        return True


@pytest.fixture(autouse=True)
def _restore_tracking_hooks() -> Generator[None, None, None]:
    """Restore every hook and module attribute these tests swap."""
    original_get_time = action_hooks.get_current_time_ms
    original_drain = action_hooks.drain_buffered_messages
    original_wait_initial = action_hooks.wait_for_initial_self_state
    original_sync_playwright = core_hooks.sync_playwright
    original_acquisition = tracking_module.run_tracked_acquisition_phase
    original_teleport = tracking_module.run_tracked_teleport_command
    original_landing = tracking_module.choose_combat_landing_tile
    original_threats = tracking_module.analyze_threats
    original_capture = tracking_module.capture_page_client_snapshot
    original_feedback = tracking_module._wait_for_shot_feedback
    original_probe_class = tracking_module.EnemyTrackingProbe
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.drain_buffered_messages = original_drain
    action_hooks.wait_for_initial_self_state = original_wait_initial
    core_hooks.sync_playwright = original_sync_playwright
    tracking_module.run_tracked_acquisition_phase = original_acquisition
    tracking_module.run_tracked_teleport_command = original_teleport
    tracking_module.choose_combat_landing_tile = original_landing
    tracking_module.analyze_threats = original_threats
    tracking_module.capture_page_client_snapshot = original_capture
    tracking_module._wait_for_shot_feedback = original_feedback
    tracking_module.EnemyTrackingProbe = original_probe_class


def _install_common_stubs(threats: list[EnemyThreatDict]) -> None:
    """Stub draining, threat analysis and snapshot capture."""

    def _drain(source: BufferedMessageSourceProtocol, ws: WorldService, /) -> int:
        _ = source
        return 0

    def _analyze(
        ws: WorldService,
        world: WorldStateDict,
        self_state: SelfStateDict,
        now_ms: int,
        *,
        human_min_rank: int = 0,
        human_max_rank: int = 8,
    ) -> list[EnemyThreatDict]:
        _ = (ws, world, self_state, now_ms, human_min_rank, human_max_rank)
        return threats

    def _capture(cdp: CDPSessionProtocol) -> PageClientSnapshotDict:
        _ = cdp
        return _make_snapshot()

    action_hooks.drain_buffered_messages = _drain
    tracking_module.analyze_threats = _analyze
    tracking_module.capture_page_client_snapshot = _capture


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

    tracking_module.run_tracked_acquisition_phase = _phase


def _teleport_result(
    target: TeleportTargetDict,
    status: str,
) -> TeleportAttemptResultDict:
    """Build a teleport outcome carrying the requested status."""
    return TeleportAttemptResultDict(
        target=target,
        teleport_cycle_id=1,
        status="teleport_timeout" if status == "teleport_timeout" else "landed_exact",
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
        landed_signal_received=status != "teleport_timeout",
        landed_x=100,
        landed_y=100,
        message_start_index=0,
        message_end_index=0,
        page_snapshots=[],
    )


def _stub_teleport(status: str) -> None:
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

    tracking_module.run_tracked_teleport_command = _command


def _stub_landing(x: int, y: int) -> None:
    """Make landing selection return a fixed tile over absent terrain."""

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
        now_ms: int,
        ws: WorldService,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain, now_ms, ws)
        return (x, y)

    tracking_module.choose_combat_landing_tile = _landing


def _stub_feedback(got_response: bool, was_hit: bool) -> None:
    """Make the shot-feedback wait resolve immediately."""

    def _wait(
        page: action_session.WaitPageProtocol,
        probe: ProbeBase,
        *,
        timeout_ms: int,
    ) -> tuple[bool, bool]:
        _ = (page, probe, timeout_ms)
        return (got_response, was_hit)

    tracking_module._wait_for_shot_feedback = _wait


def test_acquire_enemies_raises_when_the_map_never_syncs() -> None:
    """No sync means no acquisition, and the probe says so loudly."""
    probe = _ProbeHarness()
    _install_common_stubs([_enemy()])
    _stub_acquisition(None)

    with pytest.raises(TeleportProbeError, match="map sync did not complete"):
        probe._acquire_enemies(cdp=probe.stub_cdp, acquisition_timeout_ms=5000)


def test_acquire_enemies_builds_a_record_per_visible_enemy() -> None:
    """Acquisition returns one tracked record for each enemy in view."""
    probe = _ProbeHarness()
    _install_common_stubs([_enemy(77), _enemy(78, x=102)])
    _stub_acquisition(1100)

    tracked, _snapshot_at, threats = probe._acquire_enemies(
        cdp=probe.stub_cdp,
        acquisition_timeout_ms=5000,
    )

    assert [record["tank_id"] for record in tracked] == [77, 78]
    assert [threat["tank_id"] for threat in threats] == [77, 78]


def test_teleport_to_closest_enemy_returns_none_without_threats() -> None:
    """Nothing visible means nothing to approach."""
    probe = _ProbeHarness()
    _install_common_stubs([])

    assert (
        probe._teleport_to_closest_enemy(
            cdp=probe.stub_cdp,
            threats=[],
            teleport_timeout_ms=10000,
            message_start_index=0,
        )
        is None
    )


def test_teleport_to_closest_enemy_returns_none_without_a_landing() -> None:
    """The no-landing sentinel ends the approach before any teleport."""
    probe = _ProbeHarness()
    _install_common_stubs([_enemy()])
    _stub_landing(-1, -1)

    assert (
        probe._teleport_to_closest_enemy(
            cdp=probe.stub_cdp,
            threats=[_enemy()],
            teleport_timeout_ms=10000,
            message_start_index=0,
        )
        is None
    )


def test_teleport_to_closest_enemy_returns_none_on_timeout() -> None:
    """A teleport that never lands yields no target to shoot."""
    probe = _ProbeHarness()
    _install_common_stubs([_enemy()])
    _stub_landing(100, 100)
    _stub_teleport("teleport_timeout")

    assert (
        probe._teleport_to_closest_enemy(
            cdp=probe.stub_cdp,
            threats=[_enemy()],
            teleport_timeout_ms=10000,
            message_start_index=0,
        )
        is None
    )


def test_teleport_to_closest_enemy_returns_the_closest_target() -> None:
    """A landed teleport hands back the enemy that was approached."""
    probe = _ProbeHarness()
    closest = _enemy(77)
    _install_common_stubs([closest])
    _stub_landing(100, 100)
    _stub_teleport("landed_exact")

    target = probe._teleport_to_closest_enemy(
        cdp=probe.stub_cdp,
        threats=[closest, _enemy(78, x=120)],
        teleport_timeout_ms=10000,
        message_start_index=0,
    )

    assert target == closest


@pytest.mark.parametrize(
    ("got_response", "was_hit", "outcome"),
    [(True, True, "hit"), (True, False, "miss"), (False, False, "timeout")],
)
def test_fire_one_shot_records_every_outcome(
    got_response: bool,
    was_hit: bool,
    outcome: str,
) -> None:
    """Hit, miss and timeout each produce their own shot record.

    ``responded_ms`` is -1 exactly when the server never answered, so
    the analysis can tell an unanswered shot from a fast miss.
    """
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()
    _install_common_stubs([])
    _stub_feedback(got_response, was_hit)

    shot = probe._fire_one_shot(target=_enemy(77), shot_feedback_timeout_ms=4000)

    assert shot["outcome"] == outcome
    assert (shot["responded_ms"] == -1) is not got_response
    assert probe.shots == [(101, 100, 77)]


def test_sample_loop_emits_a_row_per_tank_per_sample() -> None:
    """The loop runs until the window closes, one row per tracked tank.

    Termination depends on the page: ``wait_for_timeout`` is what
    advances the clock, so a page that did not tick would spin here
    forever. Four 250 ms waits close a 1000 ms window.
    """
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()
    probe._fake_page = ClockAdvancingPage(clock)
    _install_common_stubs([_enemy(77)])

    rows: list[TrackingObservationDict] = probe._sample_loop(
        cdp=probe.stub_cdp,
        tracked=[_make_tracked()],
        sample_interval_ms=250,
        sample_duration_ms=1000,
    )

    assert [row["sample_index"] for row in rows] == [0, 1, 2, 3]
    assert {row["bot_mode_state"] for row in rows} == {"OBSERVE"}
    assert {row["bot_combat_target_id"] for row in rows} == {-1}
