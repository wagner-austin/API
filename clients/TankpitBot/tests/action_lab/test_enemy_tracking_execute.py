"""Tests for the enemy-tracking probe's live session.

The probe answers one question -- does the bot drop a target the JS
client still lists -- so every give-up path matters: a run that quietly
returns no observations looks the same as a run that found no
divergence. These drive acquisition, the approach, the single shot,
the sampling loop, and the run wiring with the browser stubbed out.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab._combat_probe_harness import _CAPTURE_PATH, _SnapshotPhase
from tests.action_lab._enemy_tracking_harness import (
    _ExecuteHarness,
    _make_shot,
    _make_snapshot,
    tracking_module,
)
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_cdp import StubSnapshotCDPSession
from tests.action_lab._replay_page import ClockAdvancingPage, ReplayClock
from tests.action_lab._teleport_attempt_harness import _snapshot
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol, PageProtocol, TerrainMapProtocol
from tankpit_bot._test_hooks.bot import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.enemy_tracking import (
    EnemyTrackingProbe,
    _create_enemy_tracking_probe,
    run_enemy_tracking_probe,
)
from tankpit_bot.action_lab.enemy_tracking_types import (
    EnemyTrackingProbeSessionDict,
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


def test_create_enemy_tracking_probe_builds_the_probe() -> None:
    """The factory returns a probe carrying the requested flags."""
    probe = _create_enemy_tracking_probe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    )
    assert (probe._target_url, probe._headless, probe._prefer_account) == (
        "https://tankpit.com/play",
        True,
        False,
    )
    assert type(probe) is tracking_module.EnemyTrackingProbe


class _FakeTrackingProbe(EnemyTrackingProbe):
    """Probe whose whole session is canned, for the run/save wiring."""

    def execute_probe(
        self,
        *,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        shot_feedback_timeout_ms: int,
        sample_interval_ms: int,
        sample_duration_ms: int,
    ) -> EnemyTrackingProbeSessionDict:
        """Echo the bounds it was handed back in a canned session."""
        from tests.action_lab._enemy_tracking_harness import _make_session

        session = _make_session()
        session["initial_sync_timeout_ms"] = initial_sync_timeout_ms
        session["acquisition_timeout_ms"] = acquisition_timeout_ms
        session["teleport_timeout_ms"] = teleport_timeout_ms
        session["shot_feedback_timeout_ms"] = shot_feedback_timeout_ms
        session["sample_interval_ms"] = sample_interval_ms
        session["sample_duration_ms"] = sample_duration_ms
        session["capture_session_path"] = ""
        return session


def test_run_enemy_tracking_probe_writes_the_session_json(fake_fs: FakeFileSystem) -> None:
    """The run wiring threads the sampling bounds through and saves."""
    tracking_module.EnemyTrackingProbe = _FakeTrackingProbe

    session = run_enemy_tracking_probe(
        "https://tankpit.com/play",
        "enemy_tracking_probe.json",
        sample_interval_ms=250,
        sample_duration_ms=9000,
    )

    written = fake_fs.read_text(Path("enemy_tracking_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["sample_interval_ms"] == 250
    assert decoded["sample_duration_ms"] == 9000
    assert decoded["capture_session_path"] == "enemy_tracking_probe.capture_session.json"
    assert session["sample_interval_ms"] == 250


class _NoCdpHarness(_ExecuteHarness):
    """Harness whose CDP session never becomes available.

    Bootstrap assigns ``_cdp`` on its way through, so a plain
    pre-set of ``None`` would be overwritten before the session body
    reads it; the property swallows that write instead.
    """

    @property
    def _cdp(self) -> CDPSessionProtocol | None:
        """Report no CDP session, whatever bootstrap assigned."""
        return None

    @_cdp.setter
    def _cdp(self, value: CDPSessionProtocol | None) -> None:
        """Discard the assignment bootstrap makes."""
        _ = value


def test_execute_probe_raises_without_a_cdp_session() -> None:
    """No CDP session means no snapshots, so the session refuses to run."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _NoCdpHarness(target=None)
    recorded = RecordedChromiumSession.from_capture_path(probe, _CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    _stub_initial_sync()

    with pytest.raises(TeleportProbeError, match="cdp session is unavailable"):
        _execute(probe)


def test_execute_probe_runs_every_stage_and_records_the_shot() -> None:
    """A found target adds the shot stage and stamps the shot on the session."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ExecuteHarness(target=_enemy(77))
    recorded = RecordedChromiumSession.from_capture_path(probe, _CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    _stub_initial_sync()

    session = _execute(probe)

    assert probe.stages == ["acquire", "teleport", "shoot", "sample"]
    assert session["shot"] == _make_shot()
    assert len(session["observations"]) == 1
    assert session["sample_interval_ms"] == 250
    assert session["sample_duration_ms"] == 1000
    assert session["capture_session_path"] == ""


def test_execute_probe_skips_the_shot_when_no_target_was_reached() -> None:
    """No landing means no shot, and the session says so rather than faking one.

    The sampling loop still runs: observations of a tank nobody shot
    at are exactly what distinguishes a lock released on its own from
    one released after combat.
    """
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ExecuteHarness(target=None)
    recorded = RecordedChromiumSession.from_capture_path(probe, _CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    _stub_initial_sync()

    session = _execute(probe)

    assert probe.stages == ["acquire", "teleport", "sample"]
    assert session["shot"] is None
    assert len(session["observations"]) == 1


def _execute(probe: EnemyTrackingProbe) -> EnemyTrackingProbeSessionDict:
    """Run one probe session with the standard bounds."""
    return probe.execute_probe(
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=5000,
        teleport_timeout_ms=10000,
        shot_feedback_timeout_ms=4000,
        sample_interval_ms=250,
        sample_duration_ms=1000,
    )


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
