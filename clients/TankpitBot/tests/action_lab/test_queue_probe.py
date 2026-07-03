"""Tests for queue probe logic — experiment runners and helpers."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    ClockAdvancingPage,
    ReplayClock,
    StubbedBootstrapMixin,
)
from tests.conftest import FakeFileSystem

import tankpit_bot.action_lab.queue_probe as queue_probe_module
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.queue_probe import (
    QueueExperimentProbeProtocol,
    QueueProbe,
    QueueProbeError,
    _build_command_timing,
    _determine_experiment_status,
    _require_self_state,
    _wait_for_fuel_change,
    _wait_for_position_change,
    _wait_for_world_timestamp_advance,
    format_queue_probe_summary,
    run_move_then_pickup_experiment,
    run_queue_probe,
    run_shoot_then_pickup_experiment,
    run_shoot_then_shoot_experiment,
    run_single_experiment,
)
from tankpit_bot.action_lab.queue_probe_types import (
    QueueCommandTimingDict,
    QueueExperimentKind,
    QueueExperimentResultDict,
    QueueProbeSessionDict,
)
from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state, make_self_state
from tankpit_bot.types import CapturedMessage

_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


@pytest.fixture(autouse=True)
def _isolate_hooks() -> Generator[None, None, None]:
    """Save and restore action hooks around each test."""
    orig_time = action_hooks.get_current_time_ms
    orig_drain = action_hooks.drain_buffered_messages
    orig_playwright = core_hooks.sync_playwright
    orig_wait = action_hooks.wait_for_initial_self_state
    orig_advance = action_hooks.advance_startup_state
    orig_run_single = queue_probe_module.run_single_experiment
    yield
    action_hooks.get_current_time_ms = orig_time
    action_hooks.drain_buffered_messages = orig_drain
    core_hooks.sync_playwright = orig_playwright
    action_hooks.wait_for_initial_self_state = orig_wait
    action_hooks.advance_startup_state = orig_advance
    queue_probe_module.run_single_experiment = orig_run_single


def _make_world(timestamp_ms: int, x: int, y: int, fuel: int) -> WorldStateDict:
    empty = make_empty_world_state()
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
        tanks=empty["tanks"],
        containers=empty["containers"],
        mines=empty["mines"],
        terrain=empty["terrain"],
        viewport=empty["viewport"],
        scanned_tiles=empty["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def _make_world_no_self(timestamp_ms: int) -> WorldStateDict:
    empty = make_empty_world_state()
    return WorldStateDict(
        self_state=None,
        tanks=empty["tanks"],
        containers=empty["containers"],
        mines=empty["mines"],
        terrain=empty["terrain"],
        viewport=empty["viewport"],
        scanned_tiles=empty["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


class _SequencedWorld:
    def __init__(self, states: list[WorldStateDict]) -> None:
        self._states = states
        self._index = 0

    def current(self) -> WorldStateDict:
        return self._states[self._index]

    def advance(self) -> None:
        if self._index + 1 < len(self._states):
            self._index += 1


class _FakeExperimentProbe:
    """Satisfies QueueExperimentProbeProtocol structurally."""

    def __init__(self, worlds: _SequencedWorld, clock: ReplayClock) -> None:
        self._worlds = worlds
        self._cdp_message_buffer: list[str] = []
        self._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        self._messages_list: list[CapturedMessage] = []
        self._commands_sent: list[str] = []

    def get_world_state(self) -> WorldStateDict:
        return self._worlds.current()

    def get_self_state(self) -> SelfStateDict | None:
        return self._worlds.current()["self_state"]

    def _update_state_from_world(self) -> None:
        pass

    def _require_page(self) -> ClockAdvancingPage:
        return self._page

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages_list

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        self._commands_sent.append(f"shoot({x},{y},id={target_id})")
        return True

    def pickup_fuel(self, x: int, y: int) -> bool:
        self._commands_sent.append(f"pickup_fuel({x},{y})")
        return True

    def move_to(self, x: int, y: int) -> bool:
        self._commands_sent.append(f"move({x},{y})")
        return True


def _setup_probe(
    worlds: _SequencedWorld,
    clock: ReplayClock,
) -> _FakeExperimentProbe:
    probe = _FakeExperimentProbe(worlds, clock)
    probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    return probe


# =========================================================================
# Helper function tests
# =========================================================================


class TestDetermineExperimentStatus:
    def test_both_processed(self) -> None:
        assert _determine_experiment_status(100, 200) == "both_processed"

    def test_second_dropped(self) -> None:
        assert _determine_experiment_status(100, None) == "second_dropped"

    def test_timeout(self) -> None:
        assert _determine_experiment_status(None, None) == "timeout"

    def test_timeout_even_with_secondary(self) -> None:
        assert _determine_experiment_status(None, 200) == "timeout"


class TestBuildCommandTiming:
    def test_with_ack(self) -> None:
        timing = _build_command_timing("shoot", 100, 250)
        assert timing == QueueCommandTimingDict(
            label="shoot", sent_ms=100, ack_ms=250, elapsed_ms=150
        )

    def test_without_ack(self) -> None:
        timing = _build_command_timing("move", 100, None)
        assert timing == QueueCommandTimingDict(
            label="move", sent_ms=100, ack_ms=None, elapsed_ms=None
        )


class TestRequireSelfState:
    def test_returns_self_state(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _setup_probe(worlds, clock)
        state = _require_self_state(probe)
        assert state["x"] == 100
        assert state["fuel"] == 900

    def test_raises_when_none(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world_no_self(1000)])
        probe = _setup_probe(worlds, clock)
        with pytest.raises(QueueProbeError, match="self state unavailable"):
            _require_self_state(probe)


# =========================================================================
# Wait function tests
# =========================================================================


class TestWaitForPositionChange:
    def test_detects_change(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 101, 100, 900),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = _wait_for_position_change(
            probe,
            baseline_x=100,
            baseline_y=100,
            started_ms=1000,
            timeout_ms=5000,
        )
        assert result == 1100

    def test_returns_none_on_timeout(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _setup_probe(worlds, clock)
        result = _wait_for_position_change(
            probe,
            baseline_x=100,
            baseline_y=100,
            started_ms=1000,
            timeout_ms=250,
        )
        assert result is None


class TestWaitForFuelChange:
    def test_detects_change(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 100, 100, 950),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = _wait_for_fuel_change(
            probe,
            baseline_fuel=900,
            started_ms=1000,
            timeout_ms=5000,
        )
        assert result == 1100

    def test_returns_none_on_timeout(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _setup_probe(worlds, clock)
        result = _wait_for_fuel_change(
            probe,
            baseline_fuel=900,
            started_ms=1000,
            timeout_ms=250,
        )
        assert result is None


class TestWaitForWorldTimestampAdvance:
    def test_detects_advance(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 100, 100, 900),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = _wait_for_world_timestamp_advance(
            probe,
            baseline_ms=1000,
            started_ms=1000,
            timeout_ms=5000,
        )
        assert result == 1100

    def test_returns_none_on_timeout(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _setup_probe(worlds, clock)
        result = _wait_for_world_timestamp_advance(
            probe,
            baseline_ms=1000,
            started_ms=1000,
            timeout_ms=250,
        )
        assert result is None


# =========================================================================
# Experiment runner tests
# =========================================================================


class TestRunShootThenPickup:
    def test_both_processed_when_fuel_changes(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 100, 100, 950),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = run_shoot_then_pickup_experiment(probe, timeout_ms=5000)
        assert result["kind"] == "shoot_then_pickup"
        assert result["status"] == "both_processed"
        assert result["primary"]["label"] == "shoot"
        assert result["secondary"]["label"] == "pickup_fuel"

    def test_second_dropped_when_fuel_unchanged(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 100, 100, 900),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = run_shoot_then_pickup_experiment(probe, timeout_ms=5000)
        assert result["status"] == "second_dropped"

    def test_timeout_when_world_unchanged(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _setup_probe(worlds, clock)
        result = run_shoot_then_pickup_experiment(probe, timeout_ms=250)
        assert result["status"] == "timeout"


class TestRunShootThenShoot:
    def test_both_processed_when_two_advances(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 100, 100, 900),
                _make_world(1200, 100, 100, 900),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = run_shoot_then_shoot_experiment(probe, timeout_ms=5000)
        assert result["kind"] == "shoot_then_shoot"
        assert result["status"] == "both_processed"
        assert result["primary"]["label"] == "shoot_1"
        assert result["secondary"]["label"] == "shoot_2"

    def test_second_dropped_on_timeout(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 100, 100, 900),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = run_shoot_then_shoot_experiment(probe, timeout_ms=250)
        assert result["status"] == "second_dropped"

    def test_timeout_when_world_unchanged(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _setup_probe(worlds, clock)
        result = run_shoot_then_shoot_experiment(probe, timeout_ms=250)
        assert result["status"] == "timeout"


class TestRunMoveThenPickup:
    def test_both_processed(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 101, 100, 950),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = run_move_then_pickup_experiment(probe, timeout_ms=5000)
        assert result["kind"] == "move_then_pickup"
        assert result["status"] == "both_processed"

    def test_second_dropped_when_fuel_unchanged(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 101, 100, 900),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = run_move_then_pickup_experiment(probe, timeout_ms=5000)
        assert result["status"] == "second_dropped"

    def test_timeout_when_position_unchanged(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _setup_probe(worlds, clock)
        result = run_move_then_pickup_experiment(probe, timeout_ms=250)
        assert result["status"] == "timeout"


# =========================================================================
# run_single_experiment dispatcher tests
# =========================================================================


class TestRunSingleExperiment:
    def test_dispatches_shoot_then_pickup(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 100, 100, 950),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = run_single_experiment(probe, "shoot_then_pickup", timeout_ms=5000)
        assert result["kind"] == "shoot_then_pickup"

    def test_dispatches_shoot_then_shoot(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 100, 100, 900),
                _make_world(1200, 100, 100, 900),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = run_single_experiment(probe, "shoot_then_shoot", timeout_ms=5000)
        assert result["kind"] == "shoot_then_shoot"

    def test_dispatches_move_then_pickup(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld(
            [
                _make_world(1000, 100, 100, 900),
                _make_world(1100, 101, 100, 950),
            ]
        )
        probe = _setup_probe(worlds, clock)
        result = run_single_experiment(probe, "move_then_pickup", timeout_ms=5000)
        assert result["kind"] == "move_then_pickup"


# =========================================================================
# Format summary tests
# =========================================================================


class TestFormatQueueProbeSummary:
    def test_formats_session(self) -> None:
        timing = TeleportStartupTimingDict(
            game_ready_timestamp_ms=1000,
            intel_ready_timestamp_ms=2000,
            initial_sync_started_ms=3000,
            initial_world_timestamp_ms=4000,
            command_ready_timestamp_ms=5000,
            first_attempt_started_ms=6000,
            game_ready_to_intel_ready_ms=1000,
            intel_ready_to_initial_world_ms=2000,
            initial_world_to_command_ready_ms=1000,
            command_ready_to_first_attempt_ms=1000,
        )
        session = QueueProbeSessionDict(
            session_id="test-001",
            start_timestamp_ms=1000,
            end_timestamp_ms=5000,
            base_url="https://tankpit.com/play",
            spawn_x=128,
            spawn_y=128,
            capture_session_path="",
            initial_sync_timeout_ms=10000,
            experiment_timeout_ms=5000,
            startup_timing=timing,
            experiments=[
                QueueExperimentResultDict(
                    kind="shoot_then_pickup",
                    status="both_processed",
                    primary=QueueCommandTimingDict(
                        label="shoot", sent_ms=100, ack_ms=200, elapsed_ms=100
                    ),
                    secondary=QueueCommandTimingDict(
                        label="pickup_fuel", sent_ms=105, ack_ms=210, elapsed_ms=105
                    ),
                    inter_send_delay_ms=5,
                    total_elapsed_ms=115,
                    message_start_index=0,
                    message_end_index=10,
                ),
            ],
        )
        summary = format_queue_probe_summary(session)
        assert "test-001" in summary
        assert "shoot_then_pickup" in summary
        assert "both_processed" in summary
        assert "(128, 128)" in summary

    def test_formats_empty_experiments(self) -> None:
        timing = TeleportStartupTimingDict(
            game_ready_timestamp_ms=1000,
            intel_ready_timestamp_ms=2000,
            initial_sync_started_ms=3000,
            initial_world_timestamp_ms=4000,
            command_ready_timestamp_ms=5000,
            first_attempt_started_ms=None,
            game_ready_to_intel_ready_ms=1000,
            intel_ready_to_initial_world_ms=2000,
            initial_world_to_command_ready_ms=1000,
            command_ready_to_first_attempt_ms=None,
        )
        session = QueueProbeSessionDict(
            session_id="test-002",
            start_timestamp_ms=1000,
            end_timestamp_ms=2000,
            base_url="https://tankpit.com/play",
            spawn_x=64,
            spawn_y=64,
            capture_session_path="",
            initial_sync_timeout_ms=10000,
            experiment_timeout_ms=5000,
            startup_timing=timing,
            experiments=[],
        )
        summary = format_queue_probe_summary(session)
        assert "Experiments: 0" in summary


# =========================================================================
# QueueProbe validation tests
# =========================================================================


class _FailingCommandProbe(_FakeExperimentProbe):
    """Probe whose commands return False to test error branches."""

    def __init__(
        self,
        worlds: _SequencedWorld,
        clock: ReplayClock,
        *,
        fail_shoot: bool = False,
        fail_pickup: bool = False,
        fail_move: bool = False,
    ) -> None:
        super().__init__(worlds, clock)
        self._fail_shoot = fail_shoot
        self._fail_pickup = fail_pickup
        self._fail_move = fail_move

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        if self._fail_shoot:
            return False
        return super().shoot(x, y, target_id)

    def pickup_fuel(self, x: int, y: int) -> bool:
        if self._fail_pickup:
            return False
        return super().pickup_fuel(x, y)

    def move_to(self, x: int, y: int) -> bool:
        if self._fail_move:
            return False
        return super().move_to(x, y)


class TestShootThenPickupErrorBranches:
    def test_shoot_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_shoot=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source: 0
        with pytest.raises(QueueProbeError, match="shoot command dispatch failed"):
            run_shoot_then_pickup_experiment(probe, timeout_ms=5000)

    def test_pickup_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_pickup=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source: 0
        with pytest.raises(QueueProbeError, match="pickup_fuel command dispatch failed"):
            run_shoot_then_pickup_experiment(probe, timeout_ms=5000)


class _SecondShootFailsProbe(_FakeExperimentProbe):
    """Probe where only the second shoot call fails."""

    def __init__(self, worlds: _SequencedWorld, clock: ReplayClock) -> None:
        super().__init__(worlds, clock)
        self._shoot_count = 0

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        self._shoot_count += 1
        if self._shoot_count >= 2:
            return False
        return super().shoot(x, y, target_id)


class TestShootThenShootErrorBranches:
    def test_first_shoot_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_shoot=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source: 0
        with pytest.raises(QueueProbeError, match="first shoot command dispatch failed"):
            run_shoot_then_shoot_experiment(probe, timeout_ms=5000)

    def test_second_shoot_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _SecondShootFailsProbe(worlds, clock)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source: 0
        with pytest.raises(QueueProbeError, match="second shoot command dispatch failed"):
            run_shoot_then_shoot_experiment(probe, timeout_ms=5000)


class TestMoveThenPickupErrorBranches:
    def test_move_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_move=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source: 0
        with pytest.raises(QueueProbeError, match="move command dispatch failed"):
            run_move_then_pickup_experiment(probe, timeout_ms=5000)

    def test_pickup_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_pickup=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source: 0
        with pytest.raises(QueueProbeError, match="pickup_fuel command dispatch failed"):
            run_move_then_pickup_experiment(probe, timeout_ms=5000)


# =========================================================================
# execute_probe integration tests via StubbedBootstrapMixin
# =========================================================================


class _SteppingClock:
    def __init__(self, start_ms: int, step_ms: int) -> None:
        self._current_ms = start_ms
        self._step_ms = step_ms

    def __call__(self) -> int:
        value = self._current_ms
        self._current_ms += self._step_ms
        return value


class _ExecuteHarness(StubbedBootstrapMixin, QueueProbe):
    def __init__(self) -> None:
        QueueProbe.__init__(
            self,
            "https://tankpit.com/play",
            headless=False,
            prefer_account=True,
        )
        self._init_bootstrap_stubs()


def _make_experiment_result(kind: QueueExperimentKind) -> QueueExperimentResultDict:
    return QueueExperimentResultDict(
        kind=kind,
        status="both_processed",
        primary=QueueCommandTimingDict(
            label="primary",
            sent_ms=100,
            ack_ms=200,
            elapsed_ms=100,
        ),
        secondary=QueueCommandTimingDict(
            label="secondary",
            sent_ms=105,
            ack_ms=210,
            elapsed_ms=105,
        ),
        inter_send_delay_ms=5,
        total_elapsed_ms=115,
        message_start_index=0,
        message_end_index=1,
    )


def _wait_for_initial_self_state_spawn(
    page: action_session.WaitPageProtocol,
    provider: action_session.BufferedWorldStateProviderProtocol,
    started_ms: int,
    timeout_ms: int,
) -> tuple[int, SelfStateDict]:
    _ = (page, provider, started_ms, timeout_ms)
    return (1500, make_self_state(1, 101, 102, 2, 1, 900, 5))


def _advance_startup_state_stub(
    bot: action_session.StartupStateDriverProtocol,
) -> None:
    _ = bot


class TestExecuteProbeIntegration:
    def test_execute_probe_runs_experiments(self) -> None:
        harness = _ExecuteHarness()
        clock = _SteppingClock(1000, 100)
        action_hooks.get_current_time_ms = clock
        recorded = RecordedChromiumSession.from_capture_path(harness, _CAPTURE_PATH)
        core_hooks.sync_playwright = recorded.sync_playwright_factory
        action_hooks.wait_for_initial_self_state = _wait_for_initial_self_state_spawn
        action_hooks.advance_startup_state = _advance_startup_state_stub

        experiment_results = [_make_experiment_result("shoot_then_pickup")]

        def _fake_run_single(
            probe: QueueExperimentProbeProtocol,
            kind: QueueExperimentKind,
            *,
            timeout_ms: int,
        ) -> QueueExperimentResultDict:
            _ = (probe, timeout_ms)
            return experiment_results.pop(0)

        queue_probe_module.run_single_experiment = _fake_run_single

        session = harness.execute_probe(
            initial_sync_timeout_ms=5000,
            experiment_timeout_ms=3000,
            experiment_kinds=["shoot_then_pickup"],
        )
        assert session["spawn_x"] == 101
        assert session["spawn_y"] == 102
        assert len(session["experiments"]) == 1
        assert session["experiments"][0]["kind"] == "shoot_then_pickup"
        assert session["experiment_timeout_ms"] == 3000
        assert harness._page is None
        assert harness._cdp is None

    def test_execute_probe_raises_when_playwright_missing(self) -> None:
        from tankpit_bot.browser.types import PlaywrightNotInstalledError

        harness = _ExecuteHarness()
        core_hooks.sync_playwright = None
        with pytest.raises(PlaywrightNotInstalledError):
            harness.execute_probe(
                initial_sync_timeout_ms=5000,
                experiment_timeout_ms=3000,
                experiment_kinds=["shoot_then_pickup"],
            )


# =========================================================================
# QueueProbe validation tests
# =========================================================================


class TestQueueProbeValidation:
    def test_negative_timeout_raises(self) -> None:
        probe = QueueProbe.__new__(QueueProbe)
        with pytest.raises(ValueError, match="experiment_timeout_ms must be positive"):
            probe.execute_probe(
                initial_sync_timeout_ms=10000,
                experiment_timeout_ms=0,
                experiment_kinds=["shoot_then_pickup"],
            )

    def test_empty_kinds_raises(self) -> None:
        probe = QueueProbe.__new__(QueueProbe)
        with pytest.raises(ValueError, match="experiment_kinds must not be empty"):
            probe.execute_probe(
                initial_sync_timeout_ms=10000,
                experiment_timeout_ms=5000,
                experiment_kinds=[],
            )


# =========================================================================
# run_queue_probe integration test
# =========================================================================


class _FakeQueueProbeForRunner(QueueProbe):
    """QueueProbe that returns a canned session from execute_probe."""

    def execute_probe(
        self,
        *,
        initial_sync_timeout_ms: int,
        experiment_timeout_ms: int,
        experiment_kinds: list[QueueExperimentKind],
    ) -> QueueProbeSessionDict:
        _ = (initial_sync_timeout_ms, experiment_timeout_ms, experiment_kinds)
        return QueueProbeSessionDict(
            session_id="fake-queue-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self._target_url,
            spawn_x=100,
            spawn_y=100,
            capture_session_path="",
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            experiment_timeout_ms=experiment_timeout_ms,
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
            experiments=[],
        )


def test_create_queue_probe_factory_returns_queue_probe() -> None:
    """The factory returns a real QueueProbe instance."""
    probe = queue_probe_module._create_queue_probe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    )
    assert probe._target_url == "https://tankpit.com/play"
    assert probe._headless is True


def test_run_queue_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    """run_queue_probe persists the session and capture JSON."""
    from pathlib import Path

    from platform_core.json_utils import load_json_str, narrow_json_to_dict

    from tankpit_bot.action_lab.queue_probe_types import decode_queue_probe_session

    original_factory = queue_probe_module._create_queue_probe
    queue_probe_module._create_queue_probe = (
        lambda target_url, *, headless, prefer_account: _FakeQueueProbeForRunner(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
        )
    )
    try:
        session = run_queue_probe(
            "https://tankpit.com/play",
            "queue_probe.json",
        )
    finally:
        queue_probe_module._create_queue_probe = original_factory

    written = fake_fs.read_text(Path("queue_probe.json"))
    decoded = decode_queue_probe_session(narrow_json_to_dict(load_json_str(written)))
    assert session == decoded
    assert session["session_id"] == "fake-queue-session"
