"""Tests for the queue probe's wait helpers.

Position, fuel, and world-timestamp waits, plus experiment status and
command timing.
"""

from __future__ import annotations

import pytest
from tests.action_lab._queue_probe_harness import (
    _make_world,
    _make_world_no_self,
    _SequencedWorld,
    _setup_probe,
)
from tests.action_lab._replay_page import ReplayClock

from tankpit_bot.action_lab.queue_experiments import (
    QueueProbeError,
    _build_command_timing,
    _determine_experiment_status,
    _require_self_state,
    _wait_for_fuel_change,
    _wait_for_position_change,
    _wait_for_world_timestamp_advance,
)
from tankpit_bot.action_lab.queue_probe_types import (
    QueueCommandTimingDict,
)


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
