"""Tests for the queue probe's experiment runners.

Each command-pair experiment and the error branch it takes when a
dispatch fails.
"""

from __future__ import annotations

import pytest
from tests.action_lab._queue_probe_harness import (
    _FailingCommandProbe,
    _make_world,
    _SecondShootFailsProbe,
    _SequencedWorld,
    _setup_probe,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.queue_experiments import (
    QueueProbeError,
    run_move_then_pickup_experiment,
    run_shoot_then_pickup_experiment,
    run_shoot_then_shoot_experiment,
    run_single_experiment,
)


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


class TestShootThenPickupErrorBranches:
    def test_shoot_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_shoot=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source, ws: 0
        with pytest.raises(QueueProbeError, match="shoot command dispatch failed"):
            run_shoot_then_pickup_experiment(probe, timeout_ms=5000)

    def test_pickup_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_pickup=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source, ws: 0
        with pytest.raises(QueueProbeError, match="pickup_fuel command dispatch failed"):
            run_shoot_then_pickup_experiment(probe, timeout_ms=5000)


class TestShootThenShootErrorBranches:
    def test_first_shoot_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_shoot=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source, ws: 0
        with pytest.raises(QueueProbeError, match="first shoot command dispatch failed"):
            run_shoot_then_shoot_experiment(probe, timeout_ms=5000)

    def test_second_shoot_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _SecondShootFailsProbe(worlds, clock)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source, ws: 0
        with pytest.raises(QueueProbeError, match="second shoot command dispatch failed"):
            run_shoot_then_shoot_experiment(probe, timeout_ms=5000)


class TestMoveThenPickupErrorBranches:
    def test_move_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_move=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source, ws: 0
        with pytest.raises(QueueProbeError, match="move command dispatch failed"):
            run_move_then_pickup_experiment(probe, timeout_ms=5000)

    def test_pickup_dispatch_failure_raises(self) -> None:
        clock = ReplayClock(1000)
        worlds = _SequencedWorld([_make_world(1000, 100, 100, 900)])
        probe = _FailingCommandProbe(worlds, clock, fail_pickup=True)
        probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        action_hooks.get_current_time_ms = clock
        action_hooks.drain_buffered_messages = lambda source, ws: 0
        with pytest.raises(QueueProbeError, match="pickup_fuel command dispatch failed"):
            run_move_then_pickup_experiment(probe, timeout_ms=5000)
