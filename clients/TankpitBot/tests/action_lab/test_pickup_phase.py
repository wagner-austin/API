"""Tests for shared move-and-pickup command-phase helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
from tests.action_lab._replay_core import ReplayClock

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import pickup_phase
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state, make_self_state
from tankpit_bot.types import CapturedMessage


class _Page:
    """Minimal page fake for pickup wait loops."""

    def __init__(self, clock: ReplayClock) -> None:
        """Initialize the page."""
        self._clock = clock
        self.waits: list[float] = []
        self.on_wait: Callable[[], None] | None = None

    def wait_for_timeout(self, timeout: float) -> None:
        """Advance the clock and invoke the wait callback."""
        self.waits.append(timeout)
        self._clock.advance(int(timeout))
        if self.on_wait is not None:
            self.on_wait()

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)


class _Probe:
    """Minimal pickup probe fake."""

    def __init__(self, world: WorldStateDict, *, move_result: bool = True) -> None:
        """Initialize the probe."""
        self._world = world
        self._cdp_message_buffer: list[str] = []
        self._messages: list[CapturedMessage] = []
        self._cycles: list[ActionPhaseCycleDict] = []
        self._ended: list[ActionPhaseCycleDict] = []
        self._move_result = move_result
        self._move_calls: list[tuple[int, int]] = []
        self.reset_count = 0

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return captured messages."""
        return self._messages

    @property
    def magic(self) -> str:
        """Return a stable fake magic key."""
        return "magic"

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""
        return self._world

    def move_to(self, x: int, y: int) -> bool:
        """Record one move dispatch."""
        self._move_calls.append((x, y))
        return self._move_result

    def _start_action_phase(self, phase: str, *, attempt_label: str) -> ActionPhaseCycleDict:
        """Start one typed phase cycle."""
        cycle = ActionPhaseCycleDict(
            phase=_require_pickup_phase_name(phase),
            cycle_id=len(self._cycles) + 1,
            started_ms=1300,
        )
        assert attempt_label == "attempt-1"
        self._cycles.append(cycle)
        return cycle

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """Record one closed phase."""
        self._ended.append(cycle)

    def _reset_probe_state_to_idle(self) -> None:
        """Record one idle reset."""
        self.reset_count += 1


def _require_pickup_phase_name(phase: str) -> Literal["move", "pickup"]:
    """Return a valid move-or-pickup phase name."""
    if phase == "move":
        return "move"
    if phase == "pickup":
        return "pickup"
    raise AssertionError(f"unexpected phase {phase}")


def _make_world(x: int, y: int, fuel: int) -> WorldStateDict:
    """Build a world with one self tank."""
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1,
            x=x,
            y=y,
            team=2,
            rank=1,
            fuel=fuel,
            leaderboard_position=1,
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=world["viewport"],
        scanned_viewports=world["scanned_viewports"],
        timestamp_ms=1000,
    )


def _require_self_state(world: WorldStateDict) -> SelfStateDict:
    """Return the current self state or raise a test failure."""
    self_state = world["self_state"]
    if self_state is None:
        raise AssertionError("expected self state")
    return self_state


def test_effective_pickup_timeout_scales_with_distance() -> None:
    """Pickup timeout budget grows with travel distance."""
    assert (
        pickup_phase.effective_pickup_timeout_ms(
            current_x=100,
            current_y=100,
            target_x=101,
            target_y=100,
            base_timeout_ms=500,
        )
        == 1500
    )
    assert (
        pickup_phase.effective_pickup_timeout_ms(
            current_x=100,
            current_y=100,
            target_x=100,
            target_y=100,
            base_timeout_ms=3000,
        )
        == 3000
    )


def test_get_completed_pickup_outcome_detects_fuel_gain() -> None:
    """Pickup completion helper returns success once fuel increases."""
    clock = ReplayClock(1600)
    probe = _Probe(_make_world(100, 100, 900))
    action_hooks.get_current_time_ms = clock

    assert (
        pickup_phase.get_completed_pickup_outcome(
            probe,
            target_x=101,
            target_y=100,
            fuel_before=900,
        )
        is None
    )

    self_state = _require_self_state(probe.get_world_state())
    self_state["fuel"] = 940

    assert pickup_phase.get_completed_pickup_outcome(
        probe,
        target_x=101,
        target_y=100,
        fuel_before=900,
    ) == ("picked_up_fuel", 1600, 940)


def test_wait_for_pickup_outcome_returns_timeout_when_fuel_never_changes() -> None:
    """Pickup wait returns timeout and current fuel after expiry."""
    clock = ReplayClock(1000)
    probe = _Probe(_make_world(100, 100, 700))
    page = _Page(clock)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    assert pickup_phase.wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=700,
        timeout_ms=200,
    ) == ("pickup_timeout", 1200, 700)


def test_run_tracked_pickup_phase_returns_immediate_pickup_without_move() -> None:
    """Pickup phase short-circuits when fuel was already credited."""
    clock = ReplayClock(1500)
    probe = _Probe(_make_world(100, 100, 800))
    page = _Page(clock)
    action_hooks.get_current_time_ms = clock

    def _drain(source: BufferedMessageSourceProtocol) -> int:
        _ = source
        self_state = _require_self_state(probe.get_world_state())
        self_state["fuel"] = 940
        return 0

    action_hooks.drain_buffered_messages = _drain

    move_cycle, pickup_cycle, pickup_started_ms, status, completion_ms, fuel_after = (
        pickup_phase.run_tracked_pickup_phase(
            page,
            probe,
            attempt_label="attempt-1",
            target_x=101,
            target_y=100,
            current_x=100,
            current_y=100,
            fuel_before_pickup=800,
            pickup_timeout_ms=500,
            dispatch_failure_error=RuntimeError,
        )
    )

    assert move_cycle == ActionPhaseCycleDict(phase="move", cycle_id=1, started_ms=1300)
    assert pickup_cycle == ActionPhaseCycleDict(phase="pickup", cycle_id=2, started_ms=1300)
    assert pickup_started_ms == 1500
    assert status == "picked_up_fuel"
    assert completion_ms == 1500
    assert fuel_after == 940
    assert probe._move_calls == []
    assert probe._ended == [move_cycle, pickup_cycle]
    assert probe.reset_count == 1


def test_run_tracked_pickup_phase_dispatches_move_and_waits_for_pickup() -> None:
    """Pickup phase dispatches movement and waits for a later fuel credit."""
    clock = ReplayClock(2000)
    probe = _Probe(_make_world(100, 100, 800))
    page = _Page(clock)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    def _on_wait() -> None:
        self_state = _require_self_state(probe.get_world_state())
        self_state["fuel"] = 950

    page.on_wait = _on_wait

    move_cycle, pickup_cycle, pickup_started_ms, status, completion_ms, fuel_after = (
        pickup_phase.run_tracked_pickup_phase(
            page,
            probe,
            attempt_label="attempt-1",
            target_x=102,
            target_y=100,
            current_x=100,
            current_y=100,
            fuel_before_pickup=800,
            pickup_timeout_ms=500,
            dispatch_failure_error=RuntimeError,
        )
    )

    assert move_cycle == ActionPhaseCycleDict(phase="move", cycle_id=1, started_ms=1300)
    assert pickup_cycle == ActionPhaseCycleDict(phase="pickup", cycle_id=2, started_ms=1300)
    assert pickup_started_ms == 2000
    assert status == "picked_up_fuel"
    assert completion_ms == 2100
    assert fuel_after == 950
    assert probe._move_calls == [(102, 100)]
    assert probe._ended == [move_cycle, pickup_cycle]
    assert probe.reset_count == 1


def test_run_tracked_pickup_phase_raises_on_move_dispatch_failure() -> None:
    """Pickup phase raises immediately when movement dispatch fails."""
    clock = ReplayClock(2500)
    probe = _Probe(_make_world(100, 100, 800), move_result=False)
    page = _Page(clock)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    with pytest.raises(
        RuntimeError,
        match="move_to command dispatch failed during fuel collection",
    ):
        pickup_phase.run_tracked_pickup_phase(
            page,
            probe,
            attempt_label="attempt-1",
            target_x=101,
            target_y=100,
            current_x=100,
            current_y=100,
            fuel_before_pickup=800,
            pickup_timeout_ms=500,
            dispatch_failure_error=RuntimeError,
        )

    assert probe._move_calls == [(101, 100)]
    assert probe._ended == [
        ActionPhaseCycleDict(phase="move", cycle_id=1, started_ms=1300),
        ActionPhaseCycleDict(phase="pickup", cycle_id=2, started_ms=1300),
    ]
    assert probe.reset_count == 0
