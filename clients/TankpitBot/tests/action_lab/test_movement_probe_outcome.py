"""Tests for ``wait_for_move_outcome``.

Arrival, timeout, and both missing-self-state failure modes.
"""

from __future__ import annotations

import pytest
from tests.action_lab._movement_probe_harness import (
    _make_world,
    _MissingSelfWaitProbe,
    _MoveWaitProbe,
    _SequencedWorld,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.movement_probe import (
    MovementProbeError,
    _wait_for_move_outcome,
)
from tankpit_bot.state import (
    make_self_state,
)


def test_wait_for_move_outcome_returns_arrived_exact() -> None:
    clock = ReplayClock(1000)
    worlds = _SequencedWorld(
        [
            _make_world(1000, 100, 100, 900),
            _make_world(1100, 101, 100, 899),
            _make_world(1200, 120, 121, 890),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=worlds.advance)
    probe = _MoveWaitProbe(worlds)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source, ws: 0
    status, completion_ms, elapsed_ms, settled_x, settled_y = _wait_for_move_outcome(
        page,
        probe,
        target_x=120,
        target_y=121,
        move_started_ms=1000,
        timeout_ms=5000,
    )
    assert status == "arrived_exact"
    assert completion_ms == 1200
    assert elapsed_ms == 200
    assert (settled_x, settled_y) == (120, 121)


def test_wait_for_move_outcome_returns_timeout() -> None:
    clock = ReplayClock(1000)
    worlds = _SequencedWorld(
        [
            _make_world(1000, 100, 100, 900),
            _make_world(1100, 101, 100, 899),
            _make_world(1200, 101, 101, 898),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=worlds.advance)
    probe = _MoveWaitProbe(worlds)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source, ws: 0
    status, completion_ms, elapsed_ms, settled_x, settled_y = _wait_for_move_outcome(
        page,
        probe,
        target_x=120,
        target_y=121,
        move_started_ms=1000,
        timeout_ms=250,
    )
    assert status == "move_timeout"
    assert completion_ms == 1300
    assert elapsed_ms == 300
    assert (settled_x, settled_y) == (101, 101)


def test_wait_for_move_outcome_raises_when_self_state_disappears_mid_wait() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _MissingSelfWaitProbe([make_self_state(1, 100, 100, 2, 1, 900, 5), None])
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source, ws: 0
    with pytest.raises(MovementProbeError, match="disappeared while waiting for movement"):
        _wait_for_move_outcome(
            page,
            probe,
            target_x=120,
            target_y=121,
            move_started_ms=1000,
            timeout_ms=5000,
        )


def test_wait_for_move_outcome_raises_when_self_state_missing_after_timeout() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _MissingSelfWaitProbe([None])
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source, ws: 0
    with pytest.raises(MovementProbeError, match="disappeared after movement timeout"):
        _wait_for_move_outcome(
            page,
            probe,
            target_x=120,
            target_y=121,
            move_started_ms=1000,
            timeout_ms=0,
        )
