"""Tests for ``probe_single_movement_target``.

Every outcome one target pass can reach, including the map-open
queueing branches. ``test_movement_probe.py`` was 1,180 lines; helpers,
outcome waiting, and execution are now siblings.
"""

from __future__ import annotations

from typing import Literal

import pytest
from tests.action_lab._movement_probe_harness import (
    _make_world,
    _MapAlreadyOpenCDPSession,
    _SequencedWorld,
    _SingleTargetHarness,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot._test_hooks import (
    BufferedMessageSourceProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import movement_probe as movement_probe_module
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.movement_probe import (
    MovementOutcomeProbeProtocol,
    MovementProbeError,
)
from tankpit_bot.action_lab.types import TeleportTargetDict


def test_probe_single_movement_target_records_queue_map_open() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    after_world = _make_world(1700, 120, 121, 880)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    def _fake_wait_for_move_outcome(
        page: action_session.WaitPageProtocol,
        probe: MovementOutcomeProbeProtocol,
        *,
        target_x: int,
        target_y: int,
        move_started_ms: int,
        timeout_ms: int,
    ) -> tuple[Literal["arrived_exact", "move_timeout"], int, int, int, int]:
        _ = (page, target_x, target_y, move_started_ms, timeout_ms)
        if not isinstance(probe, _SingleTargetHarness):
            raise AssertionError("expected single-target harness")
        probe._world = after_world
        probe._self_state = after_world["self_state"]
        return ("arrived_exact", 1600, 600, 120, 121)

    movement_probe_module._wait_for_move_outcome = _fake_wait_for_move_outcome
    result = probe._probe_single_movement_target(
        TeleportTargetDict(label="move_1", x=120, y=121),
        move_timeout_ms=5000,
        queue_map_open_during_move=True,
        map_open_delay_ms=150,
        settle_delay_ms=200,
    )
    assert result["status"] == "arrived_exact"
    assert result["map_open_requested_ms"] == 1150
    assert result["map_open_message_timestamp_ms"] == 1150
    assert result["fuel_before"] == 900
    assert result["fuel_after"] == 880
    assert result["world_timestamp_after"] == 1700
    assert result["message_start_index"] == 0
    assert result["message_end_index"] == 1
    assert probe.move_calls == [(120, 121)]
    assert probe.open_map_calls == 1
    assert probe.reset_calls == 2
    assert page.waits == [150.0, 200.0]


def test_probe_single_movement_target_raises_when_cdp_session_unavailable() -> None:
    """The attempt fails fast when no CDP session is attached.

    The movement probe captures a page-client snapshot before and after
    each attempt; if CDP is unavailable there is no live source to read
    from and the probe must not silently proceed.
    """
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    probe._cdp = None
    action_hooks.get_current_time_ms = clock
    with pytest.raises(MovementProbeError, match="cdp session is unavailable"):
        probe._probe_single_movement_target(
            TeleportTargetDict(label="move_1", x=120, y=121),
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )


def test_probe_single_movement_target_records_snapshots_before_and_after() -> None:
    """The attempt result carries both bracketing page-client snapshots."""
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    after_world = _make_world(1700, 120, 121, 880)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    def _fake_wait_for_move_outcome(
        page: action_session.WaitPageProtocol,
        probe: MovementOutcomeProbeProtocol,
        *,
        target_x: int,
        target_y: int,
        move_started_ms: int,
        timeout_ms: int,
    ) -> tuple[Literal["arrived_exact", "move_timeout"], int, int, int, int]:
        _ = (page, target_x, target_y, move_started_ms, timeout_ms)
        if not isinstance(probe, _SingleTargetHarness):
            raise AssertionError("expected single-target harness")
        probe._world = after_world
        probe._self_state = after_world["self_state"]
        return ("arrived_exact", 1600, 600, 120, 121)

    movement_probe_module._wait_for_move_outcome = _fake_wait_for_move_outcome
    result = probe._probe_single_movement_target(
        TeleportTargetDict(label="move_1", x=120, y=121),
        move_timeout_ms=5000,
        queue_map_open_during_move=False,
        map_open_delay_ms=0,
        settle_delay_ms=0,
    )

    assert result["snapshot_before"]["client_present"] is True
    assert result["snapshot_after"]["client_present"] is True
    assert result["snapshot_after"]["timestamp_ms"] > result["snapshot_before"]["timestamp_ms"]


def test_probe_single_movement_target_raises_on_move_dispatch_failure() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    probe.move_result = False
    action_hooks.get_current_time_ms = clock
    with pytest.raises(MovementProbeError, match="move command dispatch failed"):
        probe._probe_single_movement_target(
            TeleportTargetDict(label="move_1", x=120, y=121),
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )


def test_probe_single_movement_target_raises_on_map_open_dispatch_failure() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    probe.open_map_result = False
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    with pytest.raises(MovementProbeError, match="map_open command dispatch failed"):
        probe._probe_single_movement_target(
            TeleportTargetDict(label="move_1", x=120, y=121),
            move_timeout_ms=5000,
            queue_map_open_during_move=True,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )


def test_probe_single_movement_target_skips_queued_map_open_when_map_already_open() -> None:
    """Mid-move queued ``map_open`` short-circuits when the JS client shows the map.

    Mirrors the ``run_tracked_acquisition_phase`` short-circuit: the
    wire ``CMD_MAP_OPEN`` is one-way, and re-sending it against an
    already-open overlay is a server-side no-op. The probe records the
    skip via ``map_open_requested_ms=None`` and refrains from calling
    ``self.open_map()``.
    """
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page, cdp=_MapAlreadyOpenCDPSession())
    action_hooks.get_current_time_ms = clock
    drain_calls = {"count": 0}

    def _count_drain(source: BufferedMessageSourceProtocol) -> int:
        _ = source
        drain_calls["count"] += 1
        return 0

    action_hooks.drain_buffered_messages = _count_drain

    def _fake_wait_for_move_outcome(
        page: action_session.WaitPageProtocol,
        probe: MovementOutcomeProbeProtocol,
        *,
        target_x: int,
        target_y: int,
        move_started_ms: int,
        timeout_ms: int,
    ) -> tuple[Literal["arrived_exact", "move_timeout"], int, int, int, int]:
        _ = (page, probe, target_x, target_y, move_started_ms, timeout_ms)
        return ("arrived_exact", 1800, 800, target_x, target_y)

    movement_probe_module._wait_for_move_outcome = _fake_wait_for_move_outcome

    result = probe._probe_single_movement_target(
        TeleportTargetDict(label="move_1", x=120, y=121),
        move_timeout_ms=5000,
        queue_map_open_during_move=True,
        map_open_delay_ms=150,
        settle_delay_ms=0,
    )

    assert probe.open_map_calls == 0
    assert result["map_open_requested_ms"] is None
    assert result["map_open_message_timestamp_ms"] is None
    assert result["snapshot_before"]["map_visible"] is True
    assert result["snapshot_after"]["map_visible"] is True
    assert drain_calls["count"] == 1
    assert result["status"] == "arrived_exact"


def test_probe_single_movement_target_raises_when_self_state_missing_after_outcome() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    def _fake_wait_for_missing_self(
        page: action_session.WaitPageProtocol,
        probe: MovementOutcomeProbeProtocol,
        *,
        target_x: int,
        target_y: int,
        move_started_ms: int,
        timeout_ms: int,
    ) -> tuple[Literal["arrived_exact", "move_timeout"], int, int, int, int]:
        _ = (page, target_x, target_y, move_started_ms, timeout_ms)
        if not isinstance(probe, _SingleTargetHarness):
            raise AssertionError("expected single-target harness")
        probe._self_state = None
        return ("move_timeout", 1600, 600, 118, 119)

    movement_probe_module._wait_for_move_outcome = _fake_wait_for_missing_self
    with pytest.raises(
        MovementProbeError,
        match="self state is unavailable after movement probe attempt",
    ):
        probe._probe_single_movement_target(
            TeleportTargetDict(label="move_1", x=120, y=121),
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )
