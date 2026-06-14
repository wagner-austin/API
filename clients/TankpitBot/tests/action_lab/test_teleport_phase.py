"""Tests for shared teleport command-phase helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
from tests.action_lab._replay_core import ReplayClock

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab import teleport_phase
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.state import WorldStateDict, make_empty_world_state
from tankpit_bot.types import CapturedMessage


class _Probe:
    """Minimal teleport probe fake for shared command-phase tests."""

    def __init__(self, *, dispatch_succeeds: bool) -> None:
        """Initialize the fake probe."""
        self.dispatch_succeeds = dispatch_succeeds
        self.teleports: list[tuple[int, int]] = []
        self.end_cycles: list[ActionPhaseCycleDict] = []
        self.reset_idle_calls = 0
        self._messages: list[CapturedMessage] = []
        self._cdp_message_buffer: list[str] = []

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return an empty message buffer."""
        return self._messages

    @property
    def magic(self) -> str:
        """Return a stable fake magic key."""
        return "magic"

    def get_world_state(self) -> WorldStateDict:
        """Return an empty world state."""
        return make_empty_world_state()

    def teleport_to(self, x: int, y: int) -> bool:
        """Record one teleport dispatch attempt."""
        self.teleports.append((x, y))
        return self.dispatch_succeeds

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """Record one phase end."""
        self.end_cycles.append(cycle)

    def _reset_probe_state_to_idle(self) -> None:
        """Record one idle reset."""
        self.reset_idle_calls += 1


def _target() -> TeleportTargetDict:
    """Build a sample teleport target."""
    return TeleportTargetDict(label="target", x=147, y=110)


def _snapshot(
    phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
) -> TeleportPageSnapshotDict:
    """Build a sample page snapshot."""
    return TeleportPageSnapshotDict(
        phase=phase,
        timestamp_ms=5000,
        client_present=True,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=10,
        last_page_client_send_age_ms=20,
        last_bot_send_age_ms=30,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        world_collections={},
        map_fields={},
    )


class _Page:
    """Minimal page fake satisfying the action-lab wait protocol."""

    def wait_for_timeout(self, timeout: float) -> None:
        """Record no-op waits."""
        _ = timeout


class _SuccessfulWaitForOutcome(teleport_phase.TeleportOutcomeWaiterProtocol):
    """Typed teleport outcome waiter fake for successful dispatches."""

    def __init__(
        self,
        *,
        page: _Page,
        probe: _Probe,
        target: TeleportTargetDict,
        wait_calls: list[int],
    ) -> None:
        """Initialize the waiter fake."""
        self._page = page
        self._probe = probe
        self._target = target
        self._wait_calls = wait_calls

    def __call__(
        self,
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target_arg: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        """Return one successful teleport result."""
        _ = (
            teleport_cycle_id,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            capture_page_snapshot,
        )
        assert page_arg is self._page
        assert provider is self._probe
        assert target_arg == self._target
        self._wait_calls.append(teleport_started_ms)
        page_snapshots.append(_snapshot("landed"))
        return _result(self._target, teleport_started_ms)


class _FailingWaitForOutcome(teleport_phase.TeleportOutcomeWaiterProtocol):
    """Typed teleport outcome waiter fake for dispatch-failure paths."""

    def __call__(
        self,
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target_arg: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        """Fail if the teleport waiter is called unexpectedly."""
        _ = (
            page_arg,
            provider,
            target_arg,
            teleport_cycle_id,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        raise AssertionError("wait_for_outcome should not run after dispatch failure")


def _result(target: TeleportTargetDict, teleport_started_ms: int) -> TeleportAttemptResultDict:
    """Build a sample teleport outcome result."""
    return TeleportAttemptResultDict(
        target=target,
        teleport_cycle_id=4,
        status="landed_exact",
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=teleport_started_ms,
        completion_timestamp_ms=1600,
        map_sync_elapsed_ms=200,
        teleport_elapsed_ms=300,
        fuel_before=1100,
        fuel_after=1004,
        world_timestamp_before=900,
        world_timestamp_after=1500,
        landed_signal_received=True,
        landed_x=147,
        landed_y=110,
        message_start_index=3,
        message_end_index=8,
        page_snapshots=[_snapshot("before_teleport"), _snapshot("landed")],
    )


def test_run_tracked_teleport_command_waits_and_resets_state() -> None:
    """Shared teleport command runner waits for outcome and resets idle state."""
    clock = ReplayClock(1400)
    original_clock = action_hooks.get_current_time_ms
    action_hooks.get_current_time_ms = clock
    page = _Page()
    probe = _Probe(dispatch_succeeds=True)
    target = _target()
    page_snapshots: list[TeleportPageSnapshotDict] = []
    capture_calls: list[str] = []
    wait_calls: list[int] = []

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        capture_calls.append(phase)
        return _snapshot(phase)

    try:
        result, started_ms = teleport_phase.run_tracked_teleport_command(
            page,
            probe,
            target,
            teleport_cycle=ActionPhaseCycleDict(phase="teleport", cycle_id=4, started_ms=1300),
            message_start_index=3,
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            fuel_before=1100,
            world_timestamp_before=900,
            timeout_ms=5000,
            page_snapshots=page_snapshots,
            capture_page_snapshot=_capture_page_snapshot,
            wait_for_outcome=_SuccessfulWaitForOutcome(
                page=page,
                probe=probe,
                target=target,
                wait_calls=wait_calls,
            ),
            dispatch_failure_error=RuntimeError,
        )
    finally:
        action_hooks.get_current_time_ms = original_clock

    assert started_ms == 1400
    assert result["teleport_started_ms"] == 1400
    assert probe.teleports == [(147, 110)]
    assert probe.end_cycles == [ActionPhaseCycleDict(phase="teleport", cycle_id=4, started_ms=1300)]
    assert probe.reset_idle_calls == 1
    assert capture_calls == ["before_teleport"]
    assert wait_calls == [1400]


def test_run_tracked_teleport_command_raises_on_dispatch_failure() -> None:
    """Shared teleport command runner raises immediately on dispatch failure."""
    clock = ReplayClock(2000)
    original_clock = action_hooks.get_current_time_ms
    action_hooks.get_current_time_ms = clock
    page = _Page()
    probe = _Probe(dispatch_succeeds=False)
    target = _target()

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return _snapshot(phase)

    try:
        with pytest.raises(RuntimeError, match="dispatch failed"):
            teleport_phase.run_tracked_teleport_command(
                page,
                probe,
                target,
                teleport_cycle=ActionPhaseCycleDict(phase="teleport", cycle_id=7, started_ms=1900),
                message_start_index=0,
                map_open_started_ms=1800,
                map_sync_timestamp_ms=1850,
                fuel_before=1000,
                world_timestamp_before=1700,
                timeout_ms=5000,
                page_snapshots=[],
                capture_page_snapshot=_capture_page_snapshot,
                wait_for_outcome=_FailingWaitForOutcome(),
                dispatch_failure_error=RuntimeError,
                dispatch_failure_message="dispatch failed",
            )
    finally:
        action_hooks.get_current_time_ms = original_clock

    assert probe.teleports == [(147, 110)]
    assert probe.end_cycles == [ActionPhaseCycleDict(phase="teleport", cycle_id=7, started_ms=1900)]
    assert probe.reset_idle_calls == 0


def test_command_dispatch_failure_diagnostic_round_trip() -> None:
    """``CommandDispatchFailureDiagnosticDict`` round-trips through JSON encoding."""
    from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

    payload = teleport_phase.CommandDispatchFailureDiagnosticDict(
        command="teleport",
        detail="websocket not ready",
    )

    encoded = teleport_phase.encode_command_dispatch_failure_diagnostic(payload)
    decoded = teleport_phase.decode_command_dispatch_failure_diagnostic(
        narrow_json_to_dict(load_json_str(dump_json_str(encoded)))
    )

    assert decoded == payload
