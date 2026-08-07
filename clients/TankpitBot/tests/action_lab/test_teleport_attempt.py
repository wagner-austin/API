"""Tests for ``run_tracked_teleport_attempt``: the acquisition path.

``test_teleport_attempt.py`` was 701 lines; the rejection and skip
outcomes are now a sibling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from tests.action_lab._teleport_attempt_harness import (
    _Page,
    _Probe,
    _result,
    _snapshot,
    _target,
    _WaitForOutcome,
)

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab import teleport_attempt
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterProtocol,
    TeleportPhaseProbeProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)


def test_run_tracked_teleport_attempt_runs_acquisition_then_teleport() -> None:
    """Shared helper returns full tracked state for a successful attempt."""
    original_acquisition = teleport_attempt.run_acquisition_phase
    original_teleport = teleport_attempt.run_teleport_phase
    expected_page = _Page()
    expected_probe = _Probe()
    expected_target = _target()
    dispatch_calls: list[str] = []
    teleport_calls: list[int] = []

    def _dispatch() -> bool:
        dispatch_calls.append("acquire")
        return True

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return _snapshot(phase)

    def _run_acquisition(
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
        Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ]:
        _ = (
            cdp,
            command_name,
            capture_before_map_open,
            wait_for_sync,
            sync_timeout_ms,
            dispatch_failure_error,
            dispatch_failure_message,
            unavailable_error,
            unavailable_message,
        )
        assert page is expected_page
        assert provider is expected_probe
        assert send_command()
        return (1500, 1700, [_snapshot("before_map_open")], _capture_page_snapshot)

    def _run_teleport(
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
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "",
    ) -> tuple[TeleportAttemptResultDict, int]:
        _ = (
            teleport_cycle,
            map_open_started_ms,
            map_sync_timestamp_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            capture_page_snapshot,
            wait_for_outcome,
            dispatch_failure_error,
            dispatch_failure_message,
        )
        assert page is expected_page
        assert probe is expected_probe
        assert target == expected_target
        assert message_start_index == 1
        assert [snapshot["phase"] for snapshot in page_snapshots] == ["before_map_open"]
        teleport_calls.append(message_start_index)
        return (_result(target), 1800)

    teleport_attempt.run_acquisition_phase = _run_acquisition
    teleport_attempt.run_teleport_phase = _run_teleport
    try:
        attempt = teleport_attempt.run_tracked_teleport_attempt(
            expected_page,
            expected_probe,
            expected_target,
            cdp=None,
            attempt_label=expected_target["label"],
            fuel_before=1100,
            world_timestamp_before=900,
            send_acquisition_command=_dispatch,
            acquisition_command_name="map_open",
            capture_before_map_open=True,
            wait_for_acquisition_sync=True,
            acquisition_timeout_ms=4000,
            teleport_timeout_ms=10000,
            wait_for_outcome=_WaitForOutcome(),
            dispatch_failure_error=RuntimeError,
            acquisition_dispatch_failure_message="acquisition failed",
            teleport_dispatch_failure_message="teleport failed",
            unavailable_error=RuntimeError,
            unavailable_message="missing",
            unexpected_result_error=RuntimeError,
            unexpected_result_message="impossible",
        )
    finally:
        teleport_attempt.run_acquisition_phase = original_acquisition
        teleport_attempt.run_teleport_phase = original_teleport

    assert expected_probe.reset_idle_calls == 1
    assert expected_probe.started_cycles == [("teleport", "target")]
    assert dispatch_calls == ["acquire"]
    assert teleport_calls == [1]
    assert attempt.message_start_index == 1
    assert attempt.teleport_cycle["cycle_id"] == 7
    assert attempt.acquisition_started_ms == 1500
    assert attempt.acquisition_sync_timestamp_ms == 1700
    assert attempt.teleport_started_ms == 1800
    assert attempt.teleport_result == _result(expected_target)


def test_run_tracked_teleport_attempt_returns_early_on_acquisition_timeout() -> None:
    """Shared helper returns without teleport dispatch when acquisition times out."""
    original_acquisition = teleport_attempt.run_acquisition_phase
    original_teleport = teleport_attempt.run_teleport_phase
    expected_page = _Page()
    expected_probe = _Probe()
    expected_target = _target()

    def _dispatch() -> bool:
        return True

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return _snapshot(phase)

    def _run_acquisition(
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
        Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ]:
        _ = (
            cdp,
            command_name,
            capture_before_map_open,
            wait_for_sync,
            sync_timeout_ms,
            dispatch_failure_error,
            dispatch_failure_message,
            unavailable_error,
            unavailable_message,
        )
        assert page is expected_page
        assert provider is expected_probe
        assert send_command()
        return (1500, None, [_snapshot("before_map_open")], _capture_page_snapshot)

    def _run_teleport(
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
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "",
    ) -> tuple[TeleportAttemptResultDict, int]:
        _ = (
            page,
            probe,
            target,
            teleport_cycle,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
            wait_for_outcome,
            dispatch_failure_error,
            dispatch_failure_message,
        )
        raise AssertionError("teleport dispatch should not run after acquisition timeout")

    teleport_attempt.run_acquisition_phase = _run_acquisition
    teleport_attempt.run_teleport_phase = _run_teleport
    try:
        attempt = teleport_attempt.run_tracked_teleport_attempt(
            expected_page,
            expected_probe,
            expected_target,
            cdp=None,
            attempt_label=expected_target["label"],
            fuel_before=1100,
            world_timestamp_before=900,
            send_acquisition_command=_dispatch,
            acquisition_command_name="map_open",
            capture_before_map_open=True,
            wait_for_acquisition_sync=True,
            acquisition_timeout_ms=4000,
            teleport_timeout_ms=10000,
            wait_for_outcome=_WaitForOutcome(),
            dispatch_failure_error=RuntimeError,
            acquisition_dispatch_failure_message="acquisition failed",
            teleport_dispatch_failure_message="teleport failed",
            unavailable_error=RuntimeError,
            unavailable_message="missing",
            unexpected_result_error=RuntimeError,
            unexpected_result_message="impossible",
        )
    finally:
        teleport_attempt.run_acquisition_phase = original_acquisition
        teleport_attempt.run_teleport_phase = original_teleport

    assert expected_probe.reset_idle_calls == 1
    assert attempt.message_start_index == 1
    assert attempt.acquisition_started_ms == 1500
    assert attempt.acquisition_sync_timestamp_ms is None
    assert attempt.teleport_result is None
    assert attempt.teleport_started_ms is None
