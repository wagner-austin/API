"""Shared tracked teleport-attempt orchestration for action-lab probes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple, Protocol

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.teleport_acquisition import run_tracked_acquisition_phase
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterProtocol,
    TeleportPhaseProbeProtocol,
    run_tracked_teleport_command,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)


class TeleportAttemptProbeProtocol(TeleportPhaseProbeProtocol, Protocol):
    """Minimal probe interface required for one tracked teleport attempt."""

    def _start_action_phase(
        self,
        phase: Literal["teleport"],
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Start one teleport action phase."""


class TrackedTeleportAttempt(NamedTuple):
    """Shared tracked teleport-attempt state returned to probe-specific callers."""

    message_start_index: int
    teleport_cycle: ActionPhaseCycleDict
    acquisition_started_ms: int
    acquisition_sync_timestamp_ms: int | None
    page_snapshots: list[TeleportPageSnapshotDict]
    capture_page_snapshot: Callable[
        [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
        TeleportPageSnapshotDict,
    ]
    teleport_result: TeleportAttemptResultDict | None
    teleport_started_ms: int | None


def run_tracked_teleport_attempt(
    page: action_session.WaitPageProtocol,
    probe: TeleportAttemptProbeProtocol,
    target: TeleportTargetDict,
    *,
    cdp: CDPSessionProtocol | None,
    attempt_label: str,
    fuel_before: int,
    world_timestamp_before: int,
    send_acquisition_command: Callable[[], bool],
    acquisition_command_name: str,
    capture_before_map_open: bool,
    wait_for_acquisition_sync: bool,
    acquisition_timeout_ms: int,
    teleport_timeout_ms: int,
    wait_for_outcome: TeleportOutcomeWaiterProtocol,
    dispatch_failure_error: type[Exception],
    acquisition_dispatch_failure_message: str,
    teleport_dispatch_failure_message: str,
    unavailable_error: type[Exception],
    unavailable_message: str,
    unexpected_result_error: type[Exception],
    unexpected_result_message: str,
    reset_to_idle_before_start: bool = True,
) -> TrackedTeleportAttempt:
    """Run one shared acquisition-plus-teleport attempt.

    Args:
        page: Page-like object used for waits.
        probe: Probe implementation dispatching acquisition and teleport commands.
        target: Requested teleport destination.
        cdp: Active CDP session for page snapshots.
        attempt_label: Stable action label used for the teleport phase.
        fuel_before: Fuel value before the attempt begins.
        world_timestamp_before: World timestamp before the attempt begins.
        send_acquisition_command: Callable that dispatches the acquisition command.
        acquisition_command_name: Structured acquisition command name.
        capture_before_map_open: Whether to capture the initial page snapshot.
        wait_for_acquisition_sync: Whether to wait for fresh world sync after
            acquisition dispatch.
        acquisition_timeout_ms: Maximum acquisition fresh-sync wait.
        teleport_timeout_ms: Maximum teleport outcome wait.
        wait_for_outcome: Teleport outcome waiter implementation.
        dispatch_failure_error: Exception type raised on command dispatch failures.
        acquisition_dispatch_failure_message: Error text for acquisition dispatch failure.
        teleport_dispatch_failure_message: Error text for teleport dispatch failure.
        unavailable_error: Exception type raised when the CDP session is unavailable.
        unavailable_message: Error text for missing CDP session failures.
        unexpected_result_error: Exception type raised for impossible teleport results.
        unexpected_result_message: Error text for impossible teleport results.
        reset_to_idle_before_start: Whether to force the probe into idle before
            opening the next attempt window.

    Returns:
        Shared tracked attempt state. ``teleport_result`` and ``teleport_started_ms``
        are ``None`` when acquisition sync timed out before teleport dispatch.

    Raises:
        Exception: Raised via the provided error types when acquisition or teleport
            dispatch fails, when the CDP session is unavailable, or when teleport
            returns an impossible result.
    """
    if reset_to_idle_before_start:
        probe._reset_probe_state_to_idle()
    message_start_index = len(probe.messages)
    teleport_cycle = probe._start_action_phase("teleport", attempt_label=attempt_label)
    (
        acquisition_started_ms,
        acquisition_sync_timestamp_ms,
        page_snapshots,
        capture_page_snapshot,
    ) = run_tracked_acquisition_phase(
        page,
        probe,
        cdp=cdp,
        send_command=send_acquisition_command,
        command_name=acquisition_command_name,
        capture_before_map_open=capture_before_map_open,
        wait_for_sync=wait_for_acquisition_sync,
        sync_timeout_ms=acquisition_timeout_ms,
        dispatch_failure_error=dispatch_failure_error,
        dispatch_failure_message=acquisition_dispatch_failure_message,
        unavailable_error=unavailable_error,
        unavailable_message=unavailable_message,
    )
    if wait_for_acquisition_sync and acquisition_sync_timestamp_ms is None:
        return TrackedTeleportAttempt(
            message_start_index=message_start_index,
            teleport_cycle=teleport_cycle,
            acquisition_started_ms=acquisition_started_ms,
            acquisition_sync_timestamp_ms=None,
            page_snapshots=page_snapshots,
            capture_page_snapshot=capture_page_snapshot,
            teleport_result=None,
            teleport_started_ms=None,
        )
    teleport_result, teleport_started_ms = run_tracked_teleport_command(
        page,
        probe,
        target,
        teleport_cycle=teleport_cycle,
        message_start_index=message_start_index,
        map_open_started_ms=acquisition_started_ms,
        map_sync_timestamp_ms=acquisition_sync_timestamp_ms,
        fuel_before=fuel_before,
        world_timestamp_before=world_timestamp_before,
        timeout_ms=teleport_timeout_ms,
        page_snapshots=page_snapshots,
        capture_page_snapshot=capture_page_snapshot,
        wait_for_outcome=wait_for_outcome,
        dispatch_failure_error=dispatch_failure_error,
        dispatch_failure_message=teleport_dispatch_failure_message,
    )
    if teleport_result["status"] == "map_sync_timeout":
        raise unexpected_result_error(unexpected_result_message)
    return TrackedTeleportAttempt(
        message_start_index=message_start_index,
        teleport_cycle=teleport_cycle,
        acquisition_started_ms=acquisition_started_ms,
        acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
        page_snapshots=page_snapshots,
        capture_page_snapshot=capture_page_snapshot,
        teleport_result=teleport_result,
        teleport_started_ms=teleport_started_ms,
    )


__all__ = [
    "TeleportAttemptProbeProtocol",
    "TrackedTeleportAttempt",
    "run_tracked_teleport_attempt",
]
