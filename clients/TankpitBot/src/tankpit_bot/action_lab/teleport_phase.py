"""Shared teleport command-phase helpers for action-lab probes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from platform_core.json_utils import JSONObject, require_str
from platform_core.logging import get_logger
from typing_extensions import TypedDict, Unpack

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.runtime_logging import emit_diagnostic

log = get_logger(__name__)


class CommandDispatchFailureDiagnosticDict(TypedDict):
    """Structured payload for a single ``command_dispatch_failure`` diagnostic.

    Attributes:
        command: Command name that failed to dispatch.
        detail: Human-readable failure detail.
    """

    command: str
    detail: str


def encode_command_dispatch_failure_diagnostic(
    payload: CommandDispatchFailureDiagnosticDict,
) -> JSONObject:
    """Encode a command_dispatch_failure diagnostic payload to JSON.

    Args:
        payload: Structured diagnostic payload.

    Returns:
        JSON-compatible representation.
    """
    return {"command": payload["command"], "detail": payload["detail"]}


def decode_command_dispatch_failure_diagnostic(
    data: JSONObject,
) -> CommandDispatchFailureDiagnosticDict:
    """Decode a command_dispatch_failure diagnostic payload from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated payload.
    """
    return CommandDispatchFailureDiagnosticDict(
        command=require_str(data, "command"),
        detail=require_str(data, "detail"),
    )


class TeleportPhaseProbeProtocol(action_session.BufferedWorldStateProviderProtocol, Protocol):
    """Minimal probe interface required to run one teleport command phase."""

    def teleport_to(self, x: int, y: int) -> bool:
        """Dispatch one teleport command."""

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """Close one active action phase cycle."""

    def _reset_probe_state_to_idle(self) -> None:
        """Reset probe state to a clean idle phase."""


class TeleportOutcomeWaiterKwargs(TypedDict):
    """Typed keyword arguments for one teleport outcome wait."""

    teleport_cycle_id: int
    message_start_index: int
    map_open_started_ms: int
    map_sync_timestamp_ms: int | None
    teleport_started_ms: int
    fuel_before: int
    world_timestamp_before: int
    timeout_ms: int
    page_snapshots: list[TeleportPageSnapshotDict]
    capture_page_snapshot: Callable[
        [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
        TeleportPageSnapshotDict,
    ]


class TeleportOutcomeWaiterProtocol(Protocol):
    """Callable protocol for waiting on one teleport outcome."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        **kwargs: Unpack[TeleportOutcomeWaiterKwargs],
    ) -> TeleportAttemptResultDict:
        """Wait for a terminal teleport outcome and return the typed result."""


def emit_command_dispatch_failure_diagnostic(command: str, detail: str) -> None:
    """Emit a structured ``command_dispatch_failure`` diagnostic event.

    Args:
        command: Command name that failed to dispatch.
        detail: Human-readable failure detail.
    """
    emit_diagnostic(
        diagnostic_kind="command_dispatch_failure",
        command=command,
        detail=detail,
    )


def _log_command_dispatch_failure(command: str, detail: str) -> None:
    """Log and emit a command dispatch failure.

    Args:
        command: Command name that failed to dispatch.
        detail: Human-readable failure detail.
    """
    log.warning("command dispatch failed: %s — %s", command, detail)
    emit_command_dispatch_failure_diagnostic(command, detail)


def run_tracked_teleport_command(
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
    dispatch_failure_message: str = "teleport command dispatch failed",
) -> tuple[TeleportAttemptResultDict, int]:
    """Run one tracked teleport command phase and wait for its outcome.

    Args:
        page: Page used for polling waits.
        probe: Probe implementation dispatching the teleport command.
        target: Requested teleport destination.
        teleport_cycle: Already-started teleport action cycle.
        message_start_index: Raw message start index for diagnostics.
        map_open_started_ms: Timestamp when the map-open/acquisition phase began.
        map_sync_timestamp_ms: Optional world-sync timestamp before teleport.
        fuel_before: Fuel value before teleport.
        world_timestamp_before: World timestamp before teleport.
        timeout_ms: Maximum teleport wait time.
        page_snapshots: Mutable page snapshot list for this attempt.
        capture_page_snapshot: Snapshot callback bound to the live page.
        wait_for_outcome: Outcome waiter implementation.
        dispatch_failure_error: Exception type raised on dispatch failure.
        dispatch_failure_message: Error text for dispatch failure.

    Returns:
        Tuple of ``(result, teleport_started_ms)``.

    Raises:
        Exception: Raised via ``dispatch_failure_error`` if the teleport command
            fails to dispatch.
    """
    teleport_started_ms = action_hooks.get_current_time_ms()
    page_snapshots.append(capture_page_snapshot("before_teleport"))
    if not probe.teleport_to(target["x"], target["y"]):
        emit_command_dispatch_failure_diagnostic("teleport", dispatch_failure_message)
        probe._end_action_phase(teleport_cycle)
        raise dispatch_failure_error(dispatch_failure_message)
    result = wait_for_outcome(
        page,
        probe,
        target,
        teleport_cycle_id=teleport_cycle["cycle_id"],
        message_start_index=message_start_index,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        fuel_before=fuel_before,
        world_timestamp_before=world_timestamp_before,
        timeout_ms=timeout_ms,
        page_snapshots=page_snapshots,
        capture_page_snapshot=capture_page_snapshot,
    )
    probe._end_action_phase(teleport_cycle)
    probe._reset_probe_state_to_idle()
    return (result, teleport_started_ms)


__all__ = [
    "CommandDispatchFailureDiagnosticDict",
    "TeleportOutcomeWaiterKwargs",
    "TeleportOutcomeWaiterProtocol",
    "TeleportPhaseProbeProtocol",
    "_log_command_dispatch_failure",
    "decode_command_dispatch_failure_diagnostic",
    "emit_command_dispatch_failure_diagnostic",
    "encode_command_dispatch_failure_diagnostic",
    "run_tracked_teleport_command",
]
