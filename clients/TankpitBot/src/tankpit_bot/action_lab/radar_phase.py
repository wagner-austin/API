"""Shared radar command-phase helpers for action-lab probes."""

from __future__ import annotations

from typing import Literal, Protocol

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.teleport_phase import emit_command_dispatch_failure_diagnostic


class RadarPhaseProbeProtocol(action_session.BufferedWorldStateProviderProtocol, Protocol):
    """Minimal probe interface required to run one tracked radar phase."""

    def use_radar(self) -> bool:
        """Dispatch one radar command."""

    def _start_action_phase(
        self,
        phase: Literal["radar"],
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Start one action phase cycle."""

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """Close one active action phase."""

    def _reset_probe_state_to_idle(self) -> None:
        """Reset probe state to idle after the phase settles."""


def clear_stale_radar_completion() -> None:
    """Drain any leaked radar-complete confirmations before a new scan."""
    while action_hooks.check_and_clear_radar_scan_complete():
        continue


def run_tracked_radar_phase(
    page: action_session.WaitPageProtocol,
    probe: RadarPhaseProbeProtocol,
    *,
    attempt_label: str,
    timeout_ms: int,
    dispatch_failure_error: type[Exception],
    dispatch_failure_message: str = "radar command dispatch failed",
) -> tuple[ActionPhaseCycleDict, int, int | None]:
    """Run one tracked radar phase and wait for sync completion.

    Args:
        page: Page used for polling waits.
        probe: Probe implementation dispatching radar.
        attempt_label: Attempt label attached to the phase cycle.
        timeout_ms: Maximum radar sync wait.
        dispatch_failure_error: Exception type raised on dispatch failure.
        dispatch_failure_message: Error text for dispatch failure.

    Returns:
        Tuple of ``(radar_cycle, radar_started_ms, radar_sync_timestamp_ms)``.

    Raises:
        Exception: Raised via ``dispatch_failure_error`` if the radar command
            fails to dispatch.
    """
    action_hooks.drain_buffered_messages(probe)
    clear_stale_radar_completion()
    radar_cycle = probe._start_action_phase("radar", attempt_label=attempt_label)
    radar_started_ms = action_hooks.get_current_time_ms()
    if not probe.use_radar():
        emit_command_dispatch_failure_diagnostic("radar", dispatch_failure_message)
        probe._end_action_phase(radar_cycle)
        raise dispatch_failure_error(dispatch_failure_message)
    radar_sync_timestamp_ms = action_session.wait_for_radar_sync(
        page,
        probe,
        radar_started_ms,
        timeout_ms,
    )
    probe._end_action_phase(radar_cycle)
    probe._reset_probe_state_to_idle()
    return (radar_cycle, radar_started_ms, radar_sync_timestamp_ms)


__all__ = [
    "RadarPhaseProbeProtocol",
    "clear_stale_radar_completion",
    "run_tracked_radar_phase",
]
