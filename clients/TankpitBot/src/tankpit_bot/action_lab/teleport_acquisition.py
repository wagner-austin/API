"""Shared pre-teleport acquisition helpers for action-lab probes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.teleport_phase import emit_command_dispatch_failure_diagnostic
from tankpit_bot.action_lab.types import TeleportPageSnapshotDict
from tankpit_bot.runtime_logging import emit_diagnostic


def start_teleport_page_snapshots(
    *,
    cdp: CDPSessionProtocol | None,
    capture_before_map_open: bool,
    unavailable_error: type[Exception],
    unavailable_message: str,
) -> tuple[
    list[TeleportPageSnapshotDict],
    Callable[
        [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
        TeleportPageSnapshotDict,
    ],
]:
    """Build shared page-snapshot state for one teleport attempt.

    Args:
        cdp: Active CDP session for the live page.
        capture_before_map_open: Whether to record the initial snapshot.
        unavailable_error: Exception type raised when CDP is unavailable.
        unavailable_message: Error message for missing CDP session failures.

    Returns:
        Mutable page snapshot list and a typed capture callback bound to the
        current CDP session.

    Raises:
        Exception: Raised via ``unavailable_error`` when the CDP session is
            unavailable.
    """
    if cdp is None:
        raise unavailable_error(unavailable_message)

    def _capture_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        """Capture one page snapshot for the requested teleport phase."""
        return action_session.capture_teleport_page_snapshot(cdp, phase)

    page_snapshots: list[TeleportPageSnapshotDict] = []
    if capture_before_map_open:
        page_snapshots.append(_capture_snapshot("before_map_open"))
    return (page_snapshots, _capture_snapshot)


def teleport_strategy_requires_map_sync(
    teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
) -> bool:
    """Return whether a teleport strategy waits for fresh world sync.

    Args:
        teleport_strategy: Selected teleport sequencing strategy.

    Returns:
        True when the strategy requires a pre-teleport fresh world sync.
    """
    return teleport_strategy == "sync_before_teleport"


def run_tracked_acquisition_phase(
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
    """Run one shared pre-teleport acquisition phase.

    Args:
        page: Page-like object used for waits.
        provider: World-state provider for sync waits.
        cdp: Active CDP session for page snapshots.
        send_command: Acquisition command dispatcher.
        command_name: Structured command name for failure logs.
        capture_before_map_open: Whether to record the initial snapshot.
        wait_for_sync: Whether to wait for fresh world sync after dispatch.
        sync_timeout_ms: Maximum fresh-sync wait.
        dispatch_failure_error: Exception type raised on dispatch failure.
        dispatch_failure_message: Error text for dispatch failure.
        unavailable_error: Exception type raised when CDP is unavailable.
        unavailable_message: Error text for missing CDP session.

    Returns:
        Tuple of ``(started_ms, sync_timestamp_ms, page_snapshots,
        capture_page_snapshot)``.

    Raises:
        Exception: Raised via ``dispatch_failure_error`` when acquisition
            dispatch fails.
        Exception: Raised via ``unavailable_error`` when the CDP session is
            unavailable.
    """
    page_snapshots, capture_page_snapshot = start_teleport_page_snapshots(
        cdp=cdp,
        capture_before_map_open=capture_before_map_open,
        unavailable_error=unavailable_error,
        unavailable_message=unavailable_message,
    )
    started_ms = action_hooks.get_current_time_ms()
    # The wire ``map_open`` command only opens the map; it does not toggle
    # closed (that requires a separate action). But re-sending it when the
    # map is already open does not produce a fresh map-sync response, so
    # the subsequent ``wait_for_world_sync`` either times out or returns a
    # stale sync and breaks the rest of the attempt. The before-map-open
    # snapshot reflects the live ``activeGame.map.h`` flag, so short-circuit
    # the dispatch and the sync wait when the goal state is already met.
    if (
        command_name == "map_open"
        and capture_before_map_open
        and page_snapshots
        and page_snapshots[-1]["map_visible"] is True
    ):
        emit_diagnostic(
            diagnostic_kind="map_open_skipped_already_open",
            origin="acquisition_phase",
            command_name=command_name,
        )
        return (started_ms, started_ms, page_snapshots, capture_page_snapshot)
    if not send_command():
        emit_command_dispatch_failure_diagnostic(command_name, dispatch_failure_message)
        raise dispatch_failure_error(dispatch_failure_message)
    if not wait_for_sync:
        return (started_ms, None, page_snapshots, capture_page_snapshot)
    return (
        started_ms,
        action_hooks.wait_for_world_sync(
            page,
            provider,
            started_ms,
            sync_timeout_ms,
        ),
        page_snapshots,
        capture_page_snapshot,
    )


__all__ = [
    "run_tracked_acquisition_phase",
    "start_teleport_page_snapshots",
    "teleport_strategy_requires_map_sync",
]
