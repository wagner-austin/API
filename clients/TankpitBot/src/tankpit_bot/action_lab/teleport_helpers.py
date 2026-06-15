"""Teleport probe helper functions — diagnostics, targets, outcome waiting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.teleport_acquisition import (
    start_teleport_page_snapshots as _shared_start_teleport_page_snapshots,
)
from tankpit_bot.action_lab.teleport_acquisition import (
    teleport_strategy_requires_map_sync as _shared_teleport_strategy_requires_map_sync,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportProbeSessionDict,
    TeleportTargetDict,
)
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.decoders import decode_message

log = get_logger(__name__)
_TELEPORT_POLL_INTERVAL_MS = 100.0


class TeleportProbeError(Exception):
    """Raised when the teleport probe cannot proceed."""


def _start_teleport_page_snapshots(
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
    """Build shared page-snapshot state for a teleport attempt."""
    return _shared_start_teleport_page_snapshots(
        cdp=cdp,
        capture_before_map_open=capture_before_map_open,
        unavailable_error=unavailable_error,
        unavailable_message=unavailable_message,
    )


def _format_attempt_window_entries(
    provider: action_session.BufferedWorldStateProviderProtocol,
    *,
    message_start_index: int,
    direction: Literal["sent", "received"],
    limit: int = 6,
) -> str:
    """Return a compact decoded message window summary for one attempt."""
    messages = provider.messages
    magic = provider.magic
    entries: list[str] = []
    for index, message in enumerate(messages[message_start_index:], start=message_start_index):
        if message["direction"] != direction:
            continue
        decoded = decode_message(message["payload"], direction, magic)
        if direction == "sent":
            sent_origin = message.get("sent_origin")
            sent_label = message.get("sent_label")
            if sent_origin == "bot_injected" and sent_label is not None and sent_label != "":
                decoded = f"{decoded} origin=bot_injected label={sent_label}"
            elif sent_origin == "page_client":
                decoded = f"{decoded} origin=page_client"
        entries.append(f"{index}:{decoded}")
        if len(entries) >= limit:
            break
    total = sum(
        1 for message in messages[message_start_index:] if message["direction"] == direction
    )
    if not entries:
        return "none"
    if total > len(entries):
        entries.append(f"...+{total - len(entries)} more")
    return " | ".join(entries)


def _format_page_snapshots(snapshots: list[TeleportPageSnapshotDict]) -> str:
    """Return a compact page-state diagnostic summary."""
    if not snapshots:
        return "none"
    entries: list[str] = []
    for snapshot in snapshots:
        entries.append(
            f"{snapshot['phase']}("
            f"client_present={snapshot['client_present']} "
            f"map_visible={snapshot['map_visible']} "
            f"state={snapshot['client_state']} "
            f"busy={snapshot['client_busy']} "
            f"pending={snapshot['pending_actions']} "
            f"heartbeat_age_ms={snapshot['heartbeat_age_ms']} "
            f"page_send_age_ms={snapshot['last_page_client_send_age_ms']} "
            f"bot_send_age_ms={snapshot['last_bot_send_age_ms']} "
            f"ws_ready_state={snapshot['ws_ready_state']} "
            f"queue={snapshot['sent_frame_meta_queue_length']})"
        )
    return " | ".join(entries)


def _log_teleport_attempt_diagnostic(
    provider: action_session.BufferedWorldStateProviderProtocol,
    *,
    target: TeleportTargetDict,
    teleport_cycle_id: int,
    status: str,
    message_start_index: int,
    page_snapshots: list[TeleportPageSnapshotDict],
) -> None:
    """Emit one structured diagnostic line for a teleport attempt."""
    emit_diagnostic(
        diagnostic_kind="teleport_attempt",
        target_x=target["x"],
        target_y=target["y"],
        cycle=teleport_cycle_id,
        status=status,
        sent=_format_attempt_window_entries(
            provider,
            message_start_index=message_start_index,
            direction="sent",
        ),
        received=_format_attempt_window_entries(
            provider,
            message_start_index=message_start_index,
            direction="received",
        ),
        page=_format_page_snapshots(page_snapshots),
    )


def _find_map_data_message_index(
    provider: action_session.BufferedWorldStateProviderProtocol,
    *,
    message_start_index: int,
    scan_start_index: int,
) -> int | None:
    """Return the index of the first received MAP_DATA message in a range."""
    messages = provider.messages
    magic = provider.magic
    for index, message in enumerate(messages[scan_start_index:], start=scan_start_index):
        if index < message_start_index:
            continue
        if message["direction"] != "received":
            continue
        decoded = decode_message(message["payload"], "received", magic)
        if "MAP_DATA" in decoded:
            return index
    return None


def _emit_teleport_attempt_diagnostic(
    provider: action_session.BufferedWorldStateProviderProtocol,
    *,
    target: TeleportTargetDict,
    teleport_cycle_id: int,
    status: str,
    message_start_index: int,
    page_snapshots: list[TeleportPageSnapshotDict],
) -> None:
    """Emit one ``teleport_attempt`` structured diagnostic event.

    Records target coordinates, cycle id, terminal status, and three
    text-rendered windows (sent message timeline, received message
    timeline, per-phase page snapshots) so consumers can reconstruct the
    full attempt timing without scraping the text log.
    """
    emit_diagnostic(
        diagnostic_kind="teleport_attempt",
        target_x=target["x"],
        target_y=target["y"],
        teleport_cycle_id=teleport_cycle_id,
        status=status,
        sent_window=_format_attempt_window_entries(
            provider,
            message_start_index=message_start_index,
            direction="sent",
        ),
        received_window=_format_attempt_window_entries(
            provider,
            message_start_index=message_start_index,
            direction="received",
        ),
        page_snapshots=_format_page_snapshots(page_snapshots),
        page_snapshot_count=len(page_snapshots),
    )


def _teleport_strategy_requires_map_sync(
    teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
) -> bool:
    """Return whether a teleport strategy waits for fresh world sync."""
    return _shared_teleport_strategy_requires_map_sync(teleport_strategy)


def _clamp_tile(value: int) -> int:
    """Clamp a world coordinate to the valid tile range.

    Args:
        value: Raw tile coordinate.

    Returns:
        Clamped tile coordinate in the inclusive ``0..255`` range.
    """
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


def build_box_targets(
    origin_x: int,
    origin_y: int,
    step_x: int,
    step_y: int,
) -> list[TeleportTargetDict]:
    """Build the default 2x5 teleport target box around an origin.

    Args:
        origin_x: Spawn X coordinate.
        origin_y: Spawn Y coordinate.
        step_x: Horizontal spacing between columns.
        step_y: Vertical spacing between rows.

    Returns:
        Ten teleport targets in row-major order.

    Raises:
        ValueError: If either step is not positive.
    """
    if step_x <= 0:
        raise ValueError("step_x must be positive")
    if step_y <= 0:
        raise ValueError("step_y must be positive")
    x_offsets = (-2 * step_x, -step_x, 0, step_x, 2 * step_x)
    y_offsets = (-step_y, step_y)
    targets: list[TeleportTargetDict] = []
    for row_index, y_offset in enumerate(y_offsets):
        for col_index, x_offset in enumerate(x_offsets):
            targets.append(
                TeleportTargetDict(
                    label=f"box_r{row_index}_c{col_index}",
                    x=_clamp_tile(origin_x + x_offset),
                    y=_clamp_tile(origin_y + y_offset),
                )
            )
    return targets


def parse_targets_arg(raw: str) -> list[TeleportTargetDict]:
    """Parse a CLI target list into teleport targets.

    The accepted format is ``x:y[,x:y...]``.

    Args:
        raw: Raw CLI argument string.

    Returns:
        Parsed teleport targets with deterministic labels.

    Raises:
        ValueError: If the argument is empty, malformed, or outside tile bounds.
    """
    stripped = raw.strip()
    if not stripped:
        raise ValueError("targets argument must not be empty")
    targets: list[TeleportTargetDict] = []
    for index, part in enumerate(stripped.split(",")):
        piece = part.strip()
        coords = piece.split(":")
        if len(coords) != 2:
            raise ValueError(f"invalid target '{piece}'; expected x:y")
        x = int(coords[0])
        y = int(coords[1])
        if x < 0 or x > 255 or y < 0 or y > 255:
            raise ValueError(f"target '{piece}' is outside 0..255 tile bounds")
        targets.append(TeleportTargetDict(label=f"target_{index}", x=x, y=y))
    return targets


def _wait_for_teleport_outcome(
    page: action_session.WaitPageProtocol,
    provider: action_session.BufferedWorldStateProviderProtocol,
    target: TeleportTargetDict,
    *,
    teleport_cycle_id: int,
    message_start_index: int = 0,
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
    """Wait for a teleport to land or time out.

    Args:
        page: Page-like object used for short waits.
        provider: World-state provider to poll.
        target: Requested teleport destination.
        map_open_started_ms: Timestamp when map-open was sent.
        map_sync_timestamp_ms: Timestamp when the map-open fresh sync arrived.
        teleport_started_ms: Timestamp when the teleport command was sent.
        fuel_before: Fuel before the attempt started.
        world_timestamp_before: World-state timestamp before the attempt started.
        timeout_ms: Maximum time to wait for landing.

    Returns:
        Terminal attempt result for the teleport.

    Raises:
        TeleportProbeError: If self state disappears during the wait.
    """
    next_scan_index = message_start_index
    map_data_snapshot_captured = False
    while action_hooks.get_current_time_ms() - teleport_started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(provider)
        if not map_data_snapshot_captured:
            map_data_index = _find_map_data_message_index(
                provider,
                message_start_index=message_start_index,
                scan_start_index=next_scan_index,
            )
            next_scan_index = len(provider.messages)
            if map_data_index is not None:
                _ = map_data_index
                page_snapshots.append(capture_page_snapshot("after_map_data"))
                map_data_snapshot_captured = True
        if action_hooks.check_and_clear_teleport_landed():
            completion_timestamp_ms = action_hooks.get_current_time_ms()
            world = provider.get_world_state()
            self_state = world["self_state"]
            if self_state is None:
                raise TeleportProbeError("self state disappeared after teleport landed")
            page_snapshots.append(capture_page_snapshot("landed"))
            status: Literal["landed_exact", "landed_offset", "map_sync_timeout", "teleport_timeout"]
            if self_state["x"] == target["x"] and self_state["y"] == target["y"]:
                status = "landed_exact"
            else:
                status = "landed_offset"
            result = TeleportAttemptResultDict(
                target=target,
                teleport_cycle_id=teleport_cycle_id,
                status=status,
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                completion_timestamp_ms=completion_timestamp_ms,
                map_sync_elapsed_ms=(
                    None
                    if map_sync_timestamp_ms is None
                    else map_sync_timestamp_ms - map_open_started_ms
                ),
                teleport_elapsed_ms=completion_timestamp_ms - teleport_started_ms,
                fuel_before=fuel_before,
                fuel_after=self_state["fuel"],
                world_timestamp_before=world_timestamp_before,
                world_timestamp_after=world["timestamp_ms"],
                landed_signal_received=True,
                landed_x=self_state["x"],
                landed_y=self_state["y"],
                message_start_index=message_start_index,
                message_end_index=len(provider.messages),
                page_snapshots=page_snapshots,
            )
            _log_teleport_attempt_diagnostic(
                provider,
                target=target,
                teleport_cycle_id=teleport_cycle_id,
                status=status,
                message_start_index=message_start_index,
                page_snapshots=page_snapshots,
            )
            return result
        page.wait_for_timeout(_TELEPORT_POLL_INTERVAL_MS)
    completion_timestamp_ms = action_hooks.get_current_time_ms()
    world = provider.get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        raise TeleportProbeError("self state disappeared while waiting for teleport timeout")
    page_snapshots.append(capture_page_snapshot("timeout"))
    result = TeleportAttemptResultDict(
        target=target,
        teleport_cycle_id=teleport_cycle_id,
        status="teleport_timeout",
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        completion_timestamp_ms=completion_timestamp_ms,
        map_sync_elapsed_ms=(
            None if map_sync_timestamp_ms is None else map_sync_timestamp_ms - map_open_started_ms
        ),
        teleport_elapsed_ms=completion_timestamp_ms - teleport_started_ms,
        fuel_before=fuel_before,
        fuel_after=self_state["fuel"],
        world_timestamp_before=world_timestamp_before,
        world_timestamp_after=world["timestamp_ms"],
        landed_signal_received=False,
        landed_x=self_state["x"],
        landed_y=self_state["y"],
        message_start_index=message_start_index,
        message_end_index=len(provider.messages),
        page_snapshots=page_snapshots,
    )
    _log_teleport_attempt_diagnostic(
        provider,
        target=target,
        teleport_cycle_id=teleport_cycle_id,
        status="teleport_timeout",
        message_start_index=message_start_index,
        page_snapshots=page_snapshots,
    )
    return result


def format_teleport_probe_summary(session: TeleportProbeSessionDict) -> str:
    """Format a compact human-readable summary line for a probe session.

    Args:
        session: Teleport probe session to summarize.

    Returns:
        One-line summary string with outcome counts.
    """
    exact = 0
    offset = 0
    map_timeouts = 0
    teleport_timeouts = 0
    for attempt in session["attempts"]:
        if attempt["status"] == "landed_exact":
            exact += 1
        elif attempt["status"] == "landed_offset":
            offset += 1
        elif attempt["status"] == "map_sync_timeout":
            map_timeouts += 1
        else:
            teleport_timeouts += 1
    startup_timing = session["startup_timing"]
    bootstrap_ms = (
        startup_timing["command_ready_timestamp_ms"] - startup_timing["initial_sync_started_ms"]
    )
    return (
        "Teleport probe complete: "
        f"strategy={session['teleport_strategy']} "
        f"attempts={len(session['attempts'])} "
        f"exact={exact} offset={offset} "
        f"map_sync_timeout={map_timeouts} teleport_timeout={teleport_timeouts} "
        "session_to_initial_sync_ms="
        f"{startup_timing['initial_sync_started_ms'] - session['start_timestamp_ms']} "
        f"initial_sync_to_command_ready_ms={bootstrap_ms}"
    )


def _limit_targets(
    targets: list[TeleportTargetDict],
    max_targets: int | None,
) -> list[TeleportTargetDict]:
    """Limit the target list when a maximum is configured.

    Args:
        targets: Candidate target list.
        max_targets: Maximum number of targets to keep, or None for no limit.

    Returns:
        Original targets, or the leading subset constrained by max_targets.

    Raises:
        ValueError: If max_targets is not positive when provided.
    """
    if max_targets is None:
        return targets
    if max_targets <= 0:
        raise ValueError("max_targets must be positive")
    return targets[:max_targets]
