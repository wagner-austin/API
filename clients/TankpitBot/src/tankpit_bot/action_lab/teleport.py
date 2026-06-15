"""Live teleport probe harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace import ActionCycleTracker, log_phase_overlaps
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseName,
    ActionPhaseOverlapDict,
)
from tankpit_bot.action_lab.probe_entrypoint import (
    run_and_save_standard_probe_session,
)
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.teleport_acquisition import (
    start_teleport_page_snapshots as _shared_start_teleport_page_snapshots,
)
from tankpit_bot.action_lab.teleport_acquisition import (
    teleport_strategy_requires_map_sync as _shared_teleport_strategy_requires_map_sync,
)
from tankpit_bot.action_lab.teleport_attempt import (
    run_tracked_teleport_attempt as _shared_run_tracked_teleport_attempt,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportProbeSessionDict,
    TeleportTargetDict,
    encode_teleport_probe_session,
)
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser import BrowserSession
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.decoders import decode_message
from tankpit_bot.sniffer.world_state import get_world_state
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)
_TELEPORT_POLL_INTERVAL_MS = 100.0
DEFAULT_TELEPORT_STRATEGY: Literal["sync_before_teleport", "immediate_after_map_open"] = (
    "immediate_after_map_open"
)
run_tracked_teleport_attempt = _shared_run_tracked_teleport_attempt


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


class TeleportProbe(BrowserSession):
    """Live teleport probe that isolates map-open and teleport behavior.

    Inherits browser lifecycle from BrowserSession. Uses CommandService
    for command dispatch instead of inheriting from Bot.
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
    ) -> None:
        """Initialize the teleport probe.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Whether to prefer account login.
        """
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
        )
        self._commands = CommandService(send_ws_bytes=self._send_websocket_bytes)
        self._cdp_message_buffer: list[str] = []
        self._action_cycle_tracker = ActionCycleTracker()
        self._attempt_phase_overlaps: list[ActionPhaseOverlapDict] = []

    # -----------------------------------------------------------------
    # Command dispatch
    # -----------------------------------------------------------------

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        """XOR encode and send command bytes via WebSocket.

        Subclasses (replay harnesses) override this to capture dispatched
        commands without hitting the wire.

        Args:
            data: Framed command bytes.
            cmd_name: Command name for logging.

        Returns:
            True if sent, False if CDP unavailable.
        """
        self._commands.cdp = self._cdp
        return self._commands.send_bytes(data, cmd_name)

    def open_map(self) -> bool:
        """Send map open command.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_MAP_OPEN, build_query_command

        return self._send_bytes(build_query_command(CMD_MAP_OPEN), "map_open")

    def teleport_to(self, x: int, y: int) -> bool:
        """Send teleport command. Map must already be open.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.bot.commands import encode_teleport_command
        from tankpit_bot.bot.types import make_teleport_command

        if self._cdp is None:
            return False
        cmd = make_teleport_command(x, y)
        return self._send_bytes(encode_teleport_command(cmd), f"teleport({x},{y})")

    def move_to(self, x: int, y: int) -> bool:
        """Send move command.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.bot.commands import encode_move_command
        from tankpit_bot.bot.types import make_move_command

        cmd = make_move_command(x, y)
        return self._send_bytes(encode_move_command(cmd), "move")

    def use_radar(self) -> bool:
        """Send radar scan command.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_RADAR, build_query_command

        return self._send_bytes(build_query_command(CMD_RADAR), "radar")

    def request_nearest_enemy(self) -> bool:
        """Send nearest enemy query command.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_NEAREST_ENEMY, build_query_command

        return self._send_bytes(build_query_command(CMD_NEAREST_ENEMY), "nearest_enemy")

    # -----------------------------------------------------------------
    # Lifecycle hooks
    # -----------------------------------------------------------------

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Buffer received messages for probe sync.

        Args:
            message: The captured message.
        """
        super()._on_message_captured(message)
        if message["direction"] == "received":
            self._cdp_message_buffer.append(message["payload"])

    def _on_magic_captured(self, magic: str) -> None:
        """Build XOR table and init trackers when magic key is captured.

        Args:
            magic: The session magic string.
        """
        from tankpit_bot.protocol.codec import (
            DEFAULT_STATIC_KEY_PATH,
            build_xor_table,
            load_static_key,
        )
        from tankpit_bot.sniffer.trackers import init_trackers_with_magic

        init_trackers_with_magic(magic)
        static_key = load_static_key(DEFAULT_STATIC_KEY_PATH)
        self._commands.xor_table = build_xor_table(static_key, magic)

    # -----------------------------------------------------------------
    # World state access
    # -----------------------------------------------------------------

    def get_world_state(self) -> WorldStateDict:
        """Get current world state.

        Returns:
            Current WorldStateDict.
        """
        return get_world_state()

    def get_state(self) -> str:
        """Get current probe state.

        Returns:
            Always "IDLE" — probes do not use the Bot HFSM.
        """
        return "IDLE"

    def _update_state_from_world(self) -> None:
        """Update state from world data. No-op for probes."""

    def get_self_state(self) -> SelfStateDict | None:
        """Get self tank state.

        Returns:
            SelfStateDict if available, None if not yet tracked.
        """
        return get_world_state()["self_state"]

    # -----------------------------------------------------------------
    # Action phase tracking
    # -----------------------------------------------------------------

    def _reset_action_cycle_tracker(self) -> None:
        """Reset action phase tracking for a new live session."""
        self._action_cycle_tracker.reset()
        self._attempt_phase_overlaps = []

    def _reset_attempt_phase_overlaps(self) -> None:
        """Reset overlap diagnostics for a new attempt."""
        self._attempt_phase_overlaps = []

    def _get_attempt_phase_overlaps(self) -> list[ActionPhaseOverlapDict]:
        """Return a copy of current overlap diagnostics for the active attempt."""
        return list(self._attempt_phase_overlaps)

    def _start_action_phase(
        self,
        phase: ActionPhaseName,
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Start a traced action phase and log invariant overlaps.

        Args:
            phase: Phase being started.
            attempt_label: Human-readable attempt label.

        Returns:
            Started phase cycle.
        """
        cycle, overlaps = self._action_cycle_tracker.begin_phase(
            phase,
            started_ms=action_hooks.get_current_time_ms(),
        )
        self._attempt_phase_overlaps.extend(overlaps)
        log_phase_overlaps(overlaps, attempt_label=attempt_label)
        return cycle

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """End a traced action phase.

        Args:
            cycle: Cycle to close.
        """
        self._action_cycle_tracker.end_phase(cycle)

    def _require_self_state(self) -> SelfStateDict:
        """Return the current self state or raise when absent.

        Returns:
            Current self tank state.

        Raises:
            TeleportProbeError: If self state is not yet available.
        """
        self_state = self.get_self_state()
        if self_state is None:
            raise TeleportProbeError("self state is unavailable")
        return self_state

    def _require_page(self) -> action_session.WaitPageProtocol:
        """Return the current Playwright page or raise when absent.

        Returns:
            Current page handle.

        Raises:
            TeleportProbeError: If the page has not been initialized.
        """
        if self._page is None:
            raise TeleportProbeError("page is unavailable")
        return self._page

    def _clear_in_flight_action(self) -> None:
        """Clear any pending action record between phases."""

    def _reset_probe_state_to_idle(self) -> None:
        """Reset the probe to idle state between attempts."""

    def _probe_single_target(
        self,
        target: TeleportTargetDict,
        *,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
    ) -> TeleportAttemptResultDict:
        """Run one teleport attempt against the live server.

        Args:
            target: Requested destination.
            map_sync_timeout_ms: Maximum wait for the map-open fresh sync.
            teleport_timeout_ms: Maximum wait for teleport confirmation.
            settle_delay_ms: Delay after completion before the next attempt.

        Returns:
            Terminal attempt result for the target.

        Raises:
            TeleportProbeError: If command dispatch fails.
        """
        page = self._require_page()
        world_before = self.get_world_state()
        self_state_before = self._require_self_state()
        fuel_before = self_state_before["fuel"]
        world_timestamp_before = world_before["timestamp_ms"]

        self._reset_attempt_phase_overlaps()
        attempt = run_tracked_teleport_attempt(
            page,
            self,
            target,
            cdp=self._cdp,
            attempt_label=target["label"],
            fuel_before=fuel_before,
            world_timestamp_before=world_timestamp_before,
            send_acquisition_command=self.open_map,
            acquisition_command_name="map_open",
            capture_before_map_open=True,
            wait_for_acquisition_sync=_teleport_strategy_requires_map_sync(teleport_strategy),
            acquisition_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            wait_for_outcome=_wait_for_teleport_outcome,
            dispatch_failure_error=TeleportProbeError,
            acquisition_dispatch_failure_message="map_open command dispatch failed",
            teleport_dispatch_failure_message="teleport command dispatch failed",
            unavailable_error=TeleportProbeError,
            unavailable_message="cdp session is unavailable",
            unexpected_result_error=TeleportProbeError,
            unexpected_result_message="teleport outcome reported impossible map_sync_timeout",
        )
        message_start_index = attempt.message_start_index
        teleport_cycle = attempt.teleport_cycle
        map_open_started_ms = attempt.acquisition_started_ms
        map_sync_timestamp_ms = attempt.acquisition_sync_timestamp_ms
        page_snapshots = attempt.page_snapshots
        if (
            _teleport_strategy_requires_map_sync(teleport_strategy)
            and map_sync_timestamp_ms is None
        ):
            completion_timestamp_ms = action_hooks.get_current_time_ms()
            self._reset_probe_state_to_idle()
            self_state_after = self._require_self_state()
            result = TeleportAttemptResultDict(
                target=target,
                teleport_cycle_id=teleport_cycle["cycle_id"],
                status="map_sync_timeout",
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=None,
                teleport_started_ms=None,
                completion_timestamp_ms=completion_timestamp_ms,
                map_sync_elapsed_ms=None,
                teleport_elapsed_ms=None,
                fuel_before=fuel_before,
                fuel_after=self_state_after["fuel"],
                world_timestamp_before=world_timestamp_before,
                world_timestamp_after=self.get_world_state()["timestamp_ms"],
                landed_signal_received=False,
                landed_x=self_state_after["x"],
                landed_y=self_state_after["y"],
                message_start_index=message_start_index,
                message_end_index=len(self.messages),
                page_snapshots=page_snapshots,
            )
            _log_teleport_attempt_diagnostic(
                self,
                target=target,
                teleport_cycle_id=teleport_cycle["cycle_id"],
                status="map_sync_timeout",
                message_start_index=message_start_index,
                page_snapshots=page_snapshots,
            )
            self._end_action_phase(teleport_cycle)
            if settle_delay_ms > 0:
                page.wait_for_timeout(float(settle_delay_ms))
            return result

        teleport_result = attempt.teleport_result
        if teleport_result is None:
            raise TeleportProbeError("teleport attempt ended before teleport dispatch")
        if settle_delay_ms > 0:
            page.wait_for_timeout(float(settle_delay_ms))
        return teleport_result

    def execute(
        self,
        *,
        explicit_targets: list[TeleportTargetDict] | None,
        box_step_x: int,
        box_step_y: int,
        max_targets: int | None,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        initial_sync_timeout_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
    ) -> TeleportProbeSessionDict:
        """Run the live teleport probe session.

        Args:
            explicit_targets: Absolute requested targets, or None to use the default box.
            box_step_x: Horizontal spacing for the default box.
            box_step_y: Vertical spacing for the default box.
            max_targets: Maximum number of targets to run, or None for all.
            teleport_strategy: Teleport sequencing strategy for each attempt.
            initial_sync_timeout_ms: Maximum wait for the initial self-state sync.
            map_sync_timeout_ms: Maximum wait for the map-open fresh sync.
            teleport_timeout_ms: Maximum wait for teleport confirmation.
            settle_delay_ms: Delay after each attempt.

        Returns:
            Complete teleport probe session.

        Raises:
            PlaywrightNotInstalledError: If Playwright is not installed.
            TeleportProbeError: If bootstrap or command dispatch fails.
        """

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> TeleportProbeSessionDict:
            targets = _limit_targets(
                (
                    explicit_targets
                    if explicit_targets is not None
                    else build_box_targets(
                        context["spawn"]["x"],
                        context["spawn"]["y"],
                        box_step_x,
                        box_step_y,
                    )
                ),
                max_targets,
            )
            if not targets:
                raise TeleportProbeError("teleport probe requires at least one target")
            attempts: list[TeleportAttemptResultDict] = []
            for target in targets:
                attempts.append(
                    self._probe_single_target(
                        target,
                        teleport_strategy=teleport_strategy,
                        map_sync_timeout_ms=map_sync_timeout_ms,
                        teleport_timeout_ms=teleport_timeout_ms,
                        settle_delay_ms=settle_delay_ms,
                    )
                )
            first_attempt_started_ms = attempts[0]["map_open_started_ms"] if attempts else None
            session_envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_attempt_started_ms,
            )
            return TeleportProbeSessionDict(
                session_id=session_envelope.session_id,
                start_timestamp_ms=session_envelope.start_timestamp_ms,
                end_timestamp_ms=session_envelope.end_timestamp_ms,
                base_url=session_envelope.base_url,
                spawn_x=session_envelope.spawn_x,
                spawn_y=session_envelope.spawn_y,
                teleport_strategy=teleport_strategy,
                max_targets=max_targets,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=session_envelope.startup_timing,
                map_sync_timeout_ms=map_sync_timeout_ms,
                teleport_timeout_ms=teleport_timeout_ms,
                settle_delay_ms=settle_delay_ms,
                targets=targets,
                attempts=attempts,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def run_teleport_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    explicit_targets: list[TeleportTargetDict] | None = None,
    box_step_x: int = 8,
    box_step_y: int = 8,
    max_targets: int | None = None,
    teleport_strategy: Literal[
        "sync_before_teleport", "immediate_after_map_open"
    ] = DEFAULT_TELEPORT_STRATEGY,
    initial_sync_timeout_ms: int = 10000,
    map_sync_timeout_ms: int = 3000,
    teleport_timeout_ms: int = 10000,
    settle_delay_ms: int = 500,
) -> TeleportProbeSessionDict:
    """Run a live teleport probe and save the session JSON.

    Args:
        target_url: URL to navigate to.
        output_path: Output path for the session JSON.
        headless: Whether to run the browser headlessly.
        prefer_account: Whether to use account login instead of guest login.
        explicit_targets: Absolute targets to test, or None for the default box.
        box_step_x: Horizontal spacing for the default box.
        box_step_y: Vertical spacing for the default box.
        max_targets: Maximum number of targets to run, or None for all.
        teleport_strategy: Teleport sequencing strategy for each attempt.
        initial_sync_timeout_ms: Maximum wait for the initial self-state sync.
        map_sync_timeout_ms: Maximum wait for the map-open fresh sync.
        teleport_timeout_ms: Maximum wait for teleport confirmation.
        settle_delay_ms: Delay after each attempt.

    Returns:
        Completed teleport probe session.
    """

    def _run_session(probe: TeleportProbe) -> TeleportProbeSessionDict:
        return probe.execute(
            explicit_targets=explicit_targets,
            box_step_x=box_step_x,
            box_step_y=box_step_y,
            max_targets=max_targets,
            teleport_strategy=teleport_strategy,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            settle_delay_ms=settle_delay_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=TeleportProbe,
        run_session=_run_session,
        encoder=encode_teleport_probe_session,
        summary_formatter=format_teleport_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "DEFAULT_TELEPORT_STRATEGY",
    "TeleportProbe",
    "TeleportProbeError",
    "_teleport_strategy_requires_map_sync",
    "build_box_targets",
    "format_teleport_probe_summary",
    "parse_targets_arg",
    "run_teleport_probe",
]
