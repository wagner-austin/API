"""Live teleport probe harness."""

from __future__ import annotations

from typing import Literal

from platform_core.logging import get_logger

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
from tankpit_bot.action_lab.teleport_attempt import (
    run_tracked_teleport_attempt as _shared_run_tracked_teleport_attempt,
)
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    _limit_targets,
    _log_teleport_attempt_diagnostic,
    _teleport_strategy_requires_map_sync,
    _wait_for_teleport_outcome,
    build_box_targets,
    format_teleport_probe_summary,
    parse_targets_arg,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportProbeSessionDict,
    TeleportTargetDict,
    encode_teleport_probe_session,
)
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser import BrowserSession
from tankpit_bot.sniffer.world_state import get_world_state
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)
DEFAULT_TELEPORT_STRATEGY: Literal["sync_before_teleport", "immediate_after_map_open"] = (
    "immediate_after_map_open"
)
run_tracked_teleport_attempt = _shared_run_tracked_teleport_attempt


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
    "_wait_for_teleport_outcome",
    "build_box_targets",
    "format_teleport_probe_summary",
    "parse_targets_arg",
    "run_teleport_probe",
]
