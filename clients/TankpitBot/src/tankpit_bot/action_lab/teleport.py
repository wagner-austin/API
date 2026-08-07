"""Live teleport probe harness."""

from __future__ import annotations

from typing import Literal, Protocol

from platform_core.logging import get_logger

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.probe_entrypoint import (
    run_and_save_standard_probe_session,
)
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.teleport_acquisition import (
    teleport_strategy_requires_map_sync,
)
from tankpit_bot.action_lab.teleport_attempt import (
    run_tracked_teleport_attempt as _shared_run_tracked_teleport_attempt,
)
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    _limit_targets,
    _log_teleport_attempt_diagnostic,
    _wait_for_teleport_outcome,
    build_box_targets,
    format_teleport_probe_summary,
    parse_targets_arg,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportProbeSessionDict,
    TeleportTargetDict,
)
from tankpit_bot.action_lab.types_codecs import encode_teleport_probe_session

log = get_logger(__name__)
DEFAULT_TELEPORT_STRATEGY: Literal["sync_before_teleport", "immediate_after_map_open"] = (
    "immediate_after_map_open"
)
run_tracked_teleport_attempt = _shared_run_tracked_teleport_attempt


class TeleportProbe(ProbeBase):
    """Live teleport probe — teleport-specific execute and outcome logic."""

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
            wait_for_acquisition_sync=teleport_strategy_requires_map_sync(teleport_strategy),
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
        if teleport_strategy_requires_map_sync(teleport_strategy) and map_sync_timestamp_ms is None:
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


class BuildTeleportProbeProtocol(Protocol):
    """Factory contract for instantiating a live ``TeleportProbe``."""

    def __call__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
    ) -> TeleportProbe:
        """Build one TeleportProbe instance.

        Args:
            target_url: URL to navigate the browser to.
            headless: Whether to run headlessly.
            prefer_account: Whether to use account login instead of guest.

        Returns:
            New TeleportProbe instance ready for live execution.
        """
        ...


def _create_teleport_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> TeleportProbe:
    """Factory for TeleportProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        TeleportProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, TeleportProbe)
    return probe


build_teleport_probe: BuildTeleportProbeProtocol = _create_teleport_probe


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
        probe_factory=build_teleport_probe,
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
    "BuildTeleportProbeProtocol",
    "TeleportProbe",
    "TeleportProbeError",
    "_wait_for_teleport_outcome",
    "build_box_targets",
    "build_teleport_probe",
    "format_teleport_probe_summary",
    "parse_targets_arg",
    "run_teleport_probe",
    "teleport_strategy_requires_map_sync",
]
