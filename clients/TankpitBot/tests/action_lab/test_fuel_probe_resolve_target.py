"""Tests for ``resolve_fuel_target`` and its reposition results.

The post-radar resolution path, including every reposition outcome the
resolver builds.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import (
    Literal,
)

import pytest
from tests.action_lab._fuel_probe_harness import (
    _ProbeHarness,
    _snapshot,
)
from tests.action_lab._fuel_probe_scenarios import (
    _resolve_with_tracked_reposition,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from typing_extensions import Unpack

from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
)
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.fuel_probe_targets import FuelProbeError
from tankpit_bot.action_lab.teleport import TeleportProbeError
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    TrackedTeleportAttempt,
)
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterKwargs,
    TeleportOutcomeWaiterProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.state import (
    ContainerStateDict,
    make_container_state,
)


def test_resolve_fuel_target_after_radar_rejects_missing_tracked_reposition_result() -> None:
    """Fuel target resolution rejects a tracked reposition without a teleport result."""
    from tests.action_lab._teleport_seams import fuel_target_phase_module

    from tankpit_bot.action_lab import fuel_target_phase

    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    page = ClockAdvancingPage(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    fuel_target = make_container_state(101, 100, True, 300)
    original_attempt_runner = fuel_target_phase_module.run_tracked_teleport_attempt

    def _requires_reposition(
        probe: fuel_target_phase.FuelTargetPhaseProbeProtocol,
        fuel_target: ContainerStateDict,
    ) -> bool:
        _ = (probe, fuel_target)
        return True

    def _landing_tile(
        probe: fuel_target_phase.FuelTargetPhaseProbeProtocol,
        fuel_target: ContainerStateDict,
    ) -> tuple[int, int] | None:
        _ = (probe, fuel_target)
        return (102, 100)

    def _make_reposition_target(target_x: int, target_y: int) -> TeleportTargetDict:
        return TeleportTargetDict(
            label=f"fuel_reposition_{target_x}_{target_y}",
            x=target_x,
            y=target_y,
        )

    def _teleport_strategy_requires_map_sync(
        strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
    ) -> bool:
        return strategy == "sync_before_teleport"

    def _wait_for_teleport_outcome_adapter(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        **kwargs: Unpack[TeleportOutcomeWaiterKwargs],
    ) -> TeleportAttemptResultDict:
        _ = (page, provider)
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=kwargs["teleport_cycle_id"],
            status="landed_exact",
            map_open_started_ms=kwargs["map_open_started_ms"],
            map_sync_timestamp_ms=kwargs["map_sync_timestamp_ms"],
            teleport_started_ms=kwargs["teleport_started_ms"],
            completion_timestamp_ms=2200,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=200,
            fuel_before=kwargs["fuel_before"],
            fuel_after=840,
            world_timestamp_before=kwargs["world_timestamp_before"],
            world_timestamp_after=2150,
            landed_signal_received=True,
            landed_x=102,
            landed_y=100,
            message_start_index=kwargs["message_start_index"],
            message_end_index=kwargs["message_start_index"],
            page_snapshots=kwargs["page_snapshots"],
        )

    wait_for_teleport_outcome: TeleportOutcomeWaiterProtocol = _wait_for_teleport_outcome_adapter

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return TeleportPageSnapshotDict(
            phase=phase,
            timestamp_ms=2000,
            client_present=True,
            map_visible=False,
            client_state=1,
            client_busy=False,
            pending_actions=0,
            heartbeat_age_ms=1,
            last_page_client_send_age_ms=2,
            last_bot_send_age_ms=3,
            ws_ready_state=1,
            current_send_label=None,
            sent_frame_meta_queue_length=0,
            self_fields={},
            world_fields={},
            map_fields={},
            world_collections={},
        )

    def _run_attempt(
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
        _ = (
            page,
            probe,
            target,
            cdp,
            attempt_label,
            fuel_before,
            world_timestamp_before,
            send_acquisition_command,
            acquisition_command_name,
            capture_before_map_open,
            wait_for_acquisition_sync,
            acquisition_timeout_ms,
            teleport_timeout_ms,
            wait_for_outcome,
            dispatch_failure_error,
            acquisition_dispatch_failure_message,
            teleport_dispatch_failure_message,
            unavailable_error,
            unavailable_message,
            unexpected_result_error,
            unexpected_result_message,
            reset_to_idle_before_start,
        )
        return TrackedTeleportAttempt(
            message_start_index=0,
            teleport_cycle=ActionPhaseCycleDict(phase="teleport", cycle_id=3, started_ms=2000),
            acquisition_started_ms=2000,
            acquisition_sync_timestamp_ms=2200,
            page_snapshots=[],
            capture_page_snapshot=_capture_page_snapshot,
            teleport_result=None,
            teleport_started_ms=None,
        )

    fuel_target_phase_module.run_tracked_teleport_attempt = _run_attempt
    try:
        with pytest.raises(FuelProbeError, match="fuel reposition ended before teleport dispatch"):
            fuel_target_phase.resolve_fuel_target_after_radar(
                page,
                probe,
                cdp=probe._cdp,
                target=target,
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1200,
                teleport_started_ms=1300,
                radar_started_ms=1600,
                radar_sync_timestamp_ms=1700,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                fuel_before=900,
                teleport_result=TeleportAttemptResultDict(
                    target=target,
                    teleport_cycle_id=1,
                    status="landed_exact",
                    map_open_started_ms=1000,
                    map_sync_timestamp_ms=1200,
                    teleport_started_ms=1300,
                    completion_timestamp_ms=1500,
                    map_sync_elapsed_ms=200,
                    teleport_elapsed_ms=200,
                    fuel_before=900,
                    fuel_after=840,
                    world_timestamp_before=950,
                    world_timestamp_after=1450,
                    landed_signal_received=True,
                    landed_x=124,
                    landed_y=100,
                    message_start_index=0,
                    message_end_index=0,
                    page_snapshots=[],
                ),
                message_start_index=0,
                teleport_cycle_ids=[1],
                radar_cycle_id=2,
                teleport_strategy="sync_before_teleport",
                snapshot_before=_snapshot(1000),
                capture_snapshot=lambda: _snapshot(1900),
                terrain_provider=lambda: None,
                find_visible_target=lambda current_probe: fuel_target,
                requires_reposition=_requires_reposition,
                find_landing_tile=_landing_tile,
                get_phase_overlaps=probe._get_attempt_phase_overlaps,
                build_no_fuel_visible_result=probe._build_no_fuel_visible_result,
                build_reposition_map_sync_timeout_result=(
                    probe._build_reposition_map_sync_timeout_result
                ),
                build_reposition_teleport_timeout_result=(
                    probe._build_reposition_teleport_timeout_result
                ),
                make_reposition_target=_make_reposition_target,
                wait_for_teleport_outcome=wait_for_teleport_outcome,
                teleport_strategy_requires_map_sync=_teleport_strategy_requires_map_sync,
                no_landing_tile_error=FuelProbeError,
                dispatch_failure_error=FuelProbeError,
                unavailable_error=FuelProbeError,
                unexpected_result_error=TeleportProbeError,
                unavailable_message="cdp session is unavailable",
                no_landing_tile_message="visible fuel target has no teleport landing tile",
                impossible_result_message=(
                    "teleport outcome reported impossible map_sync_timeout during fuel reposition"
                ),
                acquisition_dispatch_failure_message=(
                    "map_open command dispatch failed during fuel reposition"
                ),
                teleport_dispatch_failure_message=(
                    "teleport command dispatch failed during fuel reposition"
                ),
            )
    finally:
        fuel_target_phase_module.run_tracked_teleport_attempt = original_attempt_runner


def test_resolve_fuel_target_builds_reposition_map_sync_timeout_result() -> None:
    """A reposition whose map sync never confirms yields the terminal timeout result."""
    resolution = _resolve_with_tracked_reposition(
        acquisition_sync_timestamp_ms=None,
        reposition_teleport_started_ms=None,
        reposition_teleport_status=None,
    )

    if resolution.terminal_result is None:
        raise AssertionError("expected a reposition_map_sync_timeout terminal result")
    assert resolution.terminal_result["status"] == "reposition_map_sync_timeout"
    if resolution.teleport_result is None:
        raise AssertionError("expected the original teleport result to be preserved")
    assert resolution.teleport_result["landed_x"] == 124


def test_resolve_fuel_target_builds_reposition_teleport_timeout_result() -> None:
    """A reposition teleport that times out yields the terminal timeout result."""
    resolution = _resolve_with_tracked_reposition(
        acquisition_sync_timestamp_ms=2200,
        reposition_teleport_started_ms=2300,
        reposition_teleport_status="teleport_timeout",
    )

    if resolution.terminal_result is None:
        raise AssertionError("expected a reposition_teleport_timeout terminal result")
    assert resolution.terminal_result["status"] == "reposition_teleport_timeout"


def test_resolve_fuel_target_adopts_successful_reposition_teleport_result() -> None:
    """A landed reposition replaces the attempt's teleport result."""
    resolution = _resolve_with_tracked_reposition(
        acquisition_sync_timestamp_ms=2200,
        reposition_teleport_started_ms=2300,
        reposition_teleport_status="landed_exact",
    )

    assert resolution.terminal_result is None
    if resolution.teleport_result is None:
        raise AssertionError("expected the reposition teleport result to be adopted")
    assert resolution.teleport_result["landed_x"] == 102
    if resolution.fuel_target is None:
        raise AssertionError("expected the fuel target to survive the reposition")
    assert resolution.fuel_target["x"] == 101
