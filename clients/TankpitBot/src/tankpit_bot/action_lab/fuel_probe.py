"""The fuel probe: teleport to fuel, pick it up, record the attempt.

Holds :class:`FuelProbe` and its entry points. The target-selection and
outcome helpers it composes are
:mod:`tankpit_bot.action_lab.fuel_probe_targets`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from tankpit_bot.action_lab import fuel_probe_operations as _fuel_probe_operations
from tankpit_bot.action_lab import fuel_probe_targets
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict, FuelDecisionBasisDict
from tankpit_bot.action_lab.fuel_collection_phase import (
    run_tracked_fuel_collection_phase,
)
from tankpit_bot.action_lab.fuel_probe_attempt import (
    run_single_fuel_target_attempt as _shared_run_single_fuel_target_attempt,
)
from tankpit_bot.action_lab.fuel_probe_diagnostics import format_fuel_probe_summary
from tankpit_bot.action_lab.fuel_probe_entrypoint import (
    run_and_save_fuel_probe_session as _shared_run_and_save_fuel_probe_session,
)
from tankpit_bot.action_lab.fuel_probe_runner import (
    execute_fuel_probe_session as _shared_execute_fuel_probe_session,
)
from tankpit_bot.action_lab.fuel_probe_targets import (
    FuelProbeError,
    _log_fuel_target_diagnostic,
    _make_reposition_target,
)
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
)
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.pickup_phase import (
    PickupPhaseError,
    effective_pickup_timeout_ms,
    run_tracked_pickup_phase,
)
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.teleport import (
    DEFAULT_TELEPORT_STRATEGY,
)
from tankpit_bot.action_lab.teleport_acquisition import (
    teleport_strategy_requires_map_sync,
)
from tankpit_bot.action_lab.teleport_attempt import (
    run_tracked_teleport_attempt,
)
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    _wait_for_teleport_outcome,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.sniffer.world_state import get_terrain_map
from tankpit_bot.state.types import ContainerStateDict

_FUEL_PROBE_TARGET_STEP = 16

_FUEL_PROBE_TARGET_MAX_RADIUS = 48

_FUEL_PROBE_TELEPORT_STRATEGY: Literal["sync_before_teleport", "immediate_after_map_open"] = (
    DEFAULT_TELEPORT_STRATEGY
)


class FuelProbe(ProbeBase):
    """Live teleport-radar-fuel probe."""

    def _build_attempt_result(
        self,
        *,
        target: TeleportTargetDict,
        status: Literal[
            "picked_up_fuel",
            "no_fuel_visible",
            "radar_timeout",
            "map_sync_timeout",
            "reposition_map_sync_timeout",
            "teleport_timeout",
            "reposition_teleport_timeout",
            "pickup_timeout",
        ],
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int | None,
        radar_started_ms: int | None,
        radar_sync_timestamp_ms: int | None,
        pickup_started_ms: int | None,
        completion_timestamp_ms: int,
        fuel_before: int,
        fuel_after: int | None,
        landed_signal_received: bool,
        landed_x: int | None,
        landed_y: int | None,
        fuel_target: ContainerStateDict | None,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        snapshot_before: PageClientSnapshotDict,
        snapshot_after: PageClientSnapshotDict,
        radar_cycle_id: int | None = None,
        move_cycle_id: int | None = None,
        pickup_cycle_id: int | None = None,
        phase_overlaps: list[ActionPhaseOverlapDict] | None = None,
        decision_basis: FuelDecisionBasisDict | None = None,
        reposition_map_open_started_ms: int | None = None,
        reposition_map_sync_timestamp_ms: int | None = None,
        reposition_teleport_started_ms: int | None = None,
    ) -> FuelProbeAttemptResultDict:
        """Create a typed attempt result payload."""
        return _fuel_probe_operations.build_attempt_result_for_probe(
            self,
            target=target,
            status=status,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            pickup_started_ms=pickup_started_ms,
            completion_timestamp_ms=completion_timestamp_ms,
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            landed_signal_received=landed_signal_received,
            landed_x=landed_x,
            landed_y=landed_y,
            fuel_target=fuel_target,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle_id,
            move_cycle_id=move_cycle_id,
            pickup_cycle_id=pickup_cycle_id,
            phase_overlaps=phase_overlaps,
            decision_basis=decision_basis,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

    def _finalize_attempt_delay(
        self,
        page: action_session.WaitPageProtocol,
        *,
        settle_delay_ms: int,
    ) -> None:
        """Apply optional settle delay after an attempt."""
        _fuel_probe_operations.finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)

    def _build_map_sync_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        fuel_before: int,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        snapshot_before: PageClientSnapshotDict,
        snapshot_after: PageClientSnapshotDict,
    ) -> FuelProbeAttemptResultDict:
        """Build a map-sync-timeout result."""
        return _fuel_probe_operations.build_map_sync_timeout_result_for_probe(
            self,
            target=target,
            map_open_started_ms=map_open_started_ms,
            fuel_before=fuel_before,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

    def _build_teleport_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        snapshot_before: PageClientSnapshotDict,
        snapshot_after: PageClientSnapshotDict,
    ) -> FuelProbeAttemptResultDict:
        """Build a teleport-timeout result."""
        return _fuel_probe_operations.build_teleport_timeout_result_for_probe(
            self,
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            fuel_before=fuel_before,
            teleport_result=teleport_result,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

    def _build_reposition_map_sync_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        reposition_map_open_started_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        fuel_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
        snapshot_before: PageClientSnapshotDict,
        snapshot_after: PageClientSnapshotDict,
    ) -> FuelProbeAttemptResultDict:
        """Build a reposition map-sync-timeout result."""
        return _fuel_probe_operations.build_reposition_map_sync_timeout_result_for_probe(
            self,
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            fuel_before=fuel_before,
            teleport_result=teleport_result,
            fuel_target=fuel_target,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle_id,
            phase_overlaps=phase_overlaps,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

    def _build_reposition_teleport_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        reposition_map_open_started_ms: int,
        reposition_map_sync_timestamp_ms: int | None,
        reposition_teleport_started_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        fuel_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
        snapshot_before: PageClientSnapshotDict,
        snapshot_after: PageClientSnapshotDict,
    ) -> FuelProbeAttemptResultDict:
        """Build a reposition teleport-timeout result."""
        return _fuel_probe_operations.build_reposition_teleport_timeout_result_for_probe(
            self,
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
            fuel_before=fuel_before,
            teleport_result=teleport_result,
            fuel_target=fuel_target,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle_id,
            phase_overlaps=phase_overlaps,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

    def _build_radar_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
        snapshot_before: PageClientSnapshotDict,
        snapshot_after: PageClientSnapshotDict,
    ) -> FuelProbeAttemptResultDict:
        """Build a radar-timeout result."""
        return _fuel_probe_operations.build_radar_timeout_result_for_probe(
            self,
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            fuel_before=fuel_before,
            teleport_result=teleport_result,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle_id,
            phase_overlaps=phase_overlaps,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

    def _build_no_fuel_visible_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
        decision_basis: FuelDecisionBasisDict | None,
        snapshot_before: PageClientSnapshotDict,
        snapshot_after: PageClientSnapshotDict,
    ) -> FuelProbeAttemptResultDict:
        """Build a no-fuel-visible result."""
        return _fuel_probe_operations.build_no_fuel_visible_result_for_probe(
            self,
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            fuel_before=fuel_before,
            teleport_result=teleport_result,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle_id,
            phase_overlaps=phase_overlaps,
            decision_basis=decision_basis,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

    def _run_pickup_attempt(
        self,
        *,
        page: action_session.WaitPageProtocol,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        reposition_map_open_started_ms: int | None,
        reposition_map_sync_timestamp_ms: int | None,
        reposition_teleport_started_ms: int | None,
        pickup_timeout_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        fuel_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        decision_basis: FuelDecisionBasisDict | None,
        snapshot_before: PageClientSnapshotDict,
        capture_snapshot: Callable[[], PageClientSnapshotDict],
    ) -> FuelProbeAttemptResultDict:
        """Run the pickup portion of a fuel attempt."""
        try:
            return _fuel_probe_operations.run_pickup_attempt_for_probe(
                self,
                page=page,
                target=target,
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                radar_started_ms=radar_started_ms,
                radar_sync_timestamp_ms=radar_sync_timestamp_ms,
                reposition_map_open_started_ms=reposition_map_open_started_ms,
                reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
                reposition_teleport_started_ms=reposition_teleport_started_ms,
                pickup_timeout_ms=pickup_timeout_ms,
                fuel_before=fuel_before,
                teleport_result=teleport_result,
                fuel_target=fuel_target,
                message_start_index=message_start_index,
                teleport_cycle_ids=teleport_cycle_ids,
                radar_cycle_id=radar_cycle_id,
                decision_basis=decision_basis,
                snapshot_before=snapshot_before,
                capture_snapshot=capture_snapshot,
                dispatch_failure_error=FuelProbeError,
                run_tracked_pickup_phase=run_tracked_pickup_phase,
                get_completed_outcome=fuel_probe_targets._get_completed_pickup_outcome,
                wait_for_outcome=fuel_probe_targets._wait_for_pickup_outcome,
                compute_timeout=effective_pickup_timeout_ms,
            )
        except PickupPhaseError as exc:
            raise FuelProbeError(str(exc)) from exc

    def _probe_single_fuel_target(
        self,
        *,
        target: TeleportTargetDict,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
        teleport_strategy: Literal[
            "sync_before_teleport", "immediate_after_map_open"
        ] = DEFAULT_TELEPORT_STRATEGY,
    ) -> FuelProbeAttemptResultDict:
        """Run one teleport-radar-fuel attempt."""
        return _shared_run_single_fuel_target_attempt(
            self,
            target=target,
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            radar_timeout_ms=radar_timeout_ms,
            pickup_timeout_ms=pickup_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            teleport_strategy=teleport_strategy,
            cdp=self._cdp,
            wait_for_teleport_outcome=_wait_for_teleport_outcome,
            run_tracked_teleport_attempt=run_tracked_teleport_attempt,
            run_tracked_fuel_collection_phase=run_tracked_fuel_collection_phase,
            build_map_sync_timeout_result=self._build_map_sync_timeout_result,
            build_teleport_timeout_result=self._build_teleport_timeout_result,
            finalize_attempt_delay=self._finalize_attempt_delay,
            terrain_provider=get_terrain_map,
            find_visible_target=fuel_probe_targets._find_visible_fuel_target,
            requires_reposition=fuel_probe_targets._visible_fuel_requires_reposition,
            find_landing_tile=fuel_probe_targets._find_visible_fuel_landing_tile,
            get_phase_overlaps=self._get_attempt_phase_overlaps,
            log_target_diagnostic=lambda radar_cycle_id, fuel_target: _log_fuel_target_diagnostic(
                self,
                radar_cycle_id=radar_cycle_id,
                fuel_target=fuel_target,
            ),
            build_radar_timeout_result=self._build_radar_timeout_result,
            build_no_fuel_visible_result=self._build_no_fuel_visible_result,
            build_reposition_map_sync_timeout_result=self._build_reposition_map_sync_timeout_result,
            build_reposition_teleport_timeout_result=self._build_reposition_teleport_timeout_result,
            run_pickup_attempt=self._run_pickup_attempt,
            make_reposition_target=_make_reposition_target,
            teleport_strategy_requires_map_sync=teleport_strategy_requires_map_sync,
            dispatch_failure_error=FuelProbeError,
            unavailable_error=FuelProbeError,
            unexpected_result_error=TeleportProbeError,
            unexpected_missing_target_error=FuelProbeError,
            no_landing_tile_error=FuelProbeError,
            missing_dispatch_error=FuelProbeError,
            acquisition_dispatch_failure_message="map_open command dispatch failed",
            teleport_dispatch_failure_message="teleport command dispatch failed",
            reposition_acquisition_dispatch_failure_message=(
                "map_open command dispatch failed during fuel reposition"
            ),
            reposition_teleport_dispatch_failure_message=(
                "teleport command dispatch failed during fuel reposition"
            ),
            unavailable_message="cdp session is unavailable",
            impossible_map_sync_timeout_message=(
                "teleport outcome reported impossible map_sync_timeout"
            ),
            reposition_impossible_result_message=(
                "teleport outcome reported impossible map_sync_timeout during fuel reposition"
            ),
            reposition_missing_target_message="visible fuel target disappeared unexpectedly",
            no_landing_tile_message="visible fuel target has no teleport landing tile",
            missing_dispatch_message="fuel attempt ended before teleport dispatch",
        )

    def execute_probe(
        self,
        *,
        target_pickups: int,
        max_attempts: int,
        initial_sync_timeout_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelProbeSessionDict:
        """Run the live fuel probe session."""
        return _shared_execute_fuel_probe_session(
            self,
            target_pickups=target_pickups,
            max_attempts=max_attempts,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            radar_timeout_ms=radar_timeout_ms,
            pickup_timeout_ms=pickup_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            target_step=_FUEL_PROBE_TARGET_STEP,
            target_max_radius=_FUEL_PROBE_TARGET_MAX_RADIUS,
            teleport_strategy=_FUEL_PROBE_TELEPORT_STRATEGY,
            terrain_provider=get_terrain_map,
            terrain_unavailable_error=FuelProbeError,
            terrain_unavailable_message="terrain map is unavailable",
        )


def _create_fuel_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> FuelProbe:
    """Factory for FuelProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        FuelProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, FuelProbe)
    return probe


def run_fuel_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    target_pickups: int = 3,
    max_attempts: int = 9,
    initial_sync_timeout_ms: int = 10000,
    map_sync_timeout_ms: int = 3000,
    teleport_timeout_ms: int = 10000,
    radar_timeout_ms: int = 3000,
    pickup_timeout_ms: int = 3000,
    settle_delay_ms: int = 500,
) -> FuelProbeSessionDict:
    """Run a live fuel probe and save the session JSON."""
    return _shared_run_and_save_fuel_probe_session(
        probe_factory=_create_fuel_probe,
        summary_formatter=format_fuel_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
        target_pickups=target_pickups,
        max_attempts=max_attempts,
        initial_sync_timeout_ms=initial_sync_timeout_ms,
        map_sync_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        radar_timeout_ms=radar_timeout_ms,
        pickup_timeout_ms=pickup_timeout_ms,
        settle_delay_ms=settle_delay_ms,
    )


__all__ = [
    "FuelProbe",
    "run_fuel_probe",
]
