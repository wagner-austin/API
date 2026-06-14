"""Live teleport-radar-fuel action probe harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from tankpit_bot.action_lab import fuel_probe_operations as _fuel_probe_operations
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict, FuelDecisionBasisDict
from tankpit_bot.action_lab.fuel_collection_phase import (
    run_tracked_fuel_collection_phase as _shared_run_tracked_fuel_collection_phase,
)
from tankpit_bot.action_lab.fuel_probe_attempt import (
    run_single_fuel_target_attempt as _shared_run_single_fuel_target_attempt,
)
from tankpit_bot.action_lab.fuel_probe_diagnostics import (
    format_fuel_probe_summary as _shared_format_fuel_probe_summary,
)
from tankpit_bot.action_lab.fuel_probe_diagnostics import (
    format_visible_fuel_entries as _shared_format_visible_fuel_entries,
)
from tankpit_bot.action_lab.fuel_probe_diagnostics import (
    log_fuel_target_diagnostic as _shared_log_fuel_target_diagnostic,
)
from tankpit_bot.action_lab.fuel_probe_entrypoint import (
    run_and_save_fuel_probe_session as _shared_run_and_save_fuel_probe_session,
)
from tankpit_bot.action_lab.fuel_probe_runner import (
    execute_fuel_probe_session as _shared_execute_fuel_probe_session,
)
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
)
from tankpit_bot.action_lab.fuel_target_phase import (
    FuelTargetPhaseProbeProtocol,
)
from tankpit_bot.action_lab.fuel_target_phase import (
    resolve_fuel_target_after_radar as _shared_resolve_fuel_target_after_radar,
)
from tankpit_bot.action_lab.fuel_targeting import (
    FuelTargetingError,
    find_visible_fuel_landing_tile,
    visible_fuel_requires_reposition,
)
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.pickup_phase import PickupPhaseError, run_tracked_pickup_phase
from tankpit_bot.action_lab.pickup_phase import (
    effective_pickup_timeout_ms as _shared_effective_pickup_timeout_ms,
)
from tankpit_bot.action_lab.pickup_phase import (
    get_completed_pickup_outcome as _shared_get_completed_pickup_outcome,
)
from tankpit_bot.action_lab.pickup_phase import (
    wait_for_pickup_outcome as _shared_wait_for_pickup_outcome,
)
from tankpit_bot.action_lab.radar_phase import (
    clear_stale_radar_completion as _shared_clear_stale_radar_completion,
)
from tankpit_bot.action_lab.teleport import (
    DEFAULT_TELEPORT_STRATEGY,
    TeleportProbe,
    TeleportProbeError,
    _teleport_strategy_requires_map_sync,
    _wait_for_teleport_outcome,
)
from tankpit_bot.action_lab.teleport_attempt import (
    run_tracked_teleport_attempt as _shared_run_tracked_teleport_attempt,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.ai.equipment import find_best_fuel
from tankpit_bot.sniffer.world_state import get_terrain_map
from tankpit_bot.state.types import ContainerStateDict

_FUEL_PROBE_TARGET_STEP = 16
_FUEL_PROBE_TARGET_MAX_RADIUS = 48
_FUEL_PROBE_TELEPORT_STRATEGY: Literal["sync_before_teleport", "immediate_after_map_open"] = (
    DEFAULT_TELEPORT_STRATEGY
)
run_tracked_teleport_attempt = _shared_run_tracked_teleport_attempt
resolve_fuel_target_after_radar_phase = _shared_resolve_fuel_target_after_radar
run_tracked_fuel_collection_phase = _shared_run_tracked_fuel_collection_phase


class FuelProbeError(Exception):
    """Raised when the fuel probe cannot proceed."""


def _log_fuel_target_diagnostic(
    probe: TeleportProbe,
    *,
    radar_cycle_id: int,
    fuel_target: ContainerStateDict | None,
) -> None:
    """Emit one structured diagnostic line after radar target resolution."""
    _shared_log_fuel_target_diagnostic(
        probe,
        radar_cycle_id=radar_cycle_id,
        fuel_target=fuel_target,
        terrain_provider=get_terrain_map,
    )


def _find_visible_fuel_target(
    probe: FuelTargetPhaseProbeProtocol,
    allow_unreachable: bool = False,
) -> ContainerStateDict | None:
    """Return the best currently visible fuel container."""
    terrain = get_terrain_map()
    if terrain is None:
        raise FuelProbeError("terrain map is unavailable")
    self_state = probe.get_self_state()
    if self_state is None:
        raise FuelProbeError("self state is unavailable")
    world = probe.get_world_state()
    return find_best_fuel(
        world,
        self_state,
        terrain,
        allow_unreachable=allow_unreachable,
        now_ms=world["timestamp_ms"],
        minimum_volume=1,
    )


def _format_visible_fuel_entries(
    probe: FuelTargetPhaseProbeProtocol,
    *,
    fuel_target: ContainerStateDict | None,
) -> str:
    """Format the currently visible fuel candidates for diagnostics.

    Args:
        probe: Probe exposing current world and self state.
        fuel_target: Selected target for the current decision, if any.

    Returns:
        ``"unavailable"`` when terrain or self state is missing, ``"none"``
        when no visible fuel is tracked, or a compact candidate summary.
    """
    return _shared_format_visible_fuel_entries(
        probe,
        fuel_target=fuel_target,
        terrain_provider=get_terrain_map,
    )


def _visible_fuel_requires_reposition(
    probe: FuelTargetPhaseProbeProtocol,
    fuel_target: ContainerStateDict,
) -> bool:
    """Return whether a visible fuel target needs a reposition teleport."""
    try:
        return visible_fuel_requires_reposition(probe, fuel_target)
    except FuelTargetingError as exc:
        raise FuelProbeError(str(exc)) from exc


def _find_visible_fuel_landing_tile(
    probe: FuelTargetPhaseProbeProtocol,
    fuel_target: ContainerStateDict,
) -> tuple[int, int] | None:
    """Return the landing tile for a blocked visible fuel target."""
    try:
        return find_visible_fuel_landing_tile(probe, fuel_target)
    except FuelTargetingError as exc:
        raise FuelProbeError(str(exc)) from exc


def _find_visible_fuel_target_for_phase(
    probe: FuelTargetPhaseProbeProtocol,
    allow_unreachable: bool,
) -> ContainerStateDict | None:
    """Typed bridge for shared fuel-target phase selection."""
    return _find_visible_fuel_target(probe, allow_unreachable)


def _visible_fuel_requires_reposition_for_phase(
    probe: FuelTargetPhaseProbeProtocol,
    fuel_target: ContainerStateDict,
) -> bool:
    """Typed bridge for shared blocked-fuel reposition checks."""
    return _visible_fuel_requires_reposition(probe, fuel_target)


def _find_visible_fuel_landing_tile_for_phase(
    probe: FuelTargetPhaseProbeProtocol,
    fuel_target: ContainerStateDict,
) -> tuple[int, int] | None:
    """Typed bridge for shared blocked-fuel landing selection."""
    return _find_visible_fuel_landing_tile(probe, fuel_target)


def _make_reposition_target(target_x: int, target_y: int) -> TeleportTargetDict:
    """Return a typed target label for a fuel reposition teleport."""
    return TeleportTargetDict(
        label=f"fuel_reposition_{target_x}_{target_y}",
        x=target_x,
        y=target_y,
    )


def _wait_for_pickup_outcome(
    page: action_session.WaitPageProtocol,
    probe: action_session.BufferedWorldStateProviderProtocol,
    *,
    target_x: int,
    target_y: int,
    pickup_started_ms: int,
    fuel_before: int,
    timeout_ms: int,
) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
    """Wait for a fuel pickup to complete or time out."""
    try:
        return _shared_wait_for_pickup_outcome(
            page,
            probe,
            target_x=target_x,
            target_y=target_y,
            pickup_started_ms=pickup_started_ms,
            fuel_before=fuel_before,
            timeout_ms=timeout_ms,
        )
    except PickupPhaseError as exc:
        raise FuelProbeError(str(exc)) from exc


def _clear_stale_radar_completion() -> None:
    """Drain any leaked radar-complete signals before starting a new scan."""
    _shared_clear_stale_radar_completion()


def _effective_pickup_timeout_ms(
    *,
    current_x: int,
    current_y: int,
    target_x: int,
    target_y: int,
    base_timeout_ms: int,
) -> int:
    """Return a pickup timeout sized for the current travel distance.

    Args:
        current_x: Current self X tile.
        current_y: Current self Y tile.
        target_x: Pickup target X tile.
        target_y: Pickup target Y tile.
        base_timeout_ms: Configured minimum pickup timeout.

    Returns:
        Timeout in milliseconds large enough for the move plus pickup settle.
    """
    return _shared_effective_pickup_timeout_ms(
        current_x=current_x,
        current_y=current_y,
        target_x=target_x,
        target_y=target_y,
        base_timeout_ms=base_timeout_ms,
    )


def _get_completed_pickup_outcome(
    probe: action_session.WorldStateProviderProtocol,
    *,
    target_x: int,
    target_y: int,
    fuel_before: int,
) -> tuple[Literal["picked_up_fuel"], int, int] | None:
    """Return a completed pickup outcome once the fuel credit is observed."""
    try:
        return _shared_get_completed_pickup_outcome(
            probe,
            target_x=target_x,
            target_y=target_y,
            fuel_before=fuel_before,
        )
    except PickupPhaseError as exc:
        raise FuelProbeError(str(exc)) from exc


def format_fuel_probe_summary(session: FuelProbeSessionDict) -> str:
    """Format a compact summary for a fuel probe session."""
    return _shared_format_fuel_probe_summary(session)


class FuelProbe(TeleportProbe):
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
                get_completed_outcome=_get_completed_pickup_outcome,
                wait_for_outcome=_wait_for_pickup_outcome,
                compute_timeout=_effective_pickup_timeout_ms,
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
            find_visible_target=_find_visible_fuel_target_for_phase,
            requires_reposition=_visible_fuel_requires_reposition_for_phase,
            find_landing_tile=_find_visible_fuel_landing_tile_for_phase,
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
            teleport_strategy_requires_map_sync=_teleport_strategy_requires_map_sync,
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
        probe_factory=FuelProbe,
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


__all__ = ["FuelProbe", "FuelProbeError", "format_fuel_probe_summary", "run_fuel_probe"]
