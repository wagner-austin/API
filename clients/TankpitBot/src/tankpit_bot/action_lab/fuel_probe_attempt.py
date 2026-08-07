"""Run one fuel-probe attempt: pickup, and the full single-target pass.

The two orchestrators that thread a probe through an attempt. The
Protocols they accept are
:mod:`tankpit_bot.action_lab.fuel_probe_attempt_contracts`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from tankpit_bot._test_hooks import CDPSessionProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseOverlapDict,
    FuelDecisionBasisDict,
)
from tankpit_bot.action_lab.fuel_collection_phase import (
    BuildRadarTimeoutResultProtocol,
    RunPickupAttemptProtocol,
)
from tankpit_bot.action_lab.fuel_probe_attempt_contracts import (
    BuildAttemptResultProtocol,
    BuildMapSyncTimeoutResultProtocol,
    BuildTeleportTimeoutResultProtocol,
    FinalizeAttemptDelayProtocol,
    FuelProbePickupAttemptProtocol,
    FuelProbeSingleAttemptProtocol,
    RunTrackedFuelCollectionPhaseProtocol,
    RunTrackedPickupPhaseProtocol,
    RunTrackedTeleportAttemptProtocol,
)
from tankpit_bot.action_lab.fuel_probe_types import FuelProbeAttemptResultDict
from tankpit_bot.action_lab.fuel_target_phase import (
    BuildNoFuelVisibleResultProtocol,
    BuildRepositionMapSyncTimeoutResultProtocol,
    BuildRepositionTeleportTimeoutResultProtocol,
    FuelTargetPhaseProbeProtocol,
)
from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    capture_page_client_snapshot,
)
from tankpit_bot.action_lab.pickup_phase import (
    PickupImmediateOutcomeProtocol,
    PickupOutcomeWaiterProtocol,
    PickupTimeoutSizerProtocol,
)
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.state import ContainerStateDict


def run_fuel_pickup_attempt(
    probe: FuelProbePickupAttemptProtocol,
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
    dispatch_failure_error: type[Exception],
    build_attempt_result: BuildAttemptResultProtocol,
    get_phase_overlaps: Callable[[], list[ActionPhaseOverlapDict]],
    run_tracked_pickup_phase: RunTrackedPickupPhaseProtocol,
    get_completed_outcome: PickupImmediateOutcomeProtocol,
    wait_for_outcome: PickupOutcomeWaiterProtocol,
    compute_timeout: PickupTimeoutSizerProtocol,
) -> FuelProbeAttemptResultDict:
    """Run the pickup portion of one fuel attempt.

    Args:
        probe: Probe exposing pickup and world-state behavior.
        page: Page used for waits.
        target: Outer attempt target label.
        map_open_started_ms: Initial map-open timestamp.
        map_sync_timestamp_ms: Optional initial map-sync timestamp.
        teleport_started_ms: Initial teleport dispatch timestamp.
        radar_started_ms: Radar dispatch timestamp.
        radar_sync_timestamp_ms: Radar completion timestamp.
        reposition_map_open_started_ms: Optional reposition map-open timestamp.
        reposition_map_sync_timestamp_ms: Optional reposition map-sync timestamp.
        reposition_teleport_started_ms: Optional reposition teleport timestamp.
        pickup_timeout_ms: Base pickup timeout budget.
        fuel_before: Fuel before the full attempt started.
        teleport_result: Teleport result that placed the tank in the viewport.
        fuel_target: Selected visible fuel target.
        message_start_index: Raw message start index for this attempt.
        teleport_cycle_ids: Teleport phase ids collected for this attempt.
        radar_cycle_id: Radar phase id for this attempt.
        decision_basis: Structured fuel-selection decision basis.
        dispatch_failure_error: Error type raised if movement dispatch fails.
        build_attempt_result: Result builder for the completed pickup attempt.
        get_phase_overlaps: Collector for phase-overlap diagnostics.
        run_tracked_pickup_phase: Shared move-and-pickup phase runner.
        get_completed_outcome: Immediate pickup completion checker.
        wait_for_outcome: Terminal pickup outcome waiter.
        compute_timeout: Effective pickup-timeout calculator.

    Returns:
        Terminal pickup result payload.

    Raises:
        PickupPhaseError: Propagated when pickup tracking fails.
        Exception: Raised via ``dispatch_failure_error`` if movement dispatch fails.
    """
    self_state_before_pickup = probe._require_self_state()
    (
        move_cycle,
        pickup_cycle,
        pickup_started_ms,
        pickup_status,
        completion_timestamp_ms,
        fuel_after,
    ) = run_tracked_pickup_phase(
        page,
        probe,
        attempt_label=target["label"],
        target_x=fuel_target["x"],
        target_y=fuel_target["y"],
        current_x=self_state_before_pickup["x"],
        current_y=self_state_before_pickup["y"],
        fuel_before_pickup=self_state_before_pickup["fuel"],
        pickup_timeout_ms=pickup_timeout_ms,
        dispatch_failure_error=dispatch_failure_error,
        get_completed_outcome=get_completed_outcome,
        wait_for_outcome=wait_for_outcome,
        compute_timeout=compute_timeout,
    )
    snapshot_after = capture_snapshot()
    return build_attempt_result(
        target=target,
        status=pickup_status,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        reposition_map_open_started_ms=reposition_map_open_started_ms,
        reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
        reposition_teleport_started_ms=reposition_teleport_started_ms,
        pickup_started_ms=pickup_started_ms,
        completion_timestamp_ms=completion_timestamp_ms,
        fuel_before=fuel_before,
        fuel_after=fuel_after,
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        fuel_target=fuel_target,
        message_start_index=message_start_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        move_cycle_id=move_cycle["cycle_id"],
        pickup_cycle_id=pickup_cycle["cycle_id"],
        phase_overlaps=get_phase_overlaps(),
        decision_basis=decision_basis,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def run_single_fuel_target_attempt(
    probe: FuelProbeSingleAttemptProtocol,
    *,
    target: TeleportTargetDict,
    map_sync_timeout_ms: int,
    teleport_timeout_ms: int,
    radar_timeout_ms: int,
    pickup_timeout_ms: int,
    settle_delay_ms: int,
    teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
    cdp: CDPSessionProtocol | None,
    wait_for_teleport_outcome: TeleportOutcomeWaiterProtocol,
    run_tracked_teleport_attempt: RunTrackedTeleportAttemptProtocol,
    run_tracked_fuel_collection_phase: RunTrackedFuelCollectionPhaseProtocol,
    build_map_sync_timeout_result: BuildMapSyncTimeoutResultProtocol,
    build_teleport_timeout_result: BuildTeleportTimeoutResultProtocol,
    finalize_attempt_delay: FinalizeAttemptDelayProtocol,
    terrain_provider: Callable[[], TerrainMapProtocol | None],
    find_visible_target: Callable[
        [FuelTargetPhaseProbeProtocol],
        ContainerStateDict | None,
    ],
    requires_reposition: Callable[
        [FuelTargetPhaseProbeProtocol, ContainerStateDict],
        bool,
    ],
    find_landing_tile: Callable[
        [FuelTargetPhaseProbeProtocol, ContainerStateDict],
        tuple[int, int] | None,
    ],
    get_phase_overlaps: Callable[[], list[ActionPhaseOverlapDict]],
    log_target_diagnostic: Callable[[int, ContainerStateDict | None], None],
    build_radar_timeout_result: BuildRadarTimeoutResultProtocol,
    build_no_fuel_visible_result: BuildNoFuelVisibleResultProtocol,
    build_reposition_map_sync_timeout_result: BuildRepositionMapSyncTimeoutResultProtocol,
    build_reposition_teleport_timeout_result: BuildRepositionTeleportTimeoutResultProtocol,
    run_pickup_attempt: RunPickupAttemptProtocol,
    make_reposition_target: Callable[[int, int], TeleportTargetDict],
    teleport_strategy_requires_map_sync: Callable[
        [Literal["sync_before_teleport", "immediate_after_map_open"]],
        bool,
    ],
    dispatch_failure_error: type[Exception],
    unavailable_error: type[Exception],
    unexpected_result_error: type[Exception],
    unexpected_missing_target_error: type[Exception],
    no_landing_tile_error: type[Exception],
    missing_dispatch_error: type[Exception],
    acquisition_dispatch_failure_message: str,
    teleport_dispatch_failure_message: str,
    reposition_acquisition_dispatch_failure_message: str,
    reposition_teleport_dispatch_failure_message: str,
    unavailable_message: str,
    impossible_map_sync_timeout_message: str,
    reposition_impossible_result_message: str,
    reposition_missing_target_message: str,
    no_landing_tile_message: str,
    missing_dispatch_message: str,
) -> FuelProbeAttemptResultDict:
    """Run one full teleport-radar-fuel attempt.

    Args:
        probe: Probe exposing teleport, radar, and world-state behavior.
        target: Requested scan target for this attempt.
        map_sync_timeout_ms: Initial map-sync timeout.
        teleport_timeout_ms: Teleport timeout budget.
        radar_timeout_ms: Radar timeout budget.
        pickup_timeout_ms: Base pickup timeout budget.
        settle_delay_ms: Optional post-attempt settle delay.
        teleport_strategy: Teleport acquisition strategy.
        cdp: Active CDP session for page snapshots.
        wait_for_teleport_outcome: Teleport outcome waiter.
        run_tracked_teleport_attempt: Shared acquisition-plus-teleport runner.
        run_tracked_fuel_collection_phase: Shared post-teleport fuel-collection runner.
        build_map_sync_timeout_result: Map-sync-timeout result builder.
        build_teleport_timeout_result: Teleport-timeout result builder.
        finalize_attempt_delay: Optional settle-delay applicator.
        terrain_provider: Terrain lookup used during fuel targeting.
        find_visible_target: Visible-fuel selector.
        requires_reposition: Blocked-fuel reposition predicate.
        find_landing_tile: Landing-tile selector for blocked visible fuel.
        get_phase_overlaps: Phase-overlap collector for diagnostics.
        log_target_diagnostic: Structured target diagnostic logger.
        build_radar_timeout_result: Radar-timeout result builder.
        build_no_fuel_visible_result: No-visible-fuel result builder.
        build_reposition_map_sync_timeout_result: Reposition map-sync-timeout builder.
        build_reposition_teleport_timeout_result: Reposition teleport-timeout builder.
        run_pickup_attempt: Pickup-attempt runner.
        make_reposition_target: Typed reposition-target builder.
        teleport_strategy_requires_map_sync: Strategy helper for acquisition sync.
        dispatch_failure_error: Error type raised on command dispatch failure.
        unavailable_error: Error type raised when runtime dependencies are missing.
        unexpected_result_error: Error type raised on impossible tracked outcomes.
        unexpected_missing_target_error: Error type raised when radar resolution
            reaches an impossible missing-target state.
        no_landing_tile_error: Error type raised when blocked visible fuel has no
            available teleport landing tile.
        missing_dispatch_error: Error type raised when the tracked attempt ends before
            teleport dispatch.
        acquisition_dispatch_failure_message: Error text for map-open dispatch failure.
        teleport_dispatch_failure_message: Error text for teleport dispatch failure.
        reposition_acquisition_dispatch_failure_message: Error text for blocked-fuel
            reposition map-open dispatch failure.
        reposition_teleport_dispatch_failure_message: Error text for blocked-fuel
            reposition teleport dispatch failure.
        unavailable_message: Error text for unavailable runtime dependencies.
        impossible_map_sync_timeout_message: Error text for impossible initial teleport
            map-sync-timeout results.
        reposition_impossible_result_message: Error text for impossible reposition
            map-sync-timeout results.
        reposition_missing_target_message: Error text for an impossible missing visible
            target after radar resolution.
        no_landing_tile_message: Error text for blocked visible fuel without a landing tile.
        missing_dispatch_message: Error text when the tracked attempt ended before
            teleport dispatch.

    Returns:
        Terminal attempt result for the teleport, radar, target-resolution, or
        pickup phase.

    Raises:
        Exception: Raised via the provided error types when tracked sub-phases fail
            or produce impossible states.
    """
    page = probe._require_page()
    if cdp is None:
        raise unavailable_error(unavailable_message)
    cdp_for_snapshot: CDPSessionProtocol = cdp

    def capture_snapshot() -> PageClientSnapshotDict:
        """Capture the live page-client snapshot via the narrowed CDP."""
        return capture_page_client_snapshot(cdp_for_snapshot)

    self_state_before = probe._require_self_state()
    fuel_before = self_state_before["fuel"]
    snapshot_before = capture_snapshot()
    probe._reset_attempt_phase_overlaps()
    attempt = run_tracked_teleport_attempt(
        page,
        probe,
        target,
        cdp=cdp,
        attempt_label=target["label"],
        fuel_before=fuel_before,
        world_timestamp_before=probe.get_world_state()["timestamp_ms"],
        send_acquisition_command=probe.open_map,
        acquisition_command_name="map_open",
        capture_before_map_open=True,
        wait_for_acquisition_sync=teleport_strategy_requires_map_sync(teleport_strategy),
        acquisition_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        wait_for_outcome=wait_for_teleport_outcome,
        dispatch_failure_error=dispatch_failure_error,
        acquisition_dispatch_failure_message=acquisition_dispatch_failure_message,
        teleport_dispatch_failure_message=teleport_dispatch_failure_message,
        unavailable_error=unavailable_error,
        unavailable_message=unavailable_message,
        unexpected_result_error=unexpected_result_error,
        unexpected_result_message=impossible_map_sync_timeout_message,
    )
    message_start_index = attempt.message_start_index
    teleport_cycle = attempt.teleport_cycle
    teleport_cycle_ids = [teleport_cycle["cycle_id"]]
    map_open_started_ms = attempt.acquisition_started_ms
    map_sync_timestamp_ms = attempt.acquisition_sync_timestamp_ms
    if teleport_strategy_requires_map_sync(teleport_strategy) and map_sync_timestamp_ms is None:
        snapshot_after = capture_snapshot()
        result = build_map_sync_timeout_result(
            target=target,
            map_open_started_ms=map_open_started_ms,
            fuel_before=fuel_before,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )
        probe._end_action_phase(teleport_cycle)
        probe._reset_probe_state_to_idle()
        finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)
        return result

    teleport_result = attempt.teleport_result
    teleport_started_ms = attempt.teleport_started_ms
    if teleport_result is None or teleport_started_ms is None:
        raise missing_dispatch_error(missing_dispatch_message)
    if teleport_result["status"] == "teleport_timeout":
        snapshot_after = capture_snapshot()
        result = build_teleport_timeout_result(
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
        finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)
        return result

    result = run_tracked_fuel_collection_phase(
        page,
        probe,
        cdp=cdp,
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        map_sync_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        radar_timeout_ms=radar_timeout_ms,
        pickup_timeout_ms=pickup_timeout_ms,
        fuel_before=fuel_before,
        teleport_result=teleport_result,
        message_start_index=message_start_index,
        teleport_cycle_ids=teleport_cycle_ids,
        teleport_strategy=teleport_strategy,
        snapshot_before=snapshot_before,
        capture_snapshot=capture_snapshot,
        terrain_provider=terrain_provider,
        find_visible_target=find_visible_target,
        requires_reposition=requires_reposition,
        find_landing_tile=find_landing_tile,
        get_phase_overlaps=get_phase_overlaps,
        log_target_diagnostic=log_target_diagnostic,
        build_radar_timeout_result=build_radar_timeout_result,
        build_no_fuel_visible_result=build_no_fuel_visible_result,
        build_reposition_map_sync_timeout_result=build_reposition_map_sync_timeout_result,
        build_reposition_teleport_timeout_result=build_reposition_teleport_timeout_result,
        run_pickup_attempt=run_pickup_attempt,
        make_reposition_target=make_reposition_target,
        wait_for_teleport_outcome=wait_for_teleport_outcome,
        teleport_strategy_requires_map_sync=teleport_strategy_requires_map_sync,
        dispatch_failure_error=dispatch_failure_error,
        unexpected_result_error=unexpected_result_error,
        unexpected_missing_target_error=unexpected_missing_target_error,
        no_landing_tile_error=no_landing_tile_error,
        unavailable_error=unavailable_error,
        unavailable_message=unavailable_message,
        no_landing_tile_message=no_landing_tile_message,
        impossible_result_message=reposition_impossible_result_message,
        acquisition_dispatch_failure_message=reposition_acquisition_dispatch_failure_message,
        teleport_dispatch_failure_message=reposition_teleport_dispatch_failure_message,
        unexpected_missing_target_message=reposition_missing_target_message,
    )
    finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)
    return result


__all__ = [
    "run_fuel_pickup_attempt",
    "run_single_fuel_target_attempt",
]
