"""Shared single-attempt helpers for the live equipment probe."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from tankpit_bot._test_hooks import CDPSessionProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict
from tankpit_bot.action_lab.equipment_collection_phase import (
    BuildEquipmentRadarTimeoutResultProtocol,
    EquipmentCollectionPhaseProbeProtocol,
    RunEquipmentPickupAttemptProtocol,
)
from tankpit_bot.action_lab.equipment_pickup import total_inventory_count
from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeAttemptResultDict
from tankpit_bot.action_lab.equipment_target_phase import (
    BuildEquipmentRepositionMapSyncTimeoutResultProtocol,
    BuildEquipmentRepositionTeleportTimeoutResultProtocol,
    BuildNoEquipmentVisibleResultProtocol,
    EquipmentTargetPhaseProbeProtocol,
)
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    TrackedTeleportAttempt,
)
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.state import ContainerStateDict, SelfStateDict, WorldStateDict


class EquipmentProbeSingleAttemptProtocol(
    EquipmentCollectionPhaseProbeProtocol,
    TeleportAttemptProbeProtocol,
    Protocol,
):
    """Minimal probe interface required for one full equipment attempt."""

    _cdp: CDPSessionProtocol | None

    def _require_page(self) -> action_session.WaitPageProtocol:
        """Return the live page."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the current self state."""

    def _reset_attempt_phase_overlaps(self) -> None:
        """Reset any per-attempt phase-overlap tracking."""

    def open_map(self) -> bool:
        """Dispatch one map-open command."""

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""


class BuildEquipmentMapSyncTimeoutResultProtocol(Protocol):
    """Callable protocol for map-sync-timeout result builders."""

    def __call__(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        inventory_count_before: int,
        message_start_index: int,
        teleport_cycle_ids: list[int],
    ) -> EquipmentProbeAttemptResultDict:
        """Build one map-sync-timeout result."""


class BuildEquipmentTeleportTimeoutResultProtocol(Protocol):
    """Callable protocol for teleport-timeout result builders."""

    def __call__(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
    ) -> EquipmentProbeAttemptResultDict:
        """Build one teleport-timeout result."""


class FinalizeAttemptDelayProtocol(Protocol):
    """Callable protocol for optional post-attempt settle delays."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        *,
        settle_delay_ms: int,
    ) -> None:
        """Apply one optional post-attempt delay."""


class RunTrackedTeleportAttemptProtocol(Protocol):
    """Callable protocol for the shared teleport-attempt runner."""

    def __call__(
        self,
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
        """Run one tracked teleport attempt."""


class RunTrackedEquipmentCollectionPhaseProtocol(Protocol):
    """Callable protocol for the shared post-teleport equipment-collection phase."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: EquipmentCollectionPhaseProbeProtocol,
        *,
        cdp: CDPSessionProtocol | None,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        terrain_provider: Callable[[], TerrainMapProtocol | None],
        find_visible_target: Callable[
            [EquipmentTargetPhaseProbeProtocol, bool],
            ContainerStateDict | None,
        ],
        requires_reposition: Callable[
            [EquipmentTargetPhaseProbeProtocol, ContainerStateDict],
            bool,
        ],
        find_landing_tile: Callable[
            [EquipmentTargetPhaseProbeProtocol, ContainerStateDict],
            tuple[int, int] | None,
        ],
        get_phase_overlaps: Callable[[], list[ActionPhaseOverlapDict]],
        build_radar_timeout_result: BuildEquipmentRadarTimeoutResultProtocol,
        build_no_equipment_visible_result: BuildNoEquipmentVisibleResultProtocol,
        build_reposition_map_sync_timeout_result: (
            BuildEquipmentRepositionMapSyncTimeoutResultProtocol
        ),
        build_reposition_teleport_timeout_result: (
            BuildEquipmentRepositionTeleportTimeoutResultProtocol
        ),
        run_pickup_attempt: RunEquipmentPickupAttemptProtocol,
        make_reposition_target: Callable[[int, int], TeleportTargetDict],
        wait_for_teleport_outcome: TeleportOutcomeWaiterProtocol,
        teleport_strategy_requires_map_sync: Callable[
            [Literal["sync_before_teleport", "immediate_after_map_open"]],
            bool,
        ],
        dispatch_failure_error: type[Exception],
        unexpected_result_error: type[Exception],
        unexpected_missing_target_error: type[Exception],
        no_landing_tile_error: type[Exception],
        unavailable_error: type[Exception],
        unavailable_message: str,
        no_landing_tile_message: str,
        impossible_result_message: str,
        acquisition_dispatch_failure_message: str,
        teleport_dispatch_failure_message: str,
        unexpected_missing_target_message: str,
    ) -> EquipmentProbeAttemptResultDict:
        """Run the shared radar-to-pickup phase."""


def run_single_equipment_target_attempt(
    probe: EquipmentProbeSingleAttemptProtocol,
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
    run_tracked_equipment_collection_phase: RunTrackedEquipmentCollectionPhaseProtocol,
    build_map_sync_timeout_result: BuildEquipmentMapSyncTimeoutResultProtocol,
    build_teleport_timeout_result: BuildEquipmentTeleportTimeoutResultProtocol,
    finalize_attempt_delay: FinalizeAttemptDelayProtocol,
    terrain_provider: Callable[[], TerrainMapProtocol | None],
    find_visible_target: Callable[
        [EquipmentTargetPhaseProbeProtocol, bool],
        ContainerStateDict | None,
    ],
    requires_reposition: Callable[
        [EquipmentTargetPhaseProbeProtocol, ContainerStateDict],
        bool,
    ],
    find_landing_tile: Callable[
        [EquipmentTargetPhaseProbeProtocol, ContainerStateDict],
        tuple[int, int] | None,
    ],
    get_phase_overlaps: Callable[[], list[ActionPhaseOverlapDict]],
    build_radar_timeout_result: BuildEquipmentRadarTimeoutResultProtocol,
    build_no_equipment_visible_result: BuildNoEquipmentVisibleResultProtocol,
    build_reposition_map_sync_timeout_result: (
        BuildEquipmentRepositionMapSyncTimeoutResultProtocol
    ),
    build_reposition_teleport_timeout_result: (
        BuildEquipmentRepositionTeleportTimeoutResultProtocol
    ),
    run_pickup_attempt: RunEquipmentPickupAttemptProtocol,
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
) -> EquipmentProbeAttemptResultDict:
    """Run one full teleport-radar-equipment attempt.

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
        run_tracked_equipment_collection_phase: Shared post-teleport
            equipment-collection runner.
        build_map_sync_timeout_result: Map-sync-timeout result builder.
        build_teleport_timeout_result: Teleport-timeout result builder.
        finalize_attempt_delay: Optional settle-delay applicator.
        terrain_provider: Terrain lookup forwarded to target resolution.
        find_visible_target: Visible-equipment selector.
        requires_reposition: Blocked-equipment predicate.
        find_landing_tile: Landing-tile selector for blocked visible equipment.
        get_phase_overlaps: Phase-overlap collector.
        build_radar_timeout_result: Radar-timeout result builder.
        build_no_equipment_visible_result: No-visible-equipment result builder.
        build_reposition_map_sync_timeout_result: Reposition map-sync builder.
        build_reposition_teleport_timeout_result: Reposition teleport builder.
        run_pickup_attempt: Pickup-attempt runner.
        make_reposition_target: Typed reposition-target builder.
        teleport_strategy_requires_map_sync: Strategy helper for acquisition sync.
        dispatch_failure_error: Error type raised on command dispatch failure.
        unavailable_error: Error type raised when runtime dependencies are missing.
        unexpected_result_error: Error type raised on impossible tracked outcomes.
        unexpected_missing_target_error: Error type raised when radar resolution
            reaches an impossible missing-target state.
        no_landing_tile_error: Error type raised when blocked visible equipment
            has no available teleport landing tile.
        missing_dispatch_error: Error type raised when the tracked attempt ends
            before teleport dispatch.
        acquisition_dispatch_failure_message: Error text for map-open dispatch
            failure.
        teleport_dispatch_failure_message: Error text for teleport dispatch
            failure.
        reposition_acquisition_dispatch_failure_message: Error text for blocked
            equipment reposition map-open failure.
        reposition_teleport_dispatch_failure_message: Error text for blocked
            equipment reposition teleport failure.
        unavailable_message: Error text for unavailable runtime dependencies.
        impossible_map_sync_timeout_message: Error text for impossible initial
            teleport map-sync-timeout outcomes.
        reposition_impossible_result_message: Error text for impossible
            reposition map-sync-timeout outcomes.
        reposition_missing_target_message: Error text for an impossible missing
            visible target after radar resolution.
        no_landing_tile_message: Error text for blocked visible equipment
            without a landing tile.
        missing_dispatch_message: Error text when the tracked attempt ended
            before teleport dispatch.

    Returns:
        Terminal attempt result for the teleport, radar, target-resolution, or
        pickup phase.

    Raises:
        Exception: Raised via the provided error types when tracked sub-phases
            fail or produce impossible states.
    """
    page = probe._require_page()
    self_state_before = probe._require_self_state()
    inventory_count_before = total_inventory_count(get_inventory_state(get_world_service()))
    probe._reset_attempt_phase_overlaps()
    attempt = run_tracked_teleport_attempt(
        page,
        probe,
        target,
        cdp=cdp,
        attempt_label=target["label"],
        fuel_before=self_state_before["fuel"],
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
        result = build_map_sync_timeout_result(
            target=target,
            map_open_started_ms=map_open_started_ms,
            inventory_count_before=inventory_count_before,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
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
        result = build_teleport_timeout_result(
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            inventory_count_before=inventory_count_before,
            teleport_result=teleport_result,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
        )
        finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)
        return result

    result = run_tracked_equipment_collection_phase(
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
        inventory_count_before=inventory_count_before,
        teleport_result=teleport_result,
        message_start_index=message_start_index,
        teleport_cycle_ids=teleport_cycle_ids,
        teleport_strategy=teleport_strategy,
        terrain_provider=terrain_provider,
        find_visible_target=find_visible_target,
        requires_reposition=requires_reposition,
        find_landing_tile=find_landing_tile,
        get_phase_overlaps=get_phase_overlaps,
        build_radar_timeout_result=build_radar_timeout_result,
        build_no_equipment_visible_result=build_no_equipment_visible_result,
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
    "BuildEquipmentMapSyncTimeoutResultProtocol",
    "BuildEquipmentTeleportTimeoutResultProtocol",
    "EquipmentProbeSingleAttemptProtocol",
    "FinalizeAttemptDelayProtocol",
    "RunTrackedEquipmentCollectionPhaseProtocol",
    "RunTrackedTeleportAttemptProtocol",
    "run_single_equipment_target_attempt",
]
