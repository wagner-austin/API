"""Shared post-teleport equipment-collection helpers for action-lab probes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from tankpit_bot._test_hooks import CDPSessionProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict, ActionPhaseOverlapDict
from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeAttemptResultDict
from tankpit_bot.action_lab.equipment_target_phase import (
    BuildEquipmentRepositionMapSyncTimeoutResultProtocol,
    BuildEquipmentRepositionTeleportTimeoutResultProtocol,
    BuildNoEquipmentVisibleResultProtocol,
    EquipmentTargetPhaseProbeProtocol,
    EquipmentTargetResolution,
    resolve_equipment_target_after_radar,
)
from tankpit_bot.action_lab.radar_phase import run_tracked_radar_phase
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.state import ContainerStateDict, SelfStateDict, WorldStateDict


class EquipmentCollectionPhaseProbeProtocol(
    action_session.BufferedWorldStateProviderProtocol,
    Protocol,
):
    """Minimal probe interface required for post-teleport equipment collection."""

    def use_radar(self) -> bool:
        """Dispatch one radar command."""

    def open_map(self) -> bool:
        """Dispatch one map-open command."""

    def teleport_to(self, x: int, y: int) -> bool:
        """Dispatch one teleport command."""

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""

    def get_self_state(self) -> SelfStateDict | None:
        """Return the current self state when available."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the required current self state."""

    def _start_action_phase(
        self,
        phase: Literal["teleport", "radar"],
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Start one action phase cycle."""

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """Close one active action phase."""

    def _reset_probe_state_to_idle(self) -> None:
        """Reset probe state to idle after a phase settles."""


class RunRadarPhaseProtocol(Protocol):
    """Callable protocol for one tracked radar phase."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: EquipmentCollectionPhaseProbeProtocol,
        *,
        attempt_label: str,
        timeout_ms: int,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "radar command dispatch failed",
    ) -> tuple[ActionPhaseCycleDict, int, int | None]:
        """Run one tracked radar phase."""


class ResolveEquipmentTargetPhaseProtocol(Protocol):
    """Callable protocol for post-radar equipment-target resolution."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: EquipmentTargetPhaseProbeProtocol,
        *,
        cdp: CDPSessionProtocol | None,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        terrain_provider: Callable[[], TerrainMapProtocol | None],
        find_visible_target: Callable[
            [EquipmentTargetPhaseProbeProtocol],
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
        build_no_equipment_visible_result: BuildNoEquipmentVisibleResultProtocol,
        build_reposition_map_sync_timeout_result: (
            BuildEquipmentRepositionMapSyncTimeoutResultProtocol
        ),
        build_reposition_teleport_timeout_result: (
            BuildEquipmentRepositionTeleportTimeoutResultProtocol
        ),
        make_reposition_target: Callable[[int, int], TeleportTargetDict],
        wait_for_teleport_outcome: TeleportOutcomeWaiterProtocol,
        teleport_strategy_requires_map_sync: Callable[
            [Literal["sync_before_teleport", "immediate_after_map_open"]],
            bool,
        ],
        no_landing_tile_error: type[Exception],
        dispatch_failure_error: type[Exception],
        unavailable_error: type[Exception],
        unexpected_result_error: type[Exception],
        unavailable_message: str,
        no_landing_tile_message: str,
        impossible_result_message: str,
        acquisition_dispatch_failure_message: str,
        teleport_dispatch_failure_message: str,
    ) -> EquipmentTargetResolution:
        """Resolve one post-radar equipment target."""


class BuildEquipmentRadarTimeoutResultProtocol(Protocol):
    """Callable protocol for radar-timeout terminal result builders."""

    def __call__(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
    ) -> EquipmentProbeAttemptResultDict:
        """Build one radar-timeout terminal result."""


class RunEquipmentPickupAttemptProtocol(Protocol):
    """Callable protocol for one tracked equipment-pickup attempt."""

    def __call__(
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
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        equipment_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
    ) -> EquipmentProbeAttemptResultDict:
        """Run one tracked equipment-pickup attempt."""


run_radar_phase: RunRadarPhaseProtocol = run_tracked_radar_phase
resolve_equipment_target_phase: ResolveEquipmentTargetPhaseProtocol = (
    resolve_equipment_target_after_radar
)


def run_tracked_equipment_collection_phase(
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
        [EquipmentTargetPhaseProbeProtocol],
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
    """Run the shared radar-to-pickup portion of one equipment attempt.

    Args:
        page: Page used for waits.
        probe: Probe implementation exposing radar, world, and pickup behavior.
        cdp: Active CDP session for any reposition teleport snapshots.
        target: Requested scan destination for the enclosing attempt.
        map_open_started_ms: Initial map-open timestamp.
        map_sync_timestamp_ms: Optional initial map-sync timestamp.
        teleport_started_ms: Initial teleport dispatch timestamp.
        map_sync_timeout_ms: Timeout for optional reposition map sync.
        teleport_timeout_ms: Timeout for optional reposition teleport outcome.
        radar_timeout_ms: Timeout for the radar phase.
        pickup_timeout_ms: Base timeout for the pickup phase.
        inventory_count_before: Inventory total before the attempt began.
        teleport_result: Successful teleport result for the enclosing attempt.
        message_start_index: Raw message start index for the enclosing attempt.
        teleport_cycle_ids: Teleport cycle ids collected so far.
        teleport_strategy: Acquisition strategy for any reposition.
        terrain_provider: Terrain lookup forwarded to target resolution.
        find_visible_target: Visible-equipment selector.
        requires_reposition: Blocked-equipment predicate.
        find_landing_tile: Landing-tile selector for blocked visible equipment.
        get_phase_overlaps: Phase-overlap collector.
        build_radar_timeout_result: Radar-timeout terminal result builder.
        build_no_equipment_visible_result: No-visible-equipment terminal builder.
        build_reposition_map_sync_timeout_result: Reposition map-sync builder.
        build_reposition_teleport_timeout_result: Reposition teleport builder.
        run_pickup_attempt: Pickup runner for actionable visible equipment.
        make_reposition_target: Reposition target builder.
        wait_for_teleport_outcome: Teleport waiter for blocked-equipment reposition.
        teleport_strategy_requires_map_sync: Reposition sync policy helper.
        dispatch_failure_error: Error type raised on dispatch failures.
        unexpected_result_error: Error type raised on impossible outcomes.
        unexpected_missing_target_error: Error type raised when resolution
            returns no target and no terminal result.
        no_landing_tile_error: Error type raised on missing landing tiles.
        unavailable_error: Error type raised for unavailable dependencies.
        unavailable_message: Error text for missing runtime dependencies.
        no_landing_tile_message: Error text for blocked equipment with no
            landing tile.
        impossible_result_message: Error text for impossible reposition outcomes.
        acquisition_dispatch_failure_message: Error text for reposition map-open
            dispatch failure.
        teleport_dispatch_failure_message: Error text for reposition teleport
            dispatch failure.
        unexpected_missing_target_message: Error text for impossible missing
            visible target after target resolution returns no terminal result.

    Returns:
        Terminal result for the radar, target-resolution, or pickup phase.

    Raises:
        Exception: Raised via the provided error types when any sub-phase fails.
    """
    radar_cycle, radar_started_ms, radar_sync_timestamp_ms = run_radar_phase(
        page,
        probe,
        attempt_label=target["label"],
        timeout_ms=radar_timeout_ms,
        dispatch_failure_error=dispatch_failure_error,
    )
    if radar_sync_timestamp_ms is None:
        return build_radar_timeout_result(
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            inventory_count_before=inventory_count_before,
            teleport_result=teleport_result,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle["cycle_id"],
            phase_overlaps=get_phase_overlaps(),
        )

    equipment_target_probe: EquipmentTargetPhaseProbeProtocol = probe
    resolution = resolve_equipment_target_phase(
        page,
        equipment_target_probe,
        cdp=cdp,
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        map_sync_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        inventory_count_before=inventory_count_before,
        teleport_result=teleport_result,
        message_start_index=message_start_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle["cycle_id"],
        teleport_strategy=teleport_strategy,
        terrain_provider=terrain_provider,
        find_visible_target=find_visible_target,
        requires_reposition=requires_reposition,
        find_landing_tile=find_landing_tile,
        get_phase_overlaps=get_phase_overlaps,
        build_no_equipment_visible_result=build_no_equipment_visible_result,
        build_reposition_map_sync_timeout_result=build_reposition_map_sync_timeout_result,
        build_reposition_teleport_timeout_result=build_reposition_teleport_timeout_result,
        make_reposition_target=make_reposition_target,
        wait_for_teleport_outcome=wait_for_teleport_outcome,
        teleport_strategy_requires_map_sync=teleport_strategy_requires_map_sync,
        no_landing_tile_error=no_landing_tile_error,
        dispatch_failure_error=dispatch_failure_error,
        unavailable_error=unavailable_error,
        unexpected_result_error=unexpected_result_error,
        unavailable_message=unavailable_message,
        no_landing_tile_message=no_landing_tile_message,
        impossible_result_message=impossible_result_message,
        acquisition_dispatch_failure_message=acquisition_dispatch_failure_message,
        teleport_dispatch_failure_message=teleport_dispatch_failure_message,
    )
    if resolution.terminal_result is not None:
        return resolution.terminal_result
    if resolution.equipment_target is None:
        raise unexpected_missing_target_error(unexpected_missing_target_message)
    return run_pickup_attempt(
        page=page,
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        reposition_map_open_started_ms=resolution.reposition_map_open_started_ms,
        reposition_map_sync_timestamp_ms=resolution.reposition_map_sync_timestamp_ms,
        reposition_teleport_started_ms=resolution.reposition_teleport_started_ms,
        pickup_timeout_ms=pickup_timeout_ms,
        inventory_count_before=inventory_count_before,
        teleport_result=resolution.teleport_result,
        equipment_target=resolution.equipment_target,
        message_start_index=message_start_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle["cycle_id"],
    )


__all__ = [
    "BuildEquipmentRadarTimeoutResultProtocol",
    "EquipmentCollectionPhaseProbeProtocol",
    "ResolveEquipmentTargetPhaseProtocol",
    "RunEquipmentPickupAttemptProtocol",
    "RunRadarPhaseProtocol",
    "resolve_equipment_target_phase",
    "run_radar_phase",
    "run_tracked_equipment_collection_phase",
]
