"""Shared equipment target-resolution helpers for action-lab probes.

Mirrors the fuel-side target-phase shape but operates on equipment containers
and omits the volume-driven decision-basis diagnostics (equipment selection
is nearest-first).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple, Protocol

from tankpit_bot._test_hooks import CDPSessionProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict
from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeAttemptResultDict
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    run_tracked_teleport_attempt,
)
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.state import ContainerStateDict, SelfStateDict, WorldStateDict

run_equipment_reposition_attempt = run_tracked_teleport_attempt


class EquipmentTargetPhaseProbeProtocol(TeleportAttemptProbeProtocol, Protocol):
    """Minimal probe interface required for post-radar equipment target resolution."""

    def open_map(self) -> bool:
        """Dispatch one map-open command."""

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""

    def get_self_state(self) -> SelfStateDict | None:
        """Return the current self state when available."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the required current self state."""


class BlockedEquipmentRepositionResult(NamedTuple):
    """Typed result for one optional blocked-equipment reposition attempt."""

    teleport_result: TeleportAttemptResultDict | None
    terminal_result: EquipmentProbeAttemptResultDict | None
    reposition_map_open_started_ms: int | None
    reposition_map_sync_timestamp_ms: int | None
    reposition_teleport_started_ms: int | None


class EquipmentTargetResolution(NamedTuple):
    """Typed result for post-radar equipment target selection and reposition."""

    equipment_target: ContainerStateDict | None
    teleport_result: TeleportAttemptResultDict
    terminal_result: EquipmentProbeAttemptResultDict | None
    reposition_map_open_started_ms: int | None
    reposition_map_sync_timestamp_ms: int | None
    reposition_teleport_started_ms: int | None


class BuildNoEquipmentVisibleResultProtocol(Protocol):
    """Callable protocol for the no-visible-equipment terminal result builder."""

    def __call__(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
    ) -> EquipmentProbeAttemptResultDict:
        """Build one no-visible-equipment terminal result."""


class BuildEquipmentRepositionMapSyncTimeoutResultProtocol(Protocol):
    """Callable protocol for blocked-equipment reposition sync-timeout results."""

    def __call__(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        reposition_map_open_started_ms: int,
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        equipment_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
    ) -> EquipmentProbeAttemptResultDict:
        """Build one blocked-equipment reposition map-sync-timeout result."""


class BuildEquipmentRepositionTeleportTimeoutResultProtocol(Protocol):
    """Callable protocol for blocked-equipment reposition teleport-timeout results."""

    def __call__(
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
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        equipment_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
    ) -> EquipmentProbeAttemptResultDict:
        """Build one blocked-equipment reposition teleport-timeout result."""


def _run_blocked_equipment_reposition(
    page: action_session.WaitPageProtocol,
    probe: EquipmentTargetPhaseProbeProtocol,
    *,
    cdp: CDPSessionProtocol | None,
    target: TeleportTargetDict,
    equipment_target: ContainerStateDict,
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
    wait_for_teleport_outcome: TeleportOutcomeWaiterProtocol,
    teleport_strategy_requires_map_sync: Callable[
        [Literal["sync_before_teleport", "immediate_after_map_open"]],
        bool,
    ],
    find_landing_tile: Callable[
        [EquipmentTargetPhaseProbeProtocol, ContainerStateDict],
        tuple[int, int] | None,
    ],
    get_phase_overlaps: Callable[[], list[ActionPhaseOverlapDict]],
    build_reposition_map_sync_timeout_result: (
        BuildEquipmentRepositionMapSyncTimeoutResultProtocol
    ),
    build_reposition_teleport_timeout_result: (
        BuildEquipmentRepositionTeleportTimeoutResultProtocol
    ),
    make_reposition_target: Callable[[int, int], TeleportTargetDict],
    dispatch_failure_error: type[Exception],
    unavailable_error: type[Exception],
    unexpected_result_error: type[Exception],
    no_landing_tile_error: type[Exception],
    unavailable_message: str,
    no_landing_tile_message: str,
    impossible_result_message: str,
    acquisition_dispatch_failure_message: str,
    teleport_dispatch_failure_message: str,
) -> BlockedEquipmentRepositionResult:
    """Run one optional blocked-equipment reposition teleport.

    Args:
        page: Page used for waits.
        probe: Probe exposing teleport, world, and self state.
        cdp: Active CDP session for snapshot capture.
        target: Outer attempt target.
        equipment_target: Blocked visible equipment container.
        map_open_started_ms: Initial map-open timestamp.
        map_sync_timestamp_ms: Optional initial map-sync timestamp.
        teleport_started_ms: Initial teleport dispatch timestamp.
        radar_started_ms: Radar dispatch timestamp.
        radar_sync_timestamp_ms: Radar completion timestamp.
        map_sync_timeout_ms: Reposition map-sync timeout.
        teleport_timeout_ms: Reposition teleport timeout.
        inventory_count_before: Inventory total before the attempt began.
        teleport_result: Initial teleport result.
        message_start_index: Message start index for the enclosing attempt.
        teleport_cycle_ids: Mutable list of teleport cycle ids.
        radar_cycle_id: Radar cycle id for this attempt.
        teleport_strategy: Acquisition strategy for the reposition teleport.
        wait_for_teleport_outcome: Teleport waiter.
        teleport_strategy_requires_map_sync: Reposition sync policy.
        find_landing_tile: Landing-tile selector.
        get_phase_overlaps: Phase overlap collector.
        build_reposition_map_sync_timeout_result: Map-sync-timeout builder.
        build_reposition_teleport_timeout_result: Teleport-timeout builder.
        make_reposition_target: Reposition target builder.
        dispatch_failure_error: Error type raised on dispatch failure.
        unavailable_error: Error type raised on unavailable dependencies.
        unexpected_result_error: Error type raised on impossible outcomes.
        no_landing_tile_error: Error type raised when no landing tile exists.
        unavailable_message: Error text for unavailable dependencies.
        no_landing_tile_message: Error text when no landing tile exists.
        impossible_result_message: Error text for impossible reposition results.
        acquisition_dispatch_failure_message: Error text for map-open failure.
        teleport_dispatch_failure_message: Error text for teleport failure.

    Returns:
        Optional terminal reposition result with collected timestamps.
    """
    landing_tile = find_landing_tile(probe, equipment_target)
    if landing_tile is None:
        raise no_landing_tile_error(no_landing_tile_message)
    reposition_target = make_reposition_target(landing_tile[0], landing_tile[1])
    self_state_before_reposition = probe._require_self_state()
    reposition_attempt = run_equipment_reposition_attempt(
        page,
        probe,
        reposition_target,
        cdp=cdp,
        attempt_label=reposition_target["label"],
        fuel_before=self_state_before_reposition["fuel"],
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
        unexpected_result_message=impossible_result_message,
    )
    reposition_cycle = reposition_attempt.teleport_cycle
    teleport_cycle_ids.append(reposition_cycle["cycle_id"])
    reposition_map_open_started_ms = reposition_attempt.acquisition_started_ms
    reposition_map_sync_timestamp_ms = reposition_attempt.acquisition_sync_timestamp_ms
    if (
        teleport_strategy_requires_map_sync(teleport_strategy)
        and reposition_map_sync_timestamp_ms is None
    ):
        probe._end_action_phase(reposition_cycle)
        return BlockedEquipmentRepositionResult(
            teleport_result=None,
            terminal_result=build_reposition_map_sync_timeout_result(
                target=target,
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                radar_started_ms=radar_started_ms,
                radar_sync_timestamp_ms=radar_sync_timestamp_ms,
                reposition_map_open_started_ms=reposition_map_open_started_ms,
                inventory_count_before=inventory_count_before,
                teleport_result=teleport_result,
                equipment_target=equipment_target,
                message_start_index=message_start_index,
                teleport_cycle_ids=teleport_cycle_ids,
                radar_cycle_id=radar_cycle_id,
                phase_overlaps=get_phase_overlaps(),
            ),
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
        )
    reposition_teleport_result = reposition_attempt.teleport_result
    reposition_teleport_started_ms = reposition_attempt.teleport_started_ms
    if reposition_teleport_result is None or reposition_teleport_started_ms is None:
        raise dispatch_failure_error("equipment reposition ended before teleport dispatch")
    if reposition_teleport_result["status"] == "teleport_timeout":
        return BlockedEquipmentRepositionResult(
            teleport_result=None,
            terminal_result=build_reposition_teleport_timeout_result(
                target=target,
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                radar_started_ms=radar_started_ms,
                radar_sync_timestamp_ms=radar_sync_timestamp_ms,
                reposition_map_open_started_ms=reposition_map_open_started_ms,
                reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
                reposition_teleport_started_ms=reposition_teleport_started_ms,
                inventory_count_before=inventory_count_before,
                teleport_result=reposition_teleport_result,
                equipment_target=equipment_target,
                message_start_index=message_start_index,
                teleport_cycle_ids=teleport_cycle_ids,
                radar_cycle_id=radar_cycle_id,
                phase_overlaps=get_phase_overlaps(),
            ),
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
        )
    return BlockedEquipmentRepositionResult(
        teleport_result=reposition_teleport_result,
        terminal_result=None,
        reposition_map_open_started_ms=reposition_map_open_started_ms,
        reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
        reposition_teleport_started_ms=reposition_teleport_started_ms,
    )


def resolve_equipment_target_after_radar(
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
    """Resolve the post-radar equipment target and any blocked reposition.

    Args:
        page: Page used for waits.
        probe: Probe providing world, self, and teleport behavior.
        cdp: Active CDP session for snapshot capture.
        target: Outer attempt target.
        map_open_started_ms: Initial map-open timestamp.
        map_sync_timestamp_ms: Optional initial map-sync timestamp.
        teleport_started_ms: Initial teleport dispatch timestamp.
        radar_started_ms: Radar dispatch timestamp.
        radar_sync_timestamp_ms: Radar completion timestamp.
        map_sync_timeout_ms: Reposition map-sync timeout.
        teleport_timeout_ms: Reposition teleport timeout.
        inventory_count_before: Inventory total before the attempt began.
        teleport_result: Initial teleport result.
        message_start_index: Message start index for the enclosing attempt.
        teleport_cycle_ids: Mutable list of teleport cycle ids.
        radar_cycle_id: Radar cycle id for this attempt.
        teleport_strategy: Acquisition strategy for any reposition teleport.
        terrain_provider: Terrain lookup (currently unused for equipment but
            preserved for diagnostic parity with the fuel probe).
        find_visible_target: Visible-equipment selector.
        requires_reposition: Blocked-equipment predicate.
        find_landing_tile: Landing-tile selector for blocked equipment.
        get_phase_overlaps: Phase-overlap collector.
        build_no_equipment_visible_result: No-visible-equipment terminal builder.
        build_reposition_map_sync_timeout_result: Reposition map-sync-timeout builder.
        build_reposition_teleport_timeout_result: Reposition teleport-timeout builder.
        make_reposition_target: Reposition target builder.
        wait_for_teleport_outcome: Teleport waiter.
        teleport_strategy_requires_map_sync: Reposition sync policy.
        no_landing_tile_error: Error type raised when no landing tile exists.
        dispatch_failure_error: Error type raised on dispatch failure.
        unavailable_error: Error type raised on unavailable dependencies.
        unexpected_result_error: Error type raised on impossible outcomes.
        unavailable_message: Error text for unavailable dependencies.
        no_landing_tile_message: Error text when no landing tile exists.
        impossible_result_message: Error text for impossible reposition results.
        acquisition_dispatch_failure_message: Error text for map-open failure.
        teleport_dispatch_failure_message: Error text for teleport failure.

    Returns:
        Resolved equipment target with optional reposition timestamps and any
        terminal result that blocks the pickup phase.
    """
    _ = terrain_provider
    equipment_target = find_visible_target(probe)
    if equipment_target is None:
        return EquipmentTargetResolution(
            equipment_target=None,
            teleport_result=teleport_result,
            terminal_result=build_no_equipment_visible_result(
                target=target,
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                radar_started_ms=radar_started_ms,
                radar_sync_timestamp_ms=radar_sync_timestamp_ms,
                inventory_count_before=inventory_count_before,
                teleport_result=teleport_result,
                message_start_index=message_start_index,
                teleport_cycle_ids=teleport_cycle_ids,
                radar_cycle_id=radar_cycle_id,
                phase_overlaps=get_phase_overlaps(),
            ),
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
        )
    if not requires_reposition(probe, equipment_target):
        return EquipmentTargetResolution(
            equipment_target=equipment_target,
            teleport_result=teleport_result,
            terminal_result=None,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
        )
    reposition_result = _run_blocked_equipment_reposition(
        page,
        probe,
        cdp=cdp,
        target=target,
        equipment_target=equipment_target,
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
        radar_cycle_id=radar_cycle_id,
        teleport_strategy=teleport_strategy,
        wait_for_teleport_outcome=wait_for_teleport_outcome,
        teleport_strategy_requires_map_sync=teleport_strategy_requires_map_sync,
        find_landing_tile=find_landing_tile,
        get_phase_overlaps=get_phase_overlaps,
        build_reposition_map_sync_timeout_result=build_reposition_map_sync_timeout_result,
        build_reposition_teleport_timeout_result=build_reposition_teleport_timeout_result,
        make_reposition_target=make_reposition_target,
        dispatch_failure_error=dispatch_failure_error,
        unavailable_error=unavailable_error,
        unexpected_result_error=unexpected_result_error,
        no_landing_tile_error=no_landing_tile_error,
        unavailable_message=unavailable_message,
        no_landing_tile_message=no_landing_tile_message,
        impossible_result_message=impossible_result_message,
        acquisition_dispatch_failure_message=acquisition_dispatch_failure_message,
        teleport_dispatch_failure_message=teleport_dispatch_failure_message,
    )
    resolved_teleport_result = teleport_result
    if reposition_result.teleport_result is not None:
        resolved_teleport_result = reposition_result.teleport_result
    return EquipmentTargetResolution(
        equipment_target=equipment_target,
        teleport_result=resolved_teleport_result,
        terminal_result=reposition_result.terminal_result,
        reposition_map_open_started_ms=reposition_result.reposition_map_open_started_ms,
        reposition_map_sync_timestamp_ms=reposition_result.reposition_map_sync_timestamp_ms,
        reposition_teleport_started_ms=reposition_result.reposition_teleport_started_ms,
    )


__all__ = [
    "BlockedEquipmentRepositionResult",
    "BuildEquipmentRepositionMapSyncTimeoutResultProtocol",
    "BuildEquipmentRepositionTeleportTimeoutResultProtocol",
    "BuildNoEquipmentVisibleResultProtocol",
    "EquipmentTargetPhaseProbeProtocol",
    "EquipmentTargetResolution",
    "resolve_equipment_target_after_radar",
    "run_equipment_reposition_attempt",
]
