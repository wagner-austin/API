"""Shared fuel target-resolution helpers for action-lab probes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple, Protocol

from tankpit_bot._test_hooks import CDPSessionProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace import build_fuel_decision_basis
from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict, FuelDecisionBasisDict
from tankpit_bot.action_lab.fuel_probe_types import FuelProbeAttemptResultDict
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    run_tracked_teleport_attempt,
)
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.state import ContainerStateDict, SelfStateDict, WorldStateDict

run_reposition_attempt = run_tracked_teleport_attempt


class FuelTargetPhaseProbeProtocol(TeleportAttemptProbeProtocol, Protocol):
    """Minimal probe interface required for post-radar fuel target resolution."""

    def open_map(self) -> bool:
        """Dispatch one map-open command."""

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""

    def get_self_state(self) -> SelfStateDict | None:
        """Return the current self state when available."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the required current self state."""


class BlockedFuelRepositionResult(NamedTuple):
    """Typed result for one optional blocked-fuel reposition attempt."""

    teleport_result: TeleportAttemptResultDict | None
    terminal_result: FuelProbeAttemptResultDict | None
    reposition_map_open_started_ms: int | None
    reposition_map_sync_timestamp_ms: int | None
    reposition_teleport_started_ms: int | None


class FuelTargetResolution(NamedTuple):
    """Typed result for post-radar fuel target selection and reposition."""

    fuel_target: ContainerStateDict | None
    teleport_result: TeleportAttemptResultDict
    terminal_result: FuelProbeAttemptResultDict | None
    decision_basis: FuelDecisionBasisDict | None
    reposition_map_open_started_ms: int | None
    reposition_map_sync_timestamp_ms: int | None
    reposition_teleport_started_ms: int | None


class BuildNoFuelVisibleResultProtocol(Protocol):
    """Callable protocol for the no-visible-fuel terminal result builder."""

    def __call__(
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
    ) -> FuelProbeAttemptResultDict:
        """Build one no-visible-fuel terminal result."""


class BuildRepositionMapSyncTimeoutResultProtocol(Protocol):
    """Callable protocol for blocked-fuel reposition sync-timeout results."""

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
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        fuel_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
    ) -> FuelProbeAttemptResultDict:
        """Build one blocked-fuel reposition map-sync-timeout result."""


class BuildRepositionTeleportTimeoutResultProtocol(Protocol):
    """Callable protocol for blocked-fuel reposition teleport-timeout results."""

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
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        fuel_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
    ) -> FuelProbeAttemptResultDict:
        """Build one blocked-fuel reposition teleport-timeout result."""


def _run_blocked_fuel_reposition(
    page: action_session.WaitPageProtocol,
    probe: FuelTargetPhaseProbeProtocol,
    *,
    cdp: CDPSessionProtocol | None,
    target: TeleportTargetDict,
    fuel_target: ContainerStateDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    radar_sync_timestamp_ms: int,
    map_sync_timeout_ms: int,
    teleport_timeout_ms: int,
    fuel_before: int,
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
        [FuelTargetPhaseProbeProtocol, ContainerStateDict],
        tuple[int, int] | None,
    ],
    get_phase_overlaps: Callable[[], list[ActionPhaseOverlapDict]],
    build_reposition_map_sync_timeout_result: BuildRepositionMapSyncTimeoutResultProtocol,
    build_reposition_teleport_timeout_result: BuildRepositionTeleportTimeoutResultProtocol,
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
) -> BlockedFuelRepositionResult:
    """Run one optional blocked-fuel reposition teleport."""
    landing_tile = find_landing_tile(probe, fuel_target)
    if landing_tile is None:
        raise no_landing_tile_error(no_landing_tile_message)
    reposition_target = make_reposition_target(landing_tile[0], landing_tile[1])
    self_state_before_reposition = probe._require_self_state()
    reposition_attempt = run_reposition_attempt(
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
        return BlockedFuelRepositionResult(
            teleport_result=None,
            terminal_result=build_reposition_map_sync_timeout_result(
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
                phase_overlaps=get_phase_overlaps(),
            ),
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
        )
    reposition_teleport_result = reposition_attempt.teleport_result
    reposition_teleport_started_ms = reposition_attempt.teleport_started_ms
    if reposition_teleport_result is None or reposition_teleport_started_ms is None:
        raise dispatch_failure_error("fuel reposition ended before teleport dispatch")
    if reposition_teleport_result["status"] == "teleport_timeout":
        return BlockedFuelRepositionResult(
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
                fuel_before=fuel_before,
                teleport_result=reposition_teleport_result,
                fuel_target=fuel_target,
                message_start_index=message_start_index,
                teleport_cycle_ids=teleport_cycle_ids,
                radar_cycle_id=radar_cycle_id,
                phase_overlaps=get_phase_overlaps(),
            ),
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
        )
    return BlockedFuelRepositionResult(
        teleport_result=reposition_teleport_result,
        terminal_result=None,
        reposition_map_open_started_ms=reposition_map_open_started_ms,
        reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
        reposition_teleport_started_ms=reposition_teleport_started_ms,
    )


def resolve_fuel_target_after_radar(
    page: action_session.WaitPageProtocol,
    probe: FuelTargetPhaseProbeProtocol,
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
    fuel_before: int,
    teleport_result: TeleportAttemptResultDict,
    message_start_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
    terrain_provider: Callable[[], TerrainMapProtocol | None],
    find_visible_target: Callable[[FuelTargetPhaseProbeProtocol, bool], ContainerStateDict | None],
    requires_reposition: Callable[[FuelTargetPhaseProbeProtocol, ContainerStateDict], bool],
    find_landing_tile: Callable[
        [FuelTargetPhaseProbeProtocol, ContainerStateDict],
        tuple[int, int] | None,
    ],
    get_phase_overlaps: Callable[[], list[ActionPhaseOverlapDict]],
    build_no_fuel_visible_result: BuildNoFuelVisibleResultProtocol,
    build_reposition_map_sync_timeout_result: BuildRepositionMapSyncTimeoutResultProtocol,
    build_reposition_teleport_timeout_result: BuildRepositionTeleportTimeoutResultProtocol,
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
) -> FuelTargetResolution:
    """Resolve the post-radar fuel target and any blocked-fuel reposition."""
    fuel_target = find_visible_target(probe, True)
    self_state = probe._require_self_state()
    decision_basis = build_fuel_decision_basis(
        probe.get_world_state(),
        self_x=self_state["x"],
        self_y=self_state["y"],
        radar_cycle_id=radar_cycle_id,
        terrain=terrain_provider(),
        fuel_target=fuel_target,
    )
    if fuel_target is None:
        return FuelTargetResolution(
            fuel_target=None,
            teleport_result=teleport_result,
            terminal_result=build_no_fuel_visible_result(
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
                phase_overlaps=get_phase_overlaps(),
                decision_basis=decision_basis,
            ),
            decision_basis=decision_basis,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
        )
    if not requires_reposition(probe, fuel_target):
        return FuelTargetResolution(
            fuel_target=fuel_target,
            teleport_result=teleport_result,
            terminal_result=None,
            decision_basis=decision_basis,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
        )
    reposition_result = _run_blocked_fuel_reposition(
        page,
        probe,
        cdp=cdp,
        target=target,
        fuel_target=fuel_target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        map_sync_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        fuel_before=fuel_before,
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
    return FuelTargetResolution(
        fuel_target=fuel_target,
        teleport_result=resolved_teleport_result,
        terminal_result=reposition_result.terminal_result,
        decision_basis=decision_basis,
        reposition_map_open_started_ms=reposition_result.reposition_map_open_started_ms,
        reposition_map_sync_timestamp_ms=reposition_result.reposition_map_sync_timestamp_ms,
        reposition_teleport_started_ms=reposition_result.reposition_teleport_started_ms,
    )


__all__ = [
    "BlockedFuelRepositionResult",
    "BuildNoFuelVisibleResultProtocol",
    "BuildRepositionMapSyncTimeoutResultProtocol",
    "BuildRepositionTeleportTimeoutResultProtocol",
    "FuelTargetPhaseProbeProtocol",
    "FuelTargetResolution",
    "resolve_fuel_target_after_radar",
    "run_reposition_attempt",
]
