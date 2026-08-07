"""Structural contracts for one fuel-probe attempt.

The nine Protocols the attempt runners take as parameters -- the probe
surface itself plus every collaborator it is handed. Kept apart from
the runners so the contract can be read without the orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from tankpit_bot._test_hooks import CDPSessionProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseOverlapDict,
    FuelDecisionBasisDict,
)
from tankpit_bot.action_lab.fuel_collection_phase import (
    BuildRadarTimeoutResultProtocol,
    FuelCollectionPhaseProbeProtocol,
    RunPickupAttemptProtocol,
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
)
from tankpit_bot.action_lab.pickup_phase import (
    PickupImmediateOutcomeProtocol,
    PickupOutcomeWaiterProtocol,
    PickupPhaseProbeProtocol,
    PickupTimeoutSizerProtocol,
)
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    TrackedTeleportAttempt,
)
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.state import ContainerStateDict, SelfStateDict, WorldStateDict


class FuelProbePickupAttemptProtocol(PickupPhaseProbeProtocol, Protocol):
    """Minimal probe interface required for one pickup attempt."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the current self state."""


class BuildAttemptResultProtocol(Protocol):
    """Callable protocol for final pickup-attempt result builders."""

    def __call__(
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
        """Build one typed fuel-attempt result."""


class RunTrackedPickupPhaseProtocol(Protocol):
    """Callable protocol for the shared move-and-pickup phase."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: PickupPhaseProbeProtocol,
        *,
        attempt_label: str,
        target_x: int,
        target_y: int,
        current_x: int,
        current_y: int,
        fuel_before_pickup: int,
        pickup_timeout_ms: int,
        dispatch_failure_error: type[Exception],
        get_completed_outcome: PickupImmediateOutcomeProtocol,
        wait_for_outcome: PickupOutcomeWaiterProtocol,
        compute_timeout: PickupTimeoutSizerProtocol,
    ) -> tuple[
        ActionPhaseCycleDict,
        ActionPhaseCycleDict,
        int,
        Literal["picked_up_fuel", "pickup_timeout"],
        int,
        int,
    ]:
        """Run one tracked pickup phase."""


class FuelProbeSingleAttemptProtocol(
    FuelCollectionPhaseProbeProtocol,
    TeleportAttemptProbeProtocol,
    Protocol,
):
    """Minimal probe interface required for one full fuel attempt."""

    _cdp: CDPSessionProtocol | None

    def _require_page(self) -> action_session.WaitPageProtocol:
        """Return the live page."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the current self state."""

    def _reset_attempt_phase_overlaps(self) -> None:
        """Reset any per-attempt phase-overlap tracking."""

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """End one active action phase."""

    def _reset_probe_state_to_idle(self) -> None:
        """Reset the probe state machine to idle."""

    def open_map(self) -> bool:
        """Dispatch one map-open command."""

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""


class BuildMapSyncTimeoutResultProtocol(Protocol):
    """Callable protocol for map-sync timeout result builders."""

    def __call__(
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
        """Build one map-sync-timeout result."""


class BuildTeleportTimeoutResultProtocol(Protocol):
    """Callable protocol for teleport-timeout result builders."""

    def __call__(
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


class RunTrackedFuelCollectionPhaseProtocol(Protocol):
    """Callable protocol for the shared post-teleport fuel-collection phase."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: FuelCollectionPhaseProbeProtocol,
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
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        snapshot_before: PageClientSnapshotDict,
        capture_snapshot: Callable[[], PageClientSnapshotDict],
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
    ) -> FuelProbeAttemptResultDict:
        """Run the shared radar-to-pickup phase."""


__all__ = [
    "BuildAttemptResultProtocol",
    "BuildMapSyncTimeoutResultProtocol",
    "BuildTeleportTimeoutResultProtocol",
    "FinalizeAttemptDelayProtocol",
    "FuelProbePickupAttemptProtocol",
    "FuelProbeSingleAttemptProtocol",
    "RunTrackedFuelCollectionPhaseProtocol",
    "RunTrackedPickupPhaseProtocol",
    "RunTrackedTeleportAttemptProtocol",
]
