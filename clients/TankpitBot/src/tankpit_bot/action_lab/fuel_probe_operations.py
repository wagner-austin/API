"""Typed helper functions backing the live fuel-probe class surface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseOverlapDict,
    FuelDecisionBasisDict,
)
from tankpit_bot.action_lab.fuel_probe_attempt import run_fuel_pickup_attempt
from tankpit_bot.action_lab.fuel_probe_attempt_contracts import (
    FuelProbePickupAttemptProtocol,
    RunTrackedPickupPhaseProtocol,
)
from tankpit_bot.action_lab.fuel_probe_results import (
    build_attempt_result as _shared_build_attempt_result,
)
from tankpit_bot.action_lab.fuel_probe_results import (
    build_map_sync_timeout_result as _shared_build_map_sync_timeout_result,
)
from tankpit_bot.action_lab.fuel_probe_results import (
    build_no_fuel_visible_result as _shared_build_no_fuel_visible_result,
)
from tankpit_bot.action_lab.fuel_probe_results import (
    build_radar_timeout_result as _shared_build_radar_timeout_result,
)
from tankpit_bot.action_lab.fuel_probe_results import (
    build_reposition_map_sync_timeout_result as _shared_build_reposition_map_sync_timeout_result,
)
from tankpit_bot.action_lab.fuel_probe_results import (
    build_reposition_teleport_timeout_result as _shared_build_reposition_teleport_timeout_result,
)
from tankpit_bot.action_lab.fuel_probe_results import (
    build_teleport_timeout_result as _shared_build_teleport_timeout_result,
)
from tankpit_bot.action_lab.fuel_probe_types import FuelProbeAttemptResultDict
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.pickup_phase import (
    PickupImmediateOutcomeProtocol,
    PickupOutcomeWaiterProtocol,
    PickupTimeoutSizerProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.state import SelfStateDict
from tankpit_bot.state.types import ContainerStateDict
from tankpit_bot.types import CapturedMessage


class FuelProbeBuilderContextProtocol(Protocol):
    """Minimal probe context required for result building."""

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return captured wire messages."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the required current self state."""


class FuelProbeOperationContextProtocol(
    FuelProbeBuilderContextProtocol,
    FuelProbePickupAttemptProtocol,
    Protocol,
):
    """Minimal probe context required for attempt-level helper operations."""

    def _get_attempt_phase_overlaps(self) -> list[ActionPhaseOverlapDict]:
        """Return the current attempt's phase-overlap diagnostics."""


def finalize_attempt_delay(
    page: action_session.WaitPageProtocol,
    *,
    settle_delay_ms: int,
) -> None:
    """Apply one optional post-attempt settle delay.

    Args:
        page: Page used for wait timing.
        settle_delay_ms: Requested settle delay in milliseconds.
    """
    if settle_delay_ms > 0:
        page.wait_for_timeout(float(settle_delay_ms))


def build_attempt_result_for_probe(
    probe: FuelProbeBuilderContextProtocol,
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
    """Build one typed attempt result for a probe."""
    return _shared_build_attempt_result(
        target=target,
        status=status,
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
        landed_signal_received=landed_signal_received,
        landed_x=landed_x,
        landed_y=landed_y,
        fuel_target=fuel_target,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        move_cycle_id=move_cycle_id,
        pickup_cycle_id=pickup_cycle_id,
        phase_overlaps=phase_overlaps,
        decision_basis=decision_basis,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_map_sync_timeout_result_for_probe(
    probe: FuelProbeBuilderContextProtocol,
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    fuel_before: int,
    message_start_index: int,
    teleport_cycle_ids: list[int],
    snapshot_before: PageClientSnapshotDict,
    snapshot_after: PageClientSnapshotDict,
) -> FuelProbeAttemptResultDict:
    """Build one map-sync-timeout result for a probe."""
    return _shared_build_map_sync_timeout_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        completion_timestamp_ms=action_hooks.get_current_time_ms(),
        fuel_before=fuel_before,
        self_state=probe._require_self_state(),
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_teleport_timeout_result_for_probe(
    probe: FuelProbeBuilderContextProtocol,
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
    """Build one teleport-timeout result for a probe."""
    return _shared_build_teleport_timeout_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        fuel_before=fuel_before,
        teleport_result=teleport_result,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_reposition_map_sync_timeout_result_for_probe(
    probe: FuelProbeBuilderContextProtocol,
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
    """Build one reposition map-sync-timeout result for a probe."""
    return _shared_build_reposition_map_sync_timeout_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        reposition_map_open_started_ms=reposition_map_open_started_ms,
        completion_timestamp_ms=action_hooks.get_current_time_ms(),
        fuel_before=fuel_before,
        self_state=probe._require_self_state(),
        teleport_result=teleport_result,
        fuel_target=fuel_target,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_reposition_teleport_timeout_result_for_probe(
    probe: FuelProbeBuilderContextProtocol,
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
    """Build one reposition teleport-timeout result for a probe."""
    return _shared_build_reposition_teleport_timeout_result(
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
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_radar_timeout_result_for_probe(
    probe: FuelProbeBuilderContextProtocol,
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
    """Build one radar-timeout result for a probe."""
    return _shared_build_radar_timeout_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        completion_timestamp_ms=action_hooks.get_current_time_ms(),
        fuel_before=fuel_before,
        self_state=probe._require_self_state(),
        teleport_result=teleport_result,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_no_fuel_visible_result_for_probe(
    probe: FuelProbeBuilderContextProtocol,
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
    """Build one no-visible-fuel result for a probe."""
    return _shared_build_no_fuel_visible_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        completion_timestamp_ms=action_hooks.get_current_time_ms(),
        fuel_before=fuel_before,
        self_state=probe._require_self_state(),
        teleport_result=teleport_result,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
        decision_basis=decision_basis,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def run_pickup_attempt_for_probe(
    probe: FuelProbeOperationContextProtocol,
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
    run_tracked_pickup_phase: RunTrackedPickupPhaseProtocol,
    get_completed_outcome: PickupImmediateOutcomeProtocol,
    wait_for_outcome: PickupOutcomeWaiterProtocol,
    compute_timeout: PickupTimeoutSizerProtocol,
) -> FuelProbeAttemptResultDict:
    """Run one pickup attempt for a probe.

    Raises:
        PickupPhaseError: Propagated when the shared pickup phase fails.
    """

    def _build_attempt_result(
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
        return build_attempt_result_for_probe(
            probe,
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

    return run_fuel_pickup_attempt(
        probe,
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
        dispatch_failure_error=dispatch_failure_error,
        build_attempt_result=_build_attempt_result,
        get_phase_overlaps=probe._get_attempt_phase_overlaps,
        run_tracked_pickup_phase=run_tracked_pickup_phase,
        get_completed_outcome=get_completed_outcome,
        wait_for_outcome=wait_for_outcome,
        compute_timeout=compute_timeout,
    )


__all__ = [
    "build_attempt_result_for_probe",
    "build_map_sync_timeout_result_for_probe",
    "build_no_fuel_visible_result_for_probe",
    "build_radar_timeout_result_for_probe",
    "build_reposition_map_sync_timeout_result_for_probe",
    "build_reposition_teleport_timeout_result_for_probe",
    "build_teleport_timeout_result_for_probe",
    "finalize_attempt_delay",
    "run_pickup_attempt_for_probe",
]
