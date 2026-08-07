"""Shared result-construction helpers for the live fuel probe."""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict, FuelDecisionBasisDict
from tankpit_bot.action_lab.fuel_probe_types import FuelProbeAttemptResultDict
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.state import SelfStateDict
from tankpit_bot.state.types import ContainerStateDict


def build_attempt_result(
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
    message_end_index: int,
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
    """Build one typed fuel-attempt result payload.

    Args:
        target: Teleport target for the attempt.
        status: Terminal attempt status.
        map_open_started_ms: Initial map-open timestamp.
        map_sync_timestamp_ms: Initial map-sync timestamp when available.
        teleport_started_ms: Initial teleport timestamp when available.
        radar_started_ms: Radar phase start timestamp when available.
        radar_sync_timestamp_ms: Radar sync timestamp when available.
        pickup_started_ms: Pickup phase start timestamp when available.
        completion_timestamp_ms: Terminal completion timestamp.
        fuel_before: Fuel before the attempt.
        fuel_after: Fuel after the attempt when known.
        landed_signal_received: Whether the landing signal was observed.
        landed_x: Landed X tile when known.
        landed_y: Landed Y tile when known.
        fuel_target: Selected fuel target when one exists.
        message_start_index: Raw message start index for the attempt.
        message_end_index: Raw message end index for the attempt.
        teleport_cycle_ids: Teleport cycle ids associated with the attempt.
        radar_cycle_id: Radar cycle id when one exists.
        move_cycle_id: Move cycle id when one exists.
        pickup_cycle_id: Pickup cycle id when one exists.
        phase_overlaps: Overlapping phase diagnostics.
        decision_basis: Fuel selection decision basis when one exists.
        reposition_map_open_started_ms: Reposition map-open timestamp.
        reposition_map_sync_timestamp_ms: Reposition map-sync timestamp.
        reposition_teleport_started_ms: Reposition teleport timestamp.

    Returns:
        Typed attempt result payload.
    """
    return FuelProbeAttemptResultDict(
        target=target,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        move_cycle_id=move_cycle_id,
        pickup_cycle_id=pickup_cycle_id,
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
        fuel_target_x=None if fuel_target is None else fuel_target["x"],
        fuel_target_y=None if fuel_target is None else fuel_target["y"],
        fuel_target_volume=None if fuel_target is None else fuel_target["volume"],
        phase_overlaps=[] if phase_overlaps is None else phase_overlaps,
        decision_basis=decision_basis,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_terminal_attempt(
    *,
    target: TeleportTargetDict,
    status: Literal[
        "no_fuel_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
    ],
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int | None,
    radar_started_ms: int | None,
    radar_sync_timestamp_ms: int | None,
    completion_timestamp_ms: int,
    fuel_before: int,
    fuel_after: int | None,
    landed_signal_received: bool,
    landed_x: int | None,
    landed_y: int | None,
    message_start_index: int,
    message_end_index: int,
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
    """Build one non-pickup terminal attempt result.

    Args:
        target: Teleport target for the attempt.
        status: Terminal attempt status.
        map_open_started_ms: Initial map-open timestamp.
        map_sync_timestamp_ms: Initial map-sync timestamp when available.
        teleport_started_ms: Initial teleport timestamp when available.
        radar_started_ms: Radar phase start timestamp when available.
        radar_sync_timestamp_ms: Radar sync timestamp when available.
        completion_timestamp_ms: Terminal completion timestamp.
        fuel_before: Fuel before the attempt.
        fuel_after: Fuel after the attempt when known.
        landed_signal_received: Whether the landing signal was observed.
        landed_x: Landed X tile when known.
        landed_y: Landed Y tile when known.
        message_start_index: Raw message start index for the attempt.
        message_end_index: Raw message end index for the attempt.
        teleport_cycle_ids: Teleport cycle ids associated with the attempt.
        radar_cycle_id: Radar cycle id when one exists.
        move_cycle_id: Move cycle id when one exists.
        pickup_cycle_id: Pickup cycle id when one exists.
        phase_overlaps: Overlapping phase diagnostics.
        decision_basis: Fuel selection decision basis when one exists.
        reposition_map_open_started_ms: Reposition map-open timestamp.
        reposition_map_sync_timestamp_ms: Reposition map-sync timestamp.
        reposition_teleport_started_ms: Reposition teleport timestamp.

    Returns:
        Typed terminal attempt payload.
    """
    return build_attempt_result(
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
        pickup_started_ms=None,
        completion_timestamp_ms=completion_timestamp_ms,
        fuel_before=fuel_before,
        fuel_after=fuel_after,
        landed_signal_received=landed_signal_received,
        landed_x=landed_x,
        landed_y=landed_y,
        fuel_target=None,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        move_cycle_id=move_cycle_id,
        pickup_cycle_id=pickup_cycle_id,
        phase_overlaps=phase_overlaps,
        decision_basis=decision_basis,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_map_sync_timeout_result(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    completion_timestamp_ms: int,
    fuel_before: int,
    self_state: SelfStateDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    snapshot_before: PageClientSnapshotDict,
    snapshot_after: PageClientSnapshotDict,
) -> FuelProbeAttemptResultDict:
    """Build one initial map-sync-timeout result."""
    return build_terminal_attempt(
        target=target,
        status="map_sync_timeout",
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=None,
        teleport_started_ms=None,
        radar_started_ms=None,
        radar_sync_timestamp_ms=None,
        completion_timestamp_ms=completion_timestamp_ms,
        fuel_before=fuel_before,
        fuel_after=self_state["fuel"],
        landed_signal_received=False,
        landed_x=self_state["x"],
        landed_y=self_state["y"],
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_teleport_timeout_result(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    fuel_before: int,
    teleport_result: TeleportAttemptResultDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    snapshot_before: PageClientSnapshotDict,
    snapshot_after: PageClientSnapshotDict,
) -> FuelProbeAttemptResultDict:
    """Build one initial teleport-timeout result."""
    return build_attempt_result(
        target=target,
        status="teleport_timeout",
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=None,
        radar_sync_timestamp_ms=None,
        pickup_started_ms=None,
        completion_timestamp_ms=teleport_result["completion_timestamp_ms"],
        fuel_before=fuel_before,
        fuel_after=teleport_result["fuel_after"],
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        fuel_target=None,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_reposition_map_sync_timeout_result(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    radar_sync_timestamp_ms: int,
    reposition_map_open_started_ms: int,
    completion_timestamp_ms: int,
    fuel_before: int,
    self_state: SelfStateDict,
    teleport_result: TeleportAttemptResultDict,
    fuel_target: ContainerStateDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
    snapshot_before: PageClientSnapshotDict,
    snapshot_after: PageClientSnapshotDict,
) -> FuelProbeAttemptResultDict:
    """Build one blocked-fuel reposition map-sync-timeout result."""
    return build_attempt_result(
        target=target,
        status="reposition_map_sync_timeout",
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        reposition_map_open_started_ms=reposition_map_open_started_ms,
        reposition_map_sync_timestamp_ms=None,
        reposition_teleport_started_ms=None,
        pickup_started_ms=None,
        completion_timestamp_ms=completion_timestamp_ms,
        fuel_before=fuel_before,
        fuel_after=self_state["fuel"],
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        fuel_target=fuel_target,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_reposition_teleport_timeout_result(
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
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
    snapshot_before: PageClientSnapshotDict,
    snapshot_after: PageClientSnapshotDict,
) -> FuelProbeAttemptResultDict:
    """Build one blocked-fuel reposition teleport-timeout result."""
    return build_attempt_result(
        target=target,
        status="reposition_teleport_timeout",
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        reposition_map_open_started_ms=reposition_map_open_started_ms,
        reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
        reposition_teleport_started_ms=reposition_teleport_started_ms,
        pickup_started_ms=None,
        completion_timestamp_ms=teleport_result["completion_timestamp_ms"],
        fuel_before=fuel_before,
        fuel_after=teleport_result["fuel_after"],
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        fuel_target=fuel_target,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_radar_timeout_result(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    completion_timestamp_ms: int,
    fuel_before: int,
    self_state: SelfStateDict,
    teleport_result: TeleportAttemptResultDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
    snapshot_before: PageClientSnapshotDict,
    snapshot_after: PageClientSnapshotDict,
) -> FuelProbeAttemptResultDict:
    """Build one radar-timeout result."""
    return build_terminal_attempt(
        target=target,
        status="radar_timeout",
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=None,
        completion_timestamp_ms=completion_timestamp_ms,
        fuel_before=fuel_before,
        fuel_after=self_state["fuel"],
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def build_no_fuel_visible_result(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    radar_sync_timestamp_ms: int,
    completion_timestamp_ms: int,
    fuel_before: int,
    self_state: SelfStateDict,
    teleport_result: TeleportAttemptResultDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
    decision_basis: FuelDecisionBasisDict | None,
    snapshot_before: PageClientSnapshotDict,
    snapshot_after: PageClientSnapshotDict,
) -> FuelProbeAttemptResultDict:
    """Build one no-visible-fuel result."""
    return build_terminal_attempt(
        target=target,
        status="no_fuel_visible",
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        completion_timestamp_ms=completion_timestamp_ms,
        fuel_before=fuel_before,
        fuel_after=self_state["fuel"],
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
        decision_basis=decision_basis,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


__all__ = [
    "build_attempt_result",
    "build_map_sync_timeout_result",
    "build_no_fuel_visible_result",
    "build_radar_timeout_result",
    "build_reposition_map_sync_timeout_result",
    "build_reposition_teleport_timeout_result",
    "build_teleport_timeout_result",
    "build_terminal_attempt",
]
