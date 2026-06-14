"""Shared result-construction helpers for the live equipment probe."""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict
from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeAttemptResultDict
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.state import SelfStateDict
from tankpit_bot.state.types import ContainerStateDict


def build_attempt_result(
    *,
    target: TeleportTargetDict,
    status: Literal[
        "picked_up_equipment",
        "no_equipment_visible",
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
    inventory_count_before: int,
    inventory_count_after: int | None,
    landed_signal_received: bool,
    landed_x: int | None,
    landed_y: int | None,
    equipment_target: ContainerStateDict | None,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int | None = None,
    move_cycle_id: int | None = None,
    pickup_cycle_id: int | None = None,
    phase_overlaps: list[ActionPhaseOverlapDict] | None = None,
    reposition_map_open_started_ms: int | None = None,
    reposition_map_sync_timestamp_ms: int | None = None,
    reposition_teleport_started_ms: int | None = None,
) -> EquipmentProbeAttemptResultDict:
    """Build one typed equipment-attempt result payload.

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
        inventory_count_before: Total inventory items before the attempt.
        inventory_count_after: Total inventory items after the attempt when known.
        landed_signal_received: Whether the landing signal was observed.
        landed_x: Landed X tile when known.
        landed_y: Landed Y tile when known.
        equipment_target: Selected equipment target when one exists.
        message_start_index: Raw message start index for the attempt.
        message_end_index: Raw message end index for the attempt.
        teleport_cycle_ids: Teleport cycle ids associated with the attempt.
        radar_cycle_id: Radar cycle id when one exists.
        move_cycle_id: Move cycle id when one exists.
        pickup_cycle_id: Pickup cycle id when one exists.
        phase_overlaps: Overlapping phase diagnostics.
        reposition_map_open_started_ms: Reposition map-open timestamp.
        reposition_map_sync_timestamp_ms: Reposition map-sync timestamp.
        reposition_teleport_started_ms: Reposition teleport timestamp.

    Returns:
        Typed attempt result payload.
    """
    return EquipmentProbeAttemptResultDict(
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=landed_signal_received,
        landed_x=landed_x,
        landed_y=landed_y,
        equipment_target_x=None if equipment_target is None else equipment_target["x"],
        equipment_target_y=None if equipment_target is None else equipment_target["y"],
        phase_overlaps=[] if phase_overlaps is None else phase_overlaps,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
    )


def build_terminal_attempt(
    *,
    target: TeleportTargetDict,
    status: Literal[
        "no_equipment_visible",
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
    inventory_count_before: int,
    inventory_count_after: int | None,
    landed_signal_received: bool,
    landed_x: int | None,
    landed_y: int | None,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int | None = None,
    move_cycle_id: int | None = None,
    pickup_cycle_id: int | None = None,
    phase_overlaps: list[ActionPhaseOverlapDict] | None = None,
    reposition_map_open_started_ms: int | None = None,
    reposition_map_sync_timestamp_ms: int | None = None,
    reposition_teleport_started_ms: int | None = None,
) -> EquipmentProbeAttemptResultDict:
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
        inventory_count_before: Inventory total before the attempt.
        inventory_count_after: Inventory total after the attempt when known.
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=landed_signal_received,
        landed_x=landed_x,
        landed_y=landed_y,
        equipment_target=None,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        move_cycle_id=move_cycle_id,
        pickup_cycle_id=pickup_cycle_id,
        phase_overlaps=phase_overlaps,
    )


def build_map_sync_timeout_result(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    completion_timestamp_ms: int,
    inventory_count_before: int,
    inventory_count_after: int,
    self_state: SelfStateDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
) -> EquipmentProbeAttemptResultDict:
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=False,
        landed_x=self_state["x"],
        landed_y=self_state["y"],
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
    )


def build_teleport_timeout_result(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    inventory_count_before: int,
    inventory_count_after: int,
    teleport_result: TeleportAttemptResultDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
) -> EquipmentProbeAttemptResultDict:
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        equipment_target=None,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
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
    inventory_count_before: int,
    inventory_count_after: int,
    teleport_result: TeleportAttemptResultDict,
    equipment_target: ContainerStateDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
) -> EquipmentProbeAttemptResultDict:
    """Build one blocked-equipment reposition map-sync-timeout result."""
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        equipment_target=equipment_target,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
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
    inventory_count_before: int,
    inventory_count_after: int,
    teleport_result: TeleportAttemptResultDict,
    equipment_target: ContainerStateDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
) -> EquipmentProbeAttemptResultDict:
    """Build one blocked-equipment reposition teleport-timeout result."""
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        equipment_target=equipment_target,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
    )


def build_radar_timeout_result(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    completion_timestamp_ms: int,
    inventory_count_before: int,
    inventory_count_after: int,
    teleport_result: TeleportAttemptResultDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
) -> EquipmentProbeAttemptResultDict:
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
    )


def build_no_equipment_visible_result(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    radar_sync_timestamp_ms: int,
    completion_timestamp_ms: int,
    inventory_count_before: int,
    inventory_count_after: int,
    teleport_result: TeleportAttemptResultDict,
    message_start_index: int,
    message_end_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
) -> EquipmentProbeAttemptResultDict:
    """Build one no-visible-equipment result."""
    return build_terminal_attempt(
        target=target,
        status="no_equipment_visible",
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        completion_timestamp_ms=completion_timestamp_ms,
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
    )


__all__ = [
    "build_attempt_result",
    "build_map_sync_timeout_result",
    "build_no_equipment_visible_result",
    "build_radar_timeout_result",
    "build_reposition_map_sync_timeout_result",
    "build_reposition_teleport_timeout_result",
    "build_teleport_timeout_result",
    "build_terminal_attempt",
]
