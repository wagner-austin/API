"""Live teleport-radar-equipment action probe harness."""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab import equipment_probe_operations as _equipment_probe_operations
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict
from tankpit_bot.action_lab.equipment_collection_phase import (
    run_tracked_equipment_collection_phase,
)
from tankpit_bot.action_lab.equipment_probe_attempt import (
    run_single_equipment_target_attempt as _shared_run_single_equipment_target_attempt,
)
from tankpit_bot.action_lab.equipment_probe_diagnostics import (
    format_equipment_probe_summary,
)
from tankpit_bot.action_lab.equipment_probe_types import (
    EquipmentProbeAttemptResultDict,
)
from tankpit_bot.action_lab.equipment_targeting import (
    EquipmentTargetingError,
    find_visible_equipment_landing_tile,
    find_visible_equipment_target,
    visible_equipment_requires_reposition,
)
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.teleport import (
    DEFAULT_TELEPORT_STRATEGY,
)
from tankpit_bot.action_lab.teleport_acquisition import (
    teleport_strategy_requires_map_sync,
)
from tankpit_bot.action_lab.teleport_attempt import (
    run_tracked_teleport_attempt as _shared_run_tracked_teleport_attempt,
)
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    _wait_for_teleport_outcome,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.sniffer.world_state import get_terrain_map
from tankpit_bot.state.types import ContainerStateDict

_EQUIPMENT_PROBE_TARGET_STEP = 16
_EQUIPMENT_PROBE_TARGET_MAX_RADIUS = 48
_EQUIPMENT_PROBE_TELEPORT_STRATEGY: Literal[
    "sync_before_teleport",
    "immediate_after_map_open",
] = DEFAULT_TELEPORT_STRATEGY
run_tracked_teleport_attempt = _shared_run_tracked_teleport_attempt


class EquipmentProbeError(Exception):
    """Raised when the equipment probe cannot proceed."""


def _make_reposition_target(target_x: int, target_y: int) -> TeleportTargetDict:
    """Return a typed target label for an equipment reposition teleport."""
    return TeleportTargetDict(
        label=f"equipment_reposition_{target_x}_{target_y}",
        x=target_x,
        y=target_y,
    )


class EquipmentProbe(ProbeBase):
    """Live teleport-radar-equipment probe."""

    def _build_attempt_result(
        self,
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
        teleport_cycle_ids: list[int],
        radar_cycle_id: int | None = None,
        move_cycle_id: int | None = None,
        pickup_cycle_id: int | None = None,
        phase_overlaps: list[ActionPhaseOverlapDict] | None = None,
        reposition_map_open_started_ms: int | None = None,
        reposition_map_sync_timestamp_ms: int | None = None,
        reposition_teleport_started_ms: int | None = None,
    ) -> EquipmentProbeAttemptResultDict:
        """Create a typed attempt result payload."""
        return _equipment_probe_operations.build_attempt_result_for_probe(
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
            inventory_count_before=inventory_count_before,
            inventory_count_after=inventory_count_after,
            landed_signal_received=landed_signal_received,
            landed_x=landed_x,
            landed_y=landed_y,
            equipment_target=equipment_target,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle_id,
            move_cycle_id=move_cycle_id,
            pickup_cycle_id=pickup_cycle_id,
            phase_overlaps=phase_overlaps,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
        )

    def _finalize_attempt_delay(
        self,
        page: action_session.WaitPageProtocol,
        *,
        settle_delay_ms: int,
    ) -> None:
        """Apply optional settle delay after an attempt."""
        _equipment_probe_operations.finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)

    def _build_map_sync_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        inventory_count_before: int,
        message_start_index: int,
        teleport_cycle_ids: list[int],
    ) -> EquipmentProbeAttemptResultDict:
        """Build a map-sync-timeout result."""
        return _equipment_probe_operations.build_map_sync_timeout_result_for_probe(
            self,
            target=target,
            map_open_started_ms=map_open_started_ms,
            inventory_count_before=inventory_count_before,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
        )

    def _build_teleport_timeout_result(
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
        """Build a teleport-timeout result."""
        return _equipment_probe_operations.build_teleport_timeout_result_for_probe(
            self,
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            inventory_count_before=inventory_count_before,
            teleport_result=teleport_result,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
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
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        equipment_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
    ) -> EquipmentProbeAttemptResultDict:
        """Build a reposition map-sync-timeout result."""
        return _equipment_probe_operations.build_reposition_map_sync_timeout_result_for_probe(
            self,
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
            phase_overlaps=phase_overlaps,
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
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        equipment_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        phase_overlaps: list[ActionPhaseOverlapDict],
    ) -> EquipmentProbeAttemptResultDict:
        """Build a reposition teleport-timeout result."""
        return _equipment_probe_operations.build_reposition_teleport_timeout_result_for_probe(
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
            inventory_count_before=inventory_count_before,
            teleport_result=teleport_result,
            equipment_target=equipment_target,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle_id,
            phase_overlaps=phase_overlaps,
        )

    def _build_radar_timeout_result(
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
        """Build a radar-timeout result."""
        return _equipment_probe_operations.build_radar_timeout_result_for_probe(
            self,
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            inventory_count_before=inventory_count_before,
            teleport_result=teleport_result,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle_id,
            phase_overlaps=phase_overlaps,
        )

    def _build_no_equipment_visible_result(
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
        """Build a no-equipment-visible result."""
        return _equipment_probe_operations.build_no_equipment_visible_result_for_probe(
            self,
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
            phase_overlaps=phase_overlaps,
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
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        equipment_target: ContainerStateDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
    ) -> EquipmentProbeAttemptResultDict:
        """Run the pickup portion of an equipment attempt."""
        return _equipment_probe_operations.run_pickup_attempt_for_probe(
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
            inventory_count_before=inventory_count_before,
            teleport_result=teleport_result,
            equipment_target=equipment_target,
            message_start_index=message_start_index,
            teleport_cycle_ids=teleport_cycle_ids,
            radar_cycle_id=radar_cycle_id,
            dispatch_failure_error=EquipmentProbeError,
            dispatch_failure_message="move_to command dispatch failed during equipment collection",
        )

    def _probe_single_equipment_target(
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
        ] = _EQUIPMENT_PROBE_TELEPORT_STRATEGY,
    ) -> EquipmentProbeAttemptResultDict:
        """Probe one equipment target through the full pipeline."""
        return _shared_run_single_equipment_target_attempt(
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
            run_tracked_equipment_collection_phase=run_tracked_equipment_collection_phase,
            build_map_sync_timeout_result=self._build_map_sync_timeout_result,
            build_teleport_timeout_result=self._build_teleport_timeout_result,
            finalize_attempt_delay=self._finalize_attempt_delay,
            terrain_provider=get_terrain_map,
            find_visible_target=find_visible_equipment_target,
            requires_reposition=visible_equipment_requires_reposition,
            find_landing_tile=find_visible_equipment_landing_tile,
            get_phase_overlaps=self._get_attempt_phase_overlaps,
            build_radar_timeout_result=self._build_radar_timeout_result,
            build_no_equipment_visible_result=self._build_no_equipment_visible_result,
            build_reposition_map_sync_timeout_result=(
                self._build_reposition_map_sync_timeout_result
            ),
            build_reposition_teleport_timeout_result=(
                self._build_reposition_teleport_timeout_result
            ),
            run_pickup_attempt=self._run_pickup_attempt,
            make_reposition_target=_make_reposition_target,
            teleport_strategy_requires_map_sync=teleport_strategy_requires_map_sync,
            dispatch_failure_error=EquipmentProbeError,
            unavailable_error=EquipmentProbeError,
            unexpected_result_error=TeleportProbeError,
            unexpected_missing_target_error=EquipmentProbeError,
            no_landing_tile_error=EquipmentProbeError,
            missing_dispatch_error=EquipmentProbeError,
            acquisition_dispatch_failure_message="map_open command dispatch failed",
            teleport_dispatch_failure_message="teleport command dispatch failed",
            reposition_acquisition_dispatch_failure_message=(
                "map_open command dispatch failed during equipment reposition"
            ),
            reposition_teleport_dispatch_failure_message=(
                "teleport command dispatch failed during equipment reposition"
            ),
            unavailable_message="cdp session is unavailable",
            impossible_map_sync_timeout_message=(
                "teleport outcome reported impossible map_sync_timeout"
            ),
            reposition_impossible_result_message=(
                "teleport outcome reported impossible map_sync_timeout during equipment reposition"
            ),
            reposition_missing_target_message="visible equipment target disappeared unexpectedly",
            no_landing_tile_message="visible equipment target has no teleport landing tile",
            missing_dispatch_message="equipment attempt ended before teleport dispatch",
        )


__all__ = [
    "EquipmentProbe",
    "EquipmentProbeError",
    "EquipmentTargetingError",
    "format_equipment_probe_summary",
]
