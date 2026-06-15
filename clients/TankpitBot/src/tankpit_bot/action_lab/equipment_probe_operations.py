"""Typed helper functions backing the live equipment-probe class surface."""

from __future__ import annotations

from typing import Literal, Protocol

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict, ActionPhaseOverlapDict
from tankpit_bot.action_lab.equipment_pickup import (
    get_completed_equipment_pickup_outcome,
    total_inventory_count,
    wait_for_equipment_pickup_outcome,
)
from tankpit_bot.action_lab.equipment_probe_results import (
    build_attempt_result as _shared_build_attempt_result,
)
from tankpit_bot.action_lab.equipment_probe_results import (
    build_map_sync_timeout_result as _shared_build_map_sync_timeout_result,
)
from tankpit_bot.action_lab.equipment_probe_results import (
    build_no_equipment_visible_result as _shared_build_no_equipment_visible_result,
)
from tankpit_bot.action_lab.equipment_probe_results import (
    build_radar_timeout_result as _shared_build_radar_timeout_result,
)
from tankpit_bot.action_lab.equipment_probe_results import (
    build_reposition_map_sync_timeout_result as _shared_build_reposition_map_sync_timeout_result,
)
from tankpit_bot.action_lab.equipment_probe_results import (
    build_reposition_teleport_timeout_result as _shared_build_reposition_teleport_timeout_result,
)
from tankpit_bot.action_lab.equipment_probe_results import (
    build_teleport_timeout_result as _shared_build_teleport_timeout_result,
)
from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeAttemptResultDict
from tankpit_bot.action_lab.teleport_phase import _log_command_dispatch_failure
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.state import SelfStateDict
from tankpit_bot.state.types import ContainerStateDict
from tankpit_bot.types import CapturedMessage

_EQUIPMENT_PICKUP_TIMEOUT_PER_TILE_MS = 500
_EQUIPMENT_PICKUP_TIMEOUT_SETTLE_GRACE_MS = 1000
_EQUIPMENT_PICKUP_POLL_INTERVAL_MS = 100.0


class EquipmentProbeBuilderContextProtocol(Protocol):
    """Minimal probe context required for result building."""

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return captured wire messages."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the required current self state."""


class EquipmentProbePickupContextProtocol(
    EquipmentProbeBuilderContextProtocol,
    action_session.BufferedWorldStateProviderProtocol,
    Protocol,
):
    """Minimal probe context required for an equipment pickup attempt."""

    def move_to(self, x: int, y: int) -> bool:
        """Dispatch one movement command."""

    def _start_action_phase(
        self,
        phase: Literal["move", "pickup"],
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Start one action phase cycle."""

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """End one active action phase."""

    def _reset_probe_state_to_idle(self) -> None:
        """Reset the probe state machine to idle."""

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


def effective_equipment_pickup_timeout_ms(
    *,
    current_x: int,
    current_y: int,
    target_x: int,
    target_y: int,
    base_timeout_ms: int,
) -> int:
    """Return a pickup timeout sized for the current travel distance.

    Args:
        current_x: Current self X tile.
        current_y: Current self Y tile.
        target_x: Pickup target X tile.
        target_y: Pickup target Y tile.
        base_timeout_ms: Configured minimum pickup timeout.

    Returns:
        Timeout in milliseconds large enough for the move plus pickup settle.
    """
    travel_distance = abs(target_x - current_x) + abs(target_y - current_y)
    distance_budget_ms = (
        travel_distance * _EQUIPMENT_PICKUP_TIMEOUT_PER_TILE_MS
    ) + _EQUIPMENT_PICKUP_TIMEOUT_SETTLE_GRACE_MS
    if distance_budget_ms > base_timeout_ms:
        return distance_budget_ms
    return base_timeout_ms


def build_attempt_result_for_probe(
    probe: EquipmentProbeBuilderContextProtocol,
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=landed_signal_received,
        landed_x=landed_x,
        landed_y=landed_y,
        equipment_target=equipment_target,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        move_cycle_id=move_cycle_id,
        pickup_cycle_id=pickup_cycle_id,
        phase_overlaps=phase_overlaps,
    )


def build_map_sync_timeout_result_for_probe(
    probe: EquipmentProbeBuilderContextProtocol,
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    inventory_count_before: int,
    message_start_index: int,
    teleport_cycle_ids: list[int],
) -> EquipmentProbeAttemptResultDict:
    """Build one map-sync-timeout result for a probe."""
    return _shared_build_map_sync_timeout_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        completion_timestamp_ms=action_hooks.get_current_time_ms(),
        inventory_count_before=inventory_count_before,
        inventory_count_after=total_inventory_count(get_inventory_state(get_world_service())),
        self_state=probe._require_self_state(),
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
    )


def build_teleport_timeout_result_for_probe(
    probe: EquipmentProbeBuilderContextProtocol,
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
    """Build one teleport-timeout result for a probe."""
    _ = probe._require_self_state()
    return _shared_build_teleport_timeout_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        inventory_count_before=inventory_count_before,
        inventory_count_after=total_inventory_count(get_inventory_state(get_world_service())),
        teleport_result=teleport_result,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
    )


def build_reposition_map_sync_timeout_result_for_probe(
    probe: EquipmentProbeBuilderContextProtocol,
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
    """Build one reposition map-sync-timeout result for a probe."""
    _ = probe._require_self_state()
    return _shared_build_reposition_map_sync_timeout_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        reposition_map_open_started_ms=reposition_map_open_started_ms,
        completion_timestamp_ms=action_hooks.get_current_time_ms(),
        inventory_count_before=inventory_count_before,
        inventory_count_after=total_inventory_count(get_inventory_state(get_world_service())),
        teleport_result=teleport_result,
        equipment_target=equipment_target,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
    )


def build_reposition_teleport_timeout_result_for_probe(
    probe: EquipmentProbeBuilderContextProtocol,
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
    """Build one reposition teleport-timeout result for a probe."""
    _ = probe._require_self_state()
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=total_inventory_count(get_inventory_state(get_world_service())),
        teleport_result=teleport_result,
        equipment_target=equipment_target,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
    )


def build_radar_timeout_result_for_probe(
    probe: EquipmentProbeBuilderContextProtocol,
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
    """Build one radar-timeout result for a probe."""
    _ = probe._require_self_state()
    return _shared_build_radar_timeout_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        completion_timestamp_ms=action_hooks.get_current_time_ms(),
        inventory_count_before=inventory_count_before,
        inventory_count_after=total_inventory_count(get_inventory_state(get_world_service())),
        teleport_result=teleport_result,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
    )


def build_no_equipment_visible_result_for_probe(
    probe: EquipmentProbeBuilderContextProtocol,
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
    """Build one no-visible-equipment result for a probe."""
    _ = probe._require_self_state()
    return _shared_build_no_equipment_visible_result(
        target=target,
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        radar_started_ms=radar_started_ms,
        radar_sync_timestamp_ms=radar_sync_timestamp_ms,
        completion_timestamp_ms=action_hooks.get_current_time_ms(),
        inventory_count_before=inventory_count_before,
        inventory_count_after=total_inventory_count(get_inventory_state(get_world_service())),
        teleport_result=teleport_result,
        message_start_index=message_start_index,
        message_end_index=len(probe.messages),
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        phase_overlaps=phase_overlaps,
    )


def run_pickup_attempt_for_probe(
    probe: EquipmentProbePickupContextProtocol,
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
    dispatch_failure_error: type[Exception],
    dispatch_failure_message: str,
) -> EquipmentProbeAttemptResultDict:
    """Run one equipment pickup attempt for a probe.

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
        inventory_count_before: Inventory total before the full attempt.
        teleport_result: Teleport result that placed the tank in the viewport.
        equipment_target: Selected visible equipment target.
        message_start_index: Raw message start index for this attempt.
        teleport_cycle_ids: Teleport phase ids collected for this attempt.
        radar_cycle_id: Radar phase id for this attempt.
        dispatch_failure_error: Error type raised on movement dispatch failure.
        dispatch_failure_message: Error text for movement dispatch failure.

    Returns:
        Terminal pickup result payload.

    Raises:
        Exception: Raised via ``dispatch_failure_error`` on movement dispatch
            failure.
    """
    self_state_before_pickup = probe._require_self_state()
    inventory_count_before_pickup = total_inventory_count(get_inventory_state(get_world_service()))
    pickup_started_ms = action_hooks.get_current_time_ms()
    move_cycle = probe._start_action_phase("move", attempt_label=target["label"])
    pickup_cycle = probe._start_action_phase("pickup", attempt_label=target["label"])
    timeout_ms = effective_equipment_pickup_timeout_ms(
        current_x=self_state_before_pickup["x"],
        current_y=self_state_before_pickup["y"],
        target_x=equipment_target["x"],
        target_y=equipment_target["y"],
        base_timeout_ms=pickup_timeout_ms,
    )
    action_hooks.drain_buffered_messages(probe)
    immediate = get_completed_equipment_pickup_outcome(
        probe,
        target_x=equipment_target["x"],
        target_y=equipment_target["y"],
        inventory_count_before=inventory_count_before_pickup,
    )
    if immediate is None:
        if not probe.move_to(equipment_target["x"], equipment_target["y"]):
            _log_command_dispatch_failure("move", dispatch_failure_message)
            probe._end_action_phase(move_cycle)
            probe._end_action_phase(pickup_cycle)
            raise dispatch_failure_error(dispatch_failure_message)
        status, completion_timestamp_ms, inventory_count_after = wait_for_equipment_pickup_outcome(
            page,
            probe,
            target_x=equipment_target["x"],
            target_y=equipment_target["y"],
            pickup_started_ms=pickup_started_ms,
            inventory_count_before=inventory_count_before_pickup,
            timeout_ms=timeout_ms,
        )
    else:
        status, completion_timestamp_ms, inventory_count_after = immediate
    probe._end_action_phase(move_cycle)
    probe._end_action_phase(pickup_cycle)
    probe._reset_probe_state_to_idle()
    return build_attempt_result_for_probe(
        probe,
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
        inventory_count_before=inventory_count_before,
        inventory_count_after=inventory_count_after,
        landed_signal_received=teleport_result["landed_signal_received"],
        landed_x=teleport_result["landed_x"],
        landed_y=teleport_result["landed_y"],
        equipment_target=equipment_target,
        message_start_index=message_start_index,
        teleport_cycle_ids=teleport_cycle_ids,
        radar_cycle_id=radar_cycle_id,
        move_cycle_id=move_cycle["cycle_id"],
        pickup_cycle_id=pickup_cycle["cycle_id"],
        phase_overlaps=probe._get_attempt_phase_overlaps(),
    )


__all__ = [
    "EquipmentProbeBuilderContextProtocol",
    "EquipmentProbePickupContextProtocol",
    "build_attempt_result_for_probe",
    "build_map_sync_timeout_result_for_probe",
    "build_no_equipment_visible_result_for_probe",
    "build_radar_timeout_result_for_probe",
    "build_reposition_map_sync_timeout_result_for_probe",
    "build_reposition_teleport_timeout_result_for_probe",
    "build_teleport_timeout_result_for_probe",
    "effective_equipment_pickup_timeout_ms",
    "finalize_attempt_delay",
    "run_pickup_attempt_for_probe",
]


# Re-exported for module-level patching parity with fuel_probe.
_ = (
    _EQUIPMENT_PICKUP_TIMEOUT_PER_TILE_MS,
    _EQUIPMENT_PICKUP_TIMEOUT_SETTLE_GRACE_MS,
    _EQUIPMENT_PICKUP_POLL_INTERVAL_MS,
)
