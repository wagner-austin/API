"""Shared builders, probe doubles, and canned results for the
equipment-attempt tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from typing_extensions import Unpack

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseOverlapDict,
)
from tankpit_bot.action_lab.equipment_collection_phase import (
    run_tracked_equipment_collection_phase,
)
from tankpit_bot.action_lab.equipment_probe_attempt import (
    RunTrackedTeleportAttemptProtocol,
    run_single_equipment_target_attempt,
)
from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeAttemptResultDict
from tankpit_bot.action_lab.equipment_target_phase import (
    EquipmentTargetPhaseProbeProtocol,
)
from tankpit_bot.action_lab.session import (
    BufferedWorldStateProviderProtocol,
    WaitPageProtocol,
)
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    TrackedTeleportAttempt,
)
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterKwargs,
    TeleportOutcomeWaiterProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import ContainerStateDict, make_viewport_state
from tankpit_bot.types import CapturedMessage


class _Page:
    def wait_for_timeout(self, timeout: float) -> None:
        pass

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)


class _Probe:
    def __init__(self) -> None:
        self.world = get_world_service()
        self._messages: list[CapturedMessage] = []
        self._w = _world()
        self._cid = 0
        self._cdp: CDPSessionProtocol | None = None
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages

    @property
    def magic(self) -> str | None:
        return None

    def open_map(self) -> bool:
        return True

    def use_radar(self) -> bool:
        return True

    def teleport_to(self, x: int, y: int) -> bool:
        return True

    def get_world_state(self) -> WorldStateDict:
        return self._w

    def get_self_state(self) -> SelfStateDict | None:
        return self._w["self_state"]

    def _require_self_state(self) -> SelfStateDict:
        return _SELF

    def _require_page(self) -> WaitPageProtocol:
        return _Page()

    def _start_action_phase(
        self,
        phase: Literal["teleport", "radar", "move", "pickup"],
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        self._cid += 1
        return ActionPhaseCycleDict(phase=phase, cycle_id=self._cid, started_ms=1000)

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        pass

    def _reset_probe_state_to_idle(self) -> None:
        pass

    def _reset_attempt_phase_overlaps(self) -> None:
        pass

    def _get_attempt_phase_overlaps(self) -> list[ActionPhaseOverlapDict]:
        return []


_TARGET = TeleportTargetDict(label="t", x=10, y=20)


_SELF = make_self_state(
    tank_id=1,
    x=100,
    y=100,
    team=2,
    rank=1,
    fuel=700,
    leaderboard_position=1,
)


_CYCLE = ActionPhaseCycleDict(phase="teleport", cycle_id=1, started_ms=1000)


def _world() -> WorldStateDict:
    b = make_empty_world_state()
    return WorldStateDict(
        self_state=_SELF,
        tanks=b["tanks"],
        containers=b["containers"],
        mines=b["mines"],
        terrain=b["terrain"],
        viewport=make_viewport_state(left=92, top=92, width=16, height=16),
        scanned_tiles=b["scanned_tiles"],
        timestamp_ms=2000,
    )


_ATTEMPT = EquipmentProbeAttemptResultDict(
    target=_TARGET,
    teleport_cycle_ids=[1],
    radar_cycle_id=None,
    move_cycle_id=None,
    pickup_cycle_id=None,
    status="no_equipment_visible",
    map_open_started_ms=1000,
    map_sync_timestamp_ms=None,
    teleport_started_ms=None,
    radar_started_ms=None,
    radar_sync_timestamp_ms=None,
    reposition_map_open_started_ms=None,
    reposition_map_sync_timestamp_ms=None,
    reposition_teleport_started_ms=None,
    pickup_started_ms=None,
    completion_timestamp_ms=2000,
    inventory_count_before=0,
    inventory_count_after=None,
    landed_signal_received=False,
    landed_x=None,
    landed_y=None,
    equipment_target_x=None,
    equipment_target_y=None,
    phase_overlaps=[],
    message_start_index=0,
    message_end_index=0,
)


_TP_RESULT = TeleportAttemptResultDict(
    target=_TARGET,
    teleport_cycle_id=1,
    status="landed_exact",
    map_open_started_ms=1000,
    map_sync_timestamp_ms=1100,
    teleport_started_ms=1200,
    completion_timestamp_ms=1500,
    map_sync_elapsed_ms=100,
    teleport_elapsed_ms=300,
    fuel_before=700,
    fuel_after=690,
    world_timestamp_before=1100,
    world_timestamp_after=1450,
    landed_signal_received=True,
    landed_x=10,
    landed_y=20,
    message_start_index=0,
    message_end_index=0,
    page_snapshots=[],
)


def _waiter(
    page: WaitPageProtocol,
    provider: BufferedWorldStateProviderProtocol,
    target: TeleportTargetDict,
    **kwargs: Unpack[TeleportOutcomeWaiterKwargs],
) -> TeleportAttemptResultDict:
    raise AssertionError("unreachable")


def _sync_policy(
    s: Literal["sync_before_teleport", "immediate_after_map_open"],
) -> bool:
    return s == "sync_before_teleport"


def _no_find(p: EquipmentTargetPhaseProbeProtocol) -> ContainerStateDict | None:
    return None


def _no_repo(p: EquipmentTargetPhaseProbeProtocol, c: ContainerStateDict) -> bool:
    return False


def _no_land(
    p: EquipmentTargetPhaseProbeProtocol,
    c: ContainerStateDict,
) -> tuple[int, int] | None:
    return None


def _build_no_vis(
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
    return _ATTEMPT


def _build_repo_map(
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
    return _ATTEMPT


def _build_repo_tp(
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
    return _ATTEMPT


def _build_radar_timeout(
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
    return _ATTEMPT


def _build_pickup(
    *,
    page: WaitPageProtocol,
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
    return _ATTEMPT


def _build_map_sync_timeout(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    inventory_count_before: int,
    message_start_index: int,
    teleport_cycle_ids: list[int],
) -> EquipmentProbeAttemptResultDict:
    return _ATTEMPT


def _build_tp_timeout(
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
    return _ATTEMPT


def _noop_finalize(page: WaitPageProtocol, *, settle_delay_ms: int) -> None:
    pass


def _snap(
    label: Literal[
        "before_map_open",
        "before_teleport",
        "after_map_data",
        "landed",
        "timeout",
    ],
) -> TeleportPageSnapshotDict:
    return TeleportPageSnapshotDict(
        phase=label,
        timestamp_ms=1000,
        client_present=True,
        map_visible=False,
        client_state=0,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=0,
        last_page_client_send_age_ms=0,
        last_bot_send_age_ms=0,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections={},
    )


def _fake_tp_sync_timeout(
    page: WaitPageProtocol,
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
    return TrackedTeleportAttempt(
        message_start_index=0,
        teleport_cycle=_CYCLE,
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=None,
        page_snapshots=[],
        capture_page_snapshot=_snap,
        teleport_result=None,
        teleport_started_ms=None,
    )


def _fake_tp_missing_dispatch(
    page: WaitPageProtocol,
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
    return TrackedTeleportAttempt(
        message_start_index=0,
        teleport_cycle=_CYCLE,
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=1100,
        page_snapshots=[],
        capture_page_snapshot=_snap,
        teleport_result=None,
        teleport_started_ms=None,
    )


def _fake_tp_landed(
    page: WaitPageProtocol,
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
    return TrackedTeleportAttempt(
        message_start_index=0,
        teleport_cycle=_CYCLE,
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=1100,
        page_snapshots=[],
        capture_page_snapshot=_snap,
        teleport_result=_TP_RESULT,
        teleport_started_ms=1200,
    )


def run_attempt(
    fake_tp: RunTrackedTeleportAttemptProtocol,
    strategy: Literal["sync_before_teleport", "immediate_after_map_open"] = "sync_before_teleport",
) -> EquipmentProbeAttemptResultDict:
    return run_single_equipment_target_attempt(
        probe=_Probe(),
        target=_TARGET,
        map_sync_timeout_ms=30000,
        teleport_timeout_ms=30000,
        radar_timeout_ms=30000,
        pickup_timeout_ms=10000,
        settle_delay_ms=0,
        teleport_strategy=strategy,
        cdp=None,
        wait_for_teleport_outcome=_waiter,
        run_tracked_teleport_attempt=fake_tp,
        run_tracked_equipment_collection_phase=run_tracked_equipment_collection_phase,
        build_map_sync_timeout_result=_build_map_sync_timeout,
        build_teleport_timeout_result=_build_tp_timeout,
        finalize_attempt_delay=_noop_finalize,
        terrain_provider=lambda: None,
        find_visible_target=_no_find,
        requires_reposition=_no_repo,
        find_landing_tile=_no_land,
        get_phase_overlaps=lambda: [],
        build_radar_timeout_result=_build_radar_timeout,
        build_no_equipment_visible_result=_build_no_vis,
        build_reposition_map_sync_timeout_result=_build_repo_map,
        build_reposition_teleport_timeout_result=_build_repo_tp,
        run_pickup_attempt=_build_pickup,
        make_reposition_target=lambda x, y: _TARGET,
        teleport_strategy_requires_map_sync=_sync_policy,
        dispatch_failure_error=RuntimeError,
        unavailable_error=RuntimeError,
        unexpected_result_error=RuntimeError,
        unexpected_missing_target_error=RuntimeError,
        no_landing_tile_error=RuntimeError,
        missing_dispatch_error=RuntimeError,
        acquisition_dispatch_failure_message="m",
        teleport_dispatch_failure_message="t",
        reposition_acquisition_dispatch_failure_message="rm",
        reposition_teleport_dispatch_failure_message="rt",
        unavailable_message="u",
        impossible_map_sync_timeout_message="i",
        reposition_impossible_result_message="ri",
        reposition_missing_target_message="rmt",
        no_landing_tile_message="nl",
        missing_dispatch_message="missing dispatch",
    )
