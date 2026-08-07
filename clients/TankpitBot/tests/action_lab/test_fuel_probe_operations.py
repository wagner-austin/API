"""Direct tests for fuel-probe helper operations."""

from __future__ import annotations

from typing import Literal

from tests.action_lab._fuel_probe_harness import _ProbeHarness
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.fuel_probe_attempt_contracts import RunTrackedPickupPhaseProtocol
from tankpit_bot.action_lab.fuel_probe_operations import (
    build_attempt_result_for_probe,
    build_map_sync_timeout_result_for_probe,
    build_no_fuel_visible_result_for_probe,
    build_radar_timeout_result_for_probe,
    build_reposition_map_sync_timeout_result_for_probe,
    build_reposition_teleport_timeout_result_for_probe,
    build_teleport_timeout_result_for_probe,
    finalize_attempt_delay,
    run_pickup_attempt_for_probe,
)
from tankpit_bot.action_lab.pickup_phase import (
    PickupImmediateOutcomeProtocol,
    PickupOutcomeWaiterProtocol,
    PickupPhaseProbeProtocol,
    PickupTimeoutSizerProtocol,
)
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.state import make_container_state


def _snapshot(timestamp_ms: int) -> PageClientSnapshotDict:
    """Build a sample page-client snapshot for fuel-probe operations tests."""
    return PageClientSnapshotDict(
        timestamp_ms=timestamp_ms,
        client_present=True,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=10,
        last_page_client_send_age_ms=20,
        last_bot_send_age_ms=30,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        world_collections={},
        map_fields={},
    )


def _target() -> TeleportTargetDict:
    """Return a stable teleport target for helper tests."""
    return TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)


def _teleport_result(
    target: TeleportTargetDict,
    *,
    status: Literal["landed_exact", "teleport_timeout"] = "landed_exact",
) -> TeleportAttemptResultDict:
    """Build a typed teleport result for helper tests."""
    return TeleportAttemptResultDict(
        target=target,
        teleport_cycle_id=1,
        status=status,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        completion_timestamp_ms=1500,
        map_sync_elapsed_ms=200,
        teleport_elapsed_ms=200,
        fuel_before=700,
        fuel_after=620,
        world_timestamp_before=950,
        world_timestamp_after=1450,
        landed_signal_received=status != "teleport_timeout",
        landed_x=124,
        landed_y=100,
        message_start_index=0,
        message_end_index=2,
        page_snapshots=[],
    )


def test_finalize_attempt_delay_waits_only_when_positive() -> None:
    """Finalize delay waits only when a positive settle delay is requested."""
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(clock)

    finalize_attempt_delay(page, settle_delay_ms=0)
    finalize_attempt_delay(page, settle_delay_ms=250)

    assert page.waits == [250.0]


def test_result_builder_helpers_emit_expected_statuses() -> None:
    """Result-builder helpers produce terminal payloads with expected metadata."""
    clock = ReplayClock(2000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = _target()
    fuel_target = make_container_state(101, 100, True, 300)
    teleport_result = _teleport_result(target)

    snapshot_before = _snapshot(1000)
    snapshot_after = _snapshot(1900)
    built = build_attempt_result_for_probe(
        probe,
        target=target,
        status="picked_up_fuel",
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        radar_started_ms=1400,
        radar_sync_timestamp_ms=1500,
        pickup_started_ms=1600,
        completion_timestamp_ms=1700,
        fuel_before=700,
        fuel_after=900,
        landed_signal_received=True,
        landed_x=124,
        landed_y=100,
        fuel_target=fuel_target,
        message_start_index=0,
        teleport_cycle_ids=[1, 2],
        radar_cycle_id=3,
        move_cycle_id=4,
        pickup_cycle_id=5,
        phase_overlaps=[],
        decision_basis=None,
        reposition_map_open_started_ms=1800,
        reposition_map_sync_timestamp_ms=1850,
        reposition_teleport_started_ms=1900,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )
    map_timeout = build_map_sync_timeout_result_for_probe(
        probe,
        target=target,
        map_open_started_ms=1000,
        fuel_before=700,
        message_start_index=0,
        teleport_cycle_ids=[1],
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )
    teleport_timeout = build_teleport_timeout_result_for_probe(
        probe,
        target=target,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        fuel_before=700,
        teleport_result=_teleport_result(target, status="teleport_timeout"),
        message_start_index=0,
        teleport_cycle_ids=[1],
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )
    reposition_map_timeout = build_reposition_map_sync_timeout_result_for_probe(
        probe,
        target=target,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        radar_started_ms=1400,
        radar_sync_timestamp_ms=1500,
        reposition_map_open_started_ms=1600,
        fuel_before=700,
        teleport_result=teleport_result,
        fuel_target=fuel_target,
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        phase_overlaps=[],
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )
    reposition_teleport_timeout = build_reposition_teleport_timeout_result_for_probe(
        probe,
        target=target,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        radar_started_ms=1400,
        radar_sync_timestamp_ms=1500,
        reposition_map_open_started_ms=1600,
        reposition_map_sync_timestamp_ms=1700,
        reposition_teleport_started_ms=1800,
        fuel_before=700,
        teleport_result=teleport_result,
        fuel_target=fuel_target,
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        phase_overlaps=[],
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )
    radar_timeout = build_radar_timeout_result_for_probe(
        probe,
        target=target,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        radar_started_ms=1400,
        fuel_before=700,
        teleport_result=teleport_result,
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        phase_overlaps=[],
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )
    no_fuel_visible = build_no_fuel_visible_result_for_probe(
        probe,
        target=target,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        radar_started_ms=1400,
        radar_sync_timestamp_ms=1500,
        fuel_before=700,
        teleport_result=teleport_result,
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        phase_overlaps=[],
        decision_basis=None,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )

    assert built["status"] == "picked_up_fuel"
    assert built["message_end_index"] == len(probe.messages)
    assert built["move_cycle_id"] == 4
    assert built["pickup_cycle_id"] == 5
    assert map_timeout["status"] == "map_sync_timeout"
    assert map_timeout["completion_timestamp_ms"] == 2000
    assert teleport_timeout["status"] == "teleport_timeout"
    assert reposition_map_timeout["status"] == "reposition_map_sync_timeout"
    assert reposition_teleport_timeout["status"] == "reposition_teleport_timeout"
    assert radar_timeout["status"] == "radar_timeout"
    assert no_fuel_visible["status"] == "no_fuel_visible"


def test_probe_build_attempt_result_wrapper_delegates_to_shared_builder() -> None:
    """FuelProbe wrapper returns the shared typed attempt result payload."""
    clock = ReplayClock(2000)
    probe = _ProbeHarness(clock)
    target = _target()
    fuel_target = make_container_state(101, 100, True, 300)

    built = probe._build_attempt_result(
        target=target,
        status="picked_up_fuel",
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        radar_started_ms=1400,
        radar_sync_timestamp_ms=1500,
        pickup_started_ms=1600,
        completion_timestamp_ms=1700,
        fuel_before=700,
        fuel_after=900,
        landed_signal_received=True,
        landed_x=124,
        landed_y=100,
        fuel_target=fuel_target,
        message_start_index=0,
        teleport_cycle_ids=[1, 2],
        radar_cycle_id=3,
        move_cycle_id=4,
        pickup_cycle_id=5,
        phase_overlaps=[],
        decision_basis=None,
        reposition_map_open_started_ms=1800,
        reposition_map_sync_timestamp_ms=1850,
        reposition_teleport_started_ms=1900,
        snapshot_before=_snapshot(1000),
        snapshot_after=_snapshot(1900),
    )

    assert built["status"] == "picked_up_fuel"
    assert built["move_cycle_id"] == 4
    assert built["pickup_cycle_id"] == 5


def test_run_pickup_attempt_for_probe_uses_injected_phase_runner() -> None:
    """Pickup helper uses the injected tracked phase runner and builds the result."""
    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    page = ClockAdvancingPage(clock)
    target = _target()
    fuel_target = make_container_state(101, 100, True, 300)
    teleport_result = _teleport_result(target)

    class _TrackedPickupPhase(RunTrackedPickupPhaseProtocol):
        def __call__(
            self,
            page_arg: action_session.WaitPageProtocol,
            probe_arg: PickupPhaseProbeProtocol,
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
            _ = (
                page_arg,
                probe_arg,
                attempt_label,
                target_x,
                target_y,
                current_x,
                current_y,
                fuel_before_pickup,
                pickup_timeout_ms,
                dispatch_failure_error,
                get_completed_outcome,
                wait_for_outcome,
                compute_timeout,
            )
            return (
                ActionPhaseCycleDict(phase="move", cycle_id=11, started_ms=1000),
                ActionPhaseCycleDict(phase="pickup", cycle_id=12, started_ms=1100),
                1200,
                "picked_up_fuel",
                1300,
                900,
            )

    class _CompletedOutcome(PickupImmediateOutcomeProtocol):
        def __call__(
            self,
            probe_arg: action_session.WorldStateProviderProtocol,
            *,
            target_x: int,
            target_y: int,
            fuel_before: int,
        ) -> tuple[Literal["picked_up_fuel"], int, int] | None:
            _ = (probe_arg, target_x, target_y, fuel_before)
            return None

    class _WaitedOutcome(PickupOutcomeWaiterProtocol):
        def __call__(
            self,
            page_arg: action_session.WaitPageProtocol,
            probe_arg: action_session.BufferedWorldStateProviderProtocol,
            *,
            target_x: int,
            target_y: int,
            pickup_started_ms: int,
            fuel_before: int,
            timeout_ms: int,
        ) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
            _ = (
                page_arg,
                probe_arg,
                target_x,
                target_y,
                pickup_started_ms,
                timeout_ms,
            )
            return ("pickup_timeout", 1500, fuel_before)

    class _TimeoutSizer(PickupTimeoutSizerProtocol):
        def __call__(
            self,
            *,
            current_x: int,
            current_y: int,
            target_x: int,
            target_y: int,
            base_timeout_ms: int,
        ) -> int:
            _ = (current_x, current_y, target_x, target_y)
            return base_timeout_ms

    result = run_pickup_attempt_for_probe(
        probe,
        page=page,
        target=target,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        radar_started_ms=1400,
        radar_sync_timestamp_ms=1500,
        reposition_map_open_started_ms=None,
        reposition_map_sync_timestamp_ms=None,
        reposition_teleport_started_ms=None,
        pickup_timeout_ms=3000,
        fuel_before=700,
        teleport_result=teleport_result,
        fuel_target=fuel_target,
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        decision_basis=None,
        snapshot_before=_snapshot(1000),
        capture_snapshot=lambda: _snapshot(1900),
        dispatch_failure_error=RuntimeError,
        run_tracked_pickup_phase=_TrackedPickupPhase(),
        get_completed_outcome=_CompletedOutcome(),
        wait_for_outcome=_WaitedOutcome(),
        compute_timeout=_TimeoutSizer(),
    )

    assert result["status"] == "picked_up_fuel"
    assert result["move_cycle_id"] == 11
    assert result["pickup_cycle_id"] == 12
    assert result["fuel_after"] == 900
