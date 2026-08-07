"""Tests for ``probe_single_target``: terminal statuses and rejections.

The happy path plus the two impossible-outcome rejections the probe
must refuse rather than silently accept.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import (
    Literal,
)

import pytest
from tests.action_lab._fuel_probe_harness import (
    _ProbeHarness,
    fuel_probe_module,
)
from tests.action_lab._fuel_probe_scenarios import (
    _run_probe_single_target_scenario,
)
from tests.action_lab._replay_page import ReplayClock

from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.fuel_probe import (
    FuelProbeError,
)
from tankpit_bot.action_lab.teleport import TeleportProbeError
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    TrackedTeleportAttempt,
)
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.state import (
    ContainerStateDict,
    make_container_state,
)


@pytest.mark.parametrize(
    (
        "status",
        "map_sync_result",
        "teleport_status",
        "radar_sync_result",
        "fuel_target",
        "pickup_status",
    ),
    [
        ("map_sync_timeout", None, None, None, None, None),
        ("teleport_timeout", 1200, "teleport_timeout", None, None, None),
        ("radar_timeout", 1200, "landed_exact", None, None, None),
        ("no_fuel_visible", 1200, "landed_exact", 1600, None, None),
        (
            "picked_up_fuel",
            1200,
            "landed_exact",
            1600,
            make_container_state(101, 100, True, 300),
            "picked_up_fuel",
        ),
        (
            "pickup_timeout",
            1200,
            "landed_exact",
            1600,
            make_container_state(101, 100, True, 300),
            "pickup_timeout",
        ),
    ],
)
def test_probe_single_target_records_terminal_statuses(
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
    map_sync_result: int | None,
    teleport_status: Literal["landed_exact", "teleport_timeout", "reposition_teleport_timeout"]
    | None,
    radar_sync_result: int | None,
    fuel_target: ContainerStateDict | None,
    pickup_status: Literal["picked_up_fuel", "pickup_timeout"] | None,
) -> None:
    """Single-target probe records all terminal outcomes."""
    _run_probe_single_target_scenario(
        status=status,
        map_sync_result=map_sync_result,
        teleport_status=teleport_status,
        radar_sync_result=radar_sync_result,
        fuel_target=fuel_target,
        pickup_status=pickup_status,
    )


def test_probe_single_target_rejects_impossible_map_sync_timeout_teleport_outcome() -> None:
    """Fuel probe rejects a teleport outcome that reports map-sync timeout after sync success."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200
    action_hooks.wait_for_radar_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _teleport_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            teleport_cycle_id,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="map_sync_timeout",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=200,
            fuel_before=700,
            fuel_after=650,
            world_timestamp_before=1000,
            world_timestamp_after=1450,
            landed_signal_received=False,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome

    with pytest.raises(
        TeleportProbeError,
        match="teleport outcome reported impossible map_sync_timeout",
    ):
        probe._probe_single_fuel_target(
            target=target,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )


def test_probe_single_target_rejects_missing_tracked_teleport_result() -> None:
    """Fuel probe rejects a tracked attempt that never produced a teleport result."""
    from tankpit_bot.action_lab import fuel_probe as fuel_probe_runtime

    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    original_attempt_runner = fuel_probe_runtime.run_tracked_teleport_attempt

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return TeleportPageSnapshotDict(
            phase=phase,
            timestamp_ms=1000,
            client_present=True,
            map_visible=False,
            client_state=1,
            client_busy=False,
            pending_actions=0,
            heartbeat_age_ms=1,
            last_page_client_send_age_ms=2,
            last_bot_send_age_ms=3,
            ws_ready_state=1,
            current_send_label=None,
            sent_frame_meta_queue_length=0,
            self_fields={},
            world_fields={},
            map_fields={},
            world_collections={},
        )

    def _run_attempt(
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
        _ = (
            page,
            probe,
            target,
            cdp,
            attempt_label,
            fuel_before,
            world_timestamp_before,
            send_acquisition_command,
            acquisition_command_name,
            capture_before_map_open,
            wait_for_acquisition_sync,
            acquisition_timeout_ms,
            teleport_timeout_ms,
            wait_for_outcome,
            dispatch_failure_error,
            acquisition_dispatch_failure_message,
            teleport_dispatch_failure_message,
            unavailable_error,
            unavailable_message,
            unexpected_result_error,
            unexpected_result_message,
            reset_to_idle_before_start,
        )
        return TrackedTeleportAttempt(
            message_start_index=0,
            teleport_cycle=ActionPhaseCycleDict(phase="teleport", cycle_id=1, started_ms=1000),
            acquisition_started_ms=1000,
            acquisition_sync_timestamp_ms=1200,
            page_snapshots=[],
            capture_page_snapshot=_capture_page_snapshot,
            teleport_result=None,
            teleport_started_ms=None,
        )

    fuel_probe_runtime.run_tracked_teleport_attempt = _run_attempt
    try:
        with pytest.raises(FuelProbeError, match="fuel attempt ended before teleport dispatch"):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
                teleport_strategy="sync_before_teleport",
            )
    finally:
        fuel_probe_runtime.run_tracked_teleport_attempt = original_attempt_runner


def test_finalize_attempt_delay_skips_wait_for_zero_delay() -> None:
    """Fuel probe does not wait when settle delay is disabled."""
    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)

    probe._finalize_attempt_delay(probe._fake_page, settle_delay_ms=0)

    assert probe._fake_page.waits == []
