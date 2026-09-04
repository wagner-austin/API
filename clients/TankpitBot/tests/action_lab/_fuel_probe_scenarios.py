"""Multi-step scenario runners for the fuel-probe tests.

The two long orchestration helpers -- one driving
``probe_single_target`` end to end, one driving the tracked-reposition
resolve path -- plus the small result builders they need. Split from
:mod:`tests.action_lab._fuel_probe_harness` so neither file carries the
other's bulk.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import (
    Literal,
)

from tests.action_lab._fuel_probe_harness import (
    _build_wait_results,
    _make_pickup_outcome_callback,
    _make_teleport_outcome_callback,
    _make_world_sync_waiter,
    _ProbeHarness,
    _snapshot,
    fuel_probe_module,
    fuel_targets_module,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from tests.action_lab.conftest import (
    ground_terrain,
    rock_wall,
)
from typing_extensions import Unpack

from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.fuel_probe_targets import FuelProbeError
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
)
from tankpit_bot.action_lab.fuel_target_phase import FuelTargetResolution
from tankpit_bot.action_lab.teleport import TeleportProbeError
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
from tankpit_bot.state import (
    ContainerStateDict,
    coord_key,
    make_container_state,
)


def _run_probe_single_target_scenario(
    *,
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
) -> FuelProbeAttemptResultDict:
    """Run one configured single-target probe scenario.

    Runs the REAL targeting helpers (``_find_visible_fuel_target``,
    ``_visible_fuel_requires_reposition``, ``_find_visible_fuel_landing_tile``)
    against a harness whose world state holds the configured fuel container
    and terrain set up to produce the desired branch:

    * reposition scenarios get a rock-wall at x=102 that spans the viewport
      height — the real BFS finds no detour, ``requires_reposition`` returns
      True, and ``find_landing_tile`` returns the container coord (since the
      container's tile is GROUND).
    * all other scenarios get fully-passable terrain.
    * ``no_fuel_visible`` simply omits the container; real finder returns None.

    Teleport-outcome and pickup-outcome are still callbacks because they drive
    the state machine's terminal status — those are the leaves the test is
    exercising. Everything between the test and those leaves is real code.
    """
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    wait_results = _build_wait_results(status, map_sync_result, radar_sync_result)

    wait_for_world_sync = _make_world_sync_waiter(wait_results)

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        return wait_for_world_sync(page, provider, started_ms, timeout_ms)

    action_hooks.wait_for_world_sync = _wait_for_world_sync
    action_hooks.wait_for_radar_sync = _wait_for_world_sync
    fuel_probe_module._wait_for_teleport_outcome = _make_teleport_outcome_callback(teleport_status)
    fuel_targets_module._wait_for_pickup_outcome = _make_pickup_outcome_callback(pickup_status)

    is_reposition_scenario = status in {
        "reposition_map_sync_timeout",
        "reposition_teleport_timeout",
    }

    def _reposition_blocking_terrain() -> TerrainMapProtocol:
        return ground_terrain(rock_wall(102, range(92, 108)))

    terrain_provider: Callable[[], TerrainMapProtocol | None] = (
        _reposition_blocking_terrain if is_reposition_scenario else ground_terrain
    )
    probe.world.terrain_map = terrain_provider()
    probe.world.terrain_map = terrain_provider()

    if fuel_target is not None:
        target_key = coord_key(fuel_target["x"], fuel_target["y"])
        probe._world_state["containers"][target_key] = fuel_target

    result = probe._probe_single_fuel_target(
        target=target,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=250,
        teleport_strategy="sync_before_teleport",
    )

    assert result["status"] == status
    assert result["target"] == target
    assert result["message_start_index"] == 0
    assert result["message_end_index"] == 0
    assert probe._fake_page.waits[-1] == 250.0
    if status in {"picked_up_fuel", "pickup_timeout"}:
        assert probe.move_calls == [(101, 100)]
    else:
        assert probe.move_calls == []
    return result


def _original_teleport_result(target: TeleportTargetDict) -> TeleportAttemptResultDict:
    """Build the pre-reposition landed teleport result for reposition-branch tests."""
    return TeleportAttemptResultDict(
        target=target,
        teleport_cycle_id=1,
        status="landed_exact",
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        completion_timestamp_ms=1500,
        map_sync_elapsed_ms=200,
        teleport_elapsed_ms=200,
        fuel_before=900,
        fuel_after=840,
        world_timestamp_before=950,
        world_timestamp_after=1450,
        landed_signal_received=True,
        landed_x=124,
        landed_y=100,
        message_start_index=0,
        message_end_index=0,
        page_snapshots=[],
    )


def _resolve_with_tracked_reposition(
    *,
    acquisition_sync_timestamp_ms: int | None,
    reposition_teleport_started_ms: int | None,
    reposition_teleport_status: Literal["landed_exact", "teleport_timeout"] | None,
) -> FuelTargetResolution:
    """Drive the real post-radar resolution through a faked reposition attempt.

    Args:
        acquisition_sync_timestamp_ms: Map-sync timestamp the tracked
            reposition reports (``None`` = map-sync timeout).
        reposition_teleport_started_ms: Teleport dispatch timestamp of the
            reposition (``None`` = never dispatched).
        reposition_teleport_status: Terminal status of the reposition
            teleport, or ``None`` for no teleport result.

    Returns:
        The ``FuelTargetResolution`` produced by the real resolver.
    """
    from tests.action_lab._teleport_seams import fuel_target_phase_module

    from tankpit_bot.action_lab import fuel_target_phase

    clock = ReplayClock(2000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    page = ClockAdvancingPage(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    fuel_target = make_container_state(101, 100, True, 300)
    original_attempt_runner = fuel_target_phase_module.run_tracked_teleport_attempt

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return TeleportPageSnapshotDict(
            phase=phase,
            timestamp_ms=2000,
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

    def _reposition_teleport_result(
        reposition_target: TeleportTargetDict,
        status: Literal["landed_exact", "teleport_timeout"],
    ) -> TeleportAttemptResultDict:
        return TeleportAttemptResultDict(
            target=reposition_target,
            teleport_cycle_id=3,
            status=status,
            map_open_started_ms=2000,
            map_sync_timestamp_ms=acquisition_sync_timestamp_ms,
            teleport_started_ms=2300,
            completion_timestamp_ms=2500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=200,
            fuel_before=840,
            fuel_after=800,
            world_timestamp_before=2100,
            world_timestamp_after=2450,
            landed_signal_received=status == "landed_exact",
            landed_x=102,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
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
            cdp,
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
        cycle = probe._start_action_phase("teleport", attempt_label=attempt_label)
        teleport_result = (
            None
            if reposition_teleport_status is None
            else _reposition_teleport_result(target, reposition_teleport_status)
        )
        return TrackedTeleportAttempt(
            message_start_index=0,
            teleport_cycle=cycle,
            acquisition_started_ms=2000,
            acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
            page_snapshots=[],
            capture_page_snapshot=_capture_page_snapshot,
            teleport_result=teleport_result,
            teleport_started_ms=reposition_teleport_started_ms,
        )

    fuel_target_phase_module.run_tracked_teleport_attempt = _run_attempt
    try:
        return fuel_target_phase.resolve_fuel_target_after_radar(
            page,
            probe,
            cdp=probe._cdp,
            target=target,
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            radar_started_ms=1600,
            radar_sync_timestamp_ms=1700,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            fuel_before=900,
            teleport_result=_original_teleport_result(target),
            message_start_index=0,
            teleport_cycle_ids=[1],
            radar_cycle_id=2,
            teleport_strategy="sync_before_teleport",
            snapshot_before=_snapshot(1000),
            capture_snapshot=lambda: _snapshot(1900),
            terrain_provider=lambda: None,
            find_visible_target=lambda current_probe: fuel_target,
            requires_reposition=lambda current_probe, current_target: True,
            find_landing_tile=lambda current_probe, current_target: (102, 100),
            get_phase_overlaps=probe._get_attempt_phase_overlaps,
            build_no_fuel_visible_result=probe._build_no_fuel_visible_result,
            build_reposition_map_sync_timeout_result=(
                probe._build_reposition_map_sync_timeout_result
            ),
            build_reposition_teleport_timeout_result=(
                probe._build_reposition_teleport_timeout_result
            ),
            make_reposition_target=lambda target_x, target_y: TeleportTargetDict(
                label=f"fuel_reposition_{target_x}_{target_y}",
                x=target_x,
                y=target_y,
            ),
            wait_for_teleport_outcome=_unused_teleport_outcome_waiter,
            teleport_strategy_requires_map_sync=lambda strategy: strategy == "sync_before_teleport",
            no_landing_tile_error=FuelProbeError,
            dispatch_failure_error=FuelProbeError,
            unavailable_error=FuelProbeError,
            unexpected_result_error=TeleportProbeError,
            unavailable_message="cdp session is unavailable",
            no_landing_tile_message="visible fuel target has no teleport landing tile",
            impossible_result_message=(
                "teleport outcome reported impossible map_sync_timeout during fuel reposition"
            ),
            acquisition_dispatch_failure_message=(
                "map_open command dispatch failed during fuel reposition"
            ),
            teleport_dispatch_failure_message=(
                "teleport command dispatch failed during fuel reposition"
            ),
        )
    finally:
        fuel_target_phase_module.run_tracked_teleport_attempt = original_attempt_runner


def _unused_teleport_outcome_waiter(
    page: action_session.WaitPageProtocol,
    provider: action_session.BufferedWorldStateProviderProtocol,
    target: TeleportTargetDict,
    **kwargs: Unpack[TeleportOutcomeWaiterKwargs],
) -> TeleportAttemptResultDict:
    """Fail loudly if the faked reposition attempt delegates to the waiter."""
    _ = (page, provider, target, kwargs)
    raise AssertionError("teleport outcome waiter must not be reached in these tests")
