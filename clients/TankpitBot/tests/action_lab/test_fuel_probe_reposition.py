"""Tests for the probe's reposition and dispatch-failure paths.

Blocked visible fuel, an already-completed pickup, and the dispatch
failure that must raise rather than record a status.
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
    fuel_targeting_module,
)
from tests.action_lab._replay_page import ReplayClock
from tests.action_lab.conftest import (
    ground_terrain,
)
from tests.fakes import InMemoryTerrainMap

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BufferedMessageSourceProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.fuel_probe import (
    FuelProbe,
    FuelProbeError,
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
    make_self_state,
)


def test_probe_single_target_repositions_for_blocked_visible_fuel() -> None:
    """Single-target fuel probe can reposition to a blocked visible fuel container."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    wait_results = [1200, 1600, 1800]

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return wait_results.pop(0)

    action_hooks.wait_for_world_sync = _wait_for_world_sync
    action_hooks.wait_for_radar_sync = _wait_for_world_sync

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
        if target["label"].startswith("fuel_reposition_"):
            landed_x = 102
            landed_y = 100
            fuel_after = 620
        else:
            landed_x = 124
            landed_y = 100
            fuel_after = 640
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="landed_exact",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=landed_x,
            landed_y=landed_y,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    def _find_target(
        current_probe: FuelProbe,
    ) -> ContainerStateDict | None:
        _ = current_probe
        fuel_target = make_container_state(101, 100, True, 300)
        current_probe.get_world_state()["containers"][coord_key(101, 100)] = fuel_target
        return fuel_target

    def _requires_reposition(
        current_probe: FuelProbe,
        current_target: ContainerStateDict,
    ) -> bool:
        _ = (current_probe, current_target)
        return True

    def _find_landing(
        current_probe: FuelProbe,
        current_target: ContainerStateDict,
    ) -> tuple[int, int] | None:
        _ = (current_probe, current_target)
        return (102, 100)

    def _pickup_outcome(
        page: action_session.WaitPageProtocol,
        probe: FuelProbe,
        *,
        target_x: int,
        target_y: int,
        pickup_started_ms: int,
        fuel_before: int,
        timeout_ms: int,
    ) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
        _ = (
            page,
            probe,
            target_x,
            target_y,
            pickup_started_ms,
            fuel_before,
            timeout_ms,
        )
        return ("picked_up_fuel", 2000, 900)

    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome
    fuel_probe_module._find_visible_fuel_target = _find_target
    fuel_probe_module._visible_fuel_requires_reposition = _requires_reposition
    fuel_probe_module._find_visible_fuel_landing_tile = _find_landing
    fuel_probe_module._wait_for_pickup_outcome = _pickup_outcome

    result = probe._probe_single_fuel_target(
        target=target,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=0,
    )

    assert result["status"] == "picked_up_fuel"
    assert result["reposition_map_open_started_ms"] == 1000
    assert result["reposition_map_sync_timestamp_ms"] is None
    assert result["reposition_teleport_started_ms"] == 1000
    assert result["landed_x"] == 102
    assert result["landed_y"] == 100
    assert result["pickup_started_ms"] == 1000
    assert probe.move_calls == [(101, 100)]


def test_probe_single_target_skips_move_when_pickup_already_completed() -> None:
    """Single-target probe does not enqueue move after an immediate fuel pickup."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    wait_results = [1200, 1600]

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return wait_results.pop(0)

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
            status="landed_exact",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=640,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    drain_calls = 0

    def _pickup_before_move(provider: BufferedMessageSourceProtocol) -> int:
        nonlocal drain_calls
        _ = provider
        drain_calls += 1
        if drain_calls < 2:
            return 0
        probe.get_world_state()["self_state"] = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        probe.get_world_state()["containers"].pop(coord_key(101, 100), None)
        return 1

    fuel_target = make_container_state(101, 100, True, 300)
    probe.get_world_state()["containers"][coord_key(101, 100)] = fuel_target
    fuel_probe_module.get_terrain_map = ground_terrain
    fuel_targeting_module.get_terrain_map = ground_terrain
    action_hooks.wait_for_world_sync = _wait_for_world_sync
    action_hooks.wait_for_radar_sync = _wait_for_world_sync
    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome
    action_hooks.drain_buffered_messages = _pickup_before_move

    result = probe._probe_single_fuel_target(
        target=target,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=0,
    )

    assert result["status"] == "picked_up_fuel"
    assert result["fuel_after"] == 900
    assert probe.move_calls == []


def test_probe_single_target_raises_when_dispatch_fails() -> None:
    """Single-target probe raises on command dispatch failures."""
    from tankpit_bot.sniffer.world_state import register_room_image, set_selected_room

    original_path_exists = core_hooks.path_exists
    original_load_terrain_map = core_hooks.load_terrain_map
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    try:
        register_room_image("1", "field01.gif")
        set_selected_room("1")
        core_hooks.path_exists = lambda path: True
        core_hooks.load_terrain_map = lambda path: InMemoryTerrainMap()

        probe = _ProbeHarness(clock)
        probe.map_open_result = False
        with pytest.raises(FuelProbeError, match="map_open command dispatch failed"):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
            )

        action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200
        action_hooks.wait_for_radar_sync = lambda page, provider, started_ms, timeout_ms: 1200
        probe = _ProbeHarness(clock)
        probe.teleport_result = False
        with pytest.raises(FuelProbeError, match="teleport command dispatch failed"):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
            )

        probe = _ProbeHarness(clock)

        def _landed_teleport_outcome(
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
                [
                    Literal[
                        "before_map_open",
                        "before_teleport",
                        "after_map_data",
                        "landed",
                        "timeout",
                    ]
                ],
                TeleportPageSnapshotDict,
            ],
        ) -> TeleportAttemptResultDict:
            _ = (
                page,
                provider,
                message_start_index,
                timeout_ms,
                page_snapshots,
                capture_page_snapshot,
            )
            return TeleportAttemptResultDict(
                target=target,
                teleport_cycle_id=teleport_cycle_id,
                status="landed_exact",
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                completion_timestamp_ms=1500,
                map_sync_elapsed_ms=200,
                teleport_elapsed_ms=200,
                fuel_before=fuel_before,
                fuel_after=650,
                world_timestamp_before=world_timestamp_before,
                world_timestamp_after=1450,
                landed_signal_received=True,
                landed_x=124,
                landed_y=100,
                message_start_index=0,
                message_end_index=0,
                page_snapshots=[],
            )

        fuel_probe_module._wait_for_teleport_outcome = _landed_teleport_outcome
        probe.radar_result = False
        with pytest.raises(FuelProbeError, match="radar command dispatch failed"):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
            )

        probe = _ProbeHarness(clock)

        def _find_target(
            current_probe: FuelProbe,
        ) -> ContainerStateDict | None:
            _ = current_probe
            fuel_target = make_container_state(101, 100, True, 300)
            current_probe.get_world_state()["containers"][coord_key(101, 100)] = fuel_target
            return fuel_target

        fuel_probe_module._find_visible_fuel_target = _find_target
        probe.move_result = False
        with pytest.raises(
            FuelProbeError,
            match="move_to command dispatch failed during fuel collection",
        ):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
            )
    finally:
        core_hooks.path_exists = original_path_exists
        core_hooks.load_terrain_map = original_load_terrain_map
