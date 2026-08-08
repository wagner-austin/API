"""Tests for the fuel probe's standalone helpers.

Visible-target selection, entry formatting, pickup-outcome waiting,
and the pickup-attempt error conversion.
"""

from __future__ import annotations

from typing import (
    Literal,
)

import pytest
from tests.action_lab._fuel_probe_harness import (
    _make_world,
    _ProbeHarness,
    _snapshot,
    _terrain,
    fuel_probe_module,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot._test_hooks import (
    BufferedMessageSourceProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.fuel_probe import FuelProbe
from tankpit_bot.action_lab.fuel_probe_targets import (
    FuelProbeError,
    _find_visible_fuel_target,
    _format_visible_fuel_entries,
    _get_completed_pickup_outcome,
    _wait_for_pickup_outcome,
)
from tankpit_bot.action_lab.pickup_phase import (
    PickupImmediateOutcomeProtocol,
    PickupOutcomeWaiterProtocol,
    PickupPhaseError,
    PickupTimeoutSizerProtocol,
    effective_pickup_timeout_ms,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    coord_key,
    make_container_state,
    make_self_state,
)


def test_effective_pickup_timeout_scales_with_distance() -> None:
    """Pickup timeout grows with travel distance and never shrinks below base."""
    assert (
        effective_pickup_timeout_ms(
            current_x=100,
            current_y=100,
            target_x=101,
            target_y=100,
            base_timeout_ms=3000,
        )
        == 3000
    )
    assert (
        effective_pickup_timeout_ms(
            current_x=162,
            current_y=94,
            target_x=160,
            target_y=86,
            base_timeout_ms=3000,
        )
        == 6000
    )


def test_find_visible_fuel_target_returns_best_visible_container() -> None:
    """Fuel target selection chooses the visible high-volume fuel container."""
    probe = _ProbeHarness(ReplayClock(1000))
    world = probe.get_world_state()
    world["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=world["timestamp_ms"],
    )
    world["containers"][coord_key(102, 100)] = make_container_state(
        102,
        100,
        True,
        500,
        timestamp_ms=world["timestamp_ms"],
    )
    probe.world.terrain_map = _terrain({(100, 100), (101, 100), (102, 100)})

    fuel_target = _find_visible_fuel_target(probe)

    assert fuel_target == world["containers"][coord_key(102, 100)]


def test_format_visible_fuel_entries_returns_unavailable_without_terrain() -> None:
    """Visible-fuel diagnostics report unavailable without a terrain map."""
    probe = _ProbeHarness(ReplayClock(1000))
    probe.world.terrain_map = None

    summary = _format_visible_fuel_entries(probe, fuel_target=None)

    assert summary == "unavailable"


def test_format_visible_fuel_entries_returns_unavailable_without_self_state() -> None:
    """Visible-fuel diagnostics report unavailable without self state."""
    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state["self_state"] = None
    probe.world.terrain_map = _terrain({(100, 100)})

    summary = _format_visible_fuel_entries(probe, fuel_target=None)

    assert summary == "unavailable"


def test_format_visible_fuel_entries_returns_none_when_no_visible_fuel_is_tracked() -> None:
    """Visible-fuel diagnostics exclude non-fuel and out-of-viewport containers."""
    probe = _ProbeHarness(ReplayClock(1000))
    world = probe.get_world_state()
    world["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        False,
        300,
        timestamp_ms=world["timestamp_ms"],
    )
    world["containers"][coord_key(200, 200)] = make_container_state(
        200,
        200,
        True,
        300,
        timestamp_ms=world["timestamp_ms"],
    )
    probe.world.terrain_map = _terrain({(100, 100), (101, 100), (200, 200)})

    summary = _format_visible_fuel_entries(probe, fuel_target=None)

    assert summary == "none"


def test_format_visible_fuel_entries_marks_selected_and_truncates() -> None:
    """Visible-fuel diagnostics stamp the selected target and truncate at 8 entries.

    The 30 s stale TTL was removed 2026-07-06 -- every viewport
    container is pursuable regardless of age -- so this test now
    covers the ``selected=True`` stamp and the 8-entry truncation
    behavior of ``_format_visible_fuel_entries``.
    """
    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state = _make_world(40001, 100, 100, 700)
    world = probe.get_world_state()
    passable_tiles = {(100, 100)}
    selected_target = make_container_state(101, 100, True, 300, timestamp_ms=0)
    world["containers"][coord_key(101, 100)] = selected_target
    passable_tiles.add((101, 100))
    extra_positions = [
        (102, 100),
        (103, 100),
        (104, 100),
        (105, 100),
        (106, 100),
        (107, 100),
        (101, 101),
        (102, 101),
    ]
    for x, y in extra_positions:
        world["containers"][coord_key(x, y)] = make_container_state(
            x,
            y,
            True,
            300,
            timestamp_ms=0,
        )
        passable_tiles.add((x, y))
    probe.world.terrain_map = _terrain(passable_tiles)

    summary = _format_visible_fuel_entries(probe, fuel_target=selected_target)

    assert "reason=actionable actionable=True selected=True" in summary
    assert "...+1 more" in summary


def test_find_visible_fuel_target_requires_terrain_and_self_state() -> None:
    """Fuel target selection raises when required state is missing."""
    probe = _ProbeHarness(ReplayClock(1000))
    probe.world.terrain_map = None
    with pytest.raises(FuelProbeError, match="terrain map is unavailable"):
        _find_visible_fuel_target(probe)

    probe.world.terrain_map = _terrain({(100, 100)})
    probe._world_state["self_state"] = None
    with pytest.raises(FuelProbeError, match="self state is unavailable"):
        _find_visible_fuel_target(probe)


def test_wait_for_pickup_outcome_detects_fuel_gain_and_disappearance() -> None:
    """Pickup wait succeeds on fuel gain or container disappearance."""
    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    page = probe._fake_page
    worlds = [_make_world(1000, 100, 100, 300), _make_world(1100, 100, 100, 450)]
    for world in worlds:
        world["containers"][coord_key(101, 100)] = make_container_state(
            101,
            100,
            True,
            300,
            timestamp_ms=world["timestamp_ms"],
        )
    probe._world_state = worlds[0]

    def _advance() -> None:
        if len(worlds) > 1:
            worlds.pop(0)
        probe._world_state = worlds[0]

    page.on_wait = _advance
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda provider, ws: 0

    status, completed_ms, fuel_after = _wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=300,
        timeout_ms=1000,
    )

    assert (status, completed_ms, fuel_after) == ("picked_up_fuel", 1100, 450)

    probe._world_state = _make_world(1000, 100, 100, 700)
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )

    def _remove_container(provider: BufferedMessageSourceProtocol, ws: WorldService) -> int:
        _ = provider
        probe.get_world_state()["containers"].pop(coord_key(101, 100), None)
        return 1

    action_hooks.drain_buffered_messages = _remove_container

    disappeared = _wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=700,
        timeout_ms=1000,
    )

    assert disappeared == ("pickup_timeout", 2000, 450)


def test_wait_for_pickup_outcome_times_out_and_handles_missing_self_state() -> None:
    """Pickup wait handles timeout and missing-self-state failures."""
    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    page = probe._fake_page
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda provider, ws: 0

    timed_out = _wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=700,
        timeout_ms=150,
    )

    assert timed_out == ("pickup_timeout", 1200, 700)

    def _clear_self(provider: BufferedMessageSourceProtocol, ws: WorldService) -> int:
        _ = provider
        probe.get_world_state()["self_state"] = None
        return 1

    probe = _ProbeHarness(clock)
    action_hooks.drain_buffered_messages = _clear_self
    with pytest.raises(FuelProbeError, match="self state disappeared while waiting"):
        _wait_for_pickup_outcome(
            page,
            probe,
            target_x=101,
            target_y=100,
            pickup_started_ms=1000,
            fuel_before=700,
            timeout_ms=1000,
        )

    probe = _ProbeHarness(clock)
    probe._world_state["self_state"] = None
    action_hooks.drain_buffered_messages = lambda provider, ws: 0
    with pytest.raises(FuelProbeError, match="self state disappeared after fuel pickup timeout"):
        _wait_for_pickup_outcome(
            page,
            probe,
            target_x=101,
            target_y=100,
            pickup_started_ms=1000,
            fuel_before=700,
            timeout_ms=0,
        )


def test_get_completed_pickup_outcome_detects_pickup_and_missing_self_state() -> None:
    """Immediate pickup helper detects queued pickup events and validates self state."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )
    probe._world_state["self_state"] = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=850,
        leaderboard_position=1,
    )

    completed = _get_completed_pickup_outcome(
        probe,
        target_x=101,
        target_y=100,
        fuel_before=700,
    )

    assert completed == ("picked_up_fuel", 1000, 850)

    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )
    probe._world_state["self_state"] = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=700,
        leaderboard_position=1,
    )
    probe._world_state["containers"].pop(coord_key(101, 100), None)

    assert (
        _get_completed_pickup_outcome(
            probe,
            target_x=101,
            target_y=100,
            fuel_before=700,
        )
        is None
    )

    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state["self_state"] = None

    with pytest.raises(FuelProbeError, match="self state disappeared while waiting"):
        _get_completed_pickup_outcome(
            probe,
            target_x=101,
            target_y=100,
            fuel_before=700,
        )


def test_run_pickup_attempt_converts_pickup_phase_error() -> None:
    """Fuel pickup wrapper converts shared pickup-phase failures."""
    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    page = ClockAdvancingPage(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    fuel_target = make_container_state(101, 100, True, 300)
    original_run_pickup = fuel_probe_module.run_tracked_pickup_phase

    def _raise_pickup_phase_error(
        page: action_session.WaitPageProtocol,
        probe: FuelProbe,
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
            page,
            probe,
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
        raise PickupPhaseError("shared pickup failure")

    fuel_probe_module.run_tracked_pickup_phase = _raise_pickup_phase_error
    try:
        with pytest.raises(FuelProbeError, match="shared pickup failure"):
            probe._run_pickup_attempt(
                page=page,
                target=target,
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1200,
                teleport_started_ms=1300,
                radar_started_ms=1600,
                radar_sync_timestamp_ms=1700,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_timeout_ms=3000,
                fuel_before=900,
                teleport_result=TeleportAttemptResultDict(
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
                ),
                fuel_target=fuel_target,
                message_start_index=0,
                teleport_cycle_ids=[1],
                radar_cycle_id=2,
                decision_basis=None,
                snapshot_before=_snapshot(1000),
                capture_snapshot=lambda: _snapshot(1900),
            )
    finally:
        fuel_probe_module.run_tracked_pickup_phase = original_run_pickup
