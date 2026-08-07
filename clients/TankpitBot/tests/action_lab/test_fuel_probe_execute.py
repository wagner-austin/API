"""Tests for ``execute_probe`` and the ``run_fuel_probe`` entry point.

Limit validation, attempt collection, the two continue-until loops, and
the session-json write.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)
from tests.action_lab._fuel_probe_harness import (
    _FUEL_CAPTURE_PATH,
    _ExecuteHarness,
    _FakeFuelProbe,
    _ProbeHarness,
    _snapshot,
    _terrain,
    fuel_probe_module,
    fuel_targets_module,
)
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_page import ReplayClock
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.fuel_probe import run_fuel_probe
from tankpit_bot.action_lab.fuel_probe_targets import FuelProbeError
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    decode_fuel_probe_session,
)
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import (
    make_self_state,
)
from tankpit_bot.types import (
    decode_capture_session,
)


def test_execute_probe_raises_for_invalid_limits_and_missing_playwright() -> None:
    """Fuel probe execute validates pickup limits and Playwright presence."""
    probe = _ProbeHarness(ReplayClock(1000))
    with pytest.raises(ValueError, match="target_pickups must be positive"):
        probe.execute_probe(
            target_pickups=0,
            max_attempts=1,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )

    with pytest.raises(ValueError, match="max_attempts must be positive"):
        probe.execute_probe(
            target_pickups=1,
            max_attempts=0,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )

    with pytest.raises(ValueError, match="max_attempts must be at least target_pickups"):
        probe.execute_probe(
            target_pickups=2,
            max_attempts=1,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )

    original_sync_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            probe.execute_probe(
                target_pickups=1,
                max_attempts=1,
                initial_sync_timeout_ms=10000,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_sync_playwright


def test_execute_probe_collects_attempts_and_requires_terrain() -> None:
    """Fuel probe execute collects attempts and rejects missing terrain."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    session_browser = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = session_browser.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = lambda page, provider, started_ms, timeout_ms: (
        1200,
        make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
    )
    probe.results = [
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_124_100", "x": 124, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="picked_up_fuel",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=1500,
            completion_timestamp_ms=1600,
            fuel_before=700,
            fuel_after=900,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            fuel_target_x=125,
            fuel_target_y=100,
            fuel_target_volume=300,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=0,
            message_end_index=1,
        )
    ]
    fuel_probe_module.get_terrain_map = lambda: _terrain(
        {
            (115, 99),
            (115, 100),
            (115, 101),
            (116, 99),
            (116, 100),
            (116, 101),
            (117, 99),
            (117, 100),
            (117, 101),
        }
    )
    fuel_targets_module.get_terrain_map = fuel_probe_module.get_terrain_map

    session = probe.execute_probe(
        target_pickups=1,
        max_attempts=1,
        initial_sync_timeout_ms=10000,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 1
    assert session["spawn_x"] == 100
    assert session["target_pickups"] == 1
    assert session["startup_timing"]["initial_world_timestamp_ms"] == 1200
    assert session_browser.browser_type.launches == [False]

    fuel_probe_module.get_terrain_map = lambda: None
    fuel_targets_module.get_terrain_map = fuel_probe_module.get_terrain_map
    with pytest.raises(FuelProbeError, match="terrain map is unavailable"):
        probe.execute_probe(
            target_pickups=1,
            max_attempts=1,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )


def test_execute_probe_continues_after_pickup_until_target_pickups_reached() -> None:
    """Fuel probe execute keeps probing after a pickup until target pickups are met."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    session_browser = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = session_browser.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = lambda page, provider, started_ms, timeout_ms: (
        1200,
        make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
    )
    probe.results = [
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_116_100", "x": 116, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="picked_up_fuel",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=1500,
            completion_timestamp_ms=1600,
            fuel_before=700,
            fuel_after=850,
            landed_signal_received=True,
            landed_x=116,
            landed_y=100,
            fuel_target_x=117,
            fuel_target_y=100,
            fuel_target_volume=150,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=0,
            message_end_index=1,
        ),
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_117_100", "x": 117, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="picked_up_fuel",
            map_open_started_ms=1700,
            map_sync_timestamp_ms=1800,
            teleport_started_ms=1900,
            radar_started_ms=2000,
            radar_sync_timestamp_ms=2100,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=2200,
            completion_timestamp_ms=2300,
            fuel_before=850,
            fuel_after=1000,
            landed_signal_received=True,
            landed_x=117,
            landed_y=100,
            fuel_target_x=118,
            fuel_target_y=100,
            fuel_target_volume=150,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=2,
            message_end_index=3,
        ),
    ]
    fuel_probe_module.get_terrain_map = lambda: _terrain(
        {(x, y) for x in range(0, 201) for y in range(0, 201)}
    )
    fuel_targets_module.get_terrain_map = fuel_probe_module.get_terrain_map

    session = probe.execute_probe(
        target_pickups=2,
        max_attempts=3,
        initial_sync_timeout_ms=10000,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 2
    assert [attempt["status"] for attempt in session["attempts"]] == [
        "picked_up_fuel",
        "picked_up_fuel",
    ]
    assert session["target_pickups"] == 2


def test_execute_probe_continues_after_miss_until_pickup_succeeds() -> None:
    """Fuel probe execute keeps probing after a miss until a later pickup succeeds."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    session_browser = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = session_browser.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = lambda page, provider, started_ms, timeout_ms: (
        1200,
        make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
    )
    probe.results = [
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_116_100", "x": 116, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="no_fuel_visible",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=None,
            completion_timestamp_ms=1600,
            fuel_before=700,
            fuel_after=650,
            landed_signal_received=True,
            landed_x=116,
            landed_y=100,
            fuel_target_x=None,
            fuel_target_y=None,
            fuel_target_volume=None,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=0,
            message_end_index=1,
        ),
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_117_100", "x": 117, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="picked_up_fuel",
            map_open_started_ms=1700,
            map_sync_timestamp_ms=1800,
            teleport_started_ms=1900,
            radar_started_ms=2000,
            radar_sync_timestamp_ms=2100,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=2200,
            completion_timestamp_ms=2300,
            fuel_before=650,
            fuel_after=900,
            landed_signal_received=True,
            landed_x=117,
            landed_y=100,
            fuel_target_x=118,
            fuel_target_y=100,
            fuel_target_volume=250,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=2,
            message_end_index=3,
        ),
    ]
    fuel_probe_module.get_terrain_map = lambda: _terrain(
        {(x, y) for x in range(0, 201) for y in range(0, 201)}
    )
    fuel_targets_module.get_terrain_map = fuel_probe_module.get_terrain_map

    session = probe.execute_probe(
        target_pickups=1,
        max_attempts=3,
        initial_sync_timeout_ms=10000,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 2
    assert [attempt["status"] for attempt in session["attempts"]] == [
        "no_fuel_visible",
        "picked_up_fuel",
    ]
    assert session["target_pickups"] == 1


def test_run_fuel_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    """Fuel probe runner writes both summary JSON and raw capture output."""
    original_probe_class = fuel_probe_module.FuelProbe
    fuel_probe_module.FuelProbe = _FakeFuelProbe
    try:
        session = run_fuel_probe(
            "https://tankpit.com/play",
            "fuel_probe.json",
            target_pickups=3,
            max_attempts=3,
        )
    finally:
        fuel_probe_module.FuelProbe = original_probe_class

    written = fake_fs.read_text(Path("fuel_probe.json"))
    decoded = decode_fuel_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("fuel_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))

    assert session == decoded
    assert session["capture_session_path"] == "fuel_probe.capture_session.json"
    assert capture_decoded["session_id"] == "fuel-session"
