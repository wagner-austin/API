"""Tests for equipment probe diagnostics summary formatter."""

from __future__ import annotations

from typing import Literal

import pytest

from tankpit_bot.action_lab.equipment_probe import EquipmentProbe
from tankpit_bot.action_lab.equipment_probe_diagnostics import (
    format_equipment_probe_summary,
)
from tankpit_bot.action_lab.equipment_probe_runner import (
    execute_equipment_probe_session,
)
from tankpit_bot.action_lab.equipment_probe_types import (
    EquipmentProbeAttemptResultDict,
    EquipmentProbeSessionDict,
)
from tankpit_bot.action_lab.types import (
    TeleportStartupTimingDict,
    TeleportTargetDict,
)


def test_format_summary_with_no_attempts() -> None:
    """An empty session produces a count-only summary."""
    session = _make_session([])
    result = format_equipment_probe_summary(session)
    assert "0 attempts" in result


def _make_session_timing() -> TeleportStartupTimingDict:
    """Build a minimal startup timing dict for tests."""
    return TeleportStartupTimingDict(
        game_ready_timestamp_ms=100,
        intel_ready_timestamp_ms=200,
        initial_sync_started_ms=150,
        initial_world_timestamp_ms=250,
        command_ready_timestamp_ms=300,
        first_attempt_started_ms=350,
        game_ready_to_intel_ready_ms=100,
        intel_ready_to_initial_world_ms=50,
        initial_world_to_command_ready_ms=50,
        command_ready_to_first_attempt_ms=50,
    )


def _make_attempt(
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
) -> EquipmentProbeAttemptResultDict:
    """Build a minimal attempt with the given status."""
    return EquipmentProbeAttemptResultDict(
        target=TeleportTargetDict(label="test", x=50, y=50),
        teleport_cycle_ids=[],
        radar_cycle_id=None,
        move_cycle_id=None,
        pickup_cycle_id=None,
        status=status,
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


def _make_session(
    attempts: list[EquipmentProbeAttemptResultDict],
) -> EquipmentProbeSessionDict:
    """Build a session with the given attempts."""
    return EquipmentProbeSessionDict(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        target_pickups=3,
        max_attempts=5,
        capture_session_path="test.json",
        initial_sync_timeout_ms=30000,
        startup_timing=_make_session_timing(),
        map_sync_timeout_ms=30000,
        teleport_timeout_ms=30000,
        radar_timeout_ms=15000,
        pickup_timeout_ms=10000,
        settle_delay_ms=500,
        attempts=attempts,
    )


def test_format_summary_with_picked_up_attempt() -> None:
    """A single picked-up attempt appears in the summary."""
    session = _make_session([_make_attempt("picked_up_equipment")])
    result = format_equipment_probe_summary(session)
    assert "1 attempts" in result
    assert "1 picked up" in result


def test_format_summary_with_no_equipment_attempt() -> None:
    """A single no-equipment attempt appears in the summary."""
    session = _make_session([_make_attempt("no_equipment_visible")])
    result = format_equipment_probe_summary(session)
    assert "1 attempts" in result
    assert "1 no equipment" in result


def test_format_summary_with_radar_timeout_attempt() -> None:
    """A single radar-timeout attempt appears in the summary."""
    session = _make_session([_make_attempt("radar_timeout")])
    result = format_equipment_probe_summary(session)
    assert "1 attempts" in result
    assert "1 radar timeout" in result


def test_format_summary_with_teleport_timeout_attempt() -> None:
    """A single teleport-timeout attempt appears in the summary."""
    session = _make_session([_make_attempt("teleport_timeout")])
    result = format_equipment_probe_summary(session)
    assert "1 attempts" in result
    assert "1 teleport timeout" in result


def test_format_summary_with_map_sync_timeout_attempt() -> None:
    """map_sync_timeout is counted with teleport timeouts."""
    session = _make_session([_make_attempt("map_sync_timeout")])
    result = format_equipment_probe_summary(session)
    assert "1 attempts" in result
    assert "1 teleport timeout" in result


def test_format_summary_with_mixed_attempts() -> None:
    """Multiple statuses produce a combined summary."""
    session = _make_session(
        [
            _make_attempt("picked_up_equipment"),
            _make_attempt("picked_up_equipment"),
            _make_attempt("no_equipment_visible"),
            _make_attempt("radar_timeout"),
            _make_attempt("teleport_timeout"),
            _make_attempt("map_sync_timeout"),
        ]
    )
    result = format_equipment_probe_summary(session)
    assert "6 attempts" in result
    assert "2 picked up" in result
    assert "1 no equipment" in result
    assert "1 radar timeout" in result
    assert "2 teleport timeout" in result


def test_format_summary_skips_unknown_status() -> None:
    """An attempt with an unrecognized status is counted but not categorized."""
    session = _make_session([_make_attempt("pickup_timeout")])
    result = format_equipment_probe_summary(session)
    assert "1 attempts" in result
    assert "picked up" not in result
    assert "no equipment" not in result
    assert "radar timeout" not in result
    assert "teleport timeout" not in result


def test_execute_equipment_probe_session_raises() -> None:
    """The stub runner raises NotImplementedError."""
    probe = EquipmentProbe.__new__(EquipmentProbe)
    with pytest.raises(NotImplementedError):
        execute_equipment_probe_session(
            probe,
            max_targets=3,
            initial_sync_timeout_ms=30000,
            map_sync_timeout_ms=30000,
            teleport_timeout_ms=30000,
            radar_timeout_ms=15000,
            pickup_timeout_ms=10000,
            settle_delay_ms=500,
        )
