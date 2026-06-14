"""Tests for equipment probe TypedDict codecs."""

from __future__ import annotations

from typing import Literal

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict
from tankpit_bot.action_lab.equipment_probe_types import (
    EquipmentProbeAttemptResultDict,
    EquipmentProbeSessionDict,
    decode_equipment_probe_attempt_result,
    decode_equipment_probe_session,
    encode_equipment_probe_attempt_result,
    encode_equipment_probe_session,
)
from tankpit_bot.action_lab.types import TeleportStartupTimingDict, TeleportTargetDict


def _target() -> TeleportTargetDict:
    """Build a sample teleport target."""
    return TeleportTargetDict(label="equip_ground_50_60", x=50, y=60)


def _startup_timing() -> TeleportStartupTimingDict:
    """Build a sample startup timing payload."""
    return TeleportStartupTimingDict(
        game_ready_timestamp_ms=300,
        intel_ready_timestamp_ms=350,
        initial_sync_started_ms=400,
        initial_world_timestamp_ms=450,
        command_ready_timestamp_ms=460,
        first_attempt_started_ms=500,
        game_ready_to_intel_ready_ms=50,
        intel_ready_to_initial_world_ms=100,
        initial_world_to_command_ready_ms=10,
        command_ready_to_first_attempt_ms=40,
    )


def _attempt(
    status: Literal[
        "picked_up_equipment",
        "no_equipment_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ] = "picked_up_equipment",
) -> EquipmentProbeAttemptResultDict:
    """Build a sample equipment probe attempt."""
    return EquipmentProbeAttemptResultDict(
        target=_target(),
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        move_cycle_id=3,
        pickup_cycle_id=4,
        status=status,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        radar_started_ms=1500,
        radar_sync_timestamp_ms=1600,
        reposition_map_open_started_ms=None,
        reposition_map_sync_timestamp_ms=None,
        reposition_teleport_started_ms=None,
        pickup_started_ms=1700,
        completion_timestamp_ms=1900,
        inventory_count_before=3,
        inventory_count_after=4,
        landed_signal_received=True,
        landed_x=50,
        landed_y=60,
        equipment_target_x=51,
        equipment_target_y=60,
        phase_overlaps=[],
        message_start_index=4,
        message_end_index=9,
    )


def _session() -> EquipmentProbeSessionDict:
    """Build a sample equipment probe session."""
    return EquipmentProbeSessionDict(
        session_id="equipment-session",
        start_timestamp_ms=100,
        end_timestamp_ms=1000,
        base_url="https://tankpit.com/play",
        spawn_x=131,
        spawn_y=126,
        target_pickups=3,
        max_attempts=3,
        capture_session_path="equipment_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing=_startup_timing(),
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
        attempts=[_attempt()],
    )


def test_equipment_probe_attempt_round_trip() -> None:
    """Equipment probe attempts encode and decode cleanly."""
    encoded = encode_equipment_probe_attempt_result(_attempt())
    decoded = decode_equipment_probe_attempt_result(encoded)
    assert decoded == _attempt()


def test_equipment_probe_attempt_round_trip_with_null_fields() -> None:
    """Optional attempt fields preserve null values through codecs."""
    attempt = EquipmentProbeAttemptResultDict(
        target=_target(),
        teleport_cycle_ids=[1],
        radar_cycle_id=None,
        move_cycle_id=None,
        pickup_cycle_id=None,
        status="map_sync_timeout",
        map_open_started_ms=1000,
        map_sync_timestamp_ms=None,
        teleport_started_ms=None,
        radar_started_ms=None,
        radar_sync_timestamp_ms=None,
        reposition_map_open_started_ms=None,
        reposition_map_sync_timestamp_ms=None,
        reposition_teleport_started_ms=None,
        pickup_started_ms=None,
        completion_timestamp_ms=1600,
        inventory_count_before=3,
        inventory_count_after=None,
        landed_signal_received=False,
        landed_x=None,
        landed_y=None,
        equipment_target_x=None,
        equipment_target_y=None,
        phase_overlaps=[],
        message_start_index=4,
        message_end_index=5,
    )

    decoded = decode_equipment_probe_attempt_result(encode_equipment_probe_attempt_result(attempt))
    assert decoded == attempt


def test_equipment_probe_attempt_round_trip_with_phase_overlap() -> None:
    """Equipment attempt codecs preserve non-empty phase overlap diagnostics."""
    attempt = _attempt()
    attempt["phase_overlaps"] = [
        ActionPhaseOverlapDict(
            active_phase="radar",
            active_cycle_id=2,
            active_started_ms=1500,
            next_phase="move",
            next_cycle_id=3,
            next_started_ms=1700,
        )
    ]

    decoded = decode_equipment_probe_attempt_result(encode_equipment_probe_attempt_result(attempt))

    assert decoded == attempt


@pytest.mark.parametrize(
    "status",
    [
        "picked_up_equipment",
        "no_equipment_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ],
)
def test_equipment_probe_attempt_round_trip_for_all_statuses(
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
) -> None:
    """Equipment attempt codecs preserve all 8 status literals."""
    attempt = _attempt(status=status)

    decoded = decode_equipment_probe_attempt_result(encode_equipment_probe_attempt_result(attempt))

    assert decoded == attempt


def test_equipment_probe_attempt_rejects_invalid_status() -> None:
    """Equipment attempt decode rejects unsupported status values."""
    encoded = encode_equipment_probe_attempt_result(_attempt())
    encoded["status"] = "bad"

    with pytest.raises(JSONTypeError, match="invalid equipment probe status"):
        decode_equipment_probe_attempt_result(encoded)


def test_equipment_probe_attempt_rejects_invalid_optional_int() -> None:
    """Equipment attempt decode rejects malformed optional integer fields."""
    encoded = encode_equipment_probe_attempt_result(_attempt())
    encoded["inventory_count_after"] = "bad"

    with pytest.raises(JSONTypeError, match="inventory_count_after"):
        decode_equipment_probe_attempt_result(encoded)


def test_equipment_probe_attempt_rejects_boolean_as_optional_int() -> None:
    """Equipment attempt decode rejects booleans in optional integer fields."""
    encoded = encode_equipment_probe_attempt_result(_attempt())
    encoded["radar_cycle_id"] = True

    with pytest.raises(JSONTypeError, match="radar_cycle_id"):
        decode_equipment_probe_attempt_result(encoded)


def test_equipment_probe_attempt_rejects_non_boolean_landed_signal() -> None:
    """Equipment attempt decode rejects non-boolean landed signal values."""
    encoded = encode_equipment_probe_attempt_result(_attempt())
    encoded["landed_signal_received"] = "bad"

    with pytest.raises(JSONTypeError, match="landed_signal_received"):
        decode_equipment_probe_attempt_result(encoded)


def test_equipment_probe_attempt_rejects_non_object_target() -> None:
    """Equipment attempt decode rejects malformed target payloads."""
    encoded = encode_equipment_probe_attempt_result(_attempt())
    encoded["target"] = "bad"

    with pytest.raises(JSONTypeError, match="target"):
        decode_equipment_probe_attempt_result(encoded)


def test_equipment_probe_attempt_rejects_non_object_phase_overlap() -> None:
    """Equipment attempt decode rejects malformed phase overlap payloads."""
    encoded = encode_equipment_probe_attempt_result(_attempt())
    encoded["phase_overlaps"] = ["bad"]

    with pytest.raises(JSONTypeError, match="phase_overlaps"):
        decode_equipment_probe_attempt_result(encoded)


def test_equipment_probe_attempt_rejects_non_integer_teleport_cycle_id() -> None:
    """Equipment attempt decode rejects malformed teleport cycle ids."""
    encoded = encode_equipment_probe_attempt_result(_attempt())
    encoded["teleport_cycle_ids"] = [True]

    with pytest.raises(JSONTypeError, match="teleport_cycle_ids"):
        decode_equipment_probe_attempt_result(encoded)


def test_equipment_probe_attempt_rejects_non_integer_string_in_cycle_ids() -> None:
    """Equipment attempt decode rejects string elements in cycle id list."""
    encoded = encode_equipment_probe_attempt_result(_attempt())
    encoded["teleport_cycle_ids"] = ["bad"]

    with pytest.raises(JSONTypeError, match="teleport_cycle_ids"):
        decode_equipment_probe_attempt_result(encoded)


def test_equipment_probe_session_round_trip() -> None:
    """Equipment probe sessions encode and decode cleanly."""
    encoded = encode_equipment_probe_session(_session())
    decoded = decode_equipment_probe_session(encoded)
    assert decoded == _session()


def test_equipment_probe_session_rejects_non_object_attempt() -> None:
    """Equipment session decode rejects non-object attempts."""
    encoded = encode_equipment_probe_session(_session())
    encoded["attempts"] = ["bad"]

    with pytest.raises(JSONTypeError, match="attempts"):
        decode_equipment_probe_session(encoded)


def test_equipment_probe_session_rejects_non_object_startup_timing() -> None:
    """Equipment session decode rejects non-object startup timing values."""
    encoded = encode_equipment_probe_session(_session())
    encoded["startup_timing"] = "bad"

    with pytest.raises(JSONTypeError, match="startup_timing"):
        decode_equipment_probe_session(encoded)
