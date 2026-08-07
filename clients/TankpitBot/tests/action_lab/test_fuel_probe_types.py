"""Tests for fuel probe TypedDict codecs."""

from __future__ import annotations

from typing import Literal

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict, FuelDecisionBasisDict
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
    decode_fuel_probe_attempt_result,
    decode_fuel_probe_session,
    encode_fuel_probe_attempt_result,
    encode_fuel_probe_session,
)
from tankpit_bot.action_lab.types import TeleportStartupTimingDict, TeleportTargetDict
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict


def _snapshot(timestamp_ms: int) -> PageClientSnapshotDict:
    """Build a sample page-client snapshot for fuel attempt fixtures."""
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
    """Build a sample teleport target."""
    return TeleportTargetDict(label="fuel_ground_100_124", x=100, y=124)


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
        "picked_up_fuel",
        "no_fuel_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ] = "picked_up_fuel",
) -> FuelProbeAttemptResultDict:
    """Build a sample fuel probe attempt."""
    return FuelProbeAttemptResultDict(
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
        fuel_before=600,
        fuel_after=850,
        landed_signal_received=True,
        landed_x=100,
        landed_y=124,
        fuel_target_x=101,
        fuel_target_y=124,
        fuel_target_volume=400,
        phase_overlaps=[],
        decision_basis=_decision_basis(),
        message_start_index=4,
        message_end_index=9,
        snapshot_before=_snapshot(1000),
        snapshot_after=_snapshot(1900),
    )


def _decision_basis() -> FuelDecisionBasisDict:
    """Build a sample fuel decision basis."""
    return FuelDecisionBasisDict(
        world_timestamp_ms=1600,
        radar_cycle_id=2,
        viewport_left=92,
        viewport_top=116,
        self_x=100,
        self_y=124,
        selected_target_x=101,
        selected_target_y=124,
        candidates=[],
    )


def _session() -> FuelProbeSessionDict:
    """Build a sample fuel probe session."""
    return FuelProbeSessionDict(
        session_id="fuel-session",
        start_timestamp_ms=100,
        end_timestamp_ms=1000,
        base_url="https://tankpit.com/play",
        spawn_x=131,
        spawn_y=126,
        target_pickups=3,
        max_attempts=3,
        capture_session_path="fuel_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing=_startup_timing(),
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
        attempts=[_attempt()],
    )


def test_fuel_probe_attempt_round_trip() -> None:
    """Fuel probe attempts encode and decode cleanly."""
    encoded = encode_fuel_probe_attempt_result(_attempt())
    decoded = decode_fuel_probe_attempt_result(encoded)
    assert decoded == _attempt()


def test_fuel_probe_attempt_round_trip_with_null_fields() -> None:
    """Optional attempt fields preserve null values through codecs."""
    attempt = FuelProbeAttemptResultDict(
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
        fuel_before=600,
        fuel_after=600,
        landed_signal_received=False,
        landed_x=131,
        landed_y=126,
        fuel_target_x=None,
        fuel_target_y=None,
        fuel_target_volume=None,
        phase_overlaps=[],
        decision_basis=None,
        message_start_index=4,
        message_end_index=5,
        snapshot_before=_snapshot(1000),
        snapshot_after=_snapshot(1600),
    )

    decoded = decode_fuel_probe_attempt_result(encode_fuel_probe_attempt_result(attempt))
    assert decoded == attempt


def test_fuel_probe_attempt_round_trip_with_phase_overlap() -> None:
    """Fuel attempt codecs preserve non-empty phase overlap diagnostics."""
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

    decoded = decode_fuel_probe_attempt_result(encode_fuel_probe_attempt_result(attempt))

    assert decoded == attempt


@pytest.mark.parametrize(
    "status",
    [
        "no_fuel_visible",
        "radar_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ],
)
def test_fuel_probe_attempt_round_trip_for_terminal_statuses(
    status: Literal[
        "no_fuel_visible",
        "radar_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ],
) -> None:
    """Fuel attempt codecs preserve all non-map-sync terminal statuses."""
    attempt = _attempt(status=status)

    decoded = decode_fuel_probe_attempt_result(encode_fuel_probe_attempt_result(attempt))

    assert decoded == attempt


def test_fuel_probe_attempt_rejects_invalid_status() -> None:
    """Fuel attempt decode rejects unsupported status values."""
    encoded = encode_fuel_probe_attempt_result(_attempt())
    encoded["status"] = "bad"

    with pytest.raises(JSONTypeError, match="invalid fuel probe status"):
        decode_fuel_probe_attempt_result(encoded)


def test_fuel_probe_attempt_rejects_invalid_optional_int() -> None:
    """Fuel attempt decode rejects malformed optional integer fields."""
    encoded = encode_fuel_probe_attempt_result(_attempt())
    encoded["fuel_after"] = "bad"

    with pytest.raises(JSONTypeError, match="fuel_after"):
        decode_fuel_probe_attempt_result(encoded)


def test_fuel_probe_attempt_rejects_non_boolean_landed_signal() -> None:
    """Fuel attempt decode rejects non-boolean landed signal values."""
    encoded = encode_fuel_probe_attempt_result(_attempt())
    encoded["landed_signal_received"] = "bad"

    with pytest.raises(JSONTypeError, match="landed_signal_received"):
        decode_fuel_probe_attempt_result(encoded)


def test_fuel_probe_attempt_rejects_non_object_snapshot() -> None:
    """Fuel attempt decode rejects non-object snapshot values."""
    encoded = encode_fuel_probe_attempt_result(_attempt())
    encoded["snapshot_before"] = "bad"

    with pytest.raises(JSONTypeError, match="snapshot_before"):
        decode_fuel_probe_attempt_result(encoded)


def test_fuel_probe_attempt_rejects_non_object_target() -> None:
    """Fuel attempt decode rejects malformed target payloads."""
    encoded = encode_fuel_probe_attempt_result(_attempt())
    encoded["target"] = "bad"

    with pytest.raises(JSONTypeError, match="target"):
        decode_fuel_probe_attempt_result(encoded)


def test_fuel_probe_attempt_rejects_non_object_phase_overlap() -> None:
    """Fuel attempt decode rejects malformed phase overlap payloads."""
    encoded = encode_fuel_probe_attempt_result(_attempt())
    encoded["phase_overlaps"] = ["bad"]

    with pytest.raises(JSONTypeError, match="phase_overlaps"):
        decode_fuel_probe_attempt_result(encoded)


def test_fuel_probe_attempt_rejects_non_object_decision_basis() -> None:
    """Fuel attempt decode rejects malformed decision basis payloads."""
    encoded = encode_fuel_probe_attempt_result(_attempt())
    encoded["decision_basis"] = "bad"

    with pytest.raises(JSONTypeError, match="decision_basis"):
        decode_fuel_probe_attempt_result(encoded)


def test_fuel_probe_attempt_rejects_non_integer_teleport_cycle_id() -> None:
    """Fuel attempt decode rejects malformed teleport cycle ids."""
    encoded = encode_fuel_probe_attempt_result(_attempt())
    encoded["teleport_cycle_ids"] = [True]

    with pytest.raises(JSONTypeError, match="teleport_cycle_ids"):
        decode_fuel_probe_attempt_result(encoded)


def test_fuel_probe_session_round_trip() -> None:
    """Fuel probe sessions encode and decode cleanly."""
    encoded = encode_fuel_probe_session(_session())
    decoded = decode_fuel_probe_session(encoded)
    assert decoded == _session()


def test_fuel_probe_session_rejects_non_object_attempt() -> None:
    """Fuel session decode rejects non-object attempts."""
    encoded = encode_fuel_probe_session(_session())
    encoded["attempts"] = ["bad"]

    with pytest.raises(JSONTypeError, match="attempts"):
        decode_fuel_probe_session(encoded)


def test_fuel_probe_session_rejects_non_object_startup_timing() -> None:
    """Fuel session decode rejects non-object startup timing values."""
    encoded = encode_fuel_probe_session(_session())
    encoded["startup_timing"] = "bad"

    with pytest.raises(JSONTypeError, match="startup_timing"):
        decode_fuel_probe_session(encoded)
