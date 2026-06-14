"""Encode/decode contract tests for fuel-dot probe session types."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.action_lab.fuel_dot_probe_types import (
    DotContainerObservationDict,
    FuelDotAttemptResultDict,
    FuelDotProbeSessionDict,
    decode_dot_container_observation,
    decode_fuel_dot_attempt_result,
    decode_fuel_dot_probe_session,
    encode_dot_container_observation,
    encode_fuel_dot_attempt_result,
    encode_fuel_dot_probe_session,
)
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.types import TeleportStartupTimingDict


def _snapshot(timestamp_ms: int) -> PageClientSnapshotDict:
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
        map_fields={},
        world_collections={},
    )


def _observation() -> DotContainerObservationDict:
    return DotContainerObservationDict(x=120, y=110, is_fuel=True, volume=750)


def _attempt(*, with_optionals: bool) -> FuelDotAttemptResultDict:
    return FuelDotAttemptResultDict(
        status="fuel_on_dot" if with_optionals else "acquisition_timeout",
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=1100 if with_optionals else None,
        dots_in_atlas=650,
        dot_x=120 if with_optionals else None,
        dot_y=110 if with_optionals else None,
        dot_distance=30 if with_optionals else None,
        teleport_started_ms=1200 if with_optionals else None,
        radar_started_ms=1400 if with_optionals else None,
        radar_sync_timestamp_ms=1500 if with_optionals else None,
        completion_timestamp_ms=1600,
        fuel_before=900,
        fuel_after=720 if with_optionals else None,
        landed_signal_received=with_optionals,
        landed_x=120 if with_optionals else None,
        landed_y=110 if with_optionals else None,
        container_on_dot=_observation() if with_optionals else None,
        viewport_fuel_containers=[_observation()] if with_optionals else [],
        message_start_index=0,
        message_end_index=3,
        snapshot_before=_snapshot(1000),
        snapshot_after=_snapshot(1600),
    )


def _startup_timing() -> TeleportStartupTimingDict:
    return TeleportStartupTimingDict(
        game_ready_timestamp_ms=100,
        intel_ready_timestamp_ms=150,
        initial_sync_started_ms=200,
        initial_world_timestamp_ms=400,
        command_ready_timestamp_ms=450,
        first_attempt_started_ms=500,
        game_ready_to_intel_ready_ms=50,
        intel_ready_to_initial_world_ms=250,
        initial_world_to_command_ready_ms=50,
        command_ready_to_first_attempt_ms=50,
    )


def _session() -> FuelDotProbeSessionDict:
    return FuelDotProbeSessionDict(
        session_id="fuel-dot-session",
        start_timestamp_ms=10,
        end_timestamp_ms=20,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        max_dots=6,
        capture_session_path="fuel_dot_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing=_startup_timing(),
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=500,
        attempts=[_attempt(with_optionals=True), _attempt(with_optionals=False)],
    )


def _round_trip(payload: JSONObject) -> JSONObject:
    return narrow_json_to_dict(load_json_str(dump_json_str(payload)))


def test_observation_round_trip() -> None:
    """Observations survive JSON encode/decode untouched."""
    observation = _observation()
    assert (
        decode_dot_container_observation(_round_trip(encode_dot_container_observation(observation)))
        == observation
    )


def test_attempt_round_trip_with_optionals() -> None:
    """A fully populated attempt round-trips exactly."""
    attempt = _attempt(with_optionals=True)
    assert (
        decode_fuel_dot_attempt_result(_round_trip(encode_fuel_dot_attempt_result(attempt)))
        == attempt
    )


def test_attempt_round_trip_with_nulls() -> None:
    """A terminal attempt full of nulls round-trips exactly."""
    attempt = _attempt(with_optionals=False)
    assert (
        decode_fuel_dot_attempt_result(_round_trip(encode_fuel_dot_attempt_result(attempt)))
        == attempt
    )


def test_session_round_trip() -> None:
    """A complete session round-trips exactly."""
    session = _session()
    assert (
        decode_fuel_dot_probe_session(_round_trip(encode_fuel_dot_probe_session(session)))
        == session
    )


@pytest.mark.parametrize(
    "status",
    [
        "fuel_on_dot",
        "equipment_on_dot",
        "empty_dot",
        "acquisition_timeout",
        "teleport_timeout",
        "radar_timeout",
    ],
)
def test_decode_attempt_accepts_every_status(status: str) -> None:
    """Each supported status literal decodes to itself."""
    encoded = encode_fuel_dot_attempt_result(_attempt(with_optionals=True))
    encoded["status"] = status
    assert decode_fuel_dot_attempt_result(encoded)["status"] == status


def test_decode_attempt_rejects_unknown_status() -> None:
    """An unsupported status literal is a hard decode error."""
    encoded = encode_fuel_dot_attempt_result(_attempt(with_optionals=True))
    encoded["status"] = "landed_on_the_moon"
    with pytest.raises(JSONTypeError, match="invalid fuel-dot status"):
        decode_fuel_dot_attempt_result(encoded)


def test_decode_attempt_rejects_non_integer_optional() -> None:
    """A non-integer optional field is a hard decode error."""
    encoded = encode_fuel_dot_attempt_result(_attempt(with_optionals=True))
    encoded["dot_x"] = "120"
    with pytest.raises(JSONTypeError, match="'dot_x' must be an integer or null"):
        decode_fuel_dot_attempt_result(encoded)


def test_decode_attempt_rejects_non_boolean_landed_signal() -> None:
    """A non-boolean landed flag is a hard decode error."""
    encoded = encode_fuel_dot_attempt_result(_attempt(with_optionals=True))
    encoded["landed_signal_received"] = 1
    with pytest.raises(JSONTypeError, match="'landed_signal_received' must be a boolean"):
        decode_fuel_dot_attempt_result(encoded)


def test_decode_attempt_rejects_non_object_observation() -> None:
    """A non-object dot observation is a hard decode error."""
    encoded = encode_fuel_dot_attempt_result(_attempt(with_optionals=True))
    encoded["container_on_dot"] = [1, 2]
    with pytest.raises(JSONTypeError, match="'container_on_dot' must be an object or null"):
        decode_fuel_dot_attempt_result(encoded)


def test_decode_attempt_rejects_non_object_viewport_entry() -> None:
    """A non-object viewport container entry is a hard decode error."""
    encoded = encode_fuel_dot_attempt_result(_attempt(with_optionals=True))
    encoded["viewport_fuel_containers"] = ["x"]
    with pytest.raises(
        JSONTypeError,
        match="'viewport_fuel_containers' must contain only objects",
    ):
        decode_fuel_dot_attempt_result(encoded)


def test_decode_attempt_rejects_non_object_snapshot() -> None:
    """A non-object snapshot field is a hard decode error."""
    encoded = encode_fuel_dot_attempt_result(_attempt(with_optionals=True))
    encoded["snapshot_before"] = None
    with pytest.raises(JSONTypeError, match="'snapshot_before' must be an object"):
        decode_fuel_dot_attempt_result(encoded)


def test_decode_session_rejects_non_object_startup_timing() -> None:
    """A non-object startup timing is a hard decode error."""
    encoded = encode_fuel_dot_probe_session(_session())
    encoded["startup_timing"] = None
    with pytest.raises(JSONTypeError, match="'startup_timing' must be an object"):
        decode_fuel_dot_probe_session(encoded)


def test_decode_session_rejects_non_object_attempt_entry() -> None:
    """A non-object attempts entry is a hard decode error."""
    encoded = encode_fuel_dot_probe_session(_session())
    encoded["attempts"] = ["x"]
    with pytest.raises(JSONTypeError, match="'attempts' must contain only objects"):
        decode_fuel_dot_probe_session(encoded)
