"""Tests for queue probe TypedDict models and encode/decode."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.action_lab.queue_probe_types import (
    QueueCommandTimingDict,
    QueueExperimentResultDict,
    QueueProbeSessionDict,
    decode_queue_command_timing,
    decode_queue_experiment_result,
    decode_queue_probe_session,
    encode_queue_command_timing,
    encode_queue_experiment_result,
    encode_queue_probe_session,
)

_STARTUP_TIMING: JSONObject = {
    "game_ready_timestamp_ms": 1000,
    "intel_ready_timestamp_ms": 2000,
    "initial_sync_started_ms": 3000,
    "initial_world_timestamp_ms": 4000,
    "command_ready_timestamp_ms": 5000,
    "first_attempt_started_ms": 6000,
    "game_ready_to_intel_ready_ms": 1000,
    "intel_ready_to_initial_world_ms": 2000,
    "initial_world_to_command_ready_ms": 1000,
    "command_ready_to_first_attempt_ms": 1000,
}


def _make_timing(
    *,
    label: str = "shoot",
    sent_ms: int = 100,
    ack_ms: int | None = 200,
) -> QueueCommandTimingDict:
    elapsed = ack_ms - sent_ms if ack_ms is not None else None
    return QueueCommandTimingDict(
        label=label,
        sent_ms=sent_ms,
        ack_ms=ack_ms,
        elapsed_ms=elapsed,
    )


def _make_experiment() -> QueueExperimentResultDict:
    return QueueExperimentResultDict(
        kind="shoot_then_pickup",
        status="both_processed",
        primary=_make_timing(label="shoot", sent_ms=100, ack_ms=200),
        secondary=_make_timing(label="pickup_fuel", sent_ms=105, ack_ms=210),
        inter_send_delay_ms=5,
        total_elapsed_ms=115,
        message_start_index=0,
        message_end_index=10,
    )


class TestQueueCommandTiming:
    """Tests for QueueCommandTimingDict encode/decode."""

    def test_roundtrip_with_ack(self) -> None:
        timing = _make_timing(label="shoot", sent_ms=100, ack_ms=200)
        encoded = encode_queue_command_timing(timing)
        decoded = decode_queue_command_timing(encoded)
        assert decoded == timing

    def test_roundtrip_without_ack(self) -> None:
        timing = _make_timing(label="move", sent_ms=100, ack_ms=None)
        encoded = encode_queue_command_timing(timing)
        decoded = decode_queue_command_timing(encoded)
        assert decoded == timing

    def test_decode_missing_label_raises(self) -> None:
        data: JSONObject = {"sent_ms": 100, "ack_ms": 200, "elapsed_ms": 100}
        with pytest.raises(JSONTypeError):
            decode_queue_command_timing(data)

    def test_decode_non_int_ack_raises(self) -> None:
        data: JSONObject = {"label": "shoot", "sent_ms": 100, "ack_ms": "bad", "elapsed_ms": None}
        with pytest.raises(JSONTypeError, match="must be an integer"):
            decode_queue_command_timing(data)

    def test_decode_bool_ack_raises(self) -> None:
        data: JSONObject = {"label": "shoot", "sent_ms": 100, "ack_ms": True, "elapsed_ms": None}
        with pytest.raises(JSONTypeError, match="must be an integer"):
            decode_queue_command_timing(data)


class TestQueueExperimentResult:
    """Tests for QueueExperimentResultDict encode/decode."""

    def test_roundtrip(self) -> None:
        exp = _make_experiment()
        encoded = encode_queue_experiment_result(exp)
        decoded = decode_queue_experiment_result(encoded)
        assert decoded == exp

    def test_decode_invalid_kind_raises(self) -> None:
        exp = _make_experiment()
        encoded = encode_queue_experiment_result(exp)
        encoded["kind"] = "invalid_kind"
        with pytest.raises(JSONTypeError, match="invalid experiment kind"):
            decode_queue_experiment_result(encoded)

    def test_decode_invalid_status_raises(self) -> None:
        exp = _make_experiment()
        encoded = encode_queue_experiment_result(exp)
        encoded["status"] = "unknown_status"
        with pytest.raises(JSONTypeError, match="invalid experiment status"):
            decode_queue_experiment_result(encoded)

    def test_decode_bad_primary_raises(self) -> None:
        exp = _make_experiment()
        encoded = encode_queue_experiment_result(exp)
        encoded["primary"] = "not_a_dict"
        with pytest.raises(JSONTypeError, match="primary"):
            decode_queue_experiment_result(encoded)

    def test_decode_bad_secondary_raises(self) -> None:
        exp = _make_experiment()
        encoded = encode_queue_experiment_result(exp)
        encoded["secondary"] = 42
        with pytest.raises(JSONTypeError, match="secondary"):
            decode_queue_experiment_result(encoded)

    def test_shoot_then_shoot_kind_roundtrip(self) -> None:
        exp = _make_experiment()
        encoded = encode_queue_experiment_result(exp)
        encoded["kind"] = "shoot_then_shoot"
        decoded = decode_queue_experiment_result(encoded)
        assert decoded["kind"] == "shoot_then_shoot"

    def test_move_then_pickup_kind_roundtrip(self) -> None:
        exp = _make_experiment()
        encoded = encode_queue_experiment_result(exp)
        encoded["kind"] = "move_then_pickup"
        decoded = decode_queue_experiment_result(encoded)
        assert decoded["kind"] == "move_then_pickup"

    def test_second_dropped_status_roundtrip(self) -> None:
        exp = _make_experiment()
        encoded = encode_queue_experiment_result(exp)
        encoded["status"] = "second_dropped"
        decoded = decode_queue_experiment_result(encoded)
        assert decoded["status"] == "second_dropped"

    def test_timeout_status_roundtrip(self) -> None:
        exp = _make_experiment()
        encoded = encode_queue_experiment_result(exp)
        encoded["status"] = "timeout"
        decoded = decode_queue_experiment_result(encoded)
        assert decoded["status"] == "timeout"


class TestQueueProbeSession:
    """Tests for QueueProbeSessionDict encode/decode."""

    def _make_session(self) -> QueueProbeSessionDict:
        from tankpit_bot.action_lab.types_codecs import (
            decode_teleport_startup_timing,
        )

        timing = decode_teleport_startup_timing(_STARTUP_TIMING)
        return QueueProbeSessionDict(
            session_id="test-session-001",
            start_timestamp_ms=1000,
            end_timestamp_ms=5000,
            base_url="https://tankpit.com/play",
            spawn_x=128,
            spawn_y=128,
            capture_session_path="",
            initial_sync_timeout_ms=10000,
            experiment_timeout_ms=5000,
            startup_timing=timing,
            experiments=[_make_experiment()],
        )

    def test_roundtrip(self) -> None:
        session = self._make_session()
        encoded = encode_queue_probe_session(session)
        decoded = decode_queue_probe_session(encoded)
        assert decoded == session

    def test_roundtrip_empty_experiments(self) -> None:
        session = self._make_session()
        session["experiments"] = []
        encoded = encode_queue_probe_session(session)
        decoded = decode_queue_probe_session(encoded)
        assert decoded["experiments"] == []

    def test_decode_bad_startup_timing_raises(self) -> None:
        session = self._make_session()
        encoded = encode_queue_probe_session(session)
        encoded["startup_timing"] = "not_a_dict"
        with pytest.raises(JSONTypeError, match="startup_timing"):
            decode_queue_probe_session(encoded)

    def test_decode_bad_experiments_item_raises(self) -> None:
        session = self._make_session()
        encoded = encode_queue_probe_session(session)
        encoded["experiments"] = ["not_an_object"]
        with pytest.raises(JSONTypeError, match="experiments"):
            decode_queue_probe_session(encoded)

    def test_decode_missing_field_raises(self) -> None:
        data: JSONObject = {"session_id": "x"}
        with pytest.raises(JSONTypeError):
            decode_queue_probe_session(data)
