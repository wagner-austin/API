"""Tests for covenant metrics events: DecodeCovenantEvent."""

from __future__ import annotations

import pytest

from platform_core.covenant_metrics_decode import (
    CovenantEventV1,
    decode_covenant_event,
    is_covenant_alert_triggered,
    is_covenant_evaluation_completed,
    is_covenant_job_completed,
    is_covenant_job_failed,
    is_covenant_job_started,
    is_covenant_measurement_received,
    is_covenant_prediction_completed,
    is_covenant_retrain_triggered,
    is_covenant_stream_lag,
)
from platform_core.covenant_metrics_events import (
    encode_covenant_metrics_event,
    make_alert_triggered_event,
    make_evaluation_completed_event,
    make_measurement_received_event,
    make_prediction_completed_event,
    make_retrain_triggered_event,
    make_stream_lag_event,
)


class TestDecodeCovenantEvent:
    def test_raises_for_non_dict(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Expected JSON object"):
            decode_covenant_event("[]")

    def test_raises_for_non_string_type(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Field 'type' must be a string"):
            decode_covenant_event('{"type": 123}')

    def test_raises_for_unknown_type(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Unknown covenant event type"):
            decode_covenant_event('{"type": "unknown.event.v1", "event_id": "e1"}')

    def test_decodes_job_started_event(self) -> None:
        payload = """{
            "type": "covenant.job.started.v1",
            "domain": "covenant",
            "job_id": "j1",
            "user_id": 1,
            "queue": "covenant-training"
        }"""
        ev = decode_covenant_event(payload)
        assert is_covenant_job_started(ev)
        assert ev["type"] == "covenant.job.started.v1"

    def test_decodes_job_completed_event(self) -> None:
        payload = """{
            "type": "covenant.job.completed.v1",
            "domain": "covenant",
            "job_id": "j1",
            "user_id": 1,
            "result_id": "r1",
            "result_bytes": 1024
        }"""
        ev = decode_covenant_event(payload)
        assert is_covenant_job_completed(ev)
        assert ev["type"] == "covenant.job.completed.v1"

    def test_decodes_job_failed_event_user(self) -> None:
        payload = """{
            "type": "covenant.job.failed.v1",
            "domain": "covenant",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "user",
            "message": "Invalid input"
        }"""
        ev = decode_covenant_event(payload)
        assert is_covenant_job_failed(ev)
        assert ev["type"] == "covenant.job.failed.v1"

    def test_decodes_job_failed_event_system(self) -> None:
        payload = """{
            "type": "covenant.job.failed.v1",
            "domain": "covenant",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "system",
            "message": "Internal error"
        }"""
        ev = decode_covenant_event(payload)
        assert is_covenant_job_failed(ev)
        assert ev["type"] == "covenant.job.failed.v1"

    def test_decodes_metrics_event(self) -> None:
        ev = make_measurement_received_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            metric_count=5,
            latency_ms=10,
            timestamp="2024-01-15T10:00:00Z",
        )
        payload = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_event(payload)
        assert is_covenant_measurement_received(decoded)
        assert decoded["type"] == "covenant.metrics.measurement.received.v1"

    def test_raises_for_wrong_domain(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.job.started.v1",
            "domain": "other",
            "job_id": "j1",
            "user_id": 1,
            "queue": "q"
        }"""
        with pytest.raises(JSONTypeError, match="Domain mismatch"):
            decode_covenant_event(payload)

    def test_raises_for_missing_queue_in_started(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.job.started.v1",
            "domain": "covenant",
            "job_id": "j1",
            "user_id": 1
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'queue'"):
            decode_covenant_event(payload)

    def test_raises_for_missing_fields_in_completed(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.job.completed.v1",
            "domain": "covenant",
            "job_id": "j1",
            "user_id": 1
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'result_id'"):
            decode_covenant_event(payload)

    def test_raises_for_missing_message_in_failed(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.job.failed.v1",
            "domain": "covenant",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "user"
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'message'"):
            decode_covenant_event(payload)

    def test_raises_for_invalid_error_kind(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.job.failed.v1",
            "domain": "covenant",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "invalid",
            "message": "msg"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid error_kind 'invalid'"):
            decode_covenant_event(payload)

    def test_raises_for_unknown_job_event_suffix(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.job.unknown.v1",
            "domain": "covenant",
            "job_id": "j1",
            "user_id": 1
        }"""
        with pytest.raises(JSONTypeError, match="Unknown covenant job event type"):
            decode_covenant_event(payload)

    def test_raises_for_unknown_metrics_type(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = '{"type": "covenant.metrics.unknown.v1", "event_id": "e1"}'
        with pytest.raises(JSONTypeError, match="Unknown covenant metrics event type"):
            decode_covenant_event(payload)


class TestCombinedTypeGuards:
    def test_is_covenant_job_started(self) -> None:
        from platform_core.covenant_metrics_decode import (
            JobStartedV1,
        )

        started: JobStartedV1 = {
            "type": "covenant.job.started.v1",
            "domain": "covenant",
            "job_id": "j",
            "user_id": 1,
            "queue": "q",
        }
        ev: CovenantEventV1 = started
        assert is_covenant_job_started(ev)

    def test_is_covenant_job_completed(self) -> None:
        from platform_core.covenant_metrics_decode import (
            JobCompletedV1,
        )

        completed: JobCompletedV1 = {
            "type": "covenant.job.completed.v1",
            "domain": "covenant",
            "job_id": "j",
            "user_id": 1,
            "result_id": "r",
            "result_bytes": 1,
        }
        ev: CovenantEventV1 = completed
        assert is_covenant_job_completed(ev)

    def test_is_covenant_job_failed(self) -> None:
        from platform_core.covenant_metrics_decode import (
            JobFailedV1,
        )

        failed: JobFailedV1 = {
            "type": "covenant.job.failed.v1",
            "domain": "covenant",
            "job_id": "j",
            "user_id": 1,
            "error_kind": "user",
            "message": "m",
        }
        ev: CovenantEventV1 = failed
        assert is_covenant_job_failed(ev)

    def test_is_covenant_measurement_received(self) -> None:
        ev: CovenantEventV1 = make_measurement_received_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            metric_count=5,
            latency_ms=10,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_covenant_measurement_received(ev)

    def test_is_covenant_evaluation_completed(self) -> None:
        ev: CovenantEventV1 = make_evaluation_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            status="OK",
            covenants_evaluated=3,
            breaches_count=0,
            latency_ms=15,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_covenant_evaluation_completed(ev)

    def test_is_covenant_prediction_completed(self) -> None:
        ev: CovenantEventV1 = make_prediction_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            risk_probability=0.85,
            risk_tier="HIGH",
            model_version="v1.0.0",
            latency_ms=25,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_covenant_prediction_completed(ev)

    def test_is_covenant_alert_triggered(self) -> None:
        ev: CovenantEventV1 = make_alert_triggered_event(
            event_id="e1",
            deal_id="d1",
            alert_type="breach",
            severity="critical",
            risk_probability=0.95,
            message="Alert!",
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_covenant_alert_triggered(ev)

    def test_is_covenant_retrain_triggered(self) -> None:
        ev: CovenantEventV1 = make_retrain_triggered_event(
            event_id="e1",
            trigger_type="drift",
            current_auc=0.72,
            threshold_auc=0.75,
            samples_since_train=5000,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_covenant_retrain_triggered(ev)

    def test_is_covenant_stream_lag(self) -> None:
        ev: CovenantEventV1 = make_stream_lag_event(
            event_id="e1",
            topic="covenant.measurements.v1",
            partition=0,
            lag_messages=100,
            lag_ms=500,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_covenant_stream_lag(ev)

    def test_type_guards_return_false_for_non_matching(self) -> None:
        ev: CovenantEventV1 = make_measurement_received_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            metric_count=5,
            latency_ms=10,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_covenant_measurement_received(ev)
        assert not is_covenant_job_started(ev)
        assert not is_covenant_job_completed(ev)
        assert not is_covenant_job_failed(ev)
        assert not is_covenant_evaluation_completed(ev)
        assert not is_covenant_prediction_completed(ev)
        assert not is_covenant_alert_triggered(ev)
        assert not is_covenant_retrain_triggered(ev)
        assert not is_covenant_stream_lag(ev)


class TestDefaultChannel:
    def test_default_channel_value(self) -> None:
        from platform_core.covenant_metrics_decode import (
            DEFAULT_COVENANT_EVENTS_CHANNEL,
        )

        assert DEFAULT_COVENANT_EVENTS_CHANNEL == "covenant:events"
