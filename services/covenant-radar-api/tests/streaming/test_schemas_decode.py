"""Tests for streaming schemas module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
)

from covenant_radar_api.streaming.schemas import (
    make_alert_event,
    make_measurement_event,
    make_prediction_event,
)
from covenant_radar_api.streaming.schemas_decode import (
    decode_alert_event,
    decode_kafka_event,
    decode_measurement_event,
    decode_prediction_event,
    is_alert_event,
    is_measurement_event,
    is_prediction_event,
)


class TestDecodeMeasurementEvent:
    """Tests for decode_measurement_event."""

    def test_decodes_valid_event(self) -> None:
        """Decode valid measurement event JSON."""
        payload = dump_json_str(
            {
                "type": "covenant.measurement.v1",
                "event_id": "evt-123",
                "deal_id": "deal-456",
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
                "metric_name": "current_ratio",
                "metric_value": 2.5,
                "timestamp": "2024-04-01T12:00:00Z",
            }
        )

        event = decode_measurement_event(payload)

        assert event["type"] == "covenant.measurement.v1"
        assert event["event_id"] == "evt-123"
        assert event["metric_value"] == 2.5

    def test_wrong_type_raises(self) -> None:
        """Raise error for wrong event type."""
        payload = dump_json_str(
            {
                "type": "covenant.prediction.v1",
                "event_id": "evt-123",
            }
        )

        with pytest.raises(JSONTypeError, match="Expected measurement event"):
            decode_measurement_event(payload)

    def test_missing_field_raises(self) -> None:
        """Raise error for missing required field."""
        payload = dump_json_str(
            {
                "type": "covenant.measurement.v1",
                "event_id": "evt-123",
                # Missing deal_id and other fields
            }
        )

        with pytest.raises(JSONTypeError, match="Missing required field"):
            decode_measurement_event(payload)


class TestDecodePredictionEvent:
    """Tests for decode_prediction_event."""

    def test_decodes_valid_event(self) -> None:
        """Decode valid prediction event JSON."""
        payload = dump_json_str(
            {
                "type": "covenant.prediction.v1",
                "event_id": "evt-789",
                "deal_id": "deal-456",
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
                "evaluation_status": "BREACH",
                "covenants_evaluated": 4,
                "breaches_count": 2,
                "risk_probability": 0.85,
                "risk_tier": "CRITICAL",
                "model_version": "v2.0.0",
                "evaluation_latency_ms": 45,
                "prediction_latency_ms": 15,
                "processed_at": "2024-04-01T12:00:00Z",
            }
        )

        event = decode_prediction_event(payload)

        assert event["type"] == "covenant.prediction.v1"
        assert event["evaluation_status"] == "BREACH"
        assert event["risk_tier"] == "CRITICAL"

    def test_wrong_type_raises(self) -> None:
        """Raise error for wrong event type."""
        payload = dump_json_str(
            {
                "type": "covenant.measurement.v1",
                "event_id": "evt-123",
            }
        )

        with pytest.raises(JSONTypeError, match="Expected prediction event"):
            decode_prediction_event(payload)

    def test_invalid_evaluation_status_raises(self) -> None:
        """Raise error for invalid evaluation status."""
        payload = dump_json_str(
            {
                "type": "covenant.prediction.v1",
                "event_id": "evt-789",
                "deal_id": "deal-456",
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
                "evaluation_status": "INVALID",
                "covenants_evaluated": 1,
                "breaches_count": 0,
                "risk_probability": 0.5,
                "risk_tier": "MEDIUM",
                "model_version": "v1.0.0",
                "evaluation_latency_ms": 10,
                "prediction_latency_ms": 5,
                "processed_at": "2024-04-01T12:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError, match="Invalid evaluation status"):
            decode_prediction_event(payload)

    def test_invalid_risk_tier_raises(self) -> None:
        """Raise error for invalid risk tier."""
        payload = dump_json_str(
            {
                "type": "covenant.prediction.v1",
                "event_id": "evt-789",
                "deal_id": "deal-456",
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
                "evaluation_status": "OK",
                "covenants_evaluated": 1,
                "breaches_count": 0,
                "risk_probability": 0.5,
                "risk_tier": "INVALID",
                "model_version": "v1.0.0",
                "evaluation_latency_ms": 10,
                "prediction_latency_ms": 5,
                "processed_at": "2024-04-01T12:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError, match="Invalid risk tier"):
            decode_prediction_event(payload)

    def test_decodes_warning_status(self) -> None:
        """Decode prediction event with WARNING evaluation status."""
        payload = dump_json_str(
            {
                "type": "covenant.prediction.v1",
                "event_id": "evt-warning",
                "deal_id": "deal-456",
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
                "evaluation_status": "WARNING",
                "covenants_evaluated": 2,
                "breaches_count": 0,
                "risk_probability": 0.35,
                "risk_tier": "MEDIUM",
                "model_version": "v1.0.0",
                "evaluation_latency_ms": 10,
                "prediction_latency_ms": 5,
                "processed_at": "2024-04-01T12:00:00Z",
            }
        )

        event = decode_prediction_event(payload)

        assert event["evaluation_status"] == "WARNING"
        assert event["risk_tier"] == "MEDIUM"

    def test_decodes_high_risk_tier(self) -> None:
        """Decode prediction event with HIGH risk tier."""
        payload = dump_json_str(
            {
                "type": "covenant.prediction.v1",
                "event_id": "evt-high",
                "deal_id": "deal-456",
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
                "evaluation_status": "OK",
                "covenants_evaluated": 3,
                "breaches_count": 0,
                "risk_probability": 0.65,
                "risk_tier": "HIGH",
                "model_version": "v1.0.0",
                "evaluation_latency_ms": 10,
                "prediction_latency_ms": 5,
                "processed_at": "2024-04-01T12:00:00Z",
            }
        )

        event = decode_prediction_event(payload)

        assert event["risk_tier"] == "HIGH"


class TestDecodeAlertEvent:
    """Tests for decode_alert_event."""

    def test_decodes_valid_event(self) -> None:
        """Decode valid alert event JSON."""
        payload = dump_json_str(
            {
                "type": "covenant.alert.v1",
                "event_id": "evt-alert",
                "deal_id": "deal-456",
                "alert_type": "high_risk",
                "severity": "critical",
                "risk_probability": 0.95,
                "gemini_summary": "Critical risk alert.",
                "triggered_at": "2024-04-01T12:00:00Z",
            }
        )

        event = decode_alert_event(payload)

        assert event["type"] == "covenant.alert.v1"
        assert event["alert_type"] == "high_risk"
        assert event["severity"] == "critical"

    def test_wrong_type_raises(self) -> None:
        """Raise error for wrong event type."""
        payload = dump_json_str(
            {
                "type": "covenant.measurement.v1",
                "event_id": "evt-123",
            }
        )

        with pytest.raises(JSONTypeError, match="Expected alert event"):
            decode_alert_event(payload)

    def test_invalid_alert_type_raises(self) -> None:
        """Raise error for invalid alert type."""
        payload = dump_json_str(
            {
                "type": "covenant.alert.v1",
                "event_id": "evt-alert",
                "deal_id": "deal-456",
                "alert_type": "invalid_type",
                "severity": "warning",
                "risk_probability": 0.8,
                "gemini_summary": "Test.",
                "triggered_at": "2024-04-01T12:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError, match="Invalid alert type"):
            decode_alert_event(payload)

    def test_invalid_severity_raises(self) -> None:
        """Raise error for invalid severity."""
        payload = dump_json_str(
            {
                "type": "covenant.alert.v1",
                "event_id": "evt-alert",
                "deal_id": "deal-456",
                "alert_type": "breach",
                "severity": "invalid_severity",
                "risk_probability": 0.8,
                "gemini_summary": "Test.",
                "triggered_at": "2024-04-01T12:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError, match="Invalid alert severity"):
            decode_alert_event(payload)


class TestDecodeKafkaEvent:
    """Tests for decode_kafka_event."""

    def test_decodes_measurement(self) -> None:
        """Decode measurement event via generic decoder."""
        payload = dump_json_str(
            {
                "type": "covenant.measurement.v1",
                "event_id": "evt-123",
                "deal_id": "deal-456",
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
                "metric_name": "test",
                "metric_value": 1.0,
                "timestamp": "2024-04-01T12:00:00Z",
            }
        )

        event = decode_kafka_event(payload)
        assert event["type"] == "covenant.measurement.v1"

    def test_decodes_prediction(self) -> None:
        """Decode prediction event via generic decoder."""
        payload = dump_json_str(
            {
                "type": "covenant.prediction.v1",
                "event_id": "evt-789",
                "deal_id": "deal-456",
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
                "evaluation_status": "OK",
                "covenants_evaluated": 1,
                "breaches_count": 0,
                "risk_probability": 0.2,
                "risk_tier": "LOW",
                "model_version": "v1.0.0",
                "evaluation_latency_ms": 10,
                "prediction_latency_ms": 5,
                "processed_at": "2024-04-01T12:00:00Z",
            }
        )

        event = decode_kafka_event(payload)
        assert event["type"] == "covenant.prediction.v1"

    def test_decodes_alert(self) -> None:
        """Decode alert event via generic decoder."""
        payload = dump_json_str(
            {
                "type": "covenant.alert.v1",
                "event_id": "evt-alert",
                "deal_id": "deal-456",
                "alert_type": "breach",
                "severity": "warning",
                "risk_probability": 0.8,
                "gemini_summary": "Test.",
                "triggered_at": "2024-04-01T12:00:00Z",
            }
        )

        event = decode_kafka_event(payload)
        assert event["type"] == "covenant.alert.v1"

    def test_unknown_type_raises(self) -> None:
        """Raise error for unknown event type."""
        payload = dump_json_str(
            {
                "type": "covenant.unknown.v1",
                "event_id": "evt-123",
            }
        )

        with pytest.raises(JSONTypeError, match="Unknown Kafka event type"):
            decode_kafka_event(payload)


class TestTypeGuards:
    """Tests for TypeGuard functions."""

    def test_is_measurement_event_true(self) -> None:
        """is_measurement_event returns True for measurement."""
        event = make_measurement_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-03-31",
            metric_name="test",
            metric_value=1.0,
            timestamp="2024-04-01T12:00:00Z",
        )
        assert is_measurement_event(event) is True

    def test_is_measurement_event_false(self) -> None:
        """is_measurement_event returns False for non-measurement."""
        event = make_prediction_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-03-31",
            evaluation_status="OK",
            covenants_evaluated=1,
            breaches_count=0,
            risk_probability=0.1,
            risk_tier="LOW",
            model_version="v1",
            evaluation_latency_ms=10,
            prediction_latency_ms=5,
            processed_at="2024-04-01T12:00:00Z",
        )
        assert is_measurement_event(event) is False

    def test_is_prediction_event_true(self) -> None:
        """is_prediction_event returns True for prediction."""
        event = make_prediction_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-03-31",
            evaluation_status="OK",
            covenants_evaluated=1,
            breaches_count=0,
            risk_probability=0.1,
            risk_tier="LOW",
            model_version="v1",
            evaluation_latency_ms=10,
            prediction_latency_ms=5,
            processed_at="2024-04-01T12:00:00Z",
        )
        assert is_prediction_event(event) is True

    def test_is_prediction_event_false(self) -> None:
        """is_prediction_event returns False for non-prediction."""
        event = make_measurement_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-03-31",
            metric_name="test",
            metric_value=1.0,
            timestamp="2024-04-01T12:00:00Z",
        )
        assert is_prediction_event(event) is False

    def test_is_alert_event_true(self) -> None:
        """is_alert_event returns True for alert."""
        event = make_alert_event(
            event_id="e1",
            deal_id="d1",
            alert_type="breach",
            severity="warning",
            risk_probability=0.8,
            gemini_summary="Test.",
            triggered_at="2024-04-01T12:00:00Z",
        )
        assert is_alert_event(event) is True

    def test_is_alert_event_false(self) -> None:
        """is_alert_event returns False for non-alert."""
        event = make_measurement_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-03-31",
            metric_name="test",
            metric_value=1.0,
            timestamp="2024-04-01T12:00:00Z",
        )
        assert is_alert_event(event) is False
