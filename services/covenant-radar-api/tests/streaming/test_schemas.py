"""Tests for streaming schemas module."""

from __future__ import annotations

from covenant_domain.features import classify_risk_tier
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)

from covenant_radar_api.streaming.schemas import (
    AlertEventV1,
    MeasurementEventV1,
    PredictionEventV1,
    encode_alert_event,
    encode_kafka_event,
    encode_measurement_event,
    encode_prediction_event,
    make_alert_event,
    make_measurement_event,
    make_prediction_event,
)


class TestMakeMeasurementEvent:
    """Tests for make_measurement_event factory."""

    def test_creates_event(self) -> None:
        """Create measurement event with all fields."""
        event = make_measurement_event(
            event_id="evt-123",
            deal_id="deal-456",
            period_start="2024-01-01",
            period_end="2024-03-31",
            metric_name="debt_to_equity",
            metric_value=1.5,
            timestamp="2024-04-01T12:00:00Z",
        )

        assert event["type"] == "covenant.measurement.v1"
        assert event["event_id"] == "evt-123"
        assert event["deal_id"] == "deal-456"
        assert event["period_start"] == "2024-01-01"
        assert event["period_end"] == "2024-03-31"
        assert event["metric_name"] == "debt_to_equity"
        assert event["metric_value"] == 1.5
        assert event["timestamp"] == "2024-04-01T12:00:00Z"


class TestMakePredictionEvent:
    """Tests for make_prediction_event factory."""

    def test_creates_event(self) -> None:
        """Create prediction event with all fields."""
        event = make_prediction_event(
            event_id="evt-789",
            deal_id="deal-456",
            period_start="2024-01-01",
            period_end="2024-03-31",
            evaluation_status="WARNING",
            covenants_evaluated=5,
            breaches_count=1,
            risk_probability=0.65,
            risk_tier="HIGH",
            model_version="v1.2.3",
            evaluation_latency_ms=50,
            prediction_latency_ms=20,
            processed_at="2024-04-01T12:00:00Z",
        )

        assert event["type"] == "covenant.prediction.v1"
        assert event["event_id"] == "evt-789"
        assert event["deal_id"] == "deal-456"
        assert event["evaluation_status"] == "WARNING"
        assert event["covenants_evaluated"] == 5
        assert event["breaches_count"] == 1
        assert event["risk_probability"] == 0.65
        assert event["risk_tier"] == "HIGH"
        assert event["model_version"] == "v1.2.3"
        assert event["evaluation_latency_ms"] == 50
        assert event["prediction_latency_ms"] == 20


class TestMakeAlertEvent:
    """Tests for make_alert_event factory."""

    def test_creates_event(self) -> None:
        """Create alert event with all fields."""
        event = make_alert_event(
            event_id="evt-alert",
            deal_id="deal-456",
            alert_type="high_risk",
            severity="critical",
            risk_probability=0.92,
            gemini_summary="High risk detected for deal-456.",
            triggered_at="2024-04-01T12:00:00Z",
        )

        assert event["type"] == "covenant.alert.v1"
        assert event["event_id"] == "evt-alert"
        assert event["deal_id"] == "deal-456"
        assert event["alert_type"] == "high_risk"
        assert event["severity"] == "critical"
        assert event["risk_probability"] == 0.92
        assert event["gemini_summary"] == "High risk detected for deal-456."


class TestEncodeMeasurementEvent:
    """Tests for encode_measurement_event."""

    def test_encodes_to_json(self) -> None:
        """Encode measurement event to JSON string."""
        event: MeasurementEventV1 = {
            "type": "covenant.measurement.v1",
            "event_id": "evt-123",
            "deal_id": "deal-456",
            "period_start": "2024-01-01",
            "period_end": "2024-03-31",
            "metric_name": "ebitda",
            "metric_value": 1000000.0,
            "timestamp": "2024-04-01T12:00:00Z",
        }

        result = encode_measurement_event(event)
        parsed = narrow_json_to_dict(load_json_str(result))

        assert parsed["type"] == "covenant.measurement.v1"
        assert parsed["event_id"] == "evt-123"


class TestEncodePredictionEvent:
    """Tests for encode_prediction_event."""

    def test_encodes_to_json(self) -> None:
        """Encode prediction event to JSON string."""
        event: PredictionEventV1 = {
            "type": "covenant.prediction.v1",
            "event_id": "evt-789",
            "deal_id": "deal-456",
            "period_start": "2024-01-01",
            "period_end": "2024-03-31",
            "evaluation_status": "OK",
            "covenants_evaluated": 3,
            "breaches_count": 0,
            "risk_probability": 0.15,
            "risk_tier": "LOW",
            "model_version": "v1.0.0",
            "evaluation_latency_ms": 30,
            "prediction_latency_ms": 10,
            "processed_at": "2024-04-01T12:00:00Z",
        }

        result = encode_prediction_event(event)
        parsed = narrow_json_to_dict(load_json_str(result))

        assert parsed["type"] == "covenant.prediction.v1"
        assert parsed["risk_tier"] == "LOW"


class TestEncodeAlertEvent:
    """Tests for encode_alert_event."""

    def test_encodes_to_json(self) -> None:
        """Encode alert event to JSON string."""
        event: AlertEventV1 = {
            "type": "covenant.alert.v1",
            "event_id": "evt-alert",
            "deal_id": "deal-456",
            "alert_type": "breach",
            "severity": "warning",
            "risk_probability": 0.75,
            "gemini_summary": "Covenant breach detected.",
            "triggered_at": "2024-04-01T12:00:00Z",
        }

        result = encode_alert_event(event)
        parsed = narrow_json_to_dict(load_json_str(result))

        assert parsed["type"] == "covenant.alert.v1"
        assert parsed["alert_type"] == "breach"


class TestEncodeKafkaEvent:
    """Tests for encode_kafka_event."""

    def test_encodes_measurement(self) -> None:
        """Encode measurement event via generic function."""
        event = make_measurement_event(
            event_id="evt-1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-03-31",
            metric_name="test",
            metric_value=1.0,
            timestamp="2024-04-01T12:00:00Z",
        )

        result = encode_kafka_event(event)
        parsed = narrow_json_to_dict(load_json_str(result))
        assert parsed["type"] == "covenant.measurement.v1"


class TestClassifyRiskTier:
    """Tests for classify_risk_tier."""

    def test_low(self) -> None:
        """Probability < 0.25 returns LOW."""
        assert classify_risk_tier(0.0) == "LOW"
        assert classify_risk_tier(0.1) == "LOW"
        assert classify_risk_tier(0.24) == "LOW"

    def test_medium(self) -> None:
        """0.25 <= probability < 0.50 returns MEDIUM."""
        assert classify_risk_tier(0.25) == "MEDIUM"
        assert classify_risk_tier(0.35) == "MEDIUM"
        assert classify_risk_tier(0.49) == "MEDIUM"

    def test_high(self) -> None:
        """0.50 <= probability < 0.80 returns HIGH."""
        assert classify_risk_tier(0.50) == "HIGH"
        assert classify_risk_tier(0.65) == "HIGH"
        assert classify_risk_tier(0.79) == "HIGH"

    def test_critical(self) -> None:
        """probability >= 0.80 returns CRITICAL."""
        assert classify_risk_tier(0.80) == "CRITICAL"
        assert classify_risk_tier(0.90) == "CRITICAL"
        assert classify_risk_tier(1.0) == "CRITICAL"
