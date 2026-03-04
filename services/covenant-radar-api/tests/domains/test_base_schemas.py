"""Tests for base event schemas."""

from __future__ import annotations

import pytest
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str

from covenant_radar_api.domains.base_schemas import (
    BaseAlertEventV1,
    BaseInputEventV1,
    BasePredictionEventV1,
    _parse_base_alert_severity,
    decode_base_alert_event,
    decode_base_input_event,
    decode_base_prediction_event,
    encode_base_alert_event,
    encode_base_input_event,
    encode_base_prediction_event,
    make_base_alert_event,
    make_base_input_event,
    make_base_prediction_event,
)

# =============================================================================
# Factory: make_base_input_event
# =============================================================================


class TestMakeBaseInputEvent:
    """Tests for make_base_input_event factory."""

    def test_creates_event(self) -> None:
        """Create base input event with all fields."""
        event = make_base_input_event(
            type="weather.observation.v1",
            event_id="evt-001",
            entity_id="station-alpha",
            timestamp="2025-06-21T14:00:00Z",
        )

        assert event["type"] == "weather.observation.v1"
        assert event["event_id"] == "evt-001"
        assert event["entity_id"] == "station-alpha"
        assert event["timestamp"] == "2025-06-21T14:00:00Z"

    def test_different_domain_type(self) -> None:
        """Create base input event with a different domain type."""
        event = make_base_input_event(
            type="esports.match.v1",
            event_id="evt-002",
            entity_id="match-123",
            timestamp="2025-01-15T08:00:00Z",
        )

        assert event["type"] == "esports.match.v1"
        assert event["entity_id"] == "match-123"


# =============================================================================
# Factory: make_base_prediction_event
# =============================================================================


class TestMakeBasePredictionEvent:
    """Tests for make_base_prediction_event factory."""

    def test_creates_event(self) -> None:
        """Create base prediction event with all fields."""
        event = make_base_prediction_event(
            type="covenant.prediction.v1",
            event_id="pred-001",
            entity_id="deal-abc",
            prediction_value=0.85,
            confidence=0.70,
            model_version="xgboost-v2.1",
            latency_ms=42,
            processed_at="2025-06-21T14:00:01Z",
        )

        assert event["type"] == "covenant.prediction.v1"
        assert event["event_id"] == "pred-001"
        assert event["entity_id"] == "deal-abc"
        assert event["prediction_value"] == 0.85
        assert event["confidence"] == 0.70
        assert event["model_version"] == "xgboost-v2.1"
        assert event["latency_ms"] == 42
        assert event["processed_at"] == "2025-06-21T14:00:01Z"

    def test_zero_latency(self) -> None:
        """Create prediction event with zero latency."""
        event = make_base_prediction_event(
            type="test.prediction.v1",
            event_id="pred-002",
            entity_id="entity-1",
            prediction_value=0.0,
            confidence=1.0,
            model_version="v1",
            latency_ms=0,
            processed_at="2025-01-01T00:00:00Z",
        )

        assert event["latency_ms"] == 0
        assert event["prediction_value"] == 0.0
        assert event["confidence"] == 1.0


# =============================================================================
# Factory: make_base_alert_event
# =============================================================================


class TestMakeBaseAlertEvent:
    """Tests for make_base_alert_event factory."""

    def test_creates_info_alert(self) -> None:
        """Create alert event with info severity."""
        event = make_base_alert_event(
            type="weather.alert.v1",
            event_id="alert-001",
            entity_id="station-beta",
            alert_type="high_temperature",
            severity="info",
            prediction_value=0.65,
            gemini_summary="Mild temperature anomaly detected.",
            triggered_at="2025-06-21T14:00:02Z",
        )

        assert event["type"] == "weather.alert.v1"
        assert event["event_id"] == "alert-001"
        assert event["entity_id"] == "station-beta"
        assert event["alert_type"] == "high_temperature"
        assert event["severity"] == "info"
        assert event["prediction_value"] == 0.65
        assert event["gemini_summary"] == "Mild temperature anomaly detected."
        assert event["triggered_at"] == "2025-06-21T14:00:02Z"

    def test_creates_warning_alert(self) -> None:
        """Create alert event with warning severity."""
        event = make_base_alert_event(
            type="covenant.alert.v1",
            event_id="alert-002",
            entity_id="deal-xyz",
            alert_type="high_risk",
            severity="warning",
            prediction_value=0.82,
            gemini_summary="Deal risk elevated.",
            triggered_at="2025-06-21T14:00:03Z",
        )

        assert event["severity"] == "warning"

    def test_creates_critical_alert(self) -> None:
        """Create alert event with critical severity."""
        event = make_base_alert_event(
            type="fire.alert.v1",
            event_id="alert-003",
            entity_id="region-9",
            alert_type="wildfire_imminent",
            severity="critical",
            prediction_value=0.95,
            gemini_summary="Critical fire risk detected.",
            triggered_at="2025-06-21T14:00:04Z",
        )

        assert event["severity"] == "critical"


# =============================================================================
# Encoder: encode_base_input_event
# =============================================================================


class TestEncodeBaseInputEvent:
    """Tests for encode_base_input_event."""

    def test_round_trip(self) -> None:
        """Encode then decode produces identical event."""
        original = make_base_input_event(
            type="test.input.v1",
            event_id="evt-100",
            entity_id="entity-100",
            timestamp="2025-07-19T12:00:00Z",
        )

        json_str = encode_base_input_event(original)
        decoded = decode_base_input_event(json_str)

        assert decoded["type"] == original["type"]
        assert decoded["event_id"] == original["event_id"]
        assert decoded["entity_id"] == original["entity_id"]
        assert decoded["timestamp"] == original["timestamp"]

    def test_produces_json_with_fields(self) -> None:
        """Encoder returns JSON containing expected field values."""
        event = make_base_input_event(
            type="test.input.v1",
            event_id="evt-101",
            entity_id="entity-101",
            timestamp="2025-01-01T00:00:00Z",
        )

        result = encode_base_input_event(event)
        assert "test.input.v1" in result
        assert "evt-101" in result


# =============================================================================
# Encoder: encode_base_prediction_event
# =============================================================================


class TestEncodeBasePredictionEvent:
    """Tests for encode_base_prediction_event."""

    def test_round_trip(self) -> None:
        """Encode then decode produces identical event."""
        original = make_base_prediction_event(
            type="test.prediction.v1",
            event_id="pred-100",
            entity_id="entity-200",
            prediction_value=0.73,
            confidence=0.46,
            model_version="lgbm-v1.0",
            latency_ms=15,
            processed_at="2025-07-19T12:00:01Z",
        )

        json_str = encode_base_prediction_event(original)
        decoded = decode_base_prediction_event(json_str)

        assert decoded["type"] == original["type"]
        assert decoded["event_id"] == original["event_id"]
        assert decoded["entity_id"] == original["entity_id"]
        assert decoded["prediction_value"] == original["prediction_value"]
        assert decoded["confidence"] == original["confidence"]
        assert decoded["model_version"] == original["model_version"]
        assert decoded["latency_ms"] == original["latency_ms"]
        assert decoded["processed_at"] == original["processed_at"]


# =============================================================================
# Encoder: encode_base_alert_event
# =============================================================================


class TestEncodeBaseAlertEvent:
    """Tests for encode_base_alert_event."""

    def test_round_trip(self) -> None:
        """Encode then decode produces identical event."""
        original = make_base_alert_event(
            type="test.alert.v1",
            event_id="alert-100",
            entity_id="entity-300",
            alert_type="threshold_exceeded",
            severity="warning",
            prediction_value=0.88,
            gemini_summary="Alert summary text.",
            triggered_at="2025-07-19T12:00:02Z",
        )

        json_str = encode_base_alert_event(original)
        decoded = decode_base_alert_event(json_str)

        assert decoded["type"] == original["type"]
        assert decoded["event_id"] == original["event_id"]
        assert decoded["entity_id"] == original["entity_id"]
        assert decoded["alert_type"] == original["alert_type"]
        assert decoded["severity"] == original["severity"]
        assert decoded["prediction_value"] == original["prediction_value"]
        assert decoded["gemini_summary"] == original["gemini_summary"]
        assert decoded["triggered_at"] == original["triggered_at"]


# =============================================================================
# Decoder: decode_base_input_event
# =============================================================================


class TestDecodeBaseInputEvent:
    """Tests for decode_base_input_event."""

    def test_decodes_valid_payload(self) -> None:
        """Decode a valid JSON payload."""
        payload_dict: BaseInputEventV1 = {
            "type": "weather.observation.v1",
            "event_id": "evt-200",
            "entity_id": "station-delta",
            "timestamp": "2025-03-31T10:00:00Z",
        }
        payload = dump_json_str(payload_dict)

        event = decode_base_input_event(payload)

        assert event["type"] == "weather.observation.v1"
        assert event["event_id"] == "evt-200"
        assert event["entity_id"] == "station-delta"
        assert event["timestamp"] == "2025-03-31T10:00:00Z"

    def test_missing_field_raises(self) -> None:
        """Raises JSONTypeError for missing required field."""
        payload = dump_json_str(
            {
                "type": "test.input.v1",
                "event_id": "evt-300",
                # missing entity_id
                "timestamp": "2025-01-01T00:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError):
            decode_base_input_event(payload)

    def test_invalid_json_raises(self) -> None:
        """Raises InvalidJsonError on invalid JSON."""
        with pytest.raises(InvalidJsonError):
            decode_base_input_event("not valid json")

    def test_non_object_json_raises(self) -> None:
        """Raises JSONTypeError on JSON that is not an object."""
        with pytest.raises(JSONTypeError):
            decode_base_input_event('"just a string"')


# =============================================================================
# Decoder: decode_base_prediction_event
# =============================================================================


class TestDecodeBasePredictionEvent:
    """Tests for decode_base_prediction_event."""

    def test_decodes_valid_payload(self) -> None:
        """Decode a valid prediction event payload."""
        payload_dict: BasePredictionEventV1 = {
            "type": "covenant.prediction.v1",
            "event_id": "pred-200",
            "entity_id": "deal-abc",
            "prediction_value": 0.72,
            "confidence": 0.44,
            "model_version": "xgb-v3",
            "latency_ms": 28,
            "processed_at": "2025-03-31T10:00:01Z",
        }
        payload = dump_json_str(payload_dict)

        event = decode_base_prediction_event(payload)

        assert event["type"] == "covenant.prediction.v1"
        assert event["event_id"] == "pred-200"
        assert event["entity_id"] == "deal-abc"
        assert event["prediction_value"] == 0.72
        assert event["confidence"] == 0.44
        assert event["model_version"] == "xgb-v3"
        assert event["latency_ms"] == 28
        assert event["processed_at"] == "2025-03-31T10:00:01Z"

    def test_missing_field_raises(self) -> None:
        """Raises JSONTypeError for missing required field."""
        payload = dump_json_str(
            {
                "type": "test.prediction.v1",
                "event_id": "pred-300",
                "entity_id": "entity-1",
                # missing prediction_value
                "confidence": 0.5,
                "model_version": "v1",
                "latency_ms": 10,
                "processed_at": "2025-01-01T00:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError):
            decode_base_prediction_event(payload)

    def test_invalid_json_raises(self) -> None:
        """Raises InvalidJsonError on invalid JSON."""
        with pytest.raises(InvalidJsonError):
            decode_base_prediction_event("{bad json")

    def test_non_object_json_raises(self) -> None:
        """Raises JSONTypeError on JSON that is not an object."""
        with pytest.raises(JSONTypeError):
            decode_base_prediction_event("[1, 2, 3]")


# =============================================================================
# Decoder: decode_base_alert_event
# =============================================================================


class TestDecodeBaseAlertEvent:
    """Tests for decode_base_alert_event."""

    def test_decodes_valid_payload(self) -> None:
        """Decode a valid alert event payload."""
        payload_dict: BaseAlertEventV1 = {
            "type": "test.alert.v1",
            "event_id": "alert-200",
            "entity_id": "entity-xyz",
            "alert_type": "high_risk",
            "severity": "critical",
            "prediction_value": 0.92,
            "gemini_summary": "Critical risk detected.",
            "triggered_at": "2025-03-31T10:00:02Z",
        }
        payload = dump_json_str(payload_dict)

        event = decode_base_alert_event(payload)

        assert event["type"] == "test.alert.v1"
        assert event["event_id"] == "alert-200"
        assert event["entity_id"] == "entity-xyz"
        assert event["alert_type"] == "high_risk"
        assert event["severity"] == "critical"
        assert event["prediction_value"] == 0.92
        assert event["gemini_summary"] == "Critical risk detected."
        assert event["triggered_at"] == "2025-03-31T10:00:02Z"

    def test_invalid_severity_raises(self) -> None:
        """Raises JSONTypeError for invalid severity value."""
        payload = dump_json_str(
            {
                "type": "test.alert.v1",
                "event_id": "alert-300",
                "entity_id": "entity-1",
                "alert_type": "high_risk",
                "severity": "INVALID",
                "prediction_value": 0.9,
                "gemini_summary": "Summary.",
                "triggered_at": "2025-01-01T00:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError, match="Invalid base alert severity"):
            decode_base_alert_event(payload)

    def test_missing_field_raises(self) -> None:
        """Raises JSONTypeError for missing required field."""
        payload = dump_json_str(
            {
                "type": "test.alert.v1",
                "event_id": "alert-400",
                "entity_id": "entity-1",
                # missing alert_type
                "severity": "warning",
                "prediction_value": 0.85,
                "gemini_summary": "Summary.",
                "triggered_at": "2025-01-01T00:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError):
            decode_base_alert_event(payload)

    def test_invalid_json_raises(self) -> None:
        """Raises InvalidJsonError on invalid JSON."""
        with pytest.raises(InvalidJsonError):
            decode_base_alert_event("<<<not json>>>")

    def test_non_object_json_raises(self) -> None:
        """Raises JSONTypeError on JSON that is not an object."""
        with pytest.raises(JSONTypeError):
            decode_base_alert_event("42")


# =============================================================================
# Literal Parser: _parse_base_alert_severity
# =============================================================================


class TestParseBaseAlertSeverity:
    """Tests for _parse_base_alert_severity."""

    def test_parses_info(self) -> None:
        """Parse 'info' severity."""
        assert _parse_base_alert_severity("info") == "info"

    def test_parses_warning(self) -> None:
        """Parse 'warning' severity."""
        assert _parse_base_alert_severity("warning") == "warning"

    def test_parses_critical(self) -> None:
        """Parse 'critical' severity."""
        assert _parse_base_alert_severity("critical") == "critical"

    def test_invalid_raises(self) -> None:
        """Raises JSONTypeError for invalid severity string."""
        with pytest.raises(JSONTypeError, match="Invalid base alert severity 'bad'"):
            _parse_base_alert_severity("bad")

    def test_case_sensitive(self) -> None:
        """Uppercase variants are rejected (case-sensitive)."""
        with pytest.raises(JSONTypeError):
            _parse_base_alert_severity("INFO")
