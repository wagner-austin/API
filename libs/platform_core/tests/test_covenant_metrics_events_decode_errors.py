"""Tests for covenant metrics events: DecodeErrors."""

from __future__ import annotations

import pytest

from platform_core.covenant_metrics_decode import (
    decode_covenant_metrics_event,
)


class TestDecodeErrors:
    def test_non_object_payload_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Expected JSON object"):
            decode_covenant_metrics_event("[]")

    def test_non_string_type_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Field 'type' must be a string"):
            decode_covenant_metrics_event('{"type": 123}')

    def test_missing_event_id_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Missing required field 'event_id'"):
            decode_covenant_metrics_event('{"type": "covenant.metrics.measurement.received.v1"}')

    def test_unknown_type_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = '{"type": "covenant.metrics.unknown.v1", "event_id": "e1"}'
        with pytest.raises(JSONTypeError, match="Unknown covenant metrics event type"):
            decode_covenant_metrics_event(payload)

    def test_measurement_missing_required_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = '{"type": "covenant.metrics.measurement.received.v1", "event_id": "e1"}'
        with pytest.raises(JSONTypeError, match="Missing required field 'deal_id'"):
            decode_covenant_metrics_event(payload)

    def test_evaluation_invalid_status_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.metrics.evaluation.completed.v1",
            "event_id": "e1",
            "deal_id": "d1",
            "period_start": "2024-01-01",
            "period_end": "2024-01-31",
            "status": "INVALID",
            "covenants_evaluated": 3,
            "breaches_count": 0,
            "latency_ms": 15,
            "timestamp": "2024-01-15T10:00:00Z"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid evaluation status"):
            decode_covenant_metrics_event(payload)

    def test_prediction_invalid_risk_tier_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.metrics.prediction.completed.v1",
            "event_id": "e1",
            "deal_id": "d1",
            "period_start": "2024-01-01",
            "period_end": "2024-01-31",
            "risk_probability": 0.85,
            "risk_tier": "INVALID",
            "model_version": "v1.0.0",
            "latency_ms": 25,
            "timestamp": "2024-01-15T10:00:00Z"
        }"""
        # The refusal now comes from the one shared narrowing in
        # platform_core.risk_tiers, so it names the field and the accepted set
        # rather than only the value -- the previous message, "Invalid risk
        # tier 'INVALID'", left a reader to go and find what was valid.
        with pytest.raises(JSONTypeError, match=r"Field 'risk_tier' must be one of"):
            decode_covenant_metrics_event(payload)

    def test_alert_invalid_alert_type_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.metrics.alert.triggered.v1",
            "event_id": "e1",
            "deal_id": "d1",
            "alert_type": "invalid",
            "severity": "critical",
            "risk_probability": 0.95,
            "message": "Alert!",
            "timestamp": "2024-01-15T10:00:00Z"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid alert type"):
            decode_covenant_metrics_event(payload)

    def test_alert_invalid_severity_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.metrics.alert.triggered.v1",
            "event_id": "e1",
            "deal_id": "d1",
            "alert_type": "breach",
            "severity": "invalid",
            "risk_probability": 0.95,
            "message": "Alert!",
            "timestamp": "2024-01-15T10:00:00Z"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid alert severity"):
            decode_covenant_metrics_event(payload)

    def test_retrain_invalid_trigger_type_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "covenant.metrics.retrain.triggered.v1",
            "event_id": "e1",
            "trigger_type": "invalid",
            "current_auc": 0.72,
            "threshold_auc": 0.75,
            "samples_since_train": 5000,
            "timestamp": "2024-01-15T10:00:00Z"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid retrain trigger type"):
            decode_covenant_metrics_event(payload)
