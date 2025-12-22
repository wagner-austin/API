"""Tests for covenant_metrics_events module."""

from __future__ import annotations

import pytest

from platform_core.covenant_metrics_events import (
    AlertTriggeredV1,
    CovenantEventV1,
    CovenantMetricsEventV1,
    EvaluationCompletedV1,
    MeasurementReceivedV1,
    PredictionCompletedV1,
    RetrainTriggeredV1,
    StreamLagV1,
    decode_covenant_event,
    decode_covenant_metrics_event,
    encode_covenant_metrics_event,
    is_alert_triggered,
    is_covenant_alert_triggered,
    is_covenant_evaluation_completed,
    is_covenant_job_completed,
    is_covenant_job_failed,
    is_covenant_job_started,
    is_covenant_measurement_received,
    is_covenant_prediction_completed,
    is_covenant_retrain_triggered,
    is_covenant_stream_lag,
    is_evaluation_completed,
    is_measurement_received,
    is_prediction_completed,
    is_retrain_triggered,
    is_stream_lag,
    make_alert_triggered_event,
    make_evaluation_completed_event,
    make_measurement_received_event,
    make_prediction_completed_event,
    make_retrain_triggered_event,
    make_stream_lag_event,
)


class TestMakeMeasurementReceivedEvent:
    def test_creates_event(self) -> None:
        ev = make_measurement_received_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            metric_count=5,
            latency_ms=10,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["type"] == "covenant.metrics.measurement.received.v1"
        assert ev["event_id"] == "e1"
        assert ev["deal_id"] == "d1"
        assert ev["period_start"] == "2024-01-01"
        assert ev["period_end"] == "2024-01-31"
        assert ev["metric_count"] == 5
        assert ev["latency_ms"] == 10
        assert ev["timestamp"] == "2024-01-15T10:00:00Z"


class TestMakeEvaluationCompletedEvent:
    def test_creates_event_ok_status(self) -> None:
        ev = make_evaluation_completed_event(
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
        assert ev["type"] == "covenant.metrics.evaluation.completed.v1"
        assert ev["status"] == "OK"
        assert ev["covenants_evaluated"] == 3
        assert ev["breaches_count"] == 0

    def test_creates_event_breach_status(self) -> None:
        ev = make_evaluation_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            status="BREACH",
            covenants_evaluated=3,
            breaches_count=2,
            latency_ms=15,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["status"] == "BREACH"
        assert ev["breaches_count"] == 2

    def test_creates_event_warning_status(self) -> None:
        ev = make_evaluation_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            status="WARNING",
            covenants_evaluated=3,
            breaches_count=0,
            latency_ms=15,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["status"] == "WARNING"


class TestMakePredictionCompletedEvent:
    def test_creates_event_low_risk(self) -> None:
        ev = make_prediction_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            risk_probability=0.15,
            risk_tier="LOW",
            model_version="v1.0.0",
            latency_ms=25,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["type"] == "covenant.metrics.prediction.completed.v1"
        assert ev["risk_probability"] == 0.15
        assert ev["risk_tier"] == "LOW"
        assert ev["model_version"] == "v1.0.0"

    def test_creates_event_critical_risk(self) -> None:
        ev = make_prediction_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            risk_probability=0.92,
            risk_tier="CRITICAL",
            model_version="v1.0.0",
            latency_ms=25,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["risk_tier"] == "CRITICAL"
        assert ev["risk_probability"] == 0.92


class TestMakeAlertTriggeredEvent:
    def test_creates_event_breach(self) -> None:
        ev = make_alert_triggered_event(
            event_id="e1",
            deal_id="d1",
            alert_type="breach",
            severity="critical",
            risk_probability=0.95,
            message="Debt covenant breached",
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["type"] == "covenant.metrics.alert.triggered.v1"
        assert ev["alert_type"] == "breach"
        assert ev["severity"] == "critical"
        assert ev["risk_probability"] == 0.95
        assert ev["message"] == "Debt covenant breached"

    def test_creates_event_high_risk(self) -> None:
        ev = make_alert_triggered_event(
            event_id="e1",
            deal_id="d1",
            alert_type="high_risk",
            severity="warning",
            risk_probability=0.85,
            message="Risk elevated",
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["alert_type"] == "high_risk"
        assert ev["severity"] == "warning"


class TestMakeRetrainTriggeredEvent:
    def test_creates_event_drift(self) -> None:
        ev = make_retrain_triggered_event(
            event_id="e1",
            trigger_type="drift",
            current_auc=0.72,
            threshold_auc=0.75,
            samples_since_train=5000,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["type"] == "covenant.metrics.retrain.triggered.v1"
        assert ev["trigger_type"] == "drift"
        assert ev["current_auc"] == 0.72
        assert ev["threshold_auc"] == 0.75
        assert ev["samples_since_train"] == 5000

    def test_creates_event_data_volume(self) -> None:
        ev = make_retrain_triggered_event(
            event_id="e1",
            trigger_type="data_volume",
            current_auc=0.80,
            threshold_auc=0.75,
            samples_since_train=10000,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["trigger_type"] == "data_volume"

    def test_creates_event_scheduled(self) -> None:
        ev = make_retrain_triggered_event(
            event_id="e1",
            trigger_type="scheduled",
            current_auc=0.82,
            threshold_auc=0.75,
            samples_since_train=3000,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["trigger_type"] == "scheduled"


class TestMakeStreamLagEvent:
    def test_creates_event(self) -> None:
        ev = make_stream_lag_event(
            event_id="e1",
            topic="covenant.measurements.v1",
            partition=0,
            lag_messages=100,
            lag_ms=500,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert ev["type"] == "covenant.metrics.stream.lag.v1"
        assert ev["topic"] == "covenant.measurements.v1"
        assert ev["partition"] == 0
        assert ev["lag_messages"] == 100
        assert ev["lag_ms"] == 500


class TestEncodeDecodeRoundtrip:
    def test_measurement_received_roundtrip(self) -> None:
        ev = make_measurement_received_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            metric_count=5,
            latency_ms=10,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert decoded["type"] == "covenant.metrics.measurement.received.v1"
        assert decoded["event_id"] == "e1"
        assert is_measurement_received(decoded)
        measurement_ev: MeasurementReceivedV1 = decoded
        assert measurement_ev["metric_count"] == 5

    def test_evaluation_completed_roundtrip(self) -> None:
        ev = make_evaluation_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            status="BREACH",
            covenants_evaluated=3,
            breaches_count=1,
            latency_ms=15,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_evaluation_completed(decoded)
        eval_ev: EvaluationCompletedV1 = decoded
        assert eval_ev["status"] == "BREACH"

    def test_evaluation_ok_status_roundtrip(self) -> None:
        ev = make_evaluation_completed_event(
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
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_evaluation_completed(decoded)
        eval_ev: EvaluationCompletedV1 = decoded
        assert eval_ev["status"] == "OK"

    def test_evaluation_warning_status_roundtrip(self) -> None:
        ev = make_evaluation_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            status="WARNING",
            covenants_evaluated=3,
            breaches_count=0,
            latency_ms=15,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_evaluation_completed(decoded)
        eval_ev: EvaluationCompletedV1 = decoded
        assert eval_ev["status"] == "WARNING"

    def test_prediction_completed_roundtrip(self) -> None:
        ev = make_prediction_completed_event(
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
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_prediction_completed(decoded)
        pred_ev: PredictionCompletedV1 = decoded
        assert pred_ev["risk_tier"] == "HIGH"

    def test_prediction_low_risk_tier_roundtrip(self) -> None:
        ev = make_prediction_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            risk_probability=0.15,
            risk_tier="LOW",
            model_version="v1.0.0",
            latency_ms=25,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_prediction_completed(decoded)
        pred_ev: PredictionCompletedV1 = decoded
        assert pred_ev["risk_tier"] == "LOW"

    def test_prediction_medium_risk_tier_roundtrip(self) -> None:
        ev = make_prediction_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            risk_probability=0.45,
            risk_tier="MEDIUM",
            model_version="v1.0.0",
            latency_ms=25,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_prediction_completed(decoded)
        pred_ev: PredictionCompletedV1 = decoded
        assert pred_ev["risk_tier"] == "MEDIUM"

    def test_prediction_critical_risk_tier_roundtrip(self) -> None:
        ev = make_prediction_completed_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            risk_probability=0.95,
            risk_tier="CRITICAL",
            model_version="v1.0.0",
            latency_ms=25,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_prediction_completed(decoded)
        pred_ev: PredictionCompletedV1 = decoded
        assert pred_ev["risk_tier"] == "CRITICAL"

    def test_alert_triggered_roundtrip(self) -> None:
        ev = make_alert_triggered_event(
            event_id="e1",
            deal_id="d1",
            alert_type="breach",
            severity="critical",
            risk_probability=0.95,
            message="Alert!",
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_alert_triggered(decoded)
        alert_ev: AlertTriggeredV1 = decoded
        assert alert_ev["severity"] == "critical"

    def test_alert_high_risk_type_roundtrip(self) -> None:
        ev = make_alert_triggered_event(
            event_id="e1",
            deal_id="d1",
            alert_type="high_risk",
            severity="critical",
            risk_probability=0.88,
            message="High risk!",
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_alert_triggered(decoded)
        alert_ev: AlertTriggeredV1 = decoded
        assert alert_ev["alert_type"] == "high_risk"

    def test_alert_warning_severity_roundtrip(self) -> None:
        ev = make_alert_triggered_event(
            event_id="e1",
            deal_id="d1",
            alert_type="breach",
            severity="warning",
            risk_probability=0.75,
            message="Warning!",
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_alert_triggered(decoded)
        alert_ev: AlertTriggeredV1 = decoded
        assert alert_ev["severity"] == "warning"

    def test_retrain_triggered_roundtrip(self) -> None:
        ev = make_retrain_triggered_event(
            event_id="e1",
            trigger_type="drift",
            current_auc=0.72,
            threshold_auc=0.75,
            samples_since_train=5000,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_retrain_triggered(decoded)
        retrain_ev: RetrainTriggeredV1 = decoded
        assert retrain_ev["trigger_type"] == "drift"

    def test_retrain_data_volume_trigger_roundtrip(self) -> None:
        ev = make_retrain_triggered_event(
            event_id="e1",
            trigger_type="data_volume",
            current_auc=0.80,
            threshold_auc=0.75,
            samples_since_train=10000,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_retrain_triggered(decoded)
        retrain_ev: RetrainTriggeredV1 = decoded
        assert retrain_ev["trigger_type"] == "data_volume"

    def test_retrain_scheduled_trigger_roundtrip(self) -> None:
        ev = make_retrain_triggered_event(
            event_id="e1",
            trigger_type="scheduled",
            current_auc=0.82,
            threshold_auc=0.75,
            samples_since_train=3000,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_retrain_triggered(decoded)
        retrain_ev: RetrainTriggeredV1 = decoded
        assert retrain_ev["trigger_type"] == "scheduled"

    def test_stream_lag_roundtrip(self) -> None:
        ev = make_stream_lag_event(
            event_id="e1",
            topic="covenant.measurements.v1",
            partition=0,
            lag_messages=100,
            lag_ms=500,
            timestamp="2024-01-15T10:00:00Z",
        )
        encoded = encode_covenant_metrics_event(ev)
        decoded = decode_covenant_metrics_event(encoded)
        assert is_stream_lag(decoded)
        lag_ev: StreamLagV1 = decoded
        assert lag_ev["lag_messages"] == 100


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
        with pytest.raises(JSONTypeError, match="Invalid risk tier"):
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


class TestTypeGuards:
    def test_is_measurement_received_true(self) -> None:
        ev: CovenantMetricsEventV1 = make_measurement_received_event(
            event_id="e1",
            deal_id="d1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            metric_count=5,
            latency_ms=10,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_measurement_received(ev)
        assert not is_evaluation_completed(ev)
        assert not is_prediction_completed(ev)
        assert not is_alert_triggered(ev)
        assert not is_retrain_triggered(ev)
        assert not is_stream_lag(ev)

    def test_is_evaluation_completed_true(self) -> None:
        ev: CovenantMetricsEventV1 = make_evaluation_completed_event(
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
        assert is_evaluation_completed(ev)
        assert not is_measurement_received(ev)

    def test_is_prediction_completed_true(self) -> None:
        ev: CovenantMetricsEventV1 = make_prediction_completed_event(
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
        assert is_prediction_completed(ev)
        assert not is_measurement_received(ev)

    def test_is_alert_triggered_true(self) -> None:
        ev: CovenantMetricsEventV1 = make_alert_triggered_event(
            event_id="e1",
            deal_id="d1",
            alert_type="breach",
            severity="critical",
            risk_probability=0.95,
            message="Alert!",
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_alert_triggered(ev)
        assert not is_measurement_received(ev)

    def test_is_retrain_triggered_true(self) -> None:
        ev: CovenantMetricsEventV1 = make_retrain_triggered_event(
            event_id="e1",
            trigger_type="drift",
            current_auc=0.72,
            threshold_auc=0.75,
            samples_since_train=5000,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_retrain_triggered(ev)
        assert not is_measurement_received(ev)

    def test_is_stream_lag_true(self) -> None:
        ev: CovenantMetricsEventV1 = make_stream_lag_event(
            event_id="e1",
            topic="covenant.measurements.v1",
            partition=0,
            lag_messages=100,
            lag_ms=500,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert is_stream_lag(ev)
        assert not is_measurement_received(ev)


class TestIntCoercionToFloat:
    def test_prediction_int_to_float(self) -> None:
        payload = """{
            "type": "covenant.metrics.prediction.completed.v1",
            "event_id": "e1",
            "deal_id": "d1",
            "period_start": "2024-01-01",
            "period_end": "2024-01-31",
            "risk_probability": 1,
            "risk_tier": "CRITICAL",
            "model_version": "v1.0.0",
            "latency_ms": 25,
            "timestamp": "2024-01-15T10:00:00Z"
        }"""
        decoded = decode_covenant_metrics_event(payload)
        assert is_prediction_completed(decoded)
        pred_ev: PredictionCompletedV1 = decoded
        assert type(pred_ev["risk_probability"]) is float

    def test_alert_int_to_float(self) -> None:
        payload = """{
            "type": "covenant.metrics.alert.triggered.v1",
            "event_id": "e1",
            "deal_id": "d1",
            "alert_type": "breach",
            "severity": "critical",
            "risk_probability": 1,
            "message": "Alert!",
            "timestamp": "2024-01-15T10:00:00Z"
        }"""
        decoded = decode_covenant_metrics_event(payload)
        assert is_alert_triggered(decoded)
        alert_ev: AlertTriggeredV1 = decoded
        assert type(alert_ev["risk_probability"]) is float

    def test_retrain_int_to_float(self) -> None:
        payload = """{
            "type": "covenant.metrics.retrain.triggered.v1",
            "event_id": "e1",
            "trigger_type": "drift",
            "current_auc": 1,
            "threshold_auc": 1,
            "samples_since_train": 5000,
            "timestamp": "2024-01-15T10:00:00Z"
        }"""
        decoded = decode_covenant_metrics_event(payload)
        assert is_retrain_triggered(decoded)
        retrain_ev: RetrainTriggeredV1 = decoded
        assert type(retrain_ev["current_auc"]) is float
        assert type(retrain_ev["threshold_auc"]) is float


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
        from platform_core.covenant_metrics_events import JobStartedV1

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
        from platform_core.covenant_metrics_events import JobCompletedV1

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
        from platform_core.covenant_metrics_events import JobFailedV1

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
        from platform_core.covenant_metrics_events import DEFAULT_COVENANT_EVENTS_CHANNEL

        assert DEFAULT_COVENANT_EVENTS_CHANNEL == "covenant:events"
