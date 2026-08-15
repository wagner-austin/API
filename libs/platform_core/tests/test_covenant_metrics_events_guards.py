"""Tests for covenant metrics events: TypeGuards."""

from __future__ import annotations

from platform_core.covenant_metrics_events import (
    AlertTriggeredV1,
    CovenantMetricsEventV1,
    PredictionCompletedV1,
    RetrainTriggeredV1,
    decode_covenant_metrics_event,
    is_alert_triggered,
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
