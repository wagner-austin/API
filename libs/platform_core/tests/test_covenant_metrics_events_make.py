"""Tests for covenant metrics events: MakeMeasurementReceivedEvent."""

from __future__ import annotations

from platform_core.covenant_metrics_events import (
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
