"""Tests for covenant metrics events: EncodeDecodeRoundtrip."""

from __future__ import annotations

from platform_core.covenant_metrics_events import (
    AlertTriggeredV1,
    EvaluationCompletedV1,
    MeasurementReceivedV1,
    PredictionCompletedV1,
    RetrainTriggeredV1,
    StreamLagV1,
    decode_covenant_metrics_event,
    encode_covenant_metrics_event,
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
