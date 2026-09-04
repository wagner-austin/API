"""Covenant streaming metrics events for Covenant Radar API.

This module provides TypedDict definitions and encoder/decoder functions
for domain-specific metrics events published during streaming inference.

Lifecycle events (started, progress, completed, failed) are handled by
platform_workers.job_context via generic job_events.

Event types:
- covenant.metrics.measurement.received.v1 -> Measurement ingestion tracking
- covenant.metrics.evaluation.completed.v1 -> Covenant evaluation tracking
- covenant.metrics.prediction.completed.v1 -> Risk prediction tracking
- covenant.metrics.alert.triggered.v1 -> High-risk alert tracking
- covenant.metrics.retrain.triggered.v1 -> Model retrain trigger tracking
- covenant.metrics.stream.lag.v1 -> Kafka consumer lag tracking
"""

from __future__ import annotations

from typing import Literal, TypedDict

from .json_utils import (
    dump_json_str,
)

CovenantMetricsEventType = Literal[
    "covenant.metrics.measurement.received.v1",
    "covenant.metrics.evaluation.completed.v1",
    "covenant.metrics.prediction.completed.v1",
    "covenant.metrics.alert.triggered.v1",
    "covenant.metrics.retrain.triggered.v1",
    "covenant.metrics.stream.lag.v1",
]


class MeasurementReceivedV1(TypedDict):
    """Measurement ingestion event published when measurements are consumed."""

    type: Literal["covenant.metrics.measurement.received.v1"]
    event_id: str
    deal_id: str
    period_start: str
    period_end: str
    metric_count: int
    latency_ms: int
    timestamp: str


class EvaluationCompletedV1(TypedDict):
    """Covenant evaluation event published after deterministic evaluation."""

    type: Literal["covenant.metrics.evaluation.completed.v1"]
    event_id: str
    deal_id: str
    period_start: str
    period_end: str
    status: Literal["OK", "BREACH", "WARNING"]
    covenants_evaluated: int
    breaches_count: int
    latency_ms: int
    timestamp: str


class PredictionCompletedV1(TypedDict):
    """Risk prediction event published after ML inference."""

    type: Literal["covenant.metrics.prediction.completed.v1"]
    event_id: str
    deal_id: str
    period_start: str
    period_end: str
    risk_probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    model_version: str
    latency_ms: int
    timestamp: str


AlertSeverity = Literal["warning", "critical"]
AlertType = Literal["breach", "high_risk"]


class AlertTriggeredV1(TypedDict):
    """Alert event published when high-risk situation detected."""

    type: Literal["covenant.metrics.alert.triggered.v1"]
    event_id: str
    deal_id: str
    alert_type: AlertType
    severity: AlertSeverity
    risk_probability: float
    message: str
    timestamp: str


RetrainTriggerType = Literal["drift", "data_volume", "scheduled"]


class RetrainTriggeredV1(TypedDict):
    """Retrain trigger event published when model retraining is needed."""

    type: Literal["covenant.metrics.retrain.triggered.v1"]
    event_id: str
    trigger_type: RetrainTriggerType
    current_auc: float
    threshold_auc: float
    samples_since_train: int
    timestamp: str


class StreamLagV1(TypedDict):
    """Consumer lag event published for monitoring Kafka lag."""

    type: Literal["covenant.metrics.stream.lag.v1"]
    event_id: str
    topic: str
    partition: int
    lag_messages: int
    lag_ms: int
    timestamp: str


CovenantMetricsEventV1 = (
    MeasurementReceivedV1
    | EvaluationCompletedV1
    | PredictionCompletedV1
    | AlertTriggeredV1
    | RetrainTriggeredV1
    | StreamLagV1
)


def encode_covenant_metrics_event(event: CovenantMetricsEventV1) -> str:
    """Serialize a covenant metrics event to a compact JSON string."""
    return dump_json_str(event)


# -----------------------------------------------------------------------------
# Factory functions for creating events
# -----------------------------------------------------------------------------


def make_measurement_received_event(
    *,
    event_id: str,
    deal_id: str,
    period_start: str,
    period_end: str,
    metric_count: int,
    latency_ms: int,
    timestamp: str,
) -> MeasurementReceivedV1:
    """Create a measurement received event."""
    return {
        "type": "covenant.metrics.measurement.received.v1",
        "event_id": event_id,
        "deal_id": deal_id,
        "period_start": period_start,
        "period_end": period_end,
        "metric_count": metric_count,
        "latency_ms": latency_ms,
        "timestamp": timestamp,
    }


def make_evaluation_completed_event(
    *,
    event_id: str,
    deal_id: str,
    period_start: str,
    period_end: str,
    status: Literal["OK", "BREACH", "WARNING"],
    covenants_evaluated: int,
    breaches_count: int,
    latency_ms: int,
    timestamp: str,
) -> EvaluationCompletedV1:
    """Create an evaluation completed event."""
    return {
        "type": "covenant.metrics.evaluation.completed.v1",
        "event_id": event_id,
        "deal_id": deal_id,
        "period_start": period_start,
        "period_end": period_end,
        "status": status,
        "covenants_evaluated": covenants_evaluated,
        "breaches_count": breaches_count,
        "latency_ms": latency_ms,
        "timestamp": timestamp,
    }


def make_prediction_completed_event(
    *,
    event_id: str,
    deal_id: str,
    period_start: str,
    period_end: str,
    risk_probability: float,
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    model_version: str,
    latency_ms: int,
    timestamp: str,
) -> PredictionCompletedV1:
    """Create a prediction completed event."""
    return {
        "type": "covenant.metrics.prediction.completed.v1",
        "event_id": event_id,
        "deal_id": deal_id,
        "period_start": period_start,
        "period_end": period_end,
        "risk_probability": risk_probability,
        "risk_tier": risk_tier,
        "model_version": model_version,
        "latency_ms": latency_ms,
        "timestamp": timestamp,
    }


def make_alert_triggered_event(
    *,
    event_id: str,
    deal_id: str,
    alert_type: AlertType,
    severity: AlertSeverity,
    risk_probability: float,
    message: str,
    timestamp: str,
) -> AlertTriggeredV1:
    """Create an alert triggered event."""
    return {
        "type": "covenant.metrics.alert.triggered.v1",
        "event_id": event_id,
        "deal_id": deal_id,
        "alert_type": alert_type,
        "severity": severity,
        "risk_probability": risk_probability,
        "message": message,
        "timestamp": timestamp,
    }


def make_retrain_triggered_event(
    *,
    event_id: str,
    trigger_type: RetrainTriggerType,
    current_auc: float,
    threshold_auc: float,
    samples_since_train: int,
    timestamp: str,
) -> RetrainTriggeredV1:
    """Create a retrain triggered event."""
    return {
        "type": "covenant.metrics.retrain.triggered.v1",
        "event_id": event_id,
        "trigger_type": trigger_type,
        "current_auc": current_auc,
        "threshold_auc": threshold_auc,
        "samples_since_train": samples_since_train,
        "timestamp": timestamp,
    }


def make_stream_lag_event(
    *,
    event_id: str,
    topic: str,
    partition: int,
    lag_messages: int,
    lag_ms: int,
    timestamp: str,
) -> StreamLagV1:
    """Create a stream lag event."""
    return {
        "type": "covenant.metrics.stream.lag.v1",
        "event_id": event_id,
        "topic": topic,
        "partition": partition,
        "lag_messages": lag_messages,
        "lag_ms": lag_ms,
        "timestamp": timestamp,
    }


# -----------------------------------------------------------------------------
# Decoder functions
# -----------------------------------------------------------------------------


__all__ = [
    "AlertSeverity",
    "AlertTriggeredV1",
    "AlertType",
    "CovenantMetricsEventType",
    "CovenantMetricsEventV1",
    "EvaluationCompletedV1",
    "MeasurementReceivedV1",
    "PredictionCompletedV1",
    "RetrainTriggerType",
    "RetrainTriggeredV1",
    "StreamLagV1",
    "encode_covenant_metrics_event",
    "make_alert_triggered_event",
    "make_evaluation_completed_event",
    "make_measurement_received_event",
    "make_prediction_completed_event",
    "make_retrain_triggered_event",
    "make_stream_lag_event",
]
