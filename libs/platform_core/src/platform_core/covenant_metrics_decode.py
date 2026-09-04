"""Decoding and narrowing for covenant events (metrics + job lifecycle).

The event shapes and their factory functions live in
:mod:`platform_core.covenant_metrics_events`; this module owns the wire
side the consumers use: payload decoding, the TypeGuard narrowers, and
the combined covenant channel type.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeGuard

from platform_core.covenant_metrics_events import (
    AlertSeverity,
    AlertTriggeredV1,
    AlertType,
    CovenantMetricsEventV1,
    EvaluationCompletedV1,
    MeasurementReceivedV1,
    PredictionCompletedV1,
    RetrainTriggeredV1,
    RetrainTriggerType,
    StreamLagV1,
)
from platform_core.job_events import (
    JobCompletedV1,
    JobEventV1,
    JobFailedV1,
    JobStartedV1,
    default_events_channel,
)
from platform_core.risk_tiers import require_risk_tier

from .json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)


def _decode_measurement_received_event(
    decoded: JSONObject,
    event_id: str,
) -> MeasurementReceivedV1:
    deal_id = require_str(decoded, "deal_id")
    period_start = require_str(decoded, "period_start")
    period_end = require_str(decoded, "period_end")
    metric_count = require_int(decoded, "metric_count")
    latency_ms = require_int(decoded, "latency_ms")
    timestamp = require_str(decoded, "timestamp")
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


def _parse_evaluation_status(raw: str) -> Literal["OK", "BREACH", "WARNING"]:
    if raw == "OK":
        return "OK"
    if raw == "BREACH":
        return "BREACH"
    if raw == "WARNING":
        return "WARNING"
    raise JSONTypeError(f"Invalid evaluation status '{raw}'")


def _decode_evaluation_completed_event(
    decoded: JSONObject,
    event_id: str,
) -> EvaluationCompletedV1:
    deal_id = require_str(decoded, "deal_id")
    period_start = require_str(decoded, "period_start")
    period_end = require_str(decoded, "period_end")
    status_raw = require_str(decoded, "status")
    status = _parse_evaluation_status(status_raw)
    covenants_evaluated = require_int(decoded, "covenants_evaluated")
    breaches_count = require_int(decoded, "breaches_count")
    latency_ms = require_int(decoded, "latency_ms")
    timestamp = require_str(decoded, "timestamp")
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


def _decode_prediction_completed_event(
    decoded: JSONObject,
    event_id: str,
) -> PredictionCompletedV1:
    deal_id = require_str(decoded, "deal_id")
    period_start = require_str(decoded, "period_start")
    period_end = require_str(decoded, "period_end")
    risk_probability = require_float(decoded, "risk_probability")
    risk_tier = require_risk_tier(decoded, "risk_tier")
    model_version = require_str(decoded, "model_version")
    latency_ms = require_int(decoded, "latency_ms")
    timestamp = require_str(decoded, "timestamp")
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


def _parse_alert_type(raw: str) -> AlertType:
    if raw == "breach":
        return "breach"
    if raw == "high_risk":
        return "high_risk"
    raise JSONTypeError(f"Invalid alert type '{raw}'")


def _parse_alert_severity(raw: str) -> AlertSeverity:
    if raw == "warning":
        return "warning"
    if raw == "critical":
        return "critical"
    raise JSONTypeError(f"Invalid alert severity '{raw}'")


def _decode_alert_triggered_event(
    decoded: JSONObject,
    event_id: str,
) -> AlertTriggeredV1:
    deal_id = require_str(decoded, "deal_id")
    alert_type_raw = require_str(decoded, "alert_type")
    alert_type = _parse_alert_type(alert_type_raw)
    severity_raw = require_str(decoded, "severity")
    severity = _parse_alert_severity(severity_raw)
    risk_probability = require_float(decoded, "risk_probability")
    message = require_str(decoded, "message")
    timestamp = require_str(decoded, "timestamp")
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


def _parse_retrain_trigger_type(raw: str) -> RetrainTriggerType:
    if raw == "drift":
        return "drift"
    if raw == "data_volume":
        return "data_volume"
    if raw == "scheduled":
        return "scheduled"
    raise JSONTypeError(f"Invalid retrain trigger type '{raw}'")


def _decode_retrain_triggered_event(
    decoded: JSONObject,
    event_id: str,
) -> RetrainTriggeredV1:
    trigger_type_raw = require_str(decoded, "trigger_type")
    trigger_type = _parse_retrain_trigger_type(trigger_type_raw)
    current_auc = require_float(decoded, "current_auc")
    threshold_auc = require_float(decoded, "threshold_auc")
    samples_since_train = require_int(decoded, "samples_since_train")
    timestamp = require_str(decoded, "timestamp")
    return {
        "type": "covenant.metrics.retrain.triggered.v1",
        "event_id": event_id,
        "trigger_type": trigger_type,
        "current_auc": current_auc,
        "threshold_auc": threshold_auc,
        "samples_since_train": samples_since_train,
        "timestamp": timestamp,
    }


def _decode_stream_lag_event(
    decoded: JSONObject,
    event_id: str,
) -> StreamLagV1:
    topic = require_str(decoded, "topic")
    partition = require_int(decoded, "partition")
    lag_messages = require_int(decoded, "lag_messages")
    lag_ms = require_int(decoded, "lag_ms")
    timestamp = require_str(decoded, "timestamp")
    return {
        "type": "covenant.metrics.stream.lag.v1",
        "event_id": event_id,
        "topic": topic,
        "partition": partition,
        "lag_messages": lag_messages,
        "lag_ms": lag_ms,
        "timestamp": timestamp,
    }


_DECODERS: dict[
    str,
    Callable[[JSONObject, str], CovenantMetricsEventV1],
] = {
    "covenant.metrics.measurement.received.v1": _decode_measurement_received_event,
    "covenant.metrics.evaluation.completed.v1": _decode_evaluation_completed_event,
    "covenant.metrics.prediction.completed.v1": _decode_prediction_completed_event,
    "covenant.metrics.alert.triggered.v1": _decode_alert_triggered_event,
    "covenant.metrics.retrain.triggered.v1": _decode_retrain_triggered_event,
    "covenant.metrics.stream.lag.v1": _decode_stream_lag_event,
}


def decode_covenant_metrics_event(payload: str) -> CovenantMetricsEventV1:
    """Parse and validate a serialized covenant metrics event.

    Raises:
        JSONTypeError: if the payload is not a well-formed covenant metrics event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))

    type_raw = require_str(decoded, "type")
    event_id = require_str(decoded, "event_id")

    decoder = _DECODERS.get(type_raw)
    if decoder is None:
        raise JSONTypeError(f"Unknown covenant metrics event type: '{type_raw}'")
    return decoder(decoded, event_id)


# -----------------------------------------------------------------------------
# TypeGuard functions for type narrowing
# -----------------------------------------------------------------------------


def is_measurement_received(ev: CovenantMetricsEventV1) -> TypeGuard[MeasurementReceivedV1]:
    """Check if the event is a measurement received event."""
    return ev.get("type") == "covenant.metrics.measurement.received.v1"


def is_evaluation_completed(ev: CovenantMetricsEventV1) -> TypeGuard[EvaluationCompletedV1]:
    """Check if the event is an evaluation completed event."""
    return ev.get("type") == "covenant.metrics.evaluation.completed.v1"


def is_prediction_completed(ev: CovenantMetricsEventV1) -> TypeGuard[PredictionCompletedV1]:
    """Check if the event is a prediction completed event."""
    return ev.get("type") == "covenant.metrics.prediction.completed.v1"


def is_alert_triggered(ev: CovenantMetricsEventV1) -> TypeGuard[AlertTriggeredV1]:
    """Check if the event is an alert triggered event."""
    return ev.get("type") == "covenant.metrics.alert.triggered.v1"


def is_retrain_triggered(ev: CovenantMetricsEventV1) -> TypeGuard[RetrainTriggeredV1]:
    """Check if the event is a retrain triggered event."""
    return ev.get("type") == "covenant.metrics.retrain.triggered.v1"


def is_stream_lag(ev: CovenantMetricsEventV1) -> TypeGuard[StreamLagV1]:
    """Check if the event is a stream lag event."""
    return ev.get("type") == "covenant.metrics.stream.lag.v1"


# -----------------------------------------------------------------------------
# Combined event type for covenant channel (job lifecycle + domain metrics)
# -----------------------------------------------------------------------------

# Combined event type for covenant channel
CovenantEventV1 = JobEventV1 | CovenantMetricsEventV1

# Default channel for covenant events
DEFAULT_COVENANT_EVENTS_CHANNEL: str = default_events_channel("covenant")


def _decode_job_started(decoded: JSONObject, job_id: str, user_id: int) -> JobStartedV1:
    """Decode a started event."""
    queue = require_str(decoded, "queue")
    return {
        "type": "covenant.job.started.v1",
        "domain": "covenant",
        "job_id": job_id,
        "user_id": user_id,
        "queue": queue,
    }


def _decode_job_completed(decoded: JSONObject, job_id: str, user_id: int) -> JobCompletedV1:
    """Decode a completed event."""
    result_id = require_str(decoded, "result_id")
    result_bytes = require_int(decoded, "result_bytes")
    return {
        "type": "covenant.job.completed.v1",
        "domain": "covenant",
        "job_id": job_id,
        "user_id": user_id,
        "result_id": result_id,
        "result_bytes": result_bytes,
    }


def _decode_job_failed(decoded: JSONObject, job_id: str, user_id: int) -> JobFailedV1:
    """Decode a failed event."""
    error_kind_raw = require_str(decoded, "error_kind")
    message = require_str(decoded, "message")
    if error_kind_raw == "user":
        error_kind: Literal["user", "system"] = "user"
    elif error_kind_raw == "system":
        error_kind = "system"
    else:
        raise JSONTypeError(f"Invalid error_kind '{error_kind_raw}' in failed event")
    return {
        "type": "covenant.job.failed.v1",
        "domain": "covenant",
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": error_kind,
        "message": message,
    }


_JOB_DECODERS: dict[str, Callable[[JSONObject, str, int], JobEventV1]] = {
    "covenant.job.started.v1": _decode_job_started,
    "covenant.job.completed.v1": _decode_job_completed,
    "covenant.job.failed.v1": _decode_job_failed,
}


def decode_covenant_event(payload: str) -> CovenantEventV1:
    """Parse and validate any event from the covenant channel.

    Handles both job lifecycle events (covenant.job.*.v1) and
    metrics events (covenant.metrics.*.v1).

    Raises:
        JSONTypeError: if the payload is not a well-formed covenant event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))

    type_raw = require_str(decoded, "type")

    # Check if it's a job lifecycle event (covenant.job.*.v1)
    if type_raw.startswith("covenant.job."):
        job_id = require_str(decoded, "job_id")
        user_id = require_int(decoded, "user_id")
        domain = require_str(decoded, "domain")
        if domain != "covenant":
            raise JSONTypeError(f"Domain mismatch: expected 'covenant', got '{domain}'")
        job_decoder = _JOB_DECODERS.get(type_raw)
        if job_decoder is None:
            raise JSONTypeError(f"Unknown covenant job event type: '{type_raw}'")
        return job_decoder(decoded, job_id, user_id)

    # Check if it's a metrics event (covenant.metrics.*.v1)
    if type_raw.startswith("covenant.metrics."):
        event_id = require_str(decoded, "event_id")
        metrics_decoder = _DECODERS.get(type_raw)
        if metrics_decoder is None:
            raise JSONTypeError(f"Unknown covenant metrics event type: '{type_raw}'")
        return metrics_decoder(decoded, event_id)

    raise JSONTypeError(f"Unknown covenant event type: '{type_raw}'")


# TypeGuard helpers for combined event type narrowing
def is_covenant_job_started(ev: CovenantEventV1) -> TypeGuard[JobStartedV1]:
    """Check if a combined event is a job started event."""
    type_val = ev.get("type")
    return isinstance(type_val, str) and type_val == "covenant.job.started.v1"


def is_covenant_job_completed(ev: CovenantEventV1) -> TypeGuard[JobCompletedV1]:
    """Check if a combined event is a job completed event."""
    type_val = ev.get("type")
    return isinstance(type_val, str) and type_val == "covenant.job.completed.v1"


def is_covenant_job_failed(ev: CovenantEventV1) -> TypeGuard[JobFailedV1]:
    """Check if a combined event is a job failed event."""
    type_val = ev.get("type")
    return isinstance(type_val, str) and type_val == "covenant.job.failed.v1"


def is_covenant_measurement_received(ev: CovenantEventV1) -> TypeGuard[MeasurementReceivedV1]:
    """Check if a combined event is a measurement received event."""
    return ev.get("type") == "covenant.metrics.measurement.received.v1"


def is_covenant_evaluation_completed(ev: CovenantEventV1) -> TypeGuard[EvaluationCompletedV1]:
    """Check if a combined event is an evaluation completed event."""
    return ev.get("type") == "covenant.metrics.evaluation.completed.v1"


def is_covenant_prediction_completed(ev: CovenantEventV1) -> TypeGuard[PredictionCompletedV1]:
    """Check if a combined event is a prediction completed event."""
    return ev.get("type") == "covenant.metrics.prediction.completed.v1"


def is_covenant_alert_triggered(ev: CovenantEventV1) -> TypeGuard[AlertTriggeredV1]:
    """Check if a combined event is an alert triggered event."""
    return ev.get("type") == "covenant.metrics.alert.triggered.v1"


def is_covenant_retrain_triggered(ev: CovenantEventV1) -> TypeGuard[RetrainTriggeredV1]:
    """Check if a combined event is a retrain triggered event."""
    return ev.get("type") == "covenant.metrics.retrain.triggered.v1"


def is_covenant_stream_lag(ev: CovenantEventV1) -> TypeGuard[StreamLagV1]:
    """Check if a combined event is a stream lag event."""
    return ev.get("type") == "covenant.metrics.stream.lag.v1"


__all__ = [
    "DEFAULT_COVENANT_EVENTS_CHANNEL",
    "CovenantEventV1",
    "JobCompletedV1",
    "JobFailedV1",
    "JobStartedV1",
    "decode_covenant_event",
    "decode_covenant_metrics_event",
    "is_alert_triggered",
    "is_covenant_alert_triggered",
    "is_covenant_evaluation_completed",
    "is_covenant_job_completed",
    "is_covenant_job_failed",
    "is_covenant_job_started",
    "is_covenant_measurement_received",
    "is_covenant_prediction_completed",
    "is_covenant_retrain_triggered",
    "is_covenant_stream_lag",
    "is_evaluation_completed",
    "is_measurement_received",
    "is_prediction_completed",
    "is_retrain_triggered",
    "is_stream_lag",
]
