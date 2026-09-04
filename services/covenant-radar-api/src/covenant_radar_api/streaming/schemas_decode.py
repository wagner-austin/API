"""Kafka event decoding and narrowing."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeGuard

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)
from platform_core.risk_tiers import as_risk_tier

from covenant_radar_api.streaming.schemas import (
    AlertEventV1,
    AlertSeverity,
    AlertType,
    EvaluationStatus,
    KafkaEventV1,
    MeasurementEventV1,
    PredictionEventV1,
)


def _parse_evaluation_status(raw: str) -> EvaluationStatus:
    """Parse evaluation status from string.

    Args:
        raw: Raw string value.

    Returns:
        Validated EvaluationStatus literal.

    Raises:
        JSONTypeError: If value is not valid.
    """
    if raw == "OK":
        return "OK"
    if raw == "BREACH":
        return "BREACH"
    if raw == "WARNING":
        return "WARNING"
    raise JSONTypeError(f"Invalid evaluation status '{raw}'")


def _parse_alert_type(raw: str) -> AlertType:
    """Parse alert type from string.

    Args:
        raw: Raw string value.

    Returns:
        Validated AlertType literal.

    Raises:
        JSONTypeError: If value is not valid.
    """
    if raw == "breach":
        return "breach"
    if raw == "high_risk":
        return "high_risk"
    raise JSONTypeError(f"Invalid alert type '{raw}'")


def _parse_alert_severity(raw: str) -> AlertSeverity:
    """Parse alert severity from string.

    Args:
        raw: Raw string value.

    Returns:
        Validated AlertSeverity literal.

    Raises:
        JSONTypeError: If value is not valid.
    """
    if raw == "warning":
        return "warning"
    if raw == "critical":
        return "critical"
    raise JSONTypeError(f"Invalid alert severity '{raw}'")


# =============================================================================
# Decoder Functions
# =============================================================================


def _decode_measurement_event(decoded: JSONObject, event_id: str) -> MeasurementEventV1:
    """Decode a measurement event from JSON object.

    Args:
        decoded: Parsed JSON object.
        event_id: Already-extracted event ID.

    Returns:
        Validated MeasurementEventV1.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    deal_id = require_str(decoded, "deal_id")
    period_start = require_str(decoded, "period_start")
    period_end = require_str(decoded, "period_end")
    metric_name = require_str(decoded, "metric_name")
    metric_value = require_float(decoded, "metric_value")
    timestamp = require_str(decoded, "timestamp")
    return {
        "type": "covenant.measurement.v1",
        "event_id": event_id,
        "deal_id": deal_id,
        "period_start": period_start,
        "period_end": period_end,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "timestamp": timestamp,
    }


def _decode_prediction_event(decoded: JSONObject, event_id: str) -> PredictionEventV1:
    """Decode a prediction event from JSON object.

    Args:
        decoded: Parsed JSON object.
        event_id: Already-extracted event ID.

    Returns:
        Validated PredictionEventV1.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    deal_id = require_str(decoded, "deal_id")
    period_start = require_str(decoded, "period_start")
    period_end = require_str(decoded, "period_end")
    evaluation_status_raw = require_str(decoded, "evaluation_status")
    evaluation_status = _parse_evaluation_status(evaluation_status_raw)
    covenants_evaluated = require_int(decoded, "covenants_evaluated")
    breaches_count = require_int(decoded, "breaches_count")
    risk_probability = require_float(decoded, "risk_probability")
    risk_tier_raw = require_str(decoded, "risk_tier")
    risk_tier = as_risk_tier(risk_tier_raw, "risk_tier")
    model_version = require_str(decoded, "model_version")
    evaluation_latency_ms = require_int(decoded, "evaluation_latency_ms")
    prediction_latency_ms = require_int(decoded, "prediction_latency_ms")
    processed_at = require_str(decoded, "processed_at")
    return {
        "type": "covenant.prediction.v1",
        "event_id": event_id,
        "deal_id": deal_id,
        "period_start": period_start,
        "period_end": period_end,
        "evaluation_status": evaluation_status,
        "covenants_evaluated": covenants_evaluated,
        "breaches_count": breaches_count,
        "risk_probability": risk_probability,
        "risk_tier": risk_tier,
        "model_version": model_version,
        "evaluation_latency_ms": evaluation_latency_ms,
        "prediction_latency_ms": prediction_latency_ms,
        "processed_at": processed_at,
    }


def _decode_alert_event(decoded: JSONObject, event_id: str) -> AlertEventV1:
    """Decode an alert event from JSON object.

    Args:
        decoded: Parsed JSON object.
        event_id: Already-extracted event ID.

    Returns:
        Validated AlertEventV1.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    deal_id = require_str(decoded, "deal_id")
    alert_type_raw = require_str(decoded, "alert_type")
    alert_type = _parse_alert_type(alert_type_raw)
    severity_raw = require_str(decoded, "severity")
    severity = _parse_alert_severity(severity_raw)
    risk_probability = require_float(decoded, "risk_probability")
    gemini_summary = require_str(decoded, "gemini_summary")
    triggered_at = require_str(decoded, "triggered_at")
    return {
        "type": "covenant.alert.v1",
        "event_id": event_id,
        "deal_id": deal_id,
        "alert_type": alert_type,
        "severity": severity,
        "risk_probability": risk_probability,
        "gemini_summary": gemini_summary,
        "triggered_at": triggered_at,
    }


# Decoder dispatch table
_DECODERS: dict[str, Callable[[JSONObject, str], KafkaEventV1]] = {
    "covenant.measurement.v1": _decode_measurement_event,
    "covenant.prediction.v1": _decode_prediction_event,
    "covenant.alert.v1": _decode_alert_event,
}


def decode_measurement_event(payload: str) -> MeasurementEventV1:
    """Parse and validate a measurement event from JSON string.

    Args:
        payload: JSON string to parse.

    Returns:
        Validated MeasurementEventV1.

    Raises:
        JSONTypeError: If payload is not a valid measurement event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))
    type_raw = require_str(decoded, "type")
    if type_raw != "covenant.measurement.v1":
        raise JSONTypeError(f"Expected measurement event, got type '{type_raw}'")
    event_id = require_str(decoded, "event_id")
    return _decode_measurement_event(decoded, event_id)


def decode_prediction_event(payload: str) -> PredictionEventV1:
    """Parse and validate a prediction event from JSON string.

    Args:
        payload: JSON string to parse.

    Returns:
        Validated PredictionEventV1.

    Raises:
        JSONTypeError: If payload is not a valid prediction event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))
    type_raw = require_str(decoded, "type")
    if type_raw != "covenant.prediction.v1":
        raise JSONTypeError(f"Expected prediction event, got type '{type_raw}'")
    event_id = require_str(decoded, "event_id")
    return _decode_prediction_event(decoded, event_id)


def decode_alert_event(payload: str) -> AlertEventV1:
    """Parse and validate an alert event from JSON string.

    Args:
        payload: JSON string to parse.

    Returns:
        Validated AlertEventV1.

    Raises:
        JSONTypeError: If payload is not a valid alert event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))
    type_raw = require_str(decoded, "type")
    if type_raw != "covenant.alert.v1":
        raise JSONTypeError(f"Expected alert event, got type '{type_raw}'")
    event_id = require_str(decoded, "event_id")
    return _decode_alert_event(decoded, event_id)


def decode_kafka_event(payload: str) -> KafkaEventV1:
    """Parse and validate any Kafka event from JSON string.

    Automatically detects event type from 'type' field and dispatches
    to the appropriate decoder.

    Args:
        payload: JSON string to parse.

    Returns:
        Validated KafkaEventV1 (one of the three event types).

    Raises:
        JSONTypeError: If payload is not a valid Kafka event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))
    type_raw = require_str(decoded, "type")
    event_id = require_str(decoded, "event_id")

    decoder = _DECODERS.get(type_raw)
    if decoder is None:
        raise JSONTypeError(f"Unknown Kafka event type: '{type_raw}'")
    return decoder(decoded, event_id)


# =============================================================================
# TypeGuard Functions
# =============================================================================


def is_measurement_event(event: KafkaEventV1) -> TypeGuard[MeasurementEventV1]:
    """Check if event is a measurement event.

    Args:
        event: Any KafkaEventV1 to check.

    Returns:
        True if event is MeasurementEventV1.
    """
    return event.get("type") == "covenant.measurement.v1"


def is_prediction_event(event: KafkaEventV1) -> TypeGuard[PredictionEventV1]:
    """Check if event is a prediction event.

    Args:
        event: Any KafkaEventV1 to check.

    Returns:
        True if event is PredictionEventV1.
    """
    return event.get("type") == "covenant.prediction.v1"


def is_alert_event(event: KafkaEventV1) -> TypeGuard[AlertEventV1]:
    """Check if event is an alert event.

    Args:
        event: Any KafkaEventV1 to check.

    Returns:
        True if event is AlertEventV1.
    """
    return event.get("type") == "covenant.alert.v1"


__all__ = [
    "decode_alert_event",
    "decode_kafka_event",
    "decode_measurement_event",
    "decode_prediction_event",
    "is_alert_event",
    "is_measurement_event",
    "is_prediction_event",
]
