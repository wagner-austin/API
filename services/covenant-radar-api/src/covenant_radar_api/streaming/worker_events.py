"""Streaming worker event shapes, keys, and evaluation helpers."""

from __future__ import annotations

import time
import uuid
from typing import TypedDict

from covenant_domain import (
    CovenantResult,
)

from .schemas import (
    AlertEventV1,
    AlertSeverity,
    AlertType,
    EvaluationStatus,
    MeasurementEventV1,
    PredictionEventV1,
)


class WorkerConfig(TypedDict, total=True):
    """Configuration for streaming worker.

    Fields:
        model_version: Version string for the ML model.
        poll_timeout_seconds: Kafka poll timeout.
        alert_threshold: Risk probability threshold for alerts (default 0.8).
        commit_interval: Number of messages between commits.
        buffer_timeout_seconds: Max time to wait for measurements before processing.
        min_metrics_per_period: Minimum metrics needed before processing a period.
        tolerance_ratio_scaled: Covenant evaluation tolerance (10% = 100_000).
    """

    model_version: str
    poll_timeout_seconds: float
    alert_threshold: float
    commit_interval: int
    buffer_timeout_seconds: float
    min_metrics_per_period: int
    tolerance_ratio_scaled: int


def make_default_worker_config() -> WorkerConfig:
    """Create default worker configuration.

    Returns:
        WorkerConfig with sensible defaults.
    """
    return {
        "model_version": "v1.0.0",
        "poll_timeout_seconds": 1.0,
        "alert_threshold": 0.80,
        "commit_interval": 10,
        "buffer_timeout_seconds": 5.0,
        "min_metrics_per_period": 3,
        "tolerance_ratio_scaled": 100_000,  # 10% tolerance
    }


# =============================================================================
# Result Types
# =============================================================================


class ProcessingResult(TypedDict, total=True):
    """Result of processing a deal/period batch.

    Fields:
        prediction: The prediction event produced.
        alert: Optional alert event if risk exceeded threshold.
        evaluation_latency_ms: Time for deterministic evaluation.
        prediction_latency_ms: Time for ML inference.
    """

    prediction: PredictionEventV1
    alert: AlertEventV1 | None
    evaluation_latency_ms: int
    prediction_latency_ms: int


# =============================================================================
# Buffer Types
# =============================================================================


class BufferKey(TypedDict, total=True):
    """Key for measurement buffer."""

    deal_id: str
    period_start: str
    period_end: str


class BufferedPeriod(TypedDict, total=True):
    """Buffered measurements for a single period.

    Fields:
        metrics: Mapping of metric_name to metric_value.
        first_received_at: Timestamp when first metric arrived.
        message_count: Number of messages received.
        offsets: Kafka positions of the messages held here, as
            (topic, partition, offset). Retained so the worker can tell which
            offsets are still unprocessed and must not be committed yet.
    """

    metrics: dict[str, float]
    first_received_at: float
    message_count: int
    offsets: list[tuple[str, int, int]]


def _make_buffer_key(event: MeasurementEventV1) -> tuple[str, str, str]:
    """Create buffer key tuple from event."""
    return (event["deal_id"], event["period_start"], event["period_end"])


# =============================================================================
# Helper Functions
# =============================================================================


def _covenant_result_period_end_key(result: CovenantResult) -> str:
    """Extract period_end_iso for sorting covenant results.

    Args:
        result: Covenant result to extract key from.

    Returns:
        The period_end_iso string.
    """
    return result["period_end_iso"]


def _generate_event_id() -> str:
    """Generate a unique event ID."""
    return str(uuid.uuid4())


def _current_iso_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _determine_evaluation_status(
    results: tuple[CovenantResult, ...],
) -> EvaluationStatus:
    """Determine overall evaluation status from covenant results.

    Args:
        results: Tuple of covenant evaluation results.

    Returns:
        EvaluationStatus: OK, BREACH, or WARNING.
    """
    has_breach = False
    has_near_breach = False

    for result in results:
        status = result["status"]
        if status == "BREACH":
            has_breach = True
        elif status == "NEAR_BREACH":
            has_near_breach = True

    if has_breach:
        return "BREACH"
    if has_near_breach:
        return "WARNING"
    return "OK"


def _count_breaches(results: tuple[CovenantResult, ...]) -> int:
    """Count number of breached covenants."""
    return sum(1 for r in results if r["status"] == "BREACH")


def _determine_alert_severity(risk_probability: float) -> AlertSeverity:
    """Determine alert severity based on risk probability.

    Args:
        risk_probability: ML-predicted probability.

    Returns:
        AlertSeverity literal.
    """
    if risk_probability >= 0.90:
        return "critical"
    return "warning"


def _determine_alert_type(
    evaluation_status: EvaluationStatus,
) -> AlertType:
    """Determine alert type based on evaluation status.

    Args:
        evaluation_status: Deterministic evaluation result.

    Returns:
        AlertType literal.
    """
    if evaluation_status == "BREACH":
        return "breach"
    return "high_risk"


def _generate_alert_message(
    deal_id: str,
    deal_name: str,
    risk_probability: float,
    evaluation_status: EvaluationStatus,
    breaches_count: int,
) -> str:
    """Generate alert message text.

    Phase 5 will enhance this with Gemini-generated text.

    Args:
        deal_id: Deal identifier.
        deal_name: Human-readable deal name.
        risk_probability: ML-predicted probability.
        evaluation_status: Deterministic evaluation result.
        breaches_count: Number of covenant breaches.

    Returns:
        Human-readable alert message.
    """
    if evaluation_status == "BREACH":
        return (
            f"Deal '{deal_name}' ({deal_id}) has {breaches_count} covenant breach(es). "
            f"ML risk probability: {risk_probability:.1%}. Immediate review required."
        )
    return (
        f"Deal '{deal_name}' ({deal_id}) shows elevated risk at {risk_probability:.1%}. "
        f"No covenant breaches detected, but ML model indicates high probability of future issues."
    )


def _scale_metrics(metrics: dict[str, float]) -> dict[str, int]:
    """Scale float metrics to integer representation.

    Uses 1M scaling factor for consistency with domain model.

    Args:
        metrics: Mapping of metric_name to float value.

    Returns:
        Mapping of metric_name to scaled integer value.
    """
    return {name: int(value * 1_000_000) for name, value in metrics.items()}


# =============================================================================
# Streaming Worker
# =============================================================================
