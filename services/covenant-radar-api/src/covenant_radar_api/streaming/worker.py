"""Streaming inference worker for covenant breach prediction.

Consumes measurement events from Kafka, aggregates by deal/period,
runs covenant evaluation and ML prediction, publishes results.

Architecture:
1. Consume measurement events from Kafka
2. Buffer measurements by (deal_id, period_start, period_end)
3. When buffer has enough metrics or timeout, process the batch
4. Query database for deal/covenant data
5. Run deterministic covenant evaluation
6. Run ML prediction
7. Produce prediction/alert events to Kafka
8. Emit metrics to Datadog

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import TypedDict

from covenant_domain import (
    Covenant,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
    evaluate_all_covenants_for_period,
)
from covenant_domain.features import (
    LoanFeatures,
    RiskTier,
    classify_risk_tier,
    extract_features,
)
from covenant_ml.predictor import predict_probabilities
from covenant_ml.types import PredictorProtocol
from covenant_persistence import (
    CovenantRepository,
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)

from ..integrations.datadog.metrics import MetricsClient
from .consumer import StreamingConsumer
from .producer import StreamingProducer
from .schemas import (
    AlertEventV1,
    AlertSeverity,
    AlertType,
    EvaluationStatus,
    MeasurementEventV1,
    PredictionEventV1,
    make_alert_event,
    make_prediction_event,
)

# =============================================================================
# Configuration Types
# =============================================================================


class WorkerConfig(TypedDict, total=True):
    """Configuration for streaming worker.

    Fields:
        model_version: Version string for the ML model.
        batch_size: Max messages to poll per iteration.
        poll_timeout_seconds: Kafka poll timeout.
        alert_threshold: Risk probability threshold for alerts (default 0.8).
        commit_interval: Number of messages between commits.
        buffer_timeout_seconds: Max time to wait for measurements before processing.
        min_metrics_per_period: Minimum metrics needed before processing a period.
        tolerance_ratio_scaled: Covenant evaluation tolerance (10% = 100_000).
    """

    model_version: str
    batch_size: int
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
        "batch_size": 100,
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
    """

    metrics: dict[str, float]
    first_received_at: float
    message_count: int


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


class StreamingWorker:
    """Kafka streaming worker for real-time inference.

    Consumes measurement events, buffers by deal/period, runs evaluation
    and ML prediction, and produces prediction/alert events.

    The worker maintains an in-memory buffer of measurements grouped by
    (deal_id, period_start, period_end). When a buffer has enough metrics
    or times out, it processes the batch:

    1. Load deal and covenant data from database
    2. Run deterministic covenant evaluation
    3. Extract ML features
    4. Run ML prediction
    5. Produce prediction event to Kafka
    6. If high risk, produce alert event
    7. Emit metrics to Datadog

    Example:
        worker = StreamingWorker(
            consumer=consumer,
            producer=producer,
            metrics=metrics_client,
            model=model,
            deal_repo=deal_repo,
            covenant_repo=covenant_repo,
            measurement_repo=measurement_repo,
            result_repo=result_repo,
            sector_encoder={"Technology": 0},
            region_encoder={"North America": 0},
            config=make_default_worker_config(),
        )
        worker.run()
    """

    def __init__(
        self,
        consumer: StreamingConsumer,
        producer: StreamingProducer,
        metrics: MetricsClient,
        model: PredictorProtocol,
        deal_repo: DealRepository,
        covenant_repo: CovenantRepository,
        measurement_repo: MeasurementRepository,
        result_repo: CovenantResultRepository,
        sector_encoder: dict[str, int],
        region_encoder: dict[str, int],
        config: WorkerConfig,
    ) -> None:
        """Initialize the streaming worker.

        Args:
            consumer: Kafka consumer for measurements.
            producer: Kafka producer for predictions/alerts.
            metrics: Datadog metrics client.
            model: ML model for predictions.
            deal_repo: Repository for deal data.
            covenant_repo: Repository for covenant data.
            measurement_repo: Repository for historical measurements.
            result_repo: Repository for covenant results.
            sector_encoder: Sector to integer mapping.
            region_encoder: Region to integer mapping.
            config: Worker configuration.
        """
        self._consumer = consumer
        self._producer = producer
        self._metrics = metrics
        self._model = model
        self._deal_repo = deal_repo
        self._covenant_repo = covenant_repo
        self._measurement_repo = measurement_repo
        self._result_repo = result_repo
        self._sector_encoder = sector_encoder
        self._region_encoder = region_encoder
        self._config = config
        self._running = False
        self._messages_since_commit = 0

        # Buffer: (deal_id, period_start, period_end) -> BufferedPeriod
        self._buffer: dict[tuple[str, str, str], BufferedPeriod] = {}

    @property
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self._running

    @property
    def buffer_size(self) -> int:
        """Get number of periods currently buffered."""
        return len(self._buffer)

    def _add_to_buffer(self, event: MeasurementEventV1) -> None:
        """Add a measurement to the buffer.

        Args:
            event: Measurement event from Kafka.
        """
        key = _make_buffer_key(event)

        if key not in self._buffer:
            self._buffer[key] = {
                "metrics": {},
                "first_received_at": time.monotonic(),
                "message_count": 0,
            }

        buffered = self._buffer[key]
        buffered["metrics"][event["metric_name"]] = event["metric_value"]
        buffered["message_count"] += 1

    def _should_process_buffer(self, key: tuple[str, str, str]) -> bool:
        """Check if a buffered period should be processed.

        A period is ready for processing if:
        1. It has minimum required metrics, OR
        2. It has timed out

        Args:
            key: Buffer key (deal_id, period_start, period_end).

        Returns:
            True if buffer should be processed.
        """
        if key not in self._buffer:
            return False

        buffered = self._buffer[key]
        metric_count = len(buffered["metrics"])
        age_seconds = time.monotonic() - buffered["first_received_at"]

        # Process if we have enough metrics or buffer has timed out
        has_enough_metrics = metric_count >= self._config["min_metrics_per_period"]
        timed_out = age_seconds >= self._config["buffer_timeout_seconds"]
        return has_enough_metrics or timed_out

    def _get_ready_buffers(self) -> list[tuple[str, str, str]]:
        """Get list of buffer keys ready for processing.

        Returns:
            List of buffer keys that should be processed.
        """
        ready: list[tuple[str, str, str]] = []
        for key in self._buffer:
            if self._should_process_buffer(key):
                ready.append(key)
        return ready

    def _load_deal(self, deal_id: str) -> Deal:
        """Load deal from repository.

        Args:
            deal_id: Deal identifier.

        Returns:
            Deal data.

        Raises:
            KeyError: If deal does not exist.
        """
        deal_id_typed = DealId(value=deal_id)
        return self._deal_repo.get(deal_id_typed)

    def _load_covenants(self, deal_id: str) -> tuple[Covenant, ...]:
        """Load covenants for a deal.

        Args:
            deal_id: Deal identifier.

        Returns:
            Tuple of covenants for the deal.
        """
        deal_id_typed = DealId(value=deal_id)
        covenants_seq = self._covenant_repo.list_for_deal(deal_id_typed)
        return tuple(covenants_seq)

    def _load_recent_results(
        self,
        deal_id: str,
        limit: int = 4,
    ) -> tuple[CovenantResult, ...]:
        """Load recent covenant results for a deal.

        Args:
            deal_id: Deal identifier.
            limit: Maximum results to load.

        Returns:
            Tuple of recent covenant results, sorted by period_end descending.
        """
        deal_id_typed = DealId(value=deal_id)
        all_results = self._result_repo.list_for_deal(deal_id_typed)
        # Sort by period_end descending (most recent first)
        sorted_results = sorted(
            all_results,
            key=_covenant_result_period_end_key,
            reverse=True,
        )
        return tuple(sorted_results[:limit])

    def _load_historical_metrics(
        self,
        deal_id: str,
        periods_back: int,
    ) -> dict[str, dict[str, int]]:
        """Load historical metrics for feature extraction.

        Args:
            deal_id: Deal identifier.
            periods_back: Number of historical periods to load.

        Returns:
            Dict mapping period_end to metrics dict.
        """
        deal_id_typed = DealId(value=deal_id)
        measurements = self._measurement_repo.list_for_deal(deal_id_typed)

        # Group by period_end, convert to scaled ints
        by_period: dict[str, dict[str, int]] = defaultdict(dict)
        for m in measurements:
            period_key = m["period_end_iso"]
            by_period[period_key][m["metric_name"]] = m["metric_value_scaled"]

        return dict(by_period)

    def _build_features(
        self,
        deal: Deal,
        current_metrics: dict[str, int],
        historical: dict[str, dict[str, int]],
        recent_results: tuple[CovenantResult, ...],
    ) -> LoanFeatures:
        """Build feature vector for ML prediction.

        Args:
            deal: Deal data.
            current_metrics: Current period metrics (scaled).
            historical: Historical metrics by period.
            recent_results: Recent covenant results.

        Returns:
            LoanFeatures for ML prediction.
        """
        # Sort periods to get 1p and 4p ago
        sorted_periods = sorted(historical.keys(), reverse=True)
        metrics_1p = historical.get(sorted_periods[0], {}) if len(sorted_periods) > 0 else {}
        metrics_4p = historical.get(sorted_periods[3], {}) if len(sorted_periods) > 3 else {}

        return extract_features(
            deal=deal,
            metrics_current=current_metrics,
            metrics_1p_ago=metrics_1p,
            metrics_4p_ago=metrics_4p,
            recent_results=list(recent_results),
            sector_encoder=self._sector_encoder,
            region_encoder=self._region_encoder,
        )

    def _run_evaluation(
        self,
        deal_id: DealId,
        covenants: tuple[Covenant, ...],
        period_start: str,
        period_end: str,
        metrics_scaled: dict[str, int],
    ) -> tuple[CovenantResult, ...]:
        """Run covenant evaluation.

        Args:
            deal_id: Deal identifier.
            covenants: Covenants to evaluate.
            period_start: Period start date.
            period_end: Period end date.
            metrics_scaled: Metrics for evaluation (scaled).

        Returns:
            Tuple of covenant results.
        """
        # Convert metrics dict to Measurement list
        measurements: list[Measurement] = []
        for name, value in metrics_scaled.items():
            measurements.append(
                {
                    "deal_id": deal_id,
                    "period_start_iso": period_start,
                    "period_end_iso": period_end,
                    "metric_name": name,
                    "metric_value_scaled": value,
                }
            )

        results = evaluate_all_covenants_for_period(
            covenants=list(covenants),
            period_start_iso=period_start,
            period_end_iso=period_end,
            measurements=measurements,
            tolerance_ratio_scaled=self._config["tolerance_ratio_scaled"],
        )
        return tuple(results)

    def _run_prediction(
        self,
        features: LoanFeatures,
    ) -> tuple[float, RiskTier, int]:
        """Run ML prediction and return results with timing.

        Args:
            features: Feature vector for prediction.

        Returns:
            Tuple of (probability, risk_tier, latency_ms).
        """
        start_time = time.perf_counter()
        features_list: list[LoanFeatures] = [features]
        probabilities = predict_probabilities(self._model, features_list)
        probability = probabilities[0]
        risk_tier = classify_risk_tier(probability)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return probability, risk_tier, latency_ms

    def _should_alert(
        self,
        evaluation_status: EvaluationStatus,
        risk_probability: float,
    ) -> bool:
        """Determine if an alert should be generated.

        Args:
            evaluation_status: Deterministic evaluation result.
            risk_probability: ML-predicted probability.

        Returns:
            True if alert should be generated.
        """
        if evaluation_status == "BREACH":
            return True
        return risk_probability >= self._config["alert_threshold"]

    def process_buffered_period(
        self,
        deal_id: str,
        period_start: str,
        period_end: str,
        metrics: dict[str, float],
    ) -> ProcessingResult:
        """Process a buffered period and generate prediction.

        Args:
            deal_id: Deal identifier.
            period_start: Period start date.
            period_end: Period end date.
            metrics: Metrics for this period (float values).

        Returns:
            ProcessingResult with prediction and optional alert.

        Raises:
            KeyError: If deal does not exist.
        """
        deal_id_typed = DealId(value=deal_id)

        # Load deal data
        deal = self._load_deal(deal_id)

        # Load covenants
        covenants = self._load_covenants(deal_id)

        # Load historical data for features
        recent_results = self._load_recent_results(deal_id)
        historical = self._load_historical_metrics(deal_id, periods_back=5)

        # Scale current metrics
        metrics_scaled = _scale_metrics(metrics)

        # Run covenant evaluation
        eval_start = time.perf_counter()
        results = self._run_evaluation(
            deal_id_typed, covenants, period_start, period_end, metrics_scaled
        )
        evaluation_status = _determine_evaluation_status(results)
        breaches_count = _count_breaches(results)
        covenants_evaluated = len(results)
        evaluation_latency_ms = int((time.perf_counter() - eval_start) * 1000)

        # Emit evaluation latency metric
        self._metrics.record_evaluation_latency(
            deal_id, evaluation_status, float(evaluation_latency_ms)
        )

        # Build features and run ML prediction
        features = self._build_features(deal, metrics_scaled, historical, recent_results)
        risk_probability, risk_tier, prediction_latency_ms = self._run_prediction(features)

        # Emit prediction metrics
        self._metrics.record_prediction_latency(deal_id, risk_tier, float(prediction_latency_ms))
        self._metrics.set_prediction_risk(deal_id, risk_probability)

        # Create prediction event
        prediction = make_prediction_event(
            event_id=_generate_event_id(),
            deal_id=deal_id,
            period_start=period_start,
            period_end=period_end,
            evaluation_status=evaluation_status,
            covenants_evaluated=covenants_evaluated,
            breaches_count=breaches_count,
            risk_probability=risk_probability,
            risk_tier=risk_tier,
            model_version=self._config["model_version"],
            evaluation_latency_ms=evaluation_latency_ms,
            prediction_latency_ms=prediction_latency_ms,
            processed_at=_current_iso_timestamp(),
        )

        # Check if alert needed
        alert: AlertEventV1 | None = None
        if self._should_alert(evaluation_status, risk_probability):
            alert_type = _determine_alert_type(evaluation_status)
            severity = _determine_alert_severity(risk_probability)
            message = _generate_alert_message(
                deal_id=deal_id,
                deal_name=deal["name"],
                risk_probability=risk_probability,
                evaluation_status=evaluation_status,
                breaches_count=breaches_count,
            )

            alert = make_alert_event(
                event_id=_generate_event_id(),
                deal_id=deal_id,
                alert_type=alert_type,
                severity=severity,
                risk_probability=risk_probability,
                gemini_summary=message,
                triggered_at=_current_iso_timestamp(),
            )

            # Emit alert metric
            self._metrics.increment_alert_triggered(deal_id, severity, alert_type)

        return {
            "prediction": prediction,
            "alert": alert,
            "evaluation_latency_ms": evaluation_latency_ms,
            "prediction_latency_ms": prediction_latency_ms,
        }

    def _process_ready_buffers(self) -> int:
        """Process all ready buffers and produce events.

        Returns:
            Number of periods processed.

        Raises:
            KeyError: If a deal does not exist.
        """
        ready_keys = self._get_ready_buffers()
        processed = 0

        for key in ready_keys:
            deal_id, period_start, period_end = key
            buffered = self._buffer.pop(key)

            result = self.process_buffered_period(
                deal_id=deal_id,
                period_start=period_start,
                period_end=period_end,
                metrics=buffered["metrics"],
            )

            # Produce prediction event
            self._producer.produce_prediction(result["prediction"])

            # Produce alert event if present
            if result["alert"] is not None:
                self._producer.produce_alert(result["alert"])

            processed += 1

        # Poll producer for delivery callbacks
        if processed > 0:
            self._producer.poll(0.0)

        return processed

    def run_once(self) -> tuple[int, int]:
        """Run a single iteration of the worker loop.

        Returns:
            Tuple of (messages_consumed, periods_processed).
        """
        # Poll for new messages
        consumed = self._consumer.poll(self._config["poll_timeout_seconds"])

        messages_consumed = 0
        if consumed is not None:
            event = consumed["event"]

            # Emit measurement received metric
            self._metrics.increment_measurement_received(event["deal_id"], event["metric_name"])

            # Add to buffer
            self._add_to_buffer(event)
            messages_consumed = 1
            self._messages_since_commit += 1

        # Process ready buffers
        periods_processed = self._process_ready_buffers()

        # Commit periodically
        if self._messages_since_commit >= self._config["commit_interval"]:
            self._consumer.commit()
            self._messages_since_commit = 0

        return messages_consumed, periods_processed

    def run(self, max_iterations: int | None = None) -> tuple[int, int]:
        """Run the worker main loop.

        Args:
            max_iterations: Optional limit on iterations (for testing).
                None means run until shutdown() is called.

        Returns:
            Tuple of (total_messages_consumed, total_periods_processed).
        """
        self._running = True
        total_messages = 0
        total_periods = 0
        iterations = 0

        while self._running:
            if max_iterations is not None and iterations >= max_iterations:
                break

            messages, periods = self.run_once()
            total_messages += messages
            total_periods += periods
            iterations += 1

        return total_messages, total_periods

    def shutdown(self) -> None:
        """Graceful shutdown of the worker.

        Processes any remaining buffered periods, commits offsets,
        and flushes producer.

        Raises:
            KeyError: If a deal does not exist during processing.
        """
        self._running = False

        # Process any remaining buffers (force timeout)
        for key in list(self._buffer.keys()):
            deal_id, period_start, period_end = key
            buffered = self._buffer.pop(key)

            result = self.process_buffered_period(
                deal_id=deal_id,
                period_start=period_start,
                period_end=period_end,
                metrics=buffered["metrics"],
            )

            self._producer.produce_prediction(result["prediction"])
            if result["alert"] is not None:
                self._producer.produce_alert(result["alert"])

        # Commit any pending offsets
        if self._messages_since_commit > 0:
            self._consumer.commit()

        # Flush producer
        self._producer.flush(timeout_seconds=10.0)

        # Close consumer
        self._consumer.close()


__all__ = [
    "BufferKey",
    "BufferedPeriod",
    "ProcessingResult",
    "StreamingWorker",
    "WorkerConfig",
    "make_default_worker_config",
]
