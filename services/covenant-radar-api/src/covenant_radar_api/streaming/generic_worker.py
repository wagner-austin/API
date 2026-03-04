"""Generic streaming worker for multi-domain ML prediction.

Domain-agnostic streaming worker that delegates all domain-specific logic
through DomainProtocol. Consumes input events from Kafka, runs feature
extraction and ML prediction, and publishes prediction/alert events.

This completes Phase 1 of the multi-domain streaming platform refactor.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from ..domains.base_schemas import (
    BaseAlertEventV1,
    BaseAlertSeverity,
    BasePredictionEventV1,
    encode_base_alert_event,
    make_base_alert_event,
    make_base_prediction_event,
)
from ..domains.protocols import (
    DomainProtocol,
    ModelProtocol,
)
from . import _test_hooks_generic_worker as _hooks
from ._test_hooks import KafkaConsumerProtocol, KafkaProducerProtocol
from ._test_hooks_generic_worker import TextGeneratorProtocol

# =============================================================================
# Configuration
# =============================================================================


class GenericWorkerConfig(TypedDict, total=True):
    """Configuration for generic streaming worker.

    Attributes:
        model_version: Version string of the ML model.
        poll_timeout_seconds: Kafka consumer poll timeout.
    """

    model_version: str
    poll_timeout_seconds: float


def make_generic_worker_config(
    *,
    model_version: str,
    poll_timeout_seconds: float,
) -> GenericWorkerConfig:
    """Create a generic worker configuration.

    Args:
        model_version: Version string of the ML model.
        poll_timeout_seconds: Kafka consumer poll timeout.

    Returns:
        GenericWorkerConfig instance.
    """
    return {
        "model_version": model_version,
        "poll_timeout_seconds": poll_timeout_seconds,
    }


# =============================================================================
# Result Types
# =============================================================================


class GenericProcessingResult(TypedDict, total=True):
    """Result of processing a single input event.

    Attributes:
        prediction: The base prediction event produced.
        alert: Optional alert event if prediction exceeded threshold.
        latency_ms: Total processing latency in milliseconds.
    """

    prediction: BasePredictionEventV1
    alert: BaseAlertEventV1 | None
    latency_ms: int


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_positive_probability(proba: NDArray[np.float64]) -> float:
    """Extract positive-class probability from model output.

    Handles binary classification output of shape (1, 2) where column 1
    is the positive class probability.

    Args:
        proba: Probability array from model.predict_proba().

    Returns:
        Positive-class probability as a Python float.
    """
    # Convert full array to nested list to avoid numpy Any indexing
    rows: list[list[float]] = proba.tolist()
    row: list[float] = rows[0]
    if len(row) == 2:
        return row[1]
    return row[0]


def _calculate_confidence(prediction_value: float) -> float:
    """Calculate model confidence from prediction value.

    Confidence is distance from the 0.5 decision boundary, scaled to [0, 1].
    A prediction of 0.5 has confidence 0.0; predictions of 0.0 or 1.0 have
    confidence 1.0.

    Args:
        prediction_value: Prediction probability (0.0-1.0).

    Returns:
        Confidence score (0.0-1.0).
    """
    return abs(prediction_value - 0.5) * 2.0


def _classify_severity(
    prediction_value: float,
    alert_threshold: float,
) -> BaseAlertSeverity:
    """Classify alert severity from prediction value.

    Args:
        prediction_value: Prediction probability.
        alert_threshold: Domain alert threshold.

    Returns:
        BaseAlertSeverity literal.
    """
    if prediction_value >= 0.9:
        return "critical"
    if prediction_value >= alert_threshold:
        return "warning"
    return "info"


def _build_alert_prompt(context: dict[str, str]) -> str:
    """Build alert text prompt from context dictionary.

    Formats context key-value pairs into a structured prompt for
    text generation.

    Args:
        context: Dictionary of context key-value pairs from domain.

    Returns:
        Formatted prompt string.
    """
    lines: list[str] = ["Generate an alert summary for the following context:"]
    for key, value in context.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


# =============================================================================
# Generic Streaming Worker
# =============================================================================


class GenericStreamingWorker:
    """Domain-agnostic streaming worker for ML prediction.

    Delegates all domain-specific logic through DomainProtocol:
    - Event decoding via domain.decode_input_event()
    - Feature extraction via domain.feature_extractor.extract()
    - Event encoding via domain.encode_prediction_event()
    - Alert context via domain.generate_alert_context()
    - Topic routing via domain.config

    All dependencies are injected. No conditionals, no fallbacks.
    """

    def __init__(
        self,
        domain: DomainProtocol,
        consumer: KafkaConsumerProtocol,
        producer: KafkaProducerProtocol,
        model: ModelProtocol,
        text_generator: TextGeneratorProtocol,
        config: GenericWorkerConfig,
    ) -> None:
        """Initialize the generic streaming worker.

        Args:
            domain: Domain implementation providing codecs and features.
            consumer: Kafka consumer for input events.
            producer: Kafka producer for output events.
            model: ML model for prediction.
            text_generator: Text generator for alert summaries.
            config: Worker configuration.
        """
        self._domain = domain
        self._consumer = consumer
        self._producer = producer
        self._model = model
        self._text_generator = text_generator
        self._config = config
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the worker run loop is active."""
        return self._running

    def process_event(self, payload: str) -> GenericProcessingResult:
        """Process a single input event through the ML pipeline.

        Decodes the event, extracts features, runs prediction, and
        optionally generates an alert if the prediction exceeds the
        domain's alert threshold.

        Args:
            payload: Raw JSON payload from Kafka message.

        Returns:
            GenericProcessingResult with prediction and optional alert.
        """
        start: float = _hooks.perf_counter()

        # Decode input event through domain
        event = self._domain.decode_input_event(payload)

        # Extract features and reshape for model
        features: NDArray[np.float64] = self._domain.feature_extractor.extract(event)
        features_2d: NDArray[np.float64] = features.reshape(1, -1)

        # Run ML prediction
        proba: NDArray[np.float64] = self._model.predict_proba(features_2d)
        prediction_value: float = _extract_positive_probability(proba)
        confidence: float = _calculate_confidence(prediction_value)

        # Calculate latency
        elapsed: float = _hooks.perf_counter() - start
        latency_ms: int = int(elapsed * 1000)

        # Build prediction event
        prediction: BasePredictionEventV1 = make_base_prediction_event(
            type=f"{self._domain.config['name']}.prediction.v1",
            event_id=_hooks.generate_uuid(),
            entity_id=event["entity_id"],
            prediction_value=prediction_value,
            confidence=confidence,
            model_version=self._config["model_version"],
            latency_ms=latency_ms,
            processed_at=_hooks.current_iso_timestamp(),
        )

        # Check alert threshold
        alert: BaseAlertEventV1 | None = None
        threshold: float = self._domain.config["alert_threshold"]
        if prediction_value >= threshold:
            context: dict[str, str] = self._domain.generate_alert_context(
                event["entity_id"],
                prediction_value,
            )
            prompt: str = _build_alert_prompt(context)
            summary: str = self._text_generator.generate_text(prompt)
            severity: BaseAlertSeverity = _classify_severity(prediction_value, threshold)

            alert = make_base_alert_event(
                type=f"{self._domain.config['name']}.alert.v1",
                event_id=_hooks.generate_uuid(),
                entity_id=event["entity_id"],
                alert_type="high_risk_prediction",
                severity=severity,
                prediction_value=prediction_value,
                gemini_summary=summary,
                triggered_at=_hooks.current_iso_timestamp(),
            )

        return {
            "prediction": prediction,
            "alert": alert,
            "latency_ms": latency_ms,
        }

    def _produce_prediction(self, prediction: BasePredictionEventV1) -> None:
        """Encode and produce a prediction event to Kafka.

        Args:
            prediction: Base prediction event to publish.
        """
        encoded: str = self._domain.encode_prediction_event(prediction)
        topic: str = self._domain.config["prediction_topic"]
        key: bytes = prediction["entity_id"].encode("utf-8")
        self._producer.produce(topic, encoded.encode("utf-8"), key)

    def _produce_alert(self, alert: BaseAlertEventV1) -> None:
        """Encode and produce an alert event to Kafka.

        Args:
            alert: Base alert event to publish.
        """
        encoded: str = encode_base_alert_event(alert)
        topic: str = self._domain.config["alert_topic"]
        key: bytes = alert["entity_id"].encode("utf-8")
        self._producer.produce(topic, encoded.encode("utf-8"), key)

    def run_once(self) -> tuple[int, int]:
        """Poll one message and process if present.

        Returns:
            Tuple of (messages_consumed, events_produced).
        """
        msg = self._consumer.poll(self._config["poll_timeout_seconds"])
        if msg is None:
            return (0, 0)

        payload: str = msg.value().decode("utf-8")
        result: GenericProcessingResult = self.process_event(payload)

        self._produce_prediction(result["prediction"])
        produced: int = 1

        if result["alert"] is not None:
            self._produce_alert(result["alert"])
            produced += 1

        return (1, produced)

    def run(self, max_iterations: int | None = None) -> tuple[int, int]:
        """Run the streaming worker main loop.

        Polls for messages and processes them until shutdown is called
        or max_iterations is reached.

        Args:
            max_iterations: Maximum iterations before stopping.
                None for unlimited (runs until shutdown).

        Returns:
            Tuple of (total_messages_consumed, total_events_produced).
        """
        self._running = True
        total_consumed: int = 0
        total_produced: int = 0
        iteration: int = 0

        while self._running:
            if max_iterations is not None and iteration >= max_iterations:
                break

            consumed, produced = self.run_once()
            total_consumed += consumed
            total_produced += produced
            iteration += 1

        return (total_consumed, total_produced)

    def shutdown(self) -> None:
        """Shutdown the worker gracefully.

        Flushes the producer and closes the consumer.
        """
        self._running = False
        self._producer.flush(5.0)
        self._consumer.close()


__all__ = [
    "GenericProcessingResult",
    "GenericStreamingWorker",
    "GenericWorkerConfig",
    "make_generic_worker_config",
]
