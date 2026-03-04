"""Shared fixtures and factories for generic streaming worker tests.

Provides test data factories and helper functions for GenericStreamingWorker.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_radar_api.domains.base_schemas import (
    BaseInputEventV1,
    BasePredictionEventV1,
    decode_base_input_event,
    encode_base_input_event,
    encode_base_prediction_event,
    make_base_input_event,
)
from covenant_radar_api.domains.protocols import (
    DomainConfig,
    FeatureExtractorProtocol,
    make_domain_config,
)
from covenant_radar_api.streaming._test_hooks import (
    FakeKafkaConsumer,
    FakeKafkaProducer,
)
from covenant_radar_api.streaming._test_hooks_generic_worker import FakeTextGenerator
from covenant_radar_api.streaming._test_hooks_model import FakePredictor
from covenant_radar_api.streaming.generic_worker import (
    GenericStreamingWorker,
    GenericWorkerConfig,
    make_generic_worker_config,
)

# =============================================================================
# Fake Feature Extractor
# =============================================================================


class FakeFeatureExtractor:
    """Fake feature extractor for testing.

    Returns a fixed feature vector based on entity_id length.
    """

    def __init__(self) -> None:
        """Initialize with fixed 3-feature configuration."""
        self._feature_names: tuple[str, ...] = ("feat_a", "feat_b", "feat_c")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return ordered tuple of feature names."""
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Return number of features produced."""
        return len(self._feature_names)

    def extract(self, event: BaseInputEventV1) -> NDArray[np.float64]:
        """Extract feature vector from event.

        Args:
            event: Base input event.

        Returns:
            1D array of shape (3,).
        """
        entity_len: float = float(len(event["entity_id"]))
        features: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        features[0] = entity_len
        features[1] = entity_len * 0.5
        features[2] = 1.0
        return features

    def extract_batch(
        self,
        events: list[BaseInputEventV1],
    ) -> NDArray[np.float64]:
        """Extract features from multiple events.

        Args:
            events: List of base input events.

        Returns:
            2D array of shape (n_events, 3).
        """
        rows: list[NDArray[np.float64]] = [self.extract(e) for e in events]
        return np.stack(rows)


# =============================================================================
# Fake Domain
# =============================================================================


class FakeDomain:
    """Fake domain implementation for testing.

    Provides a minimal domain with predictable behavior.
    """

    def __init__(
        self,
        domain_config: DomainConfig | None = None,
        extractor: FeatureExtractorProtocol | None = None,
    ) -> None:
        """Initialize fake domain.

        Args:
            domain_config: Domain config. Uses default if None.
            extractor: Feature extractor. Uses FakeFeatureExtractor if None.
        """
        self._config: DomainConfig = domain_config or make_domain_config(
            name="test",
            display_name="Test Domain",
            input_topic="test-input",
            prediction_topic="test-predictions",
            alert_topic="test-alerts",
            alert_threshold=0.80,
        )
        self._extractor: FeatureExtractorProtocol = extractor or FakeFeatureExtractor()
        self.decode_calls: list[str] = []
        self.encode_calls: list[str] = []
        self.alert_context_calls: list[tuple[str, float]] = []

    @property
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        return self._config

    @property
    def feature_extractor(self) -> FeatureExtractorProtocol:
        """Return feature extractor."""
        return self._extractor

    def decode_input_event(self, payload: str) -> BaseInputEventV1:
        """Decode input event from JSON payload.

        Args:
            payload: Raw JSON string.

        Returns:
            Decoded BaseInputEventV1.
        """
        self.decode_calls.append(payload)
        return decode_base_input_event(payload)

    def encode_prediction_event(
        self,
        event: BasePredictionEventV1,
    ) -> str:
        """Encode prediction event to JSON.

        Args:
            event: Base prediction event.

        Returns:
            JSON string.
        """
        encoded: str = encode_base_prediction_event(event)
        self.encode_calls.append(encoded)
        return encoded

    def generate_alert_context(
        self,
        entity_id: str,
        prediction_value: float,
    ) -> dict[str, str]:
        """Generate alert context dictionary.

        Args:
            entity_id: Primary entity identifier.
            prediction_value: Prediction value.

        Returns:
            Context dictionary for alert prompt.
        """
        self.alert_context_calls.append((entity_id, prediction_value))
        return {
            "entity_id": entity_id,
            "prediction_value": f"{prediction_value:.4f}",
            "domain": self._config["name"],
        }


# =============================================================================
# Factory Functions
# =============================================================================


def make_test_domain_config() -> DomainConfig:
    """Create a test domain config."""
    return make_domain_config(
        name="test",
        display_name="Test Domain",
        input_topic="test-input",
        prediction_topic="test-predictions",
        alert_topic="test-alerts",
        alert_threshold=0.80,
    )


def make_test_generic_worker_config() -> GenericWorkerConfig:
    """Create a test generic worker config."""
    return make_generic_worker_config(
        model_version="test-v1",
        poll_timeout_seconds=0.1,
    )


def make_base_input_payload(
    entity_id: str = "entity-001",
    timestamp: str = "2026-01-15T10:00:00Z",
) -> str:
    """Create a serialized base input event payload.

    Args:
        entity_id: Entity identifier.
        timestamp: Event timestamp.

    Returns:
        JSON string of BaseInputEventV1.
    """
    event: BaseInputEventV1 = make_base_input_event(
        type="test.input.v1",
        event_id="evt-001",
        entity_id=entity_id,
        timestamp=timestamp,
    )
    return encode_base_input_event(event)


def make_generic_streaming_worker(
    probability: float = 0.25,
    alert_threshold: float = 0.80,
) -> tuple[
    GenericStreamingWorker,
    FakeDomain,
    FakeKafkaConsumer,
    FakeKafkaProducer,
    FakePredictor,
    FakeTextGenerator,
]:
    """Create a GenericStreamingWorker with all fake dependencies.

    Args:
        probability: Default prediction probability.
        alert_threshold: Domain alert threshold.

    Returns:
        Tuple of (worker, domain, consumer, producer, predictor, text_generator).
    """
    domain_config: DomainConfig = make_domain_config(
        name="test",
        display_name="Test Domain",
        input_topic="test-input",
        prediction_topic="test-predictions",
        alert_topic="test-alerts",
        alert_threshold=alert_threshold,
    )
    domain = FakeDomain(domain_config=domain_config)
    consumer = FakeKafkaConsumer()
    producer = FakeKafkaProducer()
    predictor = FakePredictor(default_probability=probability)
    text_generator = FakeTextGenerator()

    config: GenericWorkerConfig = make_test_generic_worker_config()

    worker = GenericStreamingWorker(
        domain=domain,
        consumer=consumer,
        producer=producer,
        model=predictor,
        text_generator=text_generator,
        config=config,
    )

    return (worker, domain, consumer, producer, predictor, text_generator)


__all__ = [
    "FakeDomain",
    "FakeFeatureExtractor",
    "make_base_input_payload",
    "make_generic_streaming_worker",
    "make_test_domain_config",
    "make_test_generic_worker_config",
]
