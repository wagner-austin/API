"""Protocol definitions for pluggable domain implementations.

Defines the contracts that domain implementations must satisfy to plug into
the multi-domain streaming platform. The GenericStreamingWorker (Phase 2)
will consume these protocols.

Protocols:
- FeatureExtractorProtocol: Extracts ML features from base input events.
- DomainProtocol: Full domain implementation with config, extraction, codecs.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from .base_schemas import BaseInputEventV1, BasePredictionEventV1

# =============================================================================
# Domain Configuration
# =============================================================================


class DomainConfig(TypedDict):
    """Configuration for a registered domain.

    Attributes:
        name: Domain identifier (e.g., "covenant", "weather", "esports").
        display_name: Human-readable name for dashboards and logs.
        input_topic: Kafka topic for consuming input events.
        prediction_topic: Kafka topic for publishing prediction events.
        alert_topic: Kafka topic for publishing alert events.
        alert_threshold: Prediction value threshold that triggers an alert.
    """

    name: str
    display_name: str
    input_topic: str
    prediction_topic: str
    alert_topic: str
    alert_threshold: float


def make_domain_config(
    *,
    name: str,
    display_name: str,
    input_topic: str,
    prediction_topic: str,
    alert_topic: str,
    alert_threshold: float,
) -> DomainConfig:
    """Create a domain configuration.

    Args:
        name: Domain identifier.
        display_name: Human-readable name.
        input_topic: Kafka topic for input events.
        prediction_topic: Kafka topic for predictions.
        alert_topic: Kafka topic for alerts.
        alert_threshold: Prediction value threshold for alerts.

    Returns:
        DomainConfig instance.
    """
    return {
        "name": name,
        "display_name": display_name,
        "input_topic": input_topic,
        "prediction_topic": prediction_topic,
        "alert_topic": alert_topic,
        "alert_threshold": alert_threshold,
    }


# =============================================================================
# Model Protocol
# =============================================================================


class ModelProtocol(Protocol):
    """Protocol for ML models used in streaming prediction.

    Minimal interface for models that predict class probabilities from
    numeric feature arrays. Both XGBoost and neural network models
    implement this interface.

    The generic streaming worker uses this to run ML inference without
    depending on any specific ML framework.
    """

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities from feature array.

        Args:
            x: 2D feature array of shape (n_samples, n_features).

        Returns:
            Probability array. Binary models return shape (n_samples, 2)
            with columns [negative_class, positive_class].
        """
        ...


# =============================================================================
# Feature Extractor Protocol
# =============================================================================


class FeatureExtractorProtocol(Protocol):
    """Protocol for extracting ML features from base input events.

    Domain implementations satisfy this protocol to transform decoded
    input events into numeric feature vectors for ML prediction.

    The generic worker calls extract() with BaseInputEventV1 instances.
    Domain adapters (Phase 2) bridge domain-specific event types to
    this interface by wrapping domain-specific extractors.
    """

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return ordered tuple of feature names."""
        ...

    @property
    def n_features(self) -> int:
        """Return number of features produced."""
        ...

    def extract(self, event: BaseInputEventV1) -> NDArray[np.float64]:
        """Extract feature vector from a single input event.

        Args:
            event: Base input event decoded from Kafka.

        Returns:
            1D numpy array of shape (n_features,) with dtype float64.
        """
        ...

    def extract_batch(
        self,
        events: list[BaseInputEventV1],
    ) -> NDArray[np.float64]:
        """Extract features from multiple input events.

        Args:
            events: List of base input events.

        Returns:
            2D numpy array of shape (n_events, n_features) with dtype float64.
        """
        ...


# =============================================================================
# Domain Protocol
# =============================================================================


class DomainProtocol(Protocol):
    """Protocol that all domain implementations must satisfy.

    A domain provides event codecs, feature extraction, alert context
    generation, and Kafka topic configuration. The GenericStreamingWorker
    delegates all domain-specific logic through this protocol.
    """

    @property
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        ...

    @property
    def feature_extractor(self) -> FeatureExtractorProtocol:
        """Return feature extractor for this domain."""
        ...

    def decode_input_event(self, payload: str) -> BaseInputEventV1:
        """Decode a domain-specific input event from JSON payload.

        Args:
            payload: Raw JSON string from Kafka.

        Returns:
            Decoded BaseInputEventV1 with at minimum the base fields.
        """
        ...

    def encode_prediction_event(self, event: BasePredictionEventV1) -> str:
        """Encode a prediction event to JSON string for Kafka.

        Args:
            event: Base prediction event to serialize.

        Returns:
            JSON string for Kafka producer.
        """
        ...

    def generate_alert_context(
        self,
        entity_id: str,
        prediction_value: float,
    ) -> dict[str, str]:
        """Generate context dictionary for Gemini alert summary.

        Args:
            entity_id: Primary entity identifier.
            prediction_value: Prediction value that triggered the alert.

        Returns:
            Dictionary of string key-value pairs for Gemini prompt context.
        """
        ...


__all__ = [
    "DomainConfig",
    "DomainProtocol",
    "FeatureExtractorProtocol",
    "ModelProtocol",
    "make_domain_config",
]
