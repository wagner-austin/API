"""Protocol definitions for pluggable domain implementations.

Defines the contracts that domain implementations must satisfy to plug into
the multi-domain streaming platform. GenericStreamingWorker consumes these
protocols and holds no domain-specific logic of its own.

Protocols:
- ModelProtocol: Predicts class probabilities from a feature array.
- DomainProtocol: Full domain implementation with config, codecs, and the
  combined decode-and-extract step.

Decoding and feature extraction are one operation rather than two. A domain's
extractor reads its own event type -- WeatherFeatureExtractor takes a
WeatherEventV1, not a BaseInputEventV1 -- and a protocol method declared to
accept the base type cannot be satisfied by one accepting a narrower type.
Splitting the step forced a cast at the boundary to bridge them. Combining it
lets each domain decode to its own type internally and hand back the base
event alongside the features, with no cast anywhere.

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
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered feature names this domain produces."""
        ...

    @property
    def n_features(self) -> int:
        """Return how many features decode_and_extract produces."""
        ...

    def decode_and_extract(
        self,
        payload: str,
    ) -> tuple[BaseInputEventV1, NDArray[np.float64]]:
        """Decode a domain event and extract its feature vector.

        One step rather than two, because the extractor reads the domain's
        own event type and the worker only ever sees the base type. Decoding
        separately would hand the worker a BaseInputEventV1 that no
        domain-specific extractor can accept without a cast.

        Args:
            payload: Raw JSON string from Kafka.

        Returns:
            The event narrowed to its base fields, and a 1D float64 array of
            shape (n_features,).

        Raises:
            JSONTypeError: If a required field is missing or has the wrong
                type.
            InvalidJsonError: If the payload is not valid JSON.
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
    "ModelProtocol",
    "make_domain_config",
]
