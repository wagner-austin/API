"""Esports domain implementation for the multi-domain streaming platform.

Binds the match state codec and the win-probability feature extractor into a
DomainProtocol implementation, so GenericStreamingWorker can run esports
without knowing anything about esports.

Unlike weather, this domain needs no fitted state: every feature is a pure
function of one snapshot, so the domain is constructed from configuration
alone.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base_schemas import (
    BaseInputEventV1,
    BasePredictionEventV1,
    encode_base_prediction_event,
    make_base_input_event,
)
from ..protocols import DomainConfig, make_domain_config
from .features import ESPORTS_FEATURE_NAMES, EsportsFeatureExtractor
from .schemas import MatchEventV1, decode_match_event

ESPORTS_DOMAIN_NAME = "esports"
ESPORTS_INPUT_TOPIC = "esports.match_state.v1"
ESPORTS_PREDICTION_TOPIC = "esports.predictions.v1"
ESPORTS_ALERT_TOPIC = "esports.alerts.v1"

# A prediction at or above this triggers an alert. Set high because a
# win-probability swing is only newsworthy once one side is decisively
# ahead; a coin-flip mid-game is the normal state, not an event.
ESPORTS_ALERT_THRESHOLD = 0.85


class EsportsDomain:
    """Esports domain plugging match snapshots into the platform.

    Consumes match state snapshots, derives which side leads and by how
    much, and reports the match as the entity the platform tracks.
    """

    def __init__(
        self,
        extractor: EsportsFeatureExtractor,
        config: DomainConfig,
    ) -> None:
        """Initialize the esports domain.

        Args:
            extractor: Stateless match state feature extractor.
            config: Domain configuration, including the Kafka topics and the
                alert threshold.
        """
        self._extractor: EsportsFeatureExtractor = extractor
        self._config: DomainConfig = config

    @property
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        return self._config

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered feature names this domain produces."""
        return ESPORTS_FEATURE_NAMES

    @property
    def n_features(self) -> int:
        """Return how many features decode_and_extract produces."""
        return len(ESPORTS_FEATURE_NAMES)

    def decode_and_extract(
        self,
        payload: str,
    ) -> tuple[BaseInputEventV1, NDArray[np.float64]]:
        """Decode a match snapshot and extract its features.

        The match_id becomes the base event's entity_id: the platform keys
        everything on entity_id, and for esports the match is the entity.

        Args:
            payload: Raw JSON string from Kafka.

        Returns:
            The snapshot narrowed to base fields, and a 1D float64 array of
            shape (n_features,).

        Raises:
            JSONTypeError: If a required field is missing or mistyped, or the
                event type is not esports.match_state.v1.
            InvalidJsonError: If the payload is not valid JSON.
        """
        event: MatchEventV1 = decode_match_event(payload)
        features: NDArray[np.float64] = self._extractor.extract(event)
        base_event: BaseInputEventV1 = make_base_input_event(
            type=event["type"],
            event_id=event["event_id"],
            entity_id=event["match_id"],
            timestamp=event["timestamp"],
        )
        return base_event, features

    def encode_prediction_event(self, event: BasePredictionEventV1) -> str:
        """Encode a prediction event to JSON for Kafka.

        Args:
            event: Base prediction event to serialize.

        Returns:
            Compact JSON string.
        """
        return encode_base_prediction_event(event)

    def generate_alert_context(
        self,
        entity_id: str,
        prediction_value: float,
    ) -> dict[str, str]:
        """Build the context an alert summary is written from.

        Args:
            entity_id: Match identifier.
            prediction_value: Predicted win probability that triggered the
                alert.

        Returns:
            String key-value pairs describing the alert.
        """
        return {
            "domain": self._config["name"],
            "match_id": entity_id,
            "blue_win_probability": f"{prediction_value:.4f}",
            "features": ", ".join(ESPORTS_FEATURE_NAMES),
        }


def make_esports_domain_config(
    *,
    alert_threshold: float = ESPORTS_ALERT_THRESHOLD,
) -> DomainConfig:
    """Create the esports domain configuration.

    Args:
        alert_threshold: Prediction value at or above which an alert fires.

    Returns:
        DomainConfig for the esports domain.
    """
    return make_domain_config(
        name=ESPORTS_DOMAIN_NAME,
        display_name="Esports",
        input_topic=ESPORTS_INPUT_TOPIC,
        prediction_topic=ESPORTS_PREDICTION_TOPIC,
        alert_topic=ESPORTS_ALERT_TOPIC,
        alert_threshold=alert_threshold,
    )


def make_esports_domain(
    *,
    alert_threshold: float = ESPORTS_ALERT_THRESHOLD,
) -> EsportsDomain:
    """Create an esports domain.

    No fitted state is required: the extractor is stateless, so the domain
    is fully determined by its configuration.

    Args:
        alert_threshold: Prediction value at or above which an alert fires.

    Returns:
        EsportsDomain ready to register.
    """
    return EsportsDomain(
        EsportsFeatureExtractor(),
        make_esports_domain_config(alert_threshold=alert_threshold),
    )


__all__ = [
    "ESPORTS_ALERT_THRESHOLD",
    "ESPORTS_ALERT_TOPIC",
    "ESPORTS_DOMAIN_NAME",
    "ESPORTS_INPUT_TOPIC",
    "ESPORTS_PREDICTION_TOPIC",
    "EsportsDomain",
    "make_esports_domain",
    "make_esports_domain_config",
]
