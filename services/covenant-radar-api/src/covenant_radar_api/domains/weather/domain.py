"""Weather domain implementation for the multi-domain streaming platform.

Binds the weather event codec and the McKinnon-style feature extractor into a
single DomainProtocol implementation, so GenericStreamingWorker can run
weather without knowing anything about weather.

Decoding and extraction are one step. WeatherFeatureExtractor reads a
WeatherEventV1, while the worker only ever holds a BaseInputEventV1; keeping
them separate would force a cast at the boundary. Decoding here to the
domain's own type and returning the base view alongside the features avoids
that entirely.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import numpy as np
from covenant_ml.datasets.types import TemporalFeatureState
from numpy.typing import NDArray

from ..base_schemas import (
    BaseInputEventV1,
    BasePredictionEventV1,
    encode_base_prediction_event,
    make_base_input_event,
)
from ..protocols import DomainConfig, make_domain_config
from .features import WEATHER_FEATURE_NAMES, WeatherFeatureExtractor
from .schemas import WeatherEventV1, decode_weather_event

WEATHER_DOMAIN_NAME = "weather"
WEATHER_INPUT_TOPIC = "weather.observations.v1"
WEATHER_PREDICTION_TOPIC = "weather.predictions.v1"
WEATHER_ALERT_TOPIC = "weather.alerts.v1"

# A prediction at or above this triggers an alert. Extreme-temperature days
# are the rare tail by construction, so the threshold sits high enough that a
# routine observation does not page anyone.
WEATHER_ALERT_THRESHOLD = 0.80


class WeatherDomain:
    """Weather domain plugging temperature observations into the platform.

    Consumes single-station temperature observations, derives McKinnon-style
    temporal features against a pre-fitted seasonal state, and reports the
    station as the entity the platform tracks.
    """

    def __init__(
        self,
        extractor: WeatherFeatureExtractor,
        config: DomainConfig,
    ) -> None:
        """Initialize the weather domain.

        Args:
            extractor: Feature extractor holding the pre-fitted temporal
                state and the station-to-location mapping.
            config: Domain configuration, including the Kafka topics and the
                alert threshold.
        """
        self._extractor: WeatherFeatureExtractor = extractor
        self._config: DomainConfig = config

    @property
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        return self._config

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered feature names this domain produces."""
        return WEATHER_FEATURE_NAMES

    @property
    def n_features(self) -> int:
        """Return how many features decode_and_extract produces."""
        return len(WEATHER_FEATURE_NAMES)

    def decode_and_extract(
        self,
        payload: str,
    ) -> tuple[BaseInputEventV1, NDArray[np.float64]]:
        """Decode a weather observation and extract its features.

        The station_id becomes the base event's entity_id: the platform keys
        everything on entity_id, and for weather the station is the entity.

        Args:
            payload: Raw JSON string from Kafka.

        Returns:
            The observation narrowed to base fields, and a 1D float64 array
            of shape (n_features,).

        Raises:
            JSONTypeError: If a required field is missing or mistyped, or the
                event type is not weather.observation.v1.
            InvalidJsonError: If the payload is not valid JSON.
            KeyError: If the station_id has no entry in the fitted
                station-to-location mapping.
        """
        event: WeatherEventV1 = decode_weather_event(payload)
        features: NDArray[np.float64] = self._extractor.extract(event)
        base_event: BaseInputEventV1 = make_base_input_event(
            type=event["type"],
            event_id=event["event_id"],
            entity_id=event["station_id"],
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
            entity_id: Weather station identifier.
            prediction_value: Predicted probability that triggered the alert.

        Returns:
            String key-value pairs describing the alert.
        """
        return {
            "domain": self._config["name"],
            "station_id": entity_id,
            "extreme_probability": f"{prediction_value:.4f}",
            "features": ", ".join(WEATHER_FEATURE_NAMES),
        }


def make_weather_domain_config(
    *,
    alert_threshold: float = WEATHER_ALERT_THRESHOLD,
) -> DomainConfig:
    """Create the weather domain configuration.

    Args:
        alert_threshold: Prediction value at or above which an alert fires.

    Returns:
        DomainConfig for the weather domain.
    """
    return make_domain_config(
        name=WEATHER_DOMAIN_NAME,
        display_name="Weather",
        input_topic=WEATHER_INPUT_TOPIC,
        prediction_topic=WEATHER_PREDICTION_TOPIC,
        alert_topic=WEATHER_ALERT_TOPIC,
        alert_threshold=alert_threshold,
    )


def make_weather_domain(
    *,
    state: TemporalFeatureState,
    station_to_location: dict[str, int],
    alert_threshold: float = WEATHER_ALERT_THRESHOLD,
) -> WeatherDomain:
    """Create a weather domain from a fitted temporal state.

    Args:
        state: Pre-fitted temporal feature state from training data.
        station_to_location: Mapping from station_id to location index in the
            fitted state arrays.
        alert_threshold: Prediction value at or above which an alert fires.

    Returns:
        WeatherDomain ready to register.
    """
    extractor = WeatherFeatureExtractor(state, station_to_location)
    return WeatherDomain(extractor, make_weather_domain_config(alert_threshold=alert_threshold))


__all__ = [
    "WEATHER_ALERT_THRESHOLD",
    "WEATHER_ALERT_TOPIC",
    "WEATHER_DOMAIN_NAME",
    "WEATHER_INPUT_TOPIC",
    "WEATHER_PREDICTION_TOPIC",
    "WeatherDomain",
    "make_weather_domain",
    "make_weather_domain_config",
]
