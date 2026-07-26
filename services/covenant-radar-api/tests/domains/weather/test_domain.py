"""Tests for the weather domain implementation.

WeatherDomain is what lets GenericStreamingWorker run weather at all: the
schemas and the extractor existed, but nothing implemented DomainProtocol, so
weather could not be registered or reached.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import JSONTypeError, dump_json_str

from covenant_radar_api.domains.base_schemas import make_base_prediction_event
from covenant_radar_api.domains.protocols import DomainProtocol
from covenant_radar_api.domains.registry import DomainRegistry
from covenant_radar_api.domains.weather.domain import (
    WEATHER_ALERT_THRESHOLD,
    WEATHER_DOMAIN_NAME,
    WeatherDomain,
    make_weather_domain,
    make_weather_domain_config,
)
from covenant_radar_api.domains.weather.features import WEATHER_FEATURE_NAMES
from covenant_radar_api.domains.weather.schemas import (
    encode_weather_event,
    make_weather_event,
)
from tests.domains.weather._test_weather_fixtures import make_flat_state

_STATIONS: dict[str, int] = {"station-a": 0}


def _make_domain(
    hot_threshold: float = 5.0,
    cold_threshold: float = -5.0,
) -> WeatherDomain:
    """Build a weather domain over a flat seasonal cycle.

    Args:
        hot_threshold: Hot-tail threshold for the single location.
        cold_threshold: Cold-tail threshold for the single location.

    Returns:
        WeatherDomain whose anomaly equals the raw temperature.
    """
    return make_weather_domain(
        state=make_flat_state(hot_threshold=hot_threshold, cold_threshold=cold_threshold),
        station_to_location=_STATIONS,
    )


def _payload(temperature: float, station_id: str = "station-a") -> str:
    """Encode a weather observation payload.

    Args:
        temperature: Observed temperature in degrees Celsius.
        station_id: Station the observation came from.

    Returns:
        JSON string as it would arrive from Kafka.
    """
    return encode_weather_event(
        make_weather_event(
            event_id="evt-1",
            station_id=station_id,
            day_of_year=180,
            temperature=temperature,
            timestamp="2026-06-29T12:00:00Z",
        )
    )


def _named_features(domain: WeatherDomain, payload: str) -> dict[str, float]:
    """Decode a payload and label each feature with its name.

    Args:
        domain: Domain under test.
        payload: JSON observation to decode.

    Returns:
        Mapping of feature name to value. The width is asserted against the
        declared names, so a mismatch fails here rather than silently
        dropping a feature.
    """
    _, features = domain.decode_and_extract(payload)
    names = domain.feature_names
    assert int(features.shape[0]) == len(names)
    values: NDArray[np.float64] = np.asarray(features, dtype=np.float64)
    return {name: float(values.flat[index]) for index, name in enumerate(names)}


class TestSatisfiesDomainProtocol:
    """WeatherDomain is usable everywhere the platform expects a domain."""

    def test_assignable_to_domain_protocol(self) -> None:
        """Structural satisfaction, checked by the annotation itself."""
        domain: DomainProtocol = _make_domain()

        assert domain.config["name"] == WEATHER_DOMAIN_NAME

    def test_registers_in_the_domain_registry(self) -> None:
        """The registry accepts it, which is how the worker reaches a domain.

        Nothing registered a domain before this existed, so the registry was
        permanently empty and the generic worker had nothing to run.
        """
        registry = DomainRegistry()

        registry.register(_make_domain())

        assert registry.list_names() == (WEATHER_DOMAIN_NAME,)
        assert registry.get(WEATHER_DOMAIN_NAME).config["name"] == WEATHER_DOMAIN_NAME

    def test_feature_names_match_the_extractor(self) -> None:
        """The domain reports exactly what the extractor produces."""
        domain = _make_domain()

        assert domain.feature_names == WEATHER_FEATURE_NAMES

    def test_n_features_matches_feature_names(self) -> None:
        """n_features is derived, not a second declaration that can drift."""
        domain = _make_domain()

        assert domain.n_features == len(domain.feature_names)


class TestDecodeAndExtract:
    """Decoding and extraction happen together, on the domain's own type."""

    def test_returns_base_event_and_features(self) -> None:
        """The vector width matches what the domain declares."""
        domain = _make_domain()

        event, features = domain.decode_and_extract(_payload(20.0))

        assert event["type"] == "weather.observation.v1"
        assert event["event_id"] == "evt-1"
        assert features.shape == (domain.n_features,)

    def test_station_id_becomes_entity_id(self) -> None:
        """The platform keys on entity_id; for weather the station is it.

        Without this mapping the prediction and alert events would carry an
        empty entity, and nothing downstream could attribute them.
        """
        domain = _make_domain()

        event, _ = domain.decode_and_extract(_payload(20.0, station_id="station-a"))

        assert event["entity_id"] == "station-a"

    def test_hot_observation_sets_hot_excess(self) -> None:
        """A temperature above the hot threshold produces positive excess.

        The seasonal cycle is flat and the baseline zero, so the anomaly is
        the temperature itself and the expected values are exact.
        """
        domain = _make_domain(hot_threshold=5.0)

        named = _named_features(domain, _payload(12.0))
        assert named["anomaly"] == pytest.approx(12.0)
        assert named["hot_excess"] == pytest.approx(7.0)
        assert named["is_hot_extreme"] == pytest.approx(1.0)
        assert named["is_cold_extreme"] == pytest.approx(0.0)

    def test_cold_observation_sets_cold_excess(self) -> None:
        """A temperature below the cold threshold produces negative excess."""
        domain = _make_domain(cold_threshold=-5.0)

        named = _named_features(domain, _payload(-12.0))
        assert named["anomaly"] == pytest.approx(-12.0)
        assert named["cold_excess"] == pytest.approx(-7.0)
        assert named["is_cold_extreme"] == pytest.approx(1.0)
        assert named["is_hot_extreme"] == pytest.approx(0.0)

    def test_ordinary_observation_is_neither_extreme(self) -> None:
        """Between the thresholds, both extreme flags stay clear."""
        domain = _make_domain(hot_threshold=5.0, cold_threshold=-5.0)

        named = _named_features(domain, _payload(1.0))
        assert named["is_hot_extreme"] == pytest.approx(0.0)
        assert named["is_cold_extreme"] == pytest.approx(0.0)
        assert named["hot_excess"] == pytest.approx(0.0)

    def test_wrong_event_type_is_rejected(self) -> None:
        """A payload from another domain fails rather than being extracted."""
        domain = _make_domain()
        payload = dump_json_str(
            {
                "type": "covenant.measurement.v1",
                "event_id": "evt-2",
                "station_id": "station-a",
                "day_of_year": 180,
                "temperature": 20.0,
                "timestamp": "2026-06-29T12:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError, match=r"weather\.observation\.v1"):
            domain.decode_and_extract(payload)

    def test_unknown_station_is_reported(self) -> None:
        """A station absent from the fitted mapping cannot be featurised."""
        domain = _make_domain()

        with pytest.raises(KeyError, match="station-z"):
            domain.decode_and_extract(_payload(20.0, station_id="station-z"))


class TestEncodePredictionEvent:
    """Prediction events round-trip through the domain's encoder."""

    def test_encodes_to_json_carrying_the_entity(self) -> None:
        """The encoded event names the station it describes."""
        domain = _make_domain()
        prediction = make_base_prediction_event(
            type="weather.prediction.v1",
            event_id="pred-1",
            entity_id="station-a",
            prediction_value=0.91,
            confidence=0.82,
            model_version="v1.0.0",
            latency_ms=4,
            processed_at="2026-06-29T12:00:01Z",
        )

        encoded = domain.encode_prediction_event(prediction)

        assert "station-a" in encoded
        assert "weather.prediction.v1" in encoded


class TestAlertContext:
    """Alert context describes the station and the prediction."""

    def test_context_carries_station_and_probability(self) -> None:
        """The summary writer gets the station and the value that fired."""
        domain = _make_domain()

        context = domain.generate_alert_context("station-a", 0.93)

        assert context["domain"] == WEATHER_DOMAIN_NAME
        assert context["station_id"] == "station-a"
        assert float(context["extreme_probability"]) == pytest.approx(0.93)

    def test_context_lists_the_features_used(self) -> None:
        """Naming the features lets a summary say what drove the alert."""
        domain = _make_domain()

        context = domain.generate_alert_context("station-a", 0.93)

        for name in WEATHER_FEATURE_NAMES:
            assert name in context["features"]


class TestDomainConfig:
    """Topic routing and the alert threshold come from the config."""

    def test_default_config_topics(self) -> None:
        """Topics are versioned and namespaced by domain."""
        config = make_weather_domain_config()

        assert config["input_topic"] == "weather.observations.v1"
        assert config["prediction_topic"] == "weather.predictions.v1"
        assert config["alert_topic"] == "weather.alerts.v1"

    def test_default_alert_threshold(self) -> None:
        """The default threshold is the published constant."""
        assert make_weather_domain_config()["alert_threshold"] == WEATHER_ALERT_THRESHOLD

    def test_alert_threshold_is_overridable(self) -> None:
        """A deployment can tune how often alerts fire."""
        config = make_weather_domain_config(alert_threshold=0.5)

        assert config["alert_threshold"] == pytest.approx(0.5)

    def test_factory_threshold_reaches_the_domain(self) -> None:
        """make_weather_domain forwards the threshold it is given."""
        domain = make_weather_domain(
            state=make_flat_state(),
            station_to_location=_STATIONS,
            alert_threshold=0.42,
        )

        assert domain.config["alert_threshold"] == pytest.approx(0.42)


class TestFeatureVectorContract:
    """The vector the model receives is well-formed."""

    def test_features_are_finite(self) -> None:
        """A non-finite feature would poison the model input silently."""
        domain = _make_domain()

        _, features = domain.decode_and_extract(_payload(20.0))

        finite: NDArray[np.bool_] = np.isfinite(features)
        assert int(np.count_nonzero(finite)) == int(features.size)

    def test_features_are_float64(self) -> None:
        """The model contract is float64; another dtype would be coerced."""
        domain = _make_domain()

        _, features = domain.decode_and_extract(_payload(20.0))

        assert features.dtype == np.float64
