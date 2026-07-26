"""Tests for domain protocol definitions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import dump_json_str

from covenant_radar_api.domains.protocols import (
    DomainProtocol,
    ModelProtocol,
    make_domain_config,
)
from tests.domains._test_domain_fixtures import make_fake_domain

# =============================================================================
# Tests: DomainConfig
# =============================================================================


class TestDomainConfig:
    """Tests for make_domain_config factory."""

    def test_make_domain_config(self) -> None:
        """Factory creates config with all fields."""
        config = make_domain_config(
            name="weather",
            display_name="Weather Monitor",
            input_topic="weather.observations.v1",
            prediction_topic="weather.predictions.v1",
            alert_topic="weather.alerts.v1",
            alert_threshold=0.75,
        )

        assert config["name"] == "weather"
        assert config["display_name"] == "Weather Monitor"
        assert config["input_topic"] == "weather.observations.v1"
        assert config["prediction_topic"] == "weather.predictions.v1"
        assert config["alert_topic"] == "weather.alerts.v1"
        assert config["alert_threshold"] == 0.75

    def test_config_is_dict(self) -> None:
        """Config is a dict (TypedDict)."""
        config = make_domain_config(
            name="test",
            display_name="Test",
            input_topic="t.in",
            prediction_topic="t.pred",
            alert_topic="t.alert",
            alert_threshold=0.5,
        )
        assert type(config) is dict


# =============================================================================
# Tests: DomainProtocol
# =============================================================================


class TestDomainProtocol:
    """Tests for DomainProtocol structural satisfaction."""

    def test_fake_satisfies_protocol(self) -> None:
        """FakeDomain satisfies DomainProtocol."""
        domain: DomainProtocol = make_fake_domain("test")
        assert domain.config["name"] == "test"

    def test_config_returns_domain_config(self) -> None:
        """Config property returns DomainConfig with correct fields."""
        domain: DomainProtocol = make_fake_domain("covenant")
        config = domain.config
        assert config["name"] == "covenant"
        assert config["display_name"] == "Fake covenant"
        assert config["input_topic"] == "covenant.input.v1"
        assert config["prediction_topic"] == "covenant.predictions.v1"
        assert config["alert_topic"] == "covenant.alerts.v1"
        assert config["alert_threshold"] == 0.8

    def test_feature_names_and_count_agree(self) -> None:
        """n_features is the length of feature_names, not a second source."""
        domain: DomainProtocol = make_fake_domain("weather")

        assert domain.n_features == len(domain.feature_names)

    def test_decode_and_extract_returns_event_and_features(self) -> None:
        """decode_and_extract yields the base event and its feature vector.

        One step rather than two: the domain decodes to its own type and
        extracts from it, so the worker never holds an event that a
        domain-specific extractor cannot accept.
        """
        domain: DomainProtocol = make_fake_domain("test")
        payload = dump_json_str(
            {
                "type": "test.input.v1",
                "event_id": "e1",
                "entity_id": "ent1",
                "timestamp": "2025-01-01T00:00:00Z",
            }
        )

        event, features = domain.decode_and_extract(payload)

        assert event["type"] == "test.input.v1"
        assert event["entity_id"] == "ent1"
        assert features.shape == (domain.n_features,)

    def test_decode_and_extract_feature_width_matches_declared_count(self) -> None:
        """The vector width matches n_features, which the model relies on."""
        domain: DomainProtocol = make_fake_domain("weather")
        payload = dump_json_str(
            {
                "type": "weather.input.v1",
                "event_id": "e2",
                "entity_id": "station-7",
                "timestamp": "2025-01-01T00:00:00Z",
            }
        )

        _, features = domain.decode_and_extract(payload)

        assert int(features.shape[0]) == domain.n_features

    def test_encode_prediction_event_returns_string(self) -> None:
        """encode_prediction_event returns JSON string."""
        domain: DomainProtocol = make_fake_domain("test")
        from covenant_radar_api.domains.base_schemas import make_base_prediction_event

        prediction = make_base_prediction_event(
            type="test.prediction.v1",
            event_id="p1",
            entity_id="ent1",
            prediction_value=0.75,
            confidence=0.50,
            model_version="v1",
            latency_ms=10,
            processed_at="2025-01-01T00:00:00Z",
        )

        result = domain.encode_prediction_event(prediction)
        assert type(result) is str
        assert "test.prediction.v1" in result

    def test_generate_alert_context_returns_dict(self) -> None:
        """generate_alert_context returns dict[str, str]."""
        domain: DomainProtocol = make_fake_domain("weather")

        context = domain.generate_alert_context("station-1", 0.92)

        assert context["domain"] == "weather"
        assert context["entity_id"] == "station-1"
        # The value must survive as a number the prompt can state, not a
        # particular decimal width -- pinning the format would test the fake's
        # formatting rather than the contract.
        assert float(context["prediction_value"]) == 0.92


# =============================================================================
# Fake Model (for ModelProtocol)
# =============================================================================


class FakeModel:
    """Fake model satisfying ModelProtocol."""

    def __init__(self, probability: float = 0.25) -> None:
        """Initialize with default probability.

        Args:
            probability: Positive class probability to return.
        """
        self._probability: float = probability

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities returning configured value.

        Args:
            x: 2D feature array of shape (n_samples, n_features).

        Returns:
            Probability array of shape (n_samples, 2).
        """
        n_samples: int = x.shape[0]
        result: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        for i in range(n_samples):
            result[i, 0] = 1.0 - self._probability
            result[i, 1] = self._probability
        return result


# =============================================================================
# Tests: ModelProtocol
# =============================================================================


class TestModelProtocol:
    """Tests for ModelProtocol structural satisfaction."""

    def test_fake_satisfies_protocol(self) -> None:
        """FakeModel satisfies ModelProtocol."""
        model: ModelProtocol = FakeModel(0.75)
        x: NDArray[np.float64] = np.zeros((1, 3), dtype=np.float64)
        result = model.predict_proba(x)
        assert result.shape == (1, 2)

    def test_returns_configured_probability(self) -> None:
        """FakeModel returns configured probability in positive column."""
        model: ModelProtocol = FakeModel(0.85)
        x: NDArray[np.float64] = np.zeros((1, 5), dtype=np.float64)
        result = model.predict_proba(x)
        rows: list[list[float]] = result.tolist()
        assert rows[0][1] == 0.85
        negative: float = rows[0][0]
        assert round(negative, 10) == 0.15

    def test_handles_multiple_samples(self) -> None:
        """FakeModel handles batch prediction."""
        model: ModelProtocol = FakeModel(0.50)
        x: NDArray[np.float64] = np.zeros((4, 2), dtype=np.float64)
        result = model.predict_proba(x)
        assert result.shape == (4, 2)
        rows: list[list[float]] = result.tolist()
        assert rows[3][1] == 0.50
