"""Tests for domain protocol definitions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_radar_api.domains.base_schemas import (
    BaseInputEventV1,
    BasePredictionEventV1,
    encode_base_prediction_event,
    make_base_input_event,
)
from covenant_radar_api.domains.protocols import (
    DomainConfig,
    DomainProtocol,
    FeatureExtractorProtocol,
    ModelProtocol,
    make_domain_config,
)

# =============================================================================
# Fake Implementations (for structural Protocol satisfaction)
# =============================================================================


class FakeFeatureExtractor:
    """Fake feature extractor satisfying FeatureExtractorProtocol."""

    def __init__(self, names: tuple[str, ...]) -> None:
        """Initialize with feature names.

        Args:
            names: Ordered tuple of feature names.
        """
        self._names: tuple[str, ...] = names

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return ordered tuple of feature names."""
        return self._names

    @property
    def n_features(self) -> int:
        """Return number of features produced."""
        return len(self._names)

    def extract(self, event: BaseInputEventV1) -> NDArray[np.float64]:
        """Extract feature vector returning zeros.

        Args:
            event: Base input event.

        Returns:
            1D numpy array of shape (n_features,).
        """
        return np.zeros(len(self._names), dtype=np.float64)

    def extract_batch(
        self,
        events: list[BaseInputEventV1],
    ) -> NDArray[np.float64]:
        """Extract features from multiple events returning zeros.

        Args:
            events: List of base input events.

        Returns:
            2D numpy array of shape (n_events, n_features).
        """
        return np.zeros((len(events), len(self._names)), dtype=np.float64)


class FakeDomain:
    """Fake domain satisfying DomainProtocol."""

    def __init__(self, name: str) -> None:
        """Initialize with domain name.

        Args:
            name: Domain identifier.
        """
        self._config: DomainConfig = make_domain_config(
            name=name,
            display_name=f"Fake {name}",
            input_topic=f"{name}.input.v1",
            prediction_topic=f"{name}.predictions.v1",
            alert_topic=f"{name}.alerts.v1",
            alert_threshold=0.8,
        )
        self._extractor: FakeFeatureExtractor = FakeFeatureExtractor(
            ("feature_a", "feature_b"),
        )

    @property
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        return self._config

    @property
    def feature_extractor(self) -> FeatureExtractorProtocol:
        """Return feature extractor for this domain."""
        return self._extractor

    def decode_input_event(self, payload: str) -> BaseInputEventV1:
        """Decode input event from JSON payload.

        Args:
            payload: Raw JSON string.

        Returns:
            Decoded BaseInputEventV1.
        """
        from covenant_radar_api.domains.base_schemas import decode_base_input_event

        return decode_base_input_event(payload)

    def encode_prediction_event(self, event: BasePredictionEventV1) -> str:
        """Encode prediction event to JSON string.

        Args:
            event: Base prediction event.

        Returns:
            JSON string.
        """
        return encode_base_prediction_event(event)

    def generate_alert_context(
        self,
        entity_id: str,
        prediction_value: float,
    ) -> dict[str, str]:
        """Generate alert context dictionary.

        Args:
            entity_id: Primary entity identifier.
            prediction_value: Prediction value that triggered alert.

        Returns:
            Context dictionary for Gemini prompt.
        """
        return {
            "domain": self._config["name"],
            "entity_id": entity_id,
            "prediction_value": f"{prediction_value:.2f}",
        }


# =============================================================================
# Tests: FeatureExtractorProtocol
# =============================================================================


class TestFeatureExtractorProtocol:
    """Tests for FeatureExtractorProtocol structural satisfaction."""

    def test_fake_satisfies_protocol(self) -> None:
        """FakeFeatureExtractor satisfies FeatureExtractorProtocol."""
        extractor: FeatureExtractorProtocol = FakeFeatureExtractor(("a", "b"))
        assert extractor.n_features == 2

    def test_extract_returns_1d_array(self) -> None:
        """Extract returns 1D array of correct shape."""
        extractor: FeatureExtractorProtocol = FakeFeatureExtractor(("x", "y", "z"))
        event = make_base_input_event(
            type="test.v1",
            event_id="e1",
            entity_id="ent1",
            timestamp="2025-01-01T00:00:00Z",
        )

        result = extractor.extract(event)

        assert result.shape == (3,)
        assert result.dtype == np.float64

    def test_extract_batch_returns_2d_array(self) -> None:
        """Extract batch returns 2D array of correct shape."""
        extractor: FeatureExtractorProtocol = FakeFeatureExtractor(("a", "b"))
        events = [
            make_base_input_event(
                type="test.v1",
                event_id=f"e{i}",
                entity_id=f"ent{i}",
                timestamp="2025-01-01T00:00:00Z",
            )
            for i in range(4)
        ]

        result = extractor.extract_batch(events)

        assert result.shape == (4, 2)
        assert result.dtype == np.float64

    def test_n_features_matches_feature_names(self) -> None:
        """n_features equals length of feature_names."""
        extractor: FeatureExtractorProtocol = FakeFeatureExtractor(
            ("f1", "f2", "f3", "f4"),
        )
        assert extractor.n_features == len(extractor.feature_names)
        assert extractor.n_features == 4

    def test_feature_names_returns_tuple(self) -> None:
        """feature_names returns a tuple."""
        extractor: FeatureExtractorProtocol = FakeFeatureExtractor(("a",))
        assert type(extractor.feature_names) is tuple


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
        domain: DomainProtocol = FakeDomain("test")
        assert domain.config["name"] == "test"

    def test_config_returns_domain_config(self) -> None:
        """Config property returns DomainConfig with correct fields."""
        domain: DomainProtocol = FakeDomain("covenant")
        config = domain.config
        assert config["name"] == "covenant"
        assert config["display_name"] == "Fake covenant"
        assert config["input_topic"] == "covenant.input.v1"
        assert config["prediction_topic"] == "covenant.predictions.v1"
        assert config["alert_topic"] == "covenant.alerts.v1"
        assert config["alert_threshold"] == 0.8

    def test_feature_extractor_returns_protocol(self) -> None:
        """feature_extractor returns FeatureExtractorProtocol."""
        domain: DomainProtocol = FakeDomain("weather")
        extractor: FeatureExtractorProtocol = domain.feature_extractor
        assert extractor.n_features == 2

    def test_decode_input_event_returns_base_event(self) -> None:
        """decode_input_event parses JSON to BaseInputEventV1."""
        domain: DomainProtocol = FakeDomain("test")
        from platform_core.json_utils import dump_json_str

        payload = dump_json_str(
            {
                "type": "test.input.v1",
                "event_id": "e1",
                "entity_id": "ent1",
                "timestamp": "2025-01-01T00:00:00Z",
            }
        )

        event = domain.decode_input_event(payload)
        assert event["type"] == "test.input.v1"
        assert event["entity_id"] == "ent1"

    def test_encode_prediction_event_returns_string(self) -> None:
        """encode_prediction_event returns JSON string."""
        domain: DomainProtocol = FakeDomain("test")
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
        domain: DomainProtocol = FakeDomain("weather")

        context = domain.generate_alert_context("station-1", 0.92)

        assert context["domain"] == "weather"
        assert context["entity_id"] == "station-1"
        assert context["prediction_value"] == "0.92"


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
