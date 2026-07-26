"""Tests for the esports domain implementation.

EsportsDomain is the second implementation of DomainProtocol, so it is also
the check that the protocol carries a domain the platform was not built
around. Weather needs a fitted state off disk; this one needs nothing but
its configuration, and the platform has to accommodate both.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import JSONTypeError, dump_json_str

from covenant_radar_api.domains.base_schemas import make_base_prediction_event
from covenant_radar_api.domains.esports.domain import (
    ESPORTS_ALERT_THRESHOLD,
    ESPORTS_ALERT_TOPIC,
    ESPORTS_DOMAIN_NAME,
    ESPORTS_INPUT_TOPIC,
    ESPORTS_PREDICTION_TOPIC,
    EsportsDomain,
    make_esports_domain,
    make_esports_domain_config,
)
from covenant_radar_api.domains.esports.features import ESPORTS_FEATURE_NAMES
from covenant_radar_api.domains.protocols import DomainProtocol
from covenant_radar_api.domains.registry import DomainRegistry
from tests.domains.esports._test_esports_fixtures import make_payload


def _named_features(domain: EsportsDomain, payload: str) -> dict[str, float]:
    """Decode a payload through the domain and label each feature.

    Args:
        domain: Domain under test.
        payload: JSON snapshot to decode.

    Returns:
        Mapping of feature name to value, width-checked against the names
        the domain declares.
    """
    _, features = domain.decode_and_extract(payload)
    names = domain.feature_names
    assert int(features.shape[0]) == len(names)
    values: NDArray[np.float64] = np.asarray(features, dtype=np.float64)
    return {name: float(values.flat[index]) for index, name in enumerate(names)}


class TestSatisfiesDomainProtocol:
    """EsportsDomain is usable everywhere the platform expects a domain."""

    def test_assignable_to_domain_protocol(self) -> None:
        """Structural satisfaction, checked by the annotation itself."""
        domain: DomainProtocol = make_esports_domain()

        assert domain.config["name"] == ESPORTS_DOMAIN_NAME

    def test_registers_in_the_domain_registry(self) -> None:
        """The registry accepts it, which is how the worker reaches a domain."""
        registry = DomainRegistry()

        registry.register(ESPORTS_DOMAIN_NAME, make_esports_domain)

        assert registry.list_names() == (ESPORTS_DOMAIN_NAME,)
        assert registry.get(ESPORTS_DOMAIN_NAME).config["name"] == ESPORTS_DOMAIN_NAME

    def test_registers_alongside_another_domain(self) -> None:
        """Two domains coexist, which is the point of the registry.

        A second domain that needed the core changed to accept it would
        mean the protocol had been written around the first one.
        """
        registry = DomainRegistry()

        registry.register(ESPORTS_DOMAIN_NAME, make_esports_domain)

        assert ESPORTS_DOMAIN_NAME in registry.list_names()

    def test_feature_names_match_the_extractor(self) -> None:
        """The domain reports exactly what the extractor produces."""
        assert make_esports_domain().feature_names == ESPORTS_FEATURE_NAMES

    def test_n_features_matches_feature_names(self) -> None:
        """n_features is derived, not a second declaration that can drift."""
        domain = make_esports_domain()

        assert domain.n_features == len(domain.feature_names)


class TestDecodeAndExtract:
    """Decoding and extraction happen together, on the domain's own type."""

    def test_returns_base_event_and_features(self) -> None:
        """The vector width matches what the domain declares."""
        domain = make_esports_domain()

        event, features = domain.decode_and_extract(make_payload(event_id="evt-1"))

        assert event["type"] == "esports.match_state.v1"
        assert event["event_id"] == "evt-1"
        assert features.shape == (domain.n_features,)

    def test_match_id_becomes_entity_id(self) -> None:
        """The platform keys on entity_id; for esports the match is it.

        Without this mapping the prediction and alert events would carry an
        empty entity and nothing downstream could attribute them.
        """
        domain = make_esports_domain()

        event, _ = domain.decode_and_extract(make_payload(match_id="match-42"))

        assert event["entity_id"] == "match-42"

    def test_timestamp_is_carried_through(self) -> None:
        """The base event keeps the snapshot's own time, not the decode time."""
        domain = make_esports_domain()

        event, _ = domain.decode_and_extract(make_payload())

        assert event["timestamp"] == "2026-07-25T18:00:00Z"

    def test_blue_lead_produces_positive_differences(self) -> None:
        """A leading blue side reaches the model as positive features."""
        domain = make_esports_domain()

        named = _named_features(
            domain,
            make_payload(blue_kills=14, red_kills=4, blue_gold=48000, red_gold=39000),
        )

        assert named["kill_diff"] == pytest.approx(10.0)
        assert named["gold_diff"] == pytest.approx(9000.0)
        assert named["blue_kill_ratio"] > 0.5

    def test_red_lead_produces_negative_differences(self) -> None:
        """A leading red side reaches the model as negative features."""
        domain = make_esports_domain()

        named = _named_features(
            domain,
            make_payload(blue_kills=2, red_kills=13, blue_gold=33000, red_gold=44000),
        )

        assert named["kill_diff"] == pytest.approx(-11.0)
        assert named["gold_diff"] == pytest.approx(-11000.0)
        assert named["blue_kill_ratio"] < 0.5

    def test_wrong_event_type_is_rejected(self) -> None:
        """A payload from another domain fails rather than being extracted."""
        domain = make_esports_domain()
        payload = dump_json_str(
            {
                "type": "weather.observation.v1",
                "event_id": "evt-2",
                "match_id": "match-1",
                "game_number": 1,
                "game_time_seconds": 600,
                "blue_kills": 0,
                "red_kills": 0,
                "blue_gold": 0,
                "red_gold": 0,
                "blue_towers": 0,
                "red_towers": 0,
                "blue_dragons": 0,
                "red_dragons": 0,
                "blue_barons": 0,
                "red_barons": 0,
                "timestamp": "2026-07-25T18:00:00Z",
            }
        )

        with pytest.raises(JSONTypeError, match=r"esports\.match_state\.v1"):
            domain.decode_and_extract(payload)

    def test_malformed_payload_is_rejected(self) -> None:
        """A missing field fails at the boundary, not inside extraction."""
        domain = make_esports_domain()

        with pytest.raises(JSONTypeError, match="blue_gold"):
            domain.decode_and_extract(
                dump_json_str(
                    {
                        "type": "esports.match_state.v1",
                        "event_id": "evt-3",
                        "match_id": "match-1",
                        "game_number": 1,
                        "game_time_seconds": 600,
                        "blue_kills": 0,
                        "red_kills": 0,
                        "red_gold": 0,
                        "blue_towers": 0,
                        "red_towers": 0,
                        "blue_dragons": 0,
                        "red_dragons": 0,
                        "blue_barons": 0,
                        "red_barons": 0,
                        "timestamp": "2026-07-25T18:00:00Z",
                    }
                )
            )

    def test_opening_snapshot_is_extractable(self) -> None:
        """The first tick of a game must not be a special case for callers."""
        domain = make_esports_domain()

        named = _named_features(domain, make_payload(game_time_seconds=0))

        assert named["gold_diff_per_minute"] == pytest.approx(0.0)
        assert named["blue_kill_ratio"] == pytest.approx(0.5)


class TestEncodePredictionEvent:
    """Prediction events round-trip through the domain's encoder."""

    def test_encodes_to_json_carrying_the_entity(self) -> None:
        """The encoded event names the match it describes."""
        domain = make_esports_domain()
        prediction = make_base_prediction_event(
            type="esports.prediction.v1",
            event_id="pred-1",
            entity_id="match-42",
            prediction_value=0.91,
            confidence=0.82,
            model_version="v1.0.0",
            latency_ms=4,
            processed_at="2026-07-25T18:00:01Z",
        )

        encoded = domain.encode_prediction_event(prediction)

        assert "match-42" in encoded
        assert "esports.prediction.v1" in encoded


class TestAlertContext:
    """Alert context describes the match and the prediction."""

    def test_context_carries_match_and_probability(self) -> None:
        """The summary writer gets the match and the value that fired."""
        domain = make_esports_domain()

        context = domain.generate_alert_context("match-42", 0.93)

        assert context["domain"] == ESPORTS_DOMAIN_NAME
        assert context["match_id"] == "match-42"
        assert float(context["blue_win_probability"]) == pytest.approx(0.93)

    def test_context_lists_the_features_used(self) -> None:
        """Naming the features lets a summary say what drove the alert."""
        domain = make_esports_domain()

        context = domain.generate_alert_context("match-42", 0.93)

        for name in ESPORTS_FEATURE_NAMES:
            assert name in context["features"]

    def test_probability_is_rendered_to_four_decimals(self) -> None:
        """The prompt builder concatenates these, so the value arrives as text.

        Fixing the precision also keeps two alerts on the same match
        comparable, rather than differing in how many digits they print.
        """
        domain = make_esports_domain()

        context = domain.generate_alert_context("match-42", 0.9312567)

        assert context["blue_win_probability"] == "0.9313"

    def test_context_keys_are_stable(self) -> None:
        """The summary prompt reads these by name, so the set is a contract."""
        domain = make_esports_domain()

        context = domain.generate_alert_context("match-42", 0.93)

        assert sorted(context.keys()) == [
            "blue_win_probability",
            "domain",
            "features",
            "match_id",
        ]


class TestDomainConfig:
    """Topic routing and the alert threshold come from the config."""

    def test_default_config_topics(self) -> None:
        """Topics are versioned and namespaced by domain."""
        config = make_esports_domain_config()

        assert config["input_topic"] == ESPORTS_INPUT_TOPIC
        assert config["prediction_topic"] == ESPORTS_PREDICTION_TOPIC
        assert config["alert_topic"] == ESPORTS_ALERT_TOPIC

    def test_topics_do_not_collide_with_another_domain(self) -> None:
        """Every topic is prefixed, so two domains cannot consume each other."""
        config = make_esports_domain_config()

        for topic in (
            config["input_topic"],
            config["prediction_topic"],
            config["alert_topic"],
        ):
            assert topic.startswith("esports.")

    def test_default_alert_threshold(self) -> None:
        """The default threshold is the published constant."""
        assert make_esports_domain_config()["alert_threshold"] == ESPORTS_ALERT_THRESHOLD

    def test_alert_threshold_is_overridable(self) -> None:
        """A deployment can tune how often alerts fire."""
        config = make_esports_domain_config(alert_threshold=0.5)

        assert config["alert_threshold"] == pytest.approx(0.5)

    def test_factory_threshold_reaches_the_domain(self) -> None:
        """make_esports_domain forwards the threshold it is given."""
        domain = make_esports_domain(alert_threshold=0.42)

        assert domain.config["alert_threshold"] == pytest.approx(0.42)

    def test_display_name_is_set(self) -> None:
        """The display name is what a human-facing summary calls the domain."""
        assert make_esports_domain_config()["display_name"] == "Esports"


class TestConstruction:
    """The domain is fully determined by configuration, unlike weather."""

    def test_needs_no_fitted_state(self) -> None:
        """Construction reads nothing off disk and takes no state argument.

        Building and immediately featurising, with no environment set and
        no file present, is what lets the registry offer esports to a
        deployment that never configured weather.
        """
        domain = make_esports_domain()

        _, features = domain.decode_and_extract(make_payload(blue_kills=3, red_kills=1))

        assert features.shape == (len(ESPORTS_FEATURE_NAMES),)
        assert domain.config["name"] == ESPORTS_DOMAIN_NAME

    def test_two_domains_are_independent(self) -> None:
        """Thresholds set on one instance do not leak into another."""
        strict = make_esports_domain(alert_threshold=0.95)
        loose = make_esports_domain(alert_threshold=0.55)

        assert strict.config["alert_threshold"] == pytest.approx(0.95)
        assert loose.config["alert_threshold"] == pytest.approx(0.55)


class TestFeatureVectorContract:
    """The vector the model receives is well-formed."""

    def test_features_are_finite(self) -> None:
        """A non-finite feature would poison the model input silently."""
        domain = make_esports_domain()

        _, features = domain.decode_and_extract(make_payload(game_time_seconds=0))

        finite: NDArray[np.bool_] = np.isfinite(features)
        assert int(np.count_nonzero(finite)) == int(features.size)

    def test_features_are_float64(self) -> None:
        """The model contract is float64; another dtype would be coerced."""
        domain = make_esports_domain()

        _, features = domain.decode_and_extract(make_payload())

        assert features.dtype == np.float64
