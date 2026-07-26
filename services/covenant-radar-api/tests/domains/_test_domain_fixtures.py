"""Shared fake domain for tests of the multi-domain platform.

One fake serves the protocol tests, the registry tests and the generic
worker tests. Each of those directories previously carried its own copy,
which meant a change to DomainProtocol had to be mirrored in several places
before the suite would compile.

The features are a deterministic function of entity_id length, so a test can
predict exactly what reaches the model from the payload it produced, without
reaching inside the fake.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_radar_api.domains.base_schemas import (
    BaseInputEventV1,
    BasePredictionEventV1,
    decode_base_input_event,
    encode_base_prediction_event,
)
from covenant_radar_api.domains.protocols import DomainConfig, make_domain_config

DEFAULT_FAKE_FEATURE_NAMES: tuple[str, ...] = ("feat_a", "feat_b", "feat_c")


class FakeDomain:
    """Fake domain satisfying DomainProtocol.

    Records every call it receives so tests can assert on the interaction
    rather than on the fake's internals.

    Attributes:
        decode_calls: Payloads passed to decode_and_extract, in order.
        encode_calls: JSON strings returned by encode_prediction_event.
        alert_context_calls: (entity_id, prediction_value) pairs received.
    """

    def __init__(
        self,
        config: DomainConfig,
        feature_names: tuple[str, ...] = DEFAULT_FAKE_FEATURE_NAMES,
    ) -> None:
        """Initialize the fake domain.

        Args:
            config: Domain configuration this fake reports.
            feature_names: Ordered feature names this fake produces.
        """
        self._config: DomainConfig = config
        self._feature_names: tuple[str, ...] = feature_names
        self.decode_calls: list[str] = []
        self.encode_calls: list[str] = []
        self.alert_context_calls: list[tuple[str, float]] = []

    @property
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        return self._config

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered feature names this domain produces."""
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Return how many features decode_and_extract produces."""
        return len(self._feature_names)

    def decode_and_extract(
        self,
        payload: str,
    ) -> tuple[BaseInputEventV1, NDArray[np.float64]]:
        """Decode the payload and derive features from it.

        Args:
            payload: Raw JSON string.

        Returns:
            The decoded event and a 1D float64 array of shape (n_features,).

        Raises:
            JSONTypeError: If a required field is missing or mistyped.
            InvalidJsonError: If the payload is not valid JSON.
        """
        self.decode_calls.append(payload)
        event: BaseInputEventV1 = decode_base_input_event(payload)
        entity_len: float = float(len(event["entity_id"]))
        features: NDArray[np.float64] = np.zeros(self.n_features, dtype=np.float64)
        for index in range(self.n_features):
            features[index] = entity_len * (index + 1)
        return event, features

    def encode_prediction_event(self, event: BasePredictionEventV1) -> str:
        """Encode a prediction event to JSON.

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
        """Generate the alert context dictionary.

        Args:
            entity_id: Primary entity identifier.
            prediction_value: Prediction value that triggered the alert.

        Returns:
            Context dictionary for the alert prompt.
        """
        self.alert_context_calls.append((entity_id, prediction_value))
        return {
            "domain": self._config["name"],
            "entity_id": entity_id,
            "prediction_value": f"{prediction_value:.4f}",
        }


def make_fake_domain_config(name: str = "test") -> DomainConfig:
    """Create a domain config for a fake domain.

    Args:
        name: Domain identifier, used to derive the topic names.

    Returns:
        DomainConfig with topics derived from the name.
    """
    return make_domain_config(
        name=name,
        display_name=f"Fake {name}",
        input_topic=f"{name}.input.v1",
        prediction_topic=f"{name}.predictions.v1",
        alert_topic=f"{name}.alerts.v1",
        alert_threshold=0.80,
    )


def make_fake_domain(
    name: str = "test",
    feature_names: tuple[str, ...] = DEFAULT_FAKE_FEATURE_NAMES,
) -> FakeDomain:
    """Create a fake domain with a config derived from its name.

    Args:
        name: Domain identifier.
        feature_names: Ordered feature names the fake produces.

    Returns:
        FakeDomain ready to register or inject.
    """
    return FakeDomain(make_fake_domain_config(name), feature_names)


__all__ = [
    "DEFAULT_FAKE_FEATURE_NAMES",
    "FakeDomain",
    "make_fake_domain",
    "make_fake_domain_config",
]
