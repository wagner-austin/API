"""Fake model and metrics implementations for streaming worker testing.

Provides test doubles for ML predictor and Datadog metrics sink.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FakePredictor:
    """Fake ML predictor for testing.

    Returns configurable probabilities for testing different scenarios.
    """

    def __init__(self, default_probability: float = 0.25) -> None:
        """Initialize with default probability.

        Args:
            default_probability: Probability to return for all predictions.
        """
        self._default_probability = default_probability
        self._call_count = 0

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities.

        Args:
            x: Feature array of shape (n_samples, n_features).

        Returns:
            Probability array of shape (n_samples, 2).
        """
        n_samples = x.shape[0]
        self._call_count += 1
        # Return [1-p, p] for each sample (binary classification)
        result = np.zeros((n_samples, 2), dtype=np.float64)
        for i in range(n_samples):
            result[i, 0] = 1.0 - self._default_probability
            result[i, 1] = self._default_probability
        return result

    @property
    def call_count(self) -> int:
        """Get number of times predict_proba was called."""
        return self._call_count


class FakeMetricsSink:
    """Fake metrics sink for testing.

    Records all metrics calls for verification.
    """

    def __init__(self) -> None:
        """Initialize with empty metric records."""
        self.increments: list[tuple[str, int, tuple[str, ...]]] = []
        self.gauges: list[tuple[str, float, tuple[str, ...]]] = []
        self.histograms: list[tuple[str, float, tuple[str, ...]]] = []

    def increment(
        self,
        metric: str,
        value: int,
        tags: tuple[str, ...],
    ) -> None:
        """Record increment call.

        Args:
            metric: Metric name.
            value: Increment value.
            tags: Metric tags.
        """
        self.increments.append((metric, value, tags))

    def gauge(
        self,
        metric: str,
        value: float,
        tags: tuple[str, ...],
    ) -> None:
        """Record gauge call.

        Args:
            metric: Metric name.
            value: Gauge value.
            tags: Metric tags.
        """
        self.gauges.append((metric, value, tags))

    def histogram(
        self,
        metric: str,
        value: float,
        tags: tuple[str, ...],
    ) -> None:
        """Record histogram call.

        Args:
            metric: Metric name.
            value: Histogram value.
            tags: Metric tags.
        """
        self.histograms.append((metric, value, tags))


__all__ = [
    "FakeMetricsSink",
    "FakePredictor",
]
