"""Weather temporal feature extraction.

Extracts McKinnon-style temporal features from single weather observation
events using a pre-fitted TemporalFeatureState. The state is injected via
the constructor; the extract() method is stateless from the caller's
perspective.

Features produced per event:
- anomaly: temperature minus reconstructed Fourier seasonal value minus
  median baseline (approximates McKinnon's residual for streaming)
- hot_excess: max(0, anomaly - hot_threshold)
- cold_excess: min(0, anomaly - cold_threshold)
- is_hot_extreme: 1.0 if anomaly > hot_threshold, else 0.0
- is_cold_extreme: 1.0 if anomaly < cold_threshold, else 0.0
"""

from __future__ import annotations

import math

import numpy as np
from covenant_ml.datasets.types import TemporalFeatureState
from numpy.typing import NDArray

from .schemas import WeatherEventV1

WEATHER_FEATURE_NAMES: tuple[str, ...] = (
    "anomaly",
    "hot_excess",
    "cold_excess",
    "is_hot_extreme",
    "is_cold_extreme",
)


class WeatherFeatureExtractor:
    """Extract temporal features from weather observation events.

    Uses pre-fitted TemporalFeatureState (Fourier seasonal cycle coefficients
    and tail-excess thresholds) to produce per-event features for ML inference.

    The state is injected via the constructor so the protocol-level extract()
    signature remains stateless. Station IDs are mapped to location indices
    via the station_to_location dictionary.

    Attributes:
        feature_names: Ordered tuple of feature names matching extract output.
    """

    def __init__(
        self,
        state: TemporalFeatureState,
        station_to_location: dict[str, int],
    ) -> None:
        """Initialize with pre-fitted state and station mapping.

        Args:
            state: Pre-fitted temporal feature state from training data.
                Contains Fourier seasonal cycle coefficients and tail-excess
                thresholds per location.
            station_to_location: Mapping from station_id strings to location
                indices in the fitted state arrays.
        """
        self._state: TemporalFeatureState = state
        self._station_to_location: dict[str, int] = station_to_location

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return ordered tuple of feature names."""
        return WEATHER_FEATURE_NAMES

    def extract(self, event: WeatherEventV1) -> NDArray[np.float64]:
        """Extract temporal features from a single weather event.

        Steps:
            1. Look up location index from station_id.
            2. Reconstruct seasonal value from fitted Fourier coefficients.
            3. Compute residual (temperature minus seasonal value minus
               median baseline).
            4. Compare residual against fitted tail-excess thresholds.

        Args:
            event: Weather observation event with station_id, day_of_year,
                and temperature fields.

        Returns:
            Feature vector of shape (5,) with dtype float64, containing
            [anomaly, hot_excess, cold_excess, is_hot_extreme, is_cold_extreme].

        Raises:
            KeyError: If event station_id is not in station_to_location mapping.
        """
        station_id: str = event["station_id"]
        loc_idx: int = self._station_to_location[station_id]
        doy: int = event["day_of_year"]
        temperature: float = event["temperature"]

        # Step 1: Reconstruct seasonal value from Fourier coefficients
        seasonal_value: float = _compute_seasonal_value(
            self._state,
            doy,
            loc_idx,
        )

        # Step 2: Compute residual (anomaly minus median baseline)
        median: float = self._state["median_baseline"][loc_idx]
        anomaly: float = temperature - seasonal_value - median

        # Step 3: Compare against tail-excess thresholds
        hot_threshold: float = self._state["thresholds"]["hot_threshold"][loc_idx]
        cold_threshold: float = self._state["thresholds"]["cold_threshold"][loc_idx]

        hot_excess: float = max(0.0, anomaly - hot_threshold)
        cold_excess: float = min(0.0, anomaly - cold_threshold)
        is_hot: float = 1.0 if anomaly > hot_threshold else 0.0
        is_cold: float = 1.0 if anomaly < cold_threshold else 0.0

        result: NDArray[np.float64] = np.zeros(5, dtype=np.float64)
        result[0] = anomaly
        result[1] = hot_excess
        result[2] = cold_excess
        result[3] = is_hot
        result[4] = is_cold
        return result


def _compute_seasonal_value(
    state: TemporalFeatureState,
    day_of_year: int,
    location_index: int,
) -> float:
    """Reconstruct seasonal value from Fourier coefficients.

    Evaluates the fitted Fourier series at the given day_of_year for
    a specific location:

        value = mean + sum_k(cos_k * cos(2*pi*k*doy/N) +
                             sin_k * sin(2*pi*k*doy/N))

    where k = 1..n_harmonics and N = n_days_per_year.

    Args:
        state: Fitted temporal feature state.
        day_of_year: Day of year (1-366).
        location_index: Index into the location dimension.

    Returns:
        Reconstructed seasonal value at the given day and location.
    """
    coeffs = state["seasonal_cycle"]
    n_harmonics: int = coeffs["n_harmonics"]
    n_days: int = coeffs["n_days_per_year"]
    value: float = coeffs["mean"][location_index]

    for k in range(n_harmonics):
        cos_coeff: float = coeffs["cos_coefficients"][k][location_index]
        sin_coeff: float = coeffs["sin_coefficients"][k][location_index]
        angle: float = 2.0 * math.pi * (k + 1) * day_of_year / n_days
        value += cos_coeff * math.cos(angle) + sin_coeff * math.sin(angle)

    return value


__all__ = [
    "WEATHER_FEATURE_NAMES",
    "WeatherFeatureExtractor",
]
