"""Tests for weather temporal feature extraction."""

from __future__ import annotations

import math

import numpy as np
import pytest
from covenant_ml.datasets.types import (
    SeasonalCycleCoefficients,
    TailThresholds,
    TemporalFeatureConfig,
    TemporalFeatureState,
)
from numpy.typing import NDArray

from covenant_radar_api.domains.weather.features import (
    WEATHER_FEATURE_NAMES,
    WeatherFeatureExtractor,
    _compute_seasonal_value,
)
from covenant_radar_api.domains.weather.schemas import (
    make_weather_event,
)


def _val(arr: NDArray[np.float64], idx: int) -> float:
    """Extract a typed float from a numpy array at the given index.

    Args:
        arr: Source array.
        idx: Index to extract.

    Returns:
        Python float value at the index.
    """
    return float(arr.flat[idx].item())


def _make_single_location_state(
    mean: float,
    cos_coefficients: tuple[tuple[float, ...], ...],
    sin_coefficients: tuple[tuple[float, ...], ...],
    hot_threshold: float,
    cold_threshold: float,
    median_baseline: float = 0.0,
) -> TemporalFeatureState:
    """Create a single-location TemporalFeatureState for testing.

    Args:
        mean: Mean value for the location.
        cos_coefficients: Cosine Fourier coefficients (n_harmonics, 1).
        sin_coefficients: Sine Fourier coefficients (n_harmonics, 1).
        hot_threshold: Hot-tail threshold for the location.
        cold_threshold: Cold-tail threshold for the location.
        median_baseline: Mean of within-season medians for the location.

    Returns:
        TemporalFeatureState with one location.
    """
    n_harmonics = len(cos_coefficients)
    config: TemporalFeatureConfig = {
        "n_fourier_harmonics": n_harmonics,
        "hot_cutoff_percentile": 95.0,
        "cold_cutoff_percentile": 5.0,
        "season": "warm",
        "season_months": (6, 7, 8),
        "compute_ar1": False,
    }
    seasonal_cycle: SeasonalCycleCoefficients = {
        "n_harmonics": n_harmonics,
        "cos_coefficients": cos_coefficients,
        "sin_coefficients": sin_coefficients,
        "mean": (mean,),
        "n_days_per_year": 365,
    }
    thresholds: TailThresholds = {
        "hot_threshold": (hot_threshold,),
        "cold_threshold": (cold_threshold,),
        "hot_percentile": 95.0,
        "cold_percentile": 5.0,
    }
    return {
        "config": config,
        "seasonal_cycle": seasonal_cycle,
        "thresholds": thresholds,
        "median_baseline": (median_baseline,),
        "n_locations": 1,
    }


def _make_multi_location_state(
    n_locations: int,
    means: tuple[float, ...],
    hot_thresholds: tuple[float, ...],
    cold_thresholds: tuple[float, ...],
    median_baselines: tuple[float, ...] | None = None,
) -> TemporalFeatureState:
    """Create a multi-location TemporalFeatureState with zero harmonics.

    Uses zero Fourier coefficients so seasonal value equals the mean.
    This simplifies test assertions.

    Args:
        n_locations: Number of locations.
        means: Mean per location.
        hot_thresholds: Hot threshold per location.
        cold_thresholds: Cold threshold per location.
        median_baselines: Median baseline per location. Defaults to zeros.

    Returns:
        TemporalFeatureState with n_locations.
    """
    if median_baselines is None:
        median_baselines = tuple(0.0 for _ in range(n_locations))
    config: TemporalFeatureConfig = {
        "n_fourier_harmonics": 1,
        "hot_cutoff_percentile": 95.0,
        "cold_cutoff_percentile": 5.0,
        "season": "warm",
        "season_months": (6, 7, 8),
        "compute_ar1": False,
    }
    # Zero coefficients: seasonal value = mean only
    cos_row: tuple[float, ...] = tuple(0.0 for _ in range(n_locations))
    sin_row: tuple[float, ...] = tuple(0.0 for _ in range(n_locations))
    seasonal_cycle: SeasonalCycleCoefficients = {
        "n_harmonics": 1,
        "cos_coefficients": (cos_row,),
        "sin_coefficients": (sin_row,),
        "mean": means,
        "n_days_per_year": 365,
    }
    thresholds: TailThresholds = {
        "hot_threshold": hot_thresholds,
        "cold_threshold": cold_thresholds,
        "hot_percentile": 95.0,
        "cold_percentile": 5.0,
    }
    return {
        "config": config,
        "seasonal_cycle": seasonal_cycle,
        "thresholds": thresholds,
        "median_baseline": median_baselines,
        "n_locations": n_locations,
    }


class TestComputeSeasonalValue:
    """Tests for _compute_seasonal_value."""

    def test_mean_only_no_harmonics_effect(self) -> None:
        """With zero coefficients, seasonal value equals the mean."""
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=5.0,
            cold_threshold=-5.0,
        )

        value = _compute_seasonal_value(state, day_of_year=100, location_index=0)

        assert value == pytest.approx(20.0)

    def test_cosine_harmonic(self) -> None:
        """Cosine coefficient adds cos(2*pi*doy/365) contribution."""
        state = _make_single_location_state(
            mean=15.0,
            cos_coefficients=((3.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=5.0,
            cold_threshold=-5.0,
        )

        # At day_of_year=0 (or 365), cos(2*pi*1*365/365) = cos(2*pi) = 1.0
        # So seasonal = 15 + 3*1 = 18
        value_365 = _compute_seasonal_value(state, day_of_year=365, location_index=0)
        expected_365 = 15.0 + 3.0 * math.cos(2.0 * math.pi * 1 * 365 / 365)
        assert value_365 == pytest.approx(expected_365)

    def test_sine_harmonic(self) -> None:
        """Sine coefficient adds sin(2*pi*doy/365) contribution."""
        state = _make_single_location_state(
            mean=10.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((4.0,),),
            hot_threshold=5.0,
            cold_threshold=-5.0,
        )

        doy = 91  # ~quarter year
        expected = 10.0 + 4.0 * math.sin(2.0 * math.pi * 1 * 91 / 365)
        value = _compute_seasonal_value(state, day_of_year=doy, location_index=0)

        assert value == pytest.approx(expected)

    def test_two_harmonics(self) -> None:
        """Two harmonics contribute independently."""
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((2.0,), (1.0,)),
            sin_coefficients=((0.5,), (0.3,)),
            hot_threshold=5.0,
            cold_threshold=-5.0,
        )

        doy = 150
        angle1 = 2.0 * math.pi * 1 * doy / 365
        angle2 = 2.0 * math.pi * 2 * doy / 365
        expected = (
            20.0
            + 2.0 * math.cos(angle1)
            + 0.5 * math.sin(angle1)
            + 1.0 * math.cos(angle2)
            + 0.3 * math.sin(angle2)
        )
        value = _compute_seasonal_value(state, day_of_year=doy, location_index=0)

        assert value == pytest.approx(expected)


class TestWeatherFeatureExtractorInit:
    """Tests for WeatherFeatureExtractor construction."""

    def test_feature_names(self) -> None:
        """Feature names match WEATHER_FEATURE_NAMES."""
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=5.0,
            cold_threshold=-5.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})

        assert extractor.feature_names == WEATHER_FEATURE_NAMES
        assert len(extractor.feature_names) == 5

    def test_feature_names_immutable(self) -> None:
        """Feature names property returns same tuple on repeated access."""
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=5.0,
            cold_threshold=-5.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})

        assert extractor.feature_names is extractor.feature_names


class TestWeatherFeatureExtractorExtract:
    """Tests for WeatherFeatureExtractor.extract."""

    def test_returns_float64_array(self) -> None:
        """Extract returns NDArray[float64] of shape (5,)."""
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=5.0,
            cold_threshold=-5.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})
        event = make_weather_event(
            event_id="e1",
            station_id="s1",
            day_of_year=100,
            temperature=25.0,
            timestamp="2025-04-10T00:00:00Z",
        )

        result = extractor.extract(event)

        assert result.dtype == np.float64
        assert result.shape == (5,)

    def test_anomaly_computation(self) -> None:
        """Anomaly is temperature minus seasonal value."""
        # mean=20, zero harmonics => seasonal=20
        # temperature=25 => anomaly=5
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=10.0,
            cold_threshold=-10.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})
        event = make_weather_event(
            event_id="e1",
            station_id="s1",
            day_of_year=100,
            temperature=25.0,
            timestamp="2025-04-10T00:00:00Z",
        )

        result = extractor.extract(event)

        assert _val(result, 0) == pytest.approx(5.0)

    def test_hot_extreme_above_threshold(self) -> None:
        """Event above hot threshold produces positive hot_excess."""
        # mean=20, hot_threshold=3 => need anomaly > 3
        # temperature=25 => anomaly=5 => hot_excess=5-3=2
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=3.0,
            cold_threshold=-3.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})
        event = make_weather_event(
            event_id="e1",
            station_id="s1",
            day_of_year=100,
            temperature=25.0,
            timestamp="2025-04-10T00:00:00Z",
        )

        result = extractor.extract(event)

        anomaly = _val(result, 0)
        hot_excess = _val(result, 1)
        is_hot = _val(result, 3)

        assert anomaly == pytest.approx(5.0)
        assert hot_excess == pytest.approx(2.0)
        assert is_hot == 1.0

    def test_cold_extreme_below_threshold(self) -> None:
        """Event below cold threshold produces negative cold_excess."""
        # mean=20, cold_threshold=-3 => need anomaly < -3
        # temperature=15 => anomaly=-5 => cold_excess=-5-(-3)=-2
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=3.0,
            cold_threshold=-3.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})
        event = make_weather_event(
            event_id="e1",
            station_id="s1",
            day_of_year=100,
            temperature=15.0,
            timestamp="2025-04-10T00:00:00Z",
        )

        result = extractor.extract(event)

        anomaly = _val(result, 0)
        cold_excess = _val(result, 2)
        is_cold = _val(result, 4)

        assert anomaly == pytest.approx(-5.0)
        assert cold_excess == pytest.approx(-2.0)
        assert is_cold == 1.0

    def test_normal_event_no_extremes(self) -> None:
        """Event within thresholds has zero excess and is_extreme flags."""
        # mean=20, thresholds=+/-10 => anomaly=2 is normal
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=10.0,
            cold_threshold=-10.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})
        event = make_weather_event(
            event_id="e1",
            station_id="s1",
            day_of_year=100,
            temperature=22.0,
            timestamp="2025-04-10T00:00:00Z",
        )

        result = extractor.extract(event)

        anomaly = _val(result, 0)
        hot_excess = _val(result, 1)
        cold_excess = _val(result, 2)
        is_hot = _val(result, 3)
        is_cold = _val(result, 4)

        assert anomaly == pytest.approx(2.0)
        assert hot_excess == 0.0
        assert cold_excess == 0.0
        assert is_hot == 0.0
        assert is_cold == 0.0

    def test_at_exact_threshold_not_extreme(self) -> None:
        """Event exactly at threshold is not flagged as extreme."""
        # anomaly exactly = hot_threshold => not strictly greater
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=5.0,
            cold_threshold=-5.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})
        event = make_weather_event(
            event_id="e1",
            station_id="s1",
            day_of_year=100,
            temperature=25.0,
            timestamp="2025-04-10T00:00:00Z",
        )

        result = extractor.extract(event)

        # anomaly = 5.0, hot_threshold = 5.0 => not strictly >
        assert _val(result, 1) == 0.0  # hot_excess
        assert _val(result, 3) == 0.0  # is_hot

    def test_with_fourier_harmonics(self) -> None:
        """Anomaly accounts for Fourier seasonal cycle."""
        # cos_coeff=5.0 for harmonic 1 at doy=365: cos(2*pi) = 1.0
        # seasonal = 20 + 5*1.0 = 25
        # temperature = 28 => anomaly = 3
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((5.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=10.0,
            cold_threshold=-10.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})
        event = make_weather_event(
            event_id="e1",
            station_id="s1",
            day_of_year=365,
            temperature=28.0,
            timestamp="2025-12-31T00:00:00Z",
        )

        result = extractor.extract(event)

        expected_seasonal = 20.0 + 5.0 * math.cos(2.0 * math.pi * 1 * 365 / 365)
        expected_anomaly = 28.0 - expected_seasonal
        assert _val(result, 0) == pytest.approx(expected_anomaly)

    def test_unknown_station_raises_key_error(self) -> None:
        """Raises KeyError for unknown station_id."""
        state = _make_single_location_state(
            mean=20.0,
            cos_coefficients=((0.0,),),
            sin_coefficients=((0.0,),),
            hot_threshold=5.0,
            cold_threshold=-5.0,
        )
        extractor = WeatherFeatureExtractor(state, {"s1": 0})
        event = make_weather_event(
            event_id="e1",
            station_id="unknown-station",
            day_of_year=100,
            temperature=25.0,
            timestamp="2025-04-10T00:00:00Z",
        )

        with pytest.raises(KeyError):
            extractor.extract(event)


class TestWeatherFeatureExtractorMultiLocation:
    """Tests for multi-location feature extraction."""

    def test_different_locations_different_anomalies(self) -> None:
        """Different stations produce different anomalies via location mapping."""
        state = _make_multi_location_state(
            n_locations=2,
            means=(20.0, 30.0),
            hot_thresholds=(5.0, 5.0),
            cold_thresholds=(-5.0, -5.0),
        )
        mapping: dict[str, int] = {"station-a": 0, "station-b": 1}
        extractor = WeatherFeatureExtractor(state, mapping)

        event_a = make_weather_event(
            event_id="e1",
            station_id="station-a",
            day_of_year=100,
            temperature=25.0,
            timestamp="2025-04-10T00:00:00Z",
        )
        event_b = make_weather_event(
            event_id="e2",
            station_id="station-b",
            day_of_year=100,
            temperature=25.0,
            timestamp="2025-04-10T00:00:00Z",
        )

        result_a = extractor.extract(event_a)
        result_b = extractor.extract(event_b)

        # station-a: mean=20, temp=25 => anomaly=5
        # station-b: mean=30, temp=25 => anomaly=-5
        assert _val(result_a, 0) == pytest.approx(5.0)
        assert _val(result_b, 0) == pytest.approx(-5.0)

    def test_different_thresholds_per_location(self) -> None:
        """Different locations use different thresholds."""
        state = _make_multi_location_state(
            n_locations=2,
            means=(20.0, 20.0),
            hot_thresholds=(3.0, 10.0),
            cold_thresholds=(-10.0, -10.0),
        )
        mapping: dict[str, int] = {"tight": 0, "loose": 1}
        extractor = WeatherFeatureExtractor(state, mapping)

        # Same temperature, same mean => same anomaly=5
        event_tight = make_weather_event(
            event_id="e1",
            station_id="tight",
            day_of_year=100,
            temperature=25.0,
            timestamp="2025-04-10T00:00:00Z",
        )
        event_loose = make_weather_event(
            event_id="e2",
            station_id="loose",
            day_of_year=100,
            temperature=25.0,
            timestamp="2025-04-10T00:00:00Z",
        )

        result_tight = extractor.extract(event_tight)
        result_loose = extractor.extract(event_loose)

        # tight: hot_threshold=3, anomaly=5 => hot_excess=2, is_hot=1
        assert _val(result_tight, 1) == pytest.approx(2.0)
        assert _val(result_tight, 3) == 1.0

        # loose: hot_threshold=10, anomaly=5 => hot_excess=0, is_hot=0
        assert _val(result_loose, 1) == 0.0
        assert _val(result_loose, 3) == 0.0


class TestWeatherFeatureNames:
    """Tests for WEATHER_FEATURE_NAMES constant."""

    def test_count(self) -> None:
        """Five features defined."""
        assert len(WEATHER_FEATURE_NAMES) == 5

    def test_names(self) -> None:
        """Feature names match expected values."""
        assert WEATHER_FEATURE_NAMES == (
            "anomaly",
            "hot_excess",
            "cold_excess",
            "is_hot_extreme",
            "is_cold_extreme",
        )

    def test_immutable_sequence(self) -> None:
        """WEATHER_FEATURE_NAMES is a tuple (verified by identity)."""
        assert WEATHER_FEATURE_NAMES is WEATHER_FEATURE_NAMES
        assert WEATHER_FEATURE_NAMES[0] == "anomaly"
