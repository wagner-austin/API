"""Shared builders for weather domain tests.

The temporal state is a nested structure of coefficient tuples, and both the
extractor tests and the domain tests need one. Building it in each file would
mean two copies drifting apart, so the builders live here.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from covenant_ml.datasets.types import (
    SeasonalCycleCoefficients,
    TailThresholds,
    TemporalFeatureConfig,
    TemporalFeatureState,
)


def make_temporal_feature_config(n_harmonics: int) -> TemporalFeatureConfig:
    """Create a temporal feature config for the given harmonic count.

    Args:
        n_harmonics: Number of Fourier harmonics in the seasonal cycle.

    Returns:
        TemporalFeatureConfig with warm-season defaults.
    """
    return {
        "n_fourier_harmonics": n_harmonics,
        "hot_cutoff_percentile": 95.0,
        "cold_cutoff_percentile": 5.0,
        "season": "warm",
        "season_months": (6, 7, 8),
        "compute_ar1": False,
    }


def make_single_location_state(
    mean: float,
    cos_coefficients: tuple[tuple[float, ...], ...],
    sin_coefficients: tuple[tuple[float, ...], ...],
    hot_threshold: float,
    cold_threshold: float,
    median_baseline: float = 0.0,
) -> TemporalFeatureState:
    """Create a single-location TemporalFeatureState.

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
        "config": make_temporal_feature_config(n_harmonics),
        "seasonal_cycle": seasonal_cycle,
        "thresholds": thresholds,
        "median_baseline": (median_baseline,),
        "n_locations": 1,
    }


def make_flat_state(
    hot_threshold: float = 5.0,
    cold_threshold: float = -5.0,
    mean: float = 0.0,
    median_baseline: float = 0.0,
) -> TemporalFeatureState:
    """Create a single-location state with no seasonal variation.

    Zero harmonics make the reconstructed seasonal value equal the mean, so a
    test can predict the anomaly from the temperature alone.

    Args:
        hot_threshold: Hot-tail threshold.
        cold_threshold: Cold-tail threshold.
        mean: Mean value for the location.
        median_baseline: Within-season median baseline.

    Returns:
        TemporalFeatureState with one location and a flat seasonal cycle.
    """
    return make_single_location_state(
        mean=mean,
        cos_coefficients=((0.0,),),
        sin_coefficients=((0.0,),),
        hot_threshold=hot_threshold,
        cold_threshold=cold_threshold,
        median_baseline=median_baseline,
    )


__all__ = [
    "make_flat_state",
    "make_single_location_state",
    "make_temporal_feature_config",
]
