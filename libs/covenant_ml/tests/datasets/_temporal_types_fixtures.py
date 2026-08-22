"""Shared fixtures and helpers for test_temporal_types splits."""

from __future__ import annotations

from covenant_ml.datasets.types_temporal import (
    SeasonalCycleCoefficients,
    TailThresholds,
    TemporalFeatureConfig,
    TemporalFeatureState,
)


def _make_test_state() -> TemporalFeatureState:
    """Create a test TemporalFeatureState for encode/decode tests."""
    return TemporalFeatureState(
        config=TemporalFeatureConfig(
            n_fourier_harmonics=2,
            hot_cutoff_percentile=95.0,
            cold_cutoff_percentile=5.0,
            season="warm",
            season_months=(6, 7, 8),
            compute_ar1=True,
        ),
        seasonal_cycle=SeasonalCycleCoefficients(
            n_harmonics=2,
            cos_coefficients=((1.5, 2.5), (3.5, 4.5)),
            sin_coefficients=((0.5, 1.5), (2.5, 3.5)),
            mean=(20.0, 22.0),
            n_days_per_year=365,
        ),
        thresholds=TailThresholds(
            hot_threshold=(2.0, 2.5),
            cold_threshold=(-2.0, -2.5),
            hot_percentile=95.0,
            cold_percentile=5.0,
        ),
        median_baseline=(0.3, -0.2),
        n_locations=2,
    )
