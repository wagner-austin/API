"""McKinnon-style temporal feature extraction for atmospheric data.

Pure numpy implementation of Fourier deseasonalization, tail-excess heat
metrics, and lag-1 autocorrelation from McKinnon (PNAS 2024, "The pace
of change of summertime temperature extremes").

All functions operate on multi-location data with shapes:
- daily_values: (n_days, n_locations)
- day_of_year: (n_days,)
- year_labels: (n_days,)

Internal module - used by the future netcdf_loader.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.types import (
    HEAT_METRIC_NAMES,
    HEAT_METRIC_NAMES_NO_AR1,
    SeasonalCycleCoefficients,
    TailThresholds,
    TemporalFeatureConfig,
    TemporalFeatureState,
)

# --- Protocol wrappers for numpy functions with insufficiently typed stubs ---


class _LinalgSolveProtocol(Protocol):
    """Protocol for numpy.linalg.solve with strict float64 typing."""

    def __call__(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _TrigProtocol(Protocol):
    """Protocol for numpy trig functions (cos, sin) with strict typing."""

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _MedianProtocol(Protocol):
    """Protocol for numpy.median returning float."""

    def __call__(self, a: NDArray[np.float64]) -> float: ...


class _PercentileProtocol(Protocol):
    """Protocol for numpy.percentile returning float."""

    def __call__(self, a: NDArray[np.float64], q: float) -> float: ...


_numpy_mod = __import__("numpy")
_linalg_mod = __import__("numpy.linalg", fromlist=["solve"])
_linalg_solve: _LinalgSolveProtocol = _linalg_mod.solve
_np_cos: _TrigProtocol = _numpy_mod.cos
_np_sin: _TrigProtocol = _numpy_mod.sin
_np_median: _MedianProtocol = _numpy_mod.median
_np_percentile: _PercentileProtocol = _numpy_mod.percentile


def _build_fourier_basis(
    day_of_year: NDArray[np.int64],
    n_harmonics: int,
    n_days_per_year: int,
) -> NDArray[np.float64]:
    """Build Fourier basis matrix for seasonal cycle fitting.

    Constructs the design matrix [1, cos1, sin1, cos2, sin2, ...] where
    cos_k = cos(2*pi*k*doy/N) and sin_k = sin(2*pi*k*doy/N).

    Args:
        day_of_year: Day-of-year values, shape (n_days,). Values 1-365/366.
        n_harmonics: Number of Fourier harmonics.
        n_days_per_year: Days per year for frequency calculation.

    Returns:
        Design matrix of shape (n_days, 1 + 2*n_harmonics).
    """
    n_days = int(day_of_year.shape[0])
    n_cols = 1 + 2 * n_harmonics
    basis = np.ones((n_days, n_cols), dtype=np.float64)

    doy_float: NDArray[np.float64] = day_of_year.astype(np.float64)
    for k in range(1, n_harmonics + 1):
        angle: NDArray[np.float64] = 2.0 * np.pi * k * doy_float / n_days_per_year
        basis[:, 2 * k - 1] = _np_cos(angle)
        basis[:, 2 * k] = _np_sin(angle)

    return basis


def fit_seasonal_cycle(
    daily_values: NDArray[np.float64],
    day_of_year: NDArray[np.int64],
    n_harmonics: int,
    n_days_per_year: int = 365,
) -> SeasonalCycleCoefficients:
    """Fit Fourier seasonal cycle to daily temperature values per location.

    Projects daily values onto Fourier basis functions using OLS:
    y(doy, loc) = mean[loc] + sum_k(a_k[loc] * cos(...) + b_k[loc] * sin(...))

    Solves beta = (X^T X)^{-1} X^T Y where Y has shape (n_days, n_locations).

    Args:
        daily_values: Daily temperature values, shape (n_days, n_locations).
        day_of_year: Day-of-year for each observation, shape (n_days,).
            Values should be 1-365 (or 1-366 for leap years).
        n_harmonics: Number of Fourier harmonics to fit.
        n_days_per_year: Days per year for frequency calculation.

    Returns:
        SeasonalCycleCoefficients with fitted Fourier coefficients per location.

    Raises:
        ValueError: If input arrays have mismatched shapes or are empty.
    """
    if len(daily_values.shape) != 2:
        raise ValueError(f"daily_values must be 2D, got shape {daily_values.shape}")
    n_days = int(daily_values.shape[0])
    n_locations = int(daily_values.shape[1])
    n_doy = int(day_of_year.shape[0])
    if n_days != n_doy:
        raise ValueError(f"Shape mismatch: daily_values ({n_days}) vs day_of_year ({n_doy})")
    if n_days == 0:
        raise ValueError("Cannot fit seasonal cycle to empty array")

    basis = _build_fourier_basis(day_of_year, n_harmonics, n_days_per_year)

    # OLS: beta = (X^T X)^{-1} X^T Y, beta shape (n_cols, n_locations)
    xtx: NDArray[np.float64] = basis.T @ basis
    xty: NDArray[np.float64] = basis.T @ daily_values
    beta: NDArray[np.float64] = _linalg_solve(xtx, xty)

    # Extract per-location mean
    mean_row: NDArray[np.float64] = beta[0, :]
    mean_vals = tuple(float(mean_row.flat[j]) for j in range(n_locations))

    # Extract per-harmonic, per-location coefficients
    cos_coeffs: list[tuple[float, ...]] = []
    sin_coeffs: list[tuple[float, ...]] = []
    for k in range(n_harmonics):
        cos_row: NDArray[np.float64] = beta[2 * k + 1, :]
        sin_row: NDArray[np.float64] = beta[2 * k + 2, :]
        cos_coeffs.append(tuple(float(cos_row.flat[j]) for j in range(n_locations)))
        sin_coeffs.append(tuple(float(sin_row.flat[j]) for j in range(n_locations)))

    return SeasonalCycleCoefficients(
        n_harmonics=n_harmonics,
        cos_coefficients=tuple(cos_coeffs),
        sin_coefficients=tuple(sin_coeffs),
        mean=mean_vals,
        n_days_per_year=n_days_per_year,
    )


def remove_seasonal_cycle(
    daily_values: NDArray[np.float64],
    day_of_year: NDArray[np.int64],
    coefficients: SeasonalCycleCoefficients,
) -> NDArray[np.float64]:
    """Remove fitted Fourier seasonal cycle from daily values per location.

    Reconstructs the seasonal cycle from coefficients and subtracts it
    to produce anomalies (deseasonalized values).

    Args:
        daily_values: Daily temperature values, shape (n_days, n_locations).
        day_of_year: Day-of-year for each observation, shape (n_days,).
        coefficients: Previously fitted Fourier coefficients.

    Returns:
        Anomalies array of shape (n_days, n_locations).
    """
    n_harmonics = coefficients["n_harmonics"]
    n_locations = len(coefficients["mean"])
    n_days_per_year = coefficients["n_days_per_year"]
    basis = _build_fourier_basis(day_of_year, n_harmonics, n_days_per_year)

    # Reconstruct beta matrix (n_cols, n_locations) from coefficients
    n_cols = 1 + 2 * n_harmonics
    beta = np.zeros((n_cols, n_locations), dtype=np.float64)
    for j in range(n_locations):
        beta[0, j] = coefficients["mean"][j]
    for k in range(n_harmonics):
        for j in range(n_locations):
            beta[2 * k + 1, j] = coefficients["cos_coefficients"][k][j]
            beta[2 * k + 2, j] = coefficients["sin_coefficients"][k][j]

    seasonal: NDArray[np.float64] = basis @ beta
    anomalies: NDArray[np.float64] = daily_values - seasonal
    return anomalies


def compute_within_season_medians(
    anomalies: NDArray[np.float64],
    year_labels: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Compute within-season median of anomalies per year per location.

    Groups anomalies by year and computes the median for each year at
    each location. This captures the year-to-year shift in the center
    of the distribution.

    Args:
        anomalies: Deseasonalized daily values, shape (n_days, n_locations).
        year_labels: Year label for each day, shape (n_days,).

    Returns:
        Tuple of (medians, unique_years) where medians has shape
        (n_unique_years, n_locations) and unique_years has shape
        (n_unique_years,).
    """
    unique_years: NDArray[np.int64] = np.unique(year_labels)
    n_unique = int(unique_years.shape[0])
    n_locations = int(anomalies.shape[1])
    medians = np.zeros((n_unique, n_locations), dtype=np.float64)

    for i in range(n_unique):
        yr = int(unique_years.flat[i])
        mask: NDArray[np.bool_] = year_labels == yr
        for j in range(n_locations):
            year_vals: NDArray[np.float64] = anomalies[mask, j]
            medians[i, j] = _np_median(year_vals)

    return medians, unique_years


def compute_residuals(
    anomalies: NDArray[np.float64],
    year_labels: NDArray[np.int64],
    medians: NDArray[np.float64],
    unique_years: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Subtract within-season median from anomalies to produce residuals.

    The residuals isolate distributional shape changes (tail behavior)
    from median shifts. This is the key insight from McKinnon's methodology.

    Args:
        anomalies: Deseasonalized daily values, shape (n_days, n_locations).
        year_labels: Year label for each day, shape (n_days,).
        medians: Per-year per-location medians, shape (n_unique_years, n_locations).
        unique_years: Unique year values, shape (n_unique_years,).

    Returns:
        Residuals array of shape (n_days, n_locations).
    """
    residuals = anomalies.copy()
    n_unique = int(unique_years.shape[0])
    for i in range(n_unique):
        yr = int(unique_years.flat[i])
        mask: NDArray[np.bool_] = year_labels == yr
        residuals[mask, :] -= medians[i, :]
    return residuals


def fit_tail_thresholds(
    residuals: NDArray[np.float64],
    hot_percentile: float,
    cold_percentile: float,
) -> TailThresholds:
    """Compute tail-excess thresholds from training residuals per location.

    Computes the hot and cold percentile thresholds used to define
    extreme days in the heat metric calculations, independently per location.

    Args:
        residuals: Training residuals, shape (n_days, n_locations).
        hot_percentile: Percentile for hot-tail threshold (e.g. 95.0).
        cold_percentile: Percentile for cold-tail threshold (e.g. 5.0).

    Returns:
        TailThresholds with per-location threshold values.
    """
    n_locations = int(residuals.shape[1])
    hot_vals: list[float] = []
    cold_vals: list[float] = []
    for j in range(n_locations):
        col: NDArray[np.float64] = residuals[:, j]
        hot_vals.append(_np_percentile(col, hot_percentile))
        cold_vals.append(_np_percentile(col, cold_percentile))

    return TailThresholds(
        hot_threshold=tuple(hot_vals),
        cold_threshold=tuple(cold_vals),
        hot_percentile=hot_percentile,
        cold_percentile=cold_percentile,
    )


def _compute_ar1_for_year(values: NDArray[np.float64]) -> float:
    """Compute lag-1 autocorrelation for a single year's residuals.

    Args:
        values: Residual values for one year at one location, shape (n_days_in_year,).

    Returns:
        Lag-1 autocorrelation coefficient. Returns 0.0 if fewer than 3 values.
    """
    n = int(values.shape[0])
    if n < 3:
        return 0.0
    x: NDArray[np.float64] = values[:-1]
    y: NDArray[np.float64] = values[1:]
    x_mean = float(np.sum(x)) / (n - 1)
    y_mean = float(np.sum(y)) / (n - 1)
    x_centered: NDArray[np.float64] = x - x_mean
    y_centered: NDArray[np.float64] = y - y_mean
    numerator = float(np.sum(x_centered * y_centered))
    denom_sq = float(np.sum(x_centered**2)) * float(np.sum(y_centered**2))
    if denom_sq < 1e-30:
        return 0.0
    return numerator / math.sqrt(denom_sq)


def _compute_year_hot_metrics(
    year_residuals: NDArray[np.float64],
    hot_thresh: float,
) -> tuple[float, float, float]:
    """Compute hot-tail metrics for a single year at a single location.

    Args:
        year_residuals: Residual values for one year at one location.
        hot_thresh: Hot-tail threshold value for this location.

    Returns:
        Tuple of (cum_excess_hot, avg_excess_hot, ndays_excess_hot).
    """
    hot_mask: NDArray[np.bool_] = year_residuals > hot_thresh
    hot_values: NDArray[np.float64] = year_residuals[hot_mask]
    n_hot = int(hot_values.shape[0])
    if n_hot == 0:
        return 0.0, 0.0, 0.0
    cum = float(np.sum(hot_values))
    avg = cum / n_hot
    return cum, avg, float(n_hot)


def _compute_year_cold_metrics(
    year_residuals: NDArray[np.float64],
    cold_thresh: float,
) -> tuple[float, float, float]:
    """Compute cold-tail metrics for a single year at a single location.

    Args:
        year_residuals: Residual values for one year at one location.
        cold_thresh: Cold-tail threshold value for this location.

    Returns:
        Tuple of (cum_excess_cold, avg_excess_cold, ndays_excess_cold).
    """
    cold_mask: NDArray[np.bool_] = year_residuals < cold_thresh
    cold_values: NDArray[np.float64] = year_residuals[cold_mask]
    n_cold = int(cold_values.shape[0])
    if n_cold == 0:
        return 0.0, 0.0, 0.0
    cum = float(np.sum(cold_values))
    avg = cum / n_cold
    return cum, avg, float(n_cold)


def compute_heat_metrics(
    residuals: NDArray[np.float64],
    year_labels: NDArray[np.int64],
    thresholds: TailThresholds,
    compute_ar1: bool,
) -> NDArray[np.float64]:
    """Compute McKinnon heat metrics from residuals per location.

    For each year and each location computes:
    - seasonal_max: Maximum residual value
    - seasonal_min: Minimum residual value
    - cum_excess_hot: Sum of residuals on days exceeding hot threshold
    - avg_excess_hot: Mean of residuals on days exceeding hot threshold
    - ndays_excess_hot: Count of days exceeding hot threshold
    - cum_excess_cold: Sum of residuals on days below cold threshold
    - avg_excess_cold: Mean of residuals on days below cold threshold
    - ndays_excess_cold: Count of days below cold threshold
    - ar1: Lag-1 autocorrelation within the year (optional)

    Args:
        residuals: Daily residual values, shape (n_days, n_locations).
        year_labels: Year label for each day, shape (n_days,).
        thresholds: Pre-computed per-location tail thresholds.
        compute_ar1: Whether to compute lag-1 autocorrelation.

    Returns:
        Heat metric array of shape (n_years, n_locations, n_metrics).
    """
    unique_years: NDArray[np.int64] = np.unique(year_labels)
    n_years = int(unique_years.shape[0])
    n_locations = int(residuals.shape[1])
    metric_names = HEAT_METRIC_NAMES if compute_ar1 else HEAT_METRIC_NAMES_NO_AR1
    n_metrics = len(metric_names)

    features = np.zeros((n_years, n_locations, n_metrics), dtype=np.float64)

    for i in range(n_years):
        yr = int(unique_years.flat[i])
        mask: NDArray[np.bool_] = year_labels == yr

        for j in range(n_locations):
            year_loc: NDArray[np.float64] = residuals[mask, j]
            hot_thresh = thresholds["hot_threshold"][j]
            cold_thresh = thresholds["cold_threshold"][j]

            # seasonal_max, seasonal_min
            features[i, j, 0] = float(np.max(year_loc))
            features[i, j, 1] = float(np.min(year_loc))

            # Hot-tail metrics
            cum_hot, avg_hot, n_hot = _compute_year_hot_metrics(year_loc, hot_thresh)
            features[i, j, 2] = cum_hot
            features[i, j, 3] = avg_hot
            features[i, j, 4] = n_hot

            # Cold-tail metrics
            cum_cold, avg_cold, n_cold = _compute_year_cold_metrics(
                year_loc,
                cold_thresh,
            )
            features[i, j, 5] = cum_cold
            features[i, j, 6] = avg_cold
            features[i, j, 7] = n_cold

            # AR(1) - optional
            if compute_ar1:
                features[i, j, 8] = _compute_ar1_for_year(year_loc)

    return features


def fit_temporal_features(
    daily_values: NDArray[np.float64],
    day_of_year: NDArray[np.int64],
    year_labels: NDArray[np.int64],
    config: TemporalFeatureConfig,
) -> TemporalFeatureState:
    """Fit temporal feature extraction state from training data.

    Orchestrates the full McKinnon pipeline on training data:
    1. Fit Fourier seasonal cycle per location
    2. Remove seasonal cycle to get anomalies
    3. Compute within-season medians per year per location
    4. Compute residuals (anomalies minus medians)
    5. Fit tail thresholds from residuals per location

    The returned state is used by transform_temporal_features() on new data.

    Args:
        daily_values: Daily temperature values, shape (n_days, n_locations).
        day_of_year: Day-of-year for each observation, shape (n_days,).
        year_labels: Year label for each day, shape (n_days,).
        config: Temporal feature configuration.

    Returns:
        TemporalFeatureState containing all fitted parameters.
    """
    n_locations = int(daily_values.shape[1])

    seasonal_cycle = fit_seasonal_cycle(
        daily_values,
        day_of_year,
        config["n_fourier_harmonics"],
    )

    anomalies = remove_seasonal_cycle(daily_values, day_of_year, seasonal_cycle)
    medians, unique_years = compute_within_season_medians(anomalies, year_labels)
    residuals = compute_residuals(anomalies, year_labels, medians, unique_years)

    thresholds = fit_tail_thresholds(
        residuals,
        config["hot_cutoff_percentile"],
        config["cold_cutoff_percentile"],
    )

    # Compute median baseline: mean of within-season medians across years
    # per location. Used by streaming extractors to approximate residuals
    # when the current season's median is not yet available.
    median_baseline_vals: list[float] = []
    for j in range(n_locations):
        col: NDArray[np.float64] = medians[:, j]
        median_baseline_vals.append(float(np.sum(col)) / int(col.shape[0]))
    median_baseline: tuple[float, ...] = tuple(median_baseline_vals)

    return TemporalFeatureState(
        config=config,
        seasonal_cycle=seasonal_cycle,
        thresholds=thresholds,
        median_baseline=median_baseline,
        n_locations=n_locations,
    )


def transform_temporal_features(
    daily_values: NDArray[np.float64],
    day_of_year: NDArray[np.int64],
    year_labels: NDArray[np.int64],
    state: TemporalFeatureState,
) -> NDArray[np.float64]:
    """Transform daily values into heat metrics using fitted state.

    Applies the fitted temporal feature extraction pipeline:
    1. Remove seasonal cycle using fitted coefficients
    2. Compute within-season medians for these years
    3. Compute residuals
    4. Compute heat metrics using fitted thresholds

    The output is flattened from (n_years, n_locations, n_metrics) to
    (n_years * n_locations, n_metrics) for feature matrix consumption.

    Args:
        daily_values: Daily temperature values, shape (n_days, n_locations).
        day_of_year: Day-of-year for each observation, shape (n_days,).
        year_labels: Year label for each day, shape (n_days,).
        state: Previously fitted temporal feature state.

    Returns:
        Feature matrix of shape (n_years * n_locations, n_metrics).
    """
    anomalies = remove_seasonal_cycle(daily_values, day_of_year, state["seasonal_cycle"])
    medians, unique_years = compute_within_season_medians(anomalies, year_labels)
    residuals = compute_residuals(anomalies, year_labels, medians, unique_years)

    # (n_years, n_locations, n_metrics)
    metrics_3d = compute_heat_metrics(
        residuals,
        year_labels,
        state["thresholds"],
        state["config"]["compute_ar1"],
    )

    # Flatten to (n_years * n_locations, n_metrics)
    n_years = int(metrics_3d.shape[0])
    n_locations = int(metrics_3d.shape[1])
    n_metrics = int(metrics_3d.shape[2])
    result: NDArray[np.float64] = metrics_3d.reshape(
        n_years * n_locations,
        n_metrics,
    )
    return result


def build_temporal_feature_names(config: TemporalFeatureConfig) -> tuple[str, ...]:
    """Build ordered tuple of temporal feature names.

    Returns the metric names that will be produced by compute_heat_metrics
    given the configuration.

    Args:
        config: Temporal feature configuration.

    Returns:
        Ordered tuple of feature name strings.
    """
    if config["compute_ar1"]:
        return HEAT_METRIC_NAMES
    return HEAT_METRIC_NAMES_NO_AR1


__all__ = [
    "build_temporal_feature_names",
    "compute_heat_metrics",
    "compute_residuals",
    "compute_within_season_medians",
    "fit_seasonal_cycle",
    "fit_tail_thresholds",
    "fit_temporal_features",
    "remove_seasonal_cycle",
    "transform_temporal_features",
]
