"""Heat-metric computation and temporal feature fit/transform."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._netcdf_temporal import (
    compute_residuals,
    compute_within_season_medians,
    fit_seasonal_cycle,
    fit_tail_thresholds,
    remove_seasonal_cycle,
    select_season,
)
from covenant_ml.datasets.types_temporal import (
    HEAT_METRIC_NAMES,
    HEAT_METRIC_NAMES_NO_AR1,
    TailThresholds,
    TemporalFeatureConfig,
    TemporalFeatureState,
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
    month_labels: NDArray[np.int64],
    year_labels: NDArray[np.int64],
    config: TemporalFeatureConfig,
) -> TemporalFeatureState:
    """Fit temporal feature extraction state from training data.

    Orchestrates the full McKinnon pipeline on training data:
    1. Fit Fourier seasonal cycle per location, across the whole year
    2. Remove seasonal cycle to get anomalies, across the whole year
    3. Restrict to config["season_months"]
    4. Compute within-season medians per year per location
    5. Compute residuals (anomalies minus medians)
    6. Fit tail thresholds from residuals per location

    Pass the full calendar year. The two stages need different spans and
    the split is what this function is for: a Fourier basis of annual period
    is only determined by observations covering the year, while the medians
    and tail thresholds are defined within one season and would otherwise
    describe the annual distribution. Handing this function pre-filtered
    summer days puts a 92-day sample against a 365-day basis, which
    fit_seasonal_cycle rejects.

    Args:
        daily_values: Daily temperature values, shape (n_days, n_locations).
        day_of_year: Day-of-year for each observation, shape (n_days,).
        month_labels: Calendar month, 1-12, for each day, shape (n_days,).
        year_labels: Year label for each day, shape (n_days,).
        config: Temporal feature configuration, whose season_months selects
            the days the thresholds are fitted on.

    Returns:
        TemporalFeatureState containing all fitted parameters.

    Raises:
        ValueError: If the observations do not determine the seasonal cycle,
            or if season_months selects no day in the input.
    """
    n_locations = int(daily_values.shape[1])

    seasonal_cycle = fit_seasonal_cycle(
        daily_values,
        day_of_year,
        config["n_fourier_harmonics"],
    )

    anomalies = remove_seasonal_cycle(daily_values, day_of_year, seasonal_cycle)

    in_season = select_season(month_labels, config["season_months"])
    season_anomalies: NDArray[np.float64] = anomalies[in_season]
    season_years: NDArray[np.int64] = year_labels[in_season]

    medians, unique_years = compute_within_season_medians(season_anomalies, season_years)
    residuals = compute_residuals(season_anomalies, season_years, medians, unique_years)

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
    month_labels: NDArray[np.int64],
    year_labels: NDArray[np.int64],
    state: TemporalFeatureState,
) -> NDArray[np.float64]:
    """Transform daily values into heat metrics using fitted state.

    Applies the fitted temporal feature extraction pipeline:
    1. Remove seasonal cycle using fitted coefficients, across the whole year
    2. Restrict to the season the state was fitted for
    3. Compute within-season medians for these years
    4. Compute residuals
    5. Compute heat metrics using fitted thresholds

    Takes the full calendar year, as fit does, and restricts at the same
    stage. Passing pre-filtered season days instead would leave the metrics
    unchanged but silently reintroduces the asymmetry that fit forbids.

    The output is flattened from (n_years, n_locations, n_metrics) to
    (n_years * n_locations, n_metrics) for feature matrix consumption.

    Args:
        daily_values: Daily temperature values, shape (n_days, n_locations).
        day_of_year: Day-of-year for each observation, shape (n_days,).
        month_labels: Calendar month, 1-12, for each day, shape (n_days,).
        year_labels: Year label for each day, shape (n_days,).
        state: Previously fitted temporal feature state.

    Returns:
        Feature matrix of shape (n_years * n_locations, n_metrics).

    Raises:
        ValueError: If the state's season_months selects no day in the input.
    """
    anomalies = remove_seasonal_cycle(daily_values, day_of_year, state["seasonal_cycle"])

    in_season = select_season(month_labels, state["config"]["season_months"])
    season_anomalies: NDArray[np.float64] = anomalies[in_season]
    season_years: NDArray[np.int64] = year_labels[in_season]

    medians, unique_years = compute_within_season_medians(season_anomalies, season_years)
    residuals = compute_residuals(season_anomalies, season_years, medians, unique_years)

    # (n_years, n_locations, n_metrics)
    metrics_3d = compute_heat_metrics(
        residuals,
        season_years,
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
    "fit_temporal_features",
    "transform_temporal_features",
]
