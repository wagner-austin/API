"""McKinnon-style temporal feature extraction for atmospheric data.

Pure numpy implementation of Fourier deseasonalization, tail-excess heat
metrics, and lag-1 autocorrelation from McKinnon (PNAS 2024, "The pace
of change of summertime temperature extremes").

All functions operate on multi-location data with shapes:
- daily_values: (n_days, n_locations)
- day_of_year: (n_days,)
- month_labels: (n_days,)
- year_labels: (n_days,)

Give fit and transform the whole calendar year. The two halves of the
pipeline need different spans: the Fourier basis has annual period and is
only determined by observations covering the year, while the medians and
tail thresholds are defined within one season. select_season is the single
place that boundary is drawn, driven by config["season_months"].

Internal module - used by the future netcdf_loader.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.types_temporal import (
    SeasonalCycleCoefficients,
    TailThresholds,
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


class _CondProtocol(Protocol):
    """Protocol for numpy.linalg.cond returning float."""

    def __call__(self, x: NDArray[np.float64]) -> float: ...


class _IntReduceProtocol(Protocol):
    """Protocol for numpy reductions over int64 arrays returning a scalar."""

    def __call__(self, a: NDArray[np.int64]) -> np.int64: ...


class _IntEqualProtocol(Protocol):
    """Protocol for numpy.equal comparing an int64 array against a scalar."""

    def __call__(self, x1: NDArray[np.int64], x2: int) -> NDArray[np.bool_]: ...


class _BoolBinaryProtocol(Protocol):
    """Protocol for numpy elementwise boolean combination."""

    def __call__(self, x1: NDArray[np.bool_], x2: NDArray[np.bool_]) -> NDArray[np.bool_]: ...


class _UniqueIntProtocol(Protocol):
    """Protocol for numpy.unique over an int64 array."""

    def __call__(self, ar: NDArray[np.int64]) -> NDArray[np.int64]: ...


_numpy_mod = __import__("numpy")
_linalg_mod = __import__("numpy.linalg", fromlist=["solve", "cond"])
_linalg_solve: _LinalgSolveProtocol = _linalg_mod.solve
_linalg_cond: _CondProtocol = _linalg_mod.cond
_np_cos: _TrigProtocol = _numpy_mod.cos
_np_sin: _TrigProtocol = _numpy_mod.sin
_np_median: _MedianProtocol = _numpy_mod.median
_np_percentile: _PercentileProtocol = _numpy_mod.percentile
_np_amin: _IntReduceProtocol = _numpy_mod.amin
_np_amax: _IntReduceProtocol = _numpy_mod.amax
_np_equal: _IntEqualProtocol = _numpy_mod.equal
_np_logical_or: _BoolBinaryProtocol = _numpy_mod.logical_or
_np_unique: _UniqueIntProtocol = _numpy_mod.unique

# Largest condition number of the normal-equations matrix that still leaves
# the fitted coefficients meaningful. The solve is beta = (X^T X)^-1 X^T Y,
# and forming X^T X squares the conditioning, so float64's ~16 significant
# digits are eroded by log10(cond) of them. At 1e8 half the digits survive,
# which is ample; a full-year daily fit sits near 2. Anything beyond this is
# not a precision nuisance but a statement that the sample cannot determine
# the basis -- most often a day-of-year range far narrower than the period.
_MAX_DESIGN_CONDITION = 1.0e8


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
        ValueError: If input arrays have mismatched shapes or are empty, or
            if the observed days do not span enough of the period to
            determine the requested harmonics.
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

    # Over a narrow slice of the year the harmonics are nearly collinear, so
    # the solve still returns coefficients -- enormous ones that cancel
    # within the observed days and diverge everywhere else. Nothing
    # downstream can detect that, because the fit looks excellent exactly
    # where it was fitted. Refusing here is the only place the caller can
    # still be told what went wrong.
    condition: float = _linalg_cond(xtx)
    if condition > _MAX_DESIGN_CONDITION:
        span = int(_np_amax(day_of_year)) - int(_np_amin(day_of_year)) + 1
        raise ValueError(
            f"Seasonal cycle is not determined by these observations: design "
            f"matrix condition {condition:.3e} exceeds {_MAX_DESIGN_CONDITION:.0e}. "
            f"The {n_days} observations span {span} of {n_days_per_year} days, "
            f"which is too narrow to fit {n_harmonics} harmonics of that period. "
            f"Fit the cycle on the full year and restrict the season afterwards."
        )

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


def select_season(
    month_labels: NDArray[np.int64],
    season_months: tuple[int, ...],
) -> NDArray[np.bool_]:
    """Select the days belonging to the configured season.

    The seasonal cycle is fitted across the whole year, but the medians,
    residuals and tail thresholds describe one season, so exactly one stage
    of the pipeline restricts its input and this is where it happens.

    Args:
        month_labels: Calendar month, 1-12, for each day, shape (n_days,).
        season_months: Month numbers defining the season, e.g. (6, 7, 8).

    Returns:
        Boolean mask of shape (n_days,), true on days within the season.

    Raises:
        ValueError: If season_months is empty, contains a value outside
            1-12, or selects no day in the input.
    """
    if not season_months:
        raise ValueError("season_months is empty; the season selects no days")
    invalid = tuple(month for month in season_months if month < 1 or month > 12)
    if invalid:
        raise ValueError(f"season_months contains non-months: {invalid}")

    mask: NDArray[np.bool_] = np.zeros(month_labels.shape[0], dtype=np.bool_)
    for month in season_months:
        matched: NDArray[np.bool_] = _np_equal(month_labels, month)
        mask = _np_logical_or(mask, matched)

    if not bool(mask.any()):
        distinct: NDArray[np.int64] = _np_unique(month_labels)
        observed = tuple(int(distinct.flat[index]) for index in range(int(distinct.shape[0])))
        raise ValueError(
            f"Season months {season_months} match none of the observed "
            f"months {observed}; the thresholds would be fitted on no data"
        )
    return mask


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


__all__ = [
    "compute_residuals",
    "compute_within_season_medians",
    "fit_seasonal_cycle",
    "fit_tail_thresholds",
    "remove_seasonal_cycle",
    "select_season",
]
