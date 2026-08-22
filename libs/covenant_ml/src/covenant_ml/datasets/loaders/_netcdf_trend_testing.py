"""McKinnon-style rank-trend hypothesis testing for atmospheric data.

Pure numpy implementation of rank conversion, spatial DOF estimation
(Bretherton et al. 1999), and Monte Carlo null distribution generation
from McKinnon (PNAS 2024, "The pace of change of summertime temperature
extremes"), steps 4-7.

All functions operate on multi-location metric arrays with shapes:
- metrics: (n_years, n_metrics, n_locations)
- latitudes: (n_locations,)

Internal module - used alongside _netcdf_temporal.py for full McKinnon
pipeline analysis.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.types_trend import (
    COLD_RANKED_METRICS,
    HOT_RANKED_METRICS,
)

# --- Protocol wrappers for numpy functions with insufficiently typed stubs ---


class _CovProtocol(Protocol):
    """Protocol for numpy.cov computing covariance matrix."""

    def __call__(
        self,
        m: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...


class _TraceProtocol(Protocol):
    """Protocol for numpy.trace returning float."""

    def __call__(self, a: NDArray[np.float64]) -> float: ...


class _NormProtocol(Protocol):
    """Protocol for numpy.linalg.norm with Frobenius norm."""

    def __call__(
        self,
        x: NDArray[np.float64],
        ord: str,
    ) -> float: ...


class _CosProtocol(Protocol):
    """Protocol for numpy.cos with strict float64 typing."""

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _Deg2radProtocol(Protocol):
    """Protocol for numpy.deg2rad with strict float64 typing."""

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


_numpy_mod = __import__("numpy")
_linalg_mod = __import__("numpy.linalg", fromlist=["norm"])

_cov: _CovProtocol = _numpy_mod.cov
_trace: _TraceProtocol = _numpy_mod.trace
_norm: _NormProtocol = _linalg_mod.norm
_cos: _CosProtocol = _numpy_mod.cos
_deg2rad: _Deg2radProtocol = _numpy_mod.deg2rad


# --- OLS slope computation ---


def compute_ols_slope(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Compute OLS regression slope of y on x.

    Manual ordinary least squares: slope = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2).
    No intercept is returned since only the slope is needed for trend testing.

    Args:
        x: Independent variable, shape (n,).
        y: Dependent variable, shape (n,).

    Returns:
        OLS slope as float.

    Raises:
        ValueError: If x and y have different lengths or length < 2.
    """
    n = int(x.shape[0])
    if n != int(y.shape[0]):
        raise ValueError(f"x length {n} != y length {int(y.shape[0])}")
    if n < 2:
        raise ValueError(f"Need at least 2 points for OLS, got {n}")

    x_mean: float = 0.0
    y_mean: float = 0.0
    for i in range(n):
        x_mean += float(x.flat[i])
        y_mean += float(y.flat[i])
    x_mean /= n
    y_mean /= n

    numerator: float = 0.0
    denominator: float = 0.0
    for i in range(n):
        x_diff = float(x.flat[i]) - x_mean
        numerator += x_diff * (float(y.flat[i]) - y_mean)
        denominator += x_diff * x_diff

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


# --- Ranking functions ---


def rank_metric_series(
    values: NDArray[np.float64],
    negate: bool,
) -> NDArray[np.float64]:
    """Convert 1D metric values to ranks (1 = most extreme).

    Uses argsort(argsort(...)) + 1 for rank conversion. When negate
    is True, values are negated before ranking so that the largest
    original value gets rank 1.

    Args:
        values: 1D array of metric values, shape (n,).
        negate: If True, negate values before ranking (for hot metrics
            where higher value = more extreme).

    Returns:
        1D array of ranks, shape (n,), dtype float64.
    """
    n = int(values.shape[0])
    work: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    for i in range(n):
        val = float(values.flat[i])
        work[i] = -val if negate else val

    order: NDArray[np.intp] = np.argsort(work)
    ranks: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ranks[int(order.flat[i])] = float(i + 1)
    return ranks


def _rank_all_metrics(
    metrics: NDArray[np.float64],
    metric_names: tuple[str, ...],
    n_years: int,
    n_metrics: int,
    n_locations: int,
) -> NDArray[np.float64]:
    """Rank each metric at each location using HOT/COLD sign conventions.

    Args:
        metrics: Metric values, shape (n_years, n_metrics, n_locations).
        metric_names: Names of metrics in order.
        n_years: Number of years.
        n_metrics: Number of metrics.
        n_locations: Number of spatial locations.

    Returns:
        Ranked values, shape (n_years, n_metrics, n_locations).
    """
    ranked: NDArray[np.float64] = np.zeros((n_years, n_metrics, n_locations), dtype=np.float64)
    for m_idx in range(n_metrics):
        name = metric_names[m_idx]
        negate = name in HOT_RANKED_METRICS
        for loc in range(n_locations):
            series: NDArray[np.float64] = np.zeros(n_years, dtype=np.float64)
            for yr in range(n_years):
                flat_idx = yr * n_metrics * n_locations + m_idx * n_locations + loc
                series[yr] = float(metrics.flat[flat_idx])
            ranked_series = rank_metric_series(series, negate=negate)
            for yr in range(n_years):
                r_flat_idx = yr * n_metrics * n_locations + m_idx * n_locations + loc
                ranked.flat[r_flat_idx] = float(ranked_series.flat[yr])
    return ranked


def _compute_composite_average(
    ranked: NDArray[np.float64],
    extended: NDArray[np.float64],
    indices: list[int],
    target_metric_idx: int,
    n_years: int,
    n_metrics: int,
    n_extended: int,
    n_locations: int,
) -> None:
    """Compute mean rank across a set of metrics and write to extended array.

    Args:
        ranked: Source ranked values, shape (n_years, n_metrics, n_locations).
        extended: Target extended array, shape (n_years, n_extended, n_locations).
        indices: Metric indices to average over.
        target_metric_idx: Index in extended array to write the average.
        n_years: Number of years.
        n_metrics: Number of original metrics.
        n_extended: Number of extended metrics.
        n_locations: Number of spatial locations.
    """
    for yr in range(n_years):
        for loc in range(n_locations):
            total: float = 0.0
            for m_idx in indices:
                r_idx = yr * n_metrics * n_locations + m_idx * n_locations + loc
                total += float(ranked.flat[r_idx])
            e_idx = yr * n_extended * n_locations + target_metric_idx * n_locations + loc
            extended.flat[e_idx] = total / len(indices)


def rank_heat_metrics(
    metrics: NDArray[np.float64],
    metric_names: tuple[str, ...],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Rank all heat metrics using HOT/COLD sign conventions.

    For each metric and location, converts the year-by-year values
    to ranks. Hot metrics are negated so rank 1 = most extreme heat.
    Cold metrics are ranked directly so rank 1 = most extreme cold.

    Also computes two composite rank averages:
    - avg_across_metrics_hot: mean rank across hot metrics (excluding ar1)
    - avg_across_metrics_cold: mean rank across cold metrics

    Args:
        metrics: Metric values, shape (n_years, n_metrics, n_locations).
        metric_names: Names of metrics in order, length n_metrics.

    Returns:
        Tuple of (ranked_metrics, extended_names) where:
        - ranked_metrics: shape (n_years, n_extended_metrics, n_locations)
        - extended_names: original names + avg_across_metrics_hot
            + avg_across_metrics_cold

    Raises:
        ValueError: If metric_names length doesn't match n_metrics dimension.
    """
    n_years = int(metrics.shape[0])
    n_metrics = int(metrics.shape[1])
    n_locations = int(metrics.shape[2])

    if len(metric_names) != n_metrics:
        raise ValueError(f"metric_names length {len(metric_names)} != n_metrics {n_metrics}")

    ranked = _rank_all_metrics(metrics, metric_names, n_years, n_metrics, n_locations)

    # Identify hot/cold metric indices
    hot_indices: list[int] = []
    cold_indices: list[int] = []
    for m_idx, name in enumerate(metric_names):
        if name in HOT_RANKED_METRICS and name != "ar1":
            hot_indices.append(m_idx)
        if name in COLD_RANKED_METRICS:
            cold_indices.append(m_idx)

    # Build extended array with 2 extra metric slots
    n_extended = n_metrics + 2
    extended: NDArray[np.float64] = np.zeros((n_years, n_extended, n_locations), dtype=np.float64)
    # Copy original ranks
    for yr in range(n_years):
        for m_idx in range(n_metrics):
            for loc in range(n_locations):
                src_idx = yr * n_metrics * n_locations + m_idx * n_locations + loc
                dst_idx = yr * n_extended * n_locations + m_idx * n_locations + loc
                extended.flat[dst_idx] = float(ranked.flat[src_idx])

    # Composite averages
    if len(hot_indices) > 0:
        _compute_composite_average(
            ranked,
            extended,
            hot_indices,
            n_metrics,
            n_years,
            n_metrics,
            n_extended,
            n_locations,
        )
    if len(cold_indices) > 0:
        _compute_composite_average(
            ranked,
            extended,
            cold_indices,
            n_metrics + 1,
            n_years,
            n_metrics,
            n_extended,
            n_locations,
        )

    extended_names = (*metric_names, "avg_across_metrics_hot", "avg_across_metrics_cold")
    return extended, extended_names


# --- Spatial weighting and DOF estimation ---


# --- Monte Carlo null distribution ---


# --- Orchestrator ---
