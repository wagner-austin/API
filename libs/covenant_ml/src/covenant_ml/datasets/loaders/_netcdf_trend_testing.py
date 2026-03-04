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

from covenant_ml.datasets.types import (
    COLD_RANKED_METRICS,
    HOT_RANKED_METRICS,
    MetricTrendResult,
    RankTrendConfig,
    RankTrendResult,
    make_metric_trend_result,
    make_rank_trend_result,
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


def compute_latitude_weights(
    latitudes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute area-based latitude weights using cosine of latitude.

    Weights are normalized to sum to 1.0. Follows the standard
    geographic weighting where grid cells at the equator have
    larger area than those near the poles.

    Args:
        latitudes: Latitude values in degrees, shape (n_locations,).

    Returns:
        Normalized weights, shape (n_locations,), summing to 1.0.

    Raises:
        ValueError: If latitudes is empty.
    """
    n = int(latitudes.shape[0])
    if n == 0:
        raise ValueError("latitudes must not be empty")

    radians: NDArray[np.float64] = _deg2rad(latitudes)
    raw_weights: NDArray[np.float64] = _cos(radians)

    # Ensure non-negative (high latitudes can give tiny values from float)
    weights: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    total: float = 0.0
    for i in range(n):
        w = abs(float(raw_weights.flat[i]))
        weights[i] = w
        total += w

    # Normalize to sum to 1.0 (total is always > 0 since abs(cos) > 0
    # for any finite float64 latitude, including ±90° where cos ≈ 6.1e-17)
    for i in range(n):
        weights[i] = float(weights.flat[i]) / total

    return weights


def compute_weighted_spatial_mean(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute weighted mean across spatial dimension.

    Args:
        values: Values with shape (n_years, n_locations).
        weights: Normalized weights, shape (n_locations,).

    Returns:
        Weighted means, shape (n_years,).

    Raises:
        ValueError: If weights length doesn't match n_locations.
    """
    n_years = int(values.shape[0])
    n_locations = int(values.shape[1])
    if int(weights.shape[0]) != n_locations:
        raise ValueError(f"weights length {int(weights.shape[0])} != n_locations {n_locations}")

    result: NDArray[np.float64] = np.zeros(n_years, dtype=np.float64)
    for yr in range(n_years):
        total: float = 0.0
        for loc in range(n_locations):
            v_idx = yr * n_locations + loc
            total += float(values.flat[v_idx]) * float(weights.flat[loc])
        result[yr] = total
    return result


def estimate_spatial_dof(
    rank_series: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> int:
    """Estimate spatial degrees of freedom using Bretherton et al. (1999).

    Computes the effective number of independent spatial locations
    from the area-weighted covariance matrix of ranks:
    DOF = trace(C)^2 / ||C||_F^2

    where C is the weighted covariance matrix and ||.||_F is the
    Frobenius norm.

    Args:
        rank_series: Rank values, shape (n_years, n_locations).
        weights: Area weights for spatial locations, shape (n_locations,).

    Returns:
        Estimated degrees of freedom (rounded to nearest integer, minimum 1).

    Raises:
        ValueError: If rank_series has fewer than 2 years or weights
            length doesn't match n_locations.
    """
    n_years = int(rank_series.shape[0])
    n_locations = int(rank_series.shape[1])

    if n_years < 2:
        raise ValueError(f"Need at least 2 years, got {n_years}")
    if int(weights.shape[0]) != n_locations:
        raise ValueError(f"weights length {int(weights.shape[0])} != n_locations {n_locations}")

    if n_locations == 1:
        return 1

    # Transpose so locations are rows for np.cov: (n_locations, n_years)
    transposed: NDArray[np.float64] = np.zeros((n_locations, n_years), dtype=np.float64)
    for loc in range(n_locations):
        for yr in range(n_years):
            src_idx = yr * n_locations + loc
            dst_idx = loc * n_years + yr
            transposed.flat[dst_idx] = float(rank_series.flat[src_idx])

    # Compute covariance matrix between locations over time
    cov_matrix: NDArray[np.float64] = _cov(transposed)

    # Bretherton formula: DOF = trace(C)^2 / ||C||_F^2
    trace_val: float = _trace(cov_matrix)
    frob_norm: float = _norm(cov_matrix, ord="fro")

    if frob_norm == 0.0:
        return 1

    dof_float = (trace_val * trace_val) / (frob_norm * frob_norm)
    dof_rounded = round(dof_float)
    return max(1, dof_rounded)


# --- Monte Carlo null distribution ---


def _build_centered_x(n_years: int) -> NDArray[np.float64]:
    """Build mean-centered year index array for OLS.

    Args:
        n_years: Number of years.

    Returns:
        Centered indices, shape (n_years,).
    """
    x_vals: NDArray[np.float64] = np.zeros(n_years, dtype=np.float64)
    for i in range(n_years):
        x_vals[i] = float(i)
    x_mean: float = 0.0
    for i in range(n_years):
        x_mean += float(x_vals.flat[i])
    x_mean /= n_years
    for i in range(n_years):
        x_vals[i] = float(x_vals.flat[i]) - x_mean
    return x_vals


def _generate_one_null_slope(
    rng: np.random.Generator,
    base_ranks: NDArray[np.int64],
    x_vals: NDArray[np.float64],
    dof: int,
    n_years: int,
) -> float:
    """Generate one null slope sample from random rank permutations.

    Args:
        rng: Random number generator.
        base_ranks: Base rank array [1, ..., n_years].
        x_vals: Centered year indices.
        dof: Number of independent permutations to average.
        n_years: Number of years.

    Returns:
        OLS slope of averaged permuted ranks.
    """
    avg_series: NDArray[np.float64] = np.zeros(n_years, dtype=np.float64)
    for _ in range(dof):
        perm: NDArray[np.int64] = rng.permutation(base_ranks)
        for i in range(n_years):
            avg_series[i] = float(avg_series.flat[i]) + float(perm.flat[i])
    for i in range(n_years):
        avg_series[i] = float(avg_series.flat[i]) / dof
    return compute_ols_slope(x_vals, avg_series)


def generate_null_trend_slopes(
    dof: int,
    n_years: int,
    n_samples: int,
    seed: int,
) -> NDArray[np.float64]:
    """Generate null distribution of rank trend slopes via Monte Carlo.

    For each sample: generates ``dof`` independent random rank
    permutations of length ``n_years``, averages them to simulate
    a spatially-averaged rank series, then computes the OLS slope.

    The null hypothesis is that ranks are uniformly distributed
    with no temporal trend.

    Args:
        dof: Spatial degrees of freedom.
        n_years: Number of years (rank values span 1 to n_years).
        n_samples: Number of Monte Carlo samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of null slopes, shape (n_samples,).

    Raises:
        ValueError: If dof < 1, n_years < 2, or n_samples < 1.
    """
    if dof < 1:
        raise ValueError(f"dof must be >= 1, got {dof}")
    if n_years < 2:
        raise ValueError(f"n_years must be >= 2, got {n_years}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    rng = np.random.default_rng(seed)
    x_vals = _build_centered_x(n_years)

    # Base ranks to permute: [1, 2, ..., n_years]
    base_ranks: NDArray[np.int64] = np.zeros(n_years, dtype=np.int64)
    for i in range(n_years):
        base_ranks[i] = i + 1

    slopes: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for sample_idx in range(n_samples):
        slopes[sample_idx] = _generate_one_null_slope(
            rng,
            base_ranks,
            x_vals,
            dof,
            n_years,
        )

    return slopes


def compute_trend_pvalue(
    observed_slope: float,
    null_slopes: NDArray[np.float64],
) -> float:
    """Compute two-sided p-value from null distribution.

    The p-value is the fraction of null slopes whose absolute value
    is greater than or equal to the absolute observed slope.

    Args:
        observed_slope: Observed OLS trend slope.
        null_slopes: Array of null distribution slopes, shape (n_samples,).

    Returns:
        Two-sided p-value in [0, 1].

    Raises:
        ValueError: If null_slopes is empty.
    """
    n = int(null_slopes.shape[0])
    if n == 0:
        raise ValueError("null_slopes must not be empty")

    abs_observed = abs(observed_slope)
    count: int = 0
    for i in range(n):
        if abs(float(null_slopes.flat[i])) >= abs_observed:
            count += 1
    return float(count) / float(n)


# --- Orchestrator ---


def run_rank_trend_analysis(
    metrics: NDArray[np.float64],
    metric_names: tuple[str, ...],
    latitudes: NDArray[np.float64],
    config: RankTrendConfig,
) -> RankTrendResult:
    """Run complete rank-trend significance analysis.

    Implements McKinnon PNAS 2024 steps 4-7:
    1. Convert metric values to ranks (step 4)
    2. Compute spatially-weighted average rank series (step 5)
    3. Estimate spatial DOF via Bretherton formula (step 6)
    4. Generate Monte Carlo null distribution and compute p-values (step 7)

    Args:
        metrics: Metric values, shape (n_years, n_metrics, n_locations).
        metric_names: Names of metrics in canonical order.
        latitudes: Latitude values in degrees, shape (n_locations,).
        config: Configuration for null distribution generation.

    Returns:
        RankTrendResult with per-metric trend test results.

    Raises:
        ValueError: If input dimensions are inconsistent.
    """
    n_years = int(metrics.shape[0])
    n_locations = int(metrics.shape[2])

    # Step 1: Rank all metrics
    ranked, extended_names = rank_heat_metrics(metrics, metric_names)
    n_extended = int(ranked.shape[1])

    # Step 2: Compute latitude weights
    weights = compute_latitude_weights(latitudes)

    # Process each metric
    results: list[MetricTrendResult] = []
    for m_idx in range(n_extended):
        name = extended_names[m_idx]

        # Extract rank series for this metric: (n_years, n_locations)
        metric_ranks: NDArray[np.float64] = np.zeros((n_years, n_locations), dtype=np.float64)
        n_ext_metrics = int(ranked.shape[1])
        for yr in range(n_years):
            for loc in range(n_locations):
                src_idx = yr * n_ext_metrics * n_locations + m_idx * n_locations + loc
                dst_idx = yr * n_locations + loc
                metric_ranks.flat[dst_idx] = float(ranked.flat[src_idx])

        # Step 3: Spatial DOF
        dof = estimate_spatial_dof(metric_ranks, weights)

        # Step 4: Spatially-weighted average rank series
        avg_ranks = compute_weighted_spatial_mean(metric_ranks, weights)

        # Step 5: Observed OLS slope
        x_years: NDArray[np.float64] = np.zeros(n_years, dtype=np.float64)
        for i in range(n_years):
            x_years[i] = float(i)
        observed_slope = compute_ols_slope(x_years, avg_ranks)

        # Step 6: Generate null distribution
        null_slopes = generate_null_trend_slopes(
            dof=dof,
            n_years=n_years,
            n_samples=config["n_null_samples"],
            seed=config["random_seed"],
        )

        # Step 7: P-value
        p_value = compute_trend_pvalue(observed_slope, null_slopes)

        results.append(
            make_metric_trend_result(
                metric_name=name,
                observed_slope=observed_slope,
                p_value=p_value,
                is_significant=p_value < 0.05,
                n_years=n_years,
                spatial_dof=dof,
            )
        )

    return make_rank_trend_result(
        metric_results=tuple(results),
        n_null_samples=config["n_null_samples"],
        random_seed=config["random_seed"],
    )
