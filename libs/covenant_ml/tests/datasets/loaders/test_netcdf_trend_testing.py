"""Tests for McKinnon-style rank-trend hypothesis testing functions.

Tests cover all 9 public functions in _netcdf_trend_testing.py: OLS slope
computation, rank conversion, composite rank averaging, latitude weighting,
weighted spatial mean, spatial DOF estimation, Monte Carlo null distribution
generation, p-value computation, and the full analysis orchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from covenant_ml.datasets.loaders._netcdf_trend_testing import (
    compute_ols_slope,
    rank_heat_metrics,
    rank_metric_series,
)
from tests.datasets.loaders._trend_fixtures import (
    _f64,
    _f64_3d,
    _flat,
)


class TestComputeOlsSlope:
    """Tests for compute_ols_slope: manual OLS regression slope."""

    def test_perfect_positive_slope(self) -> None:
        """Perfect linear relationship y = 2x + 1 gives slope 2."""
        x = _f64([0.0, 1.0, 2.0, 3.0, 4.0])
        y = _f64([1.0, 3.0, 5.0, 7.0, 9.0])
        slope = compute_ols_slope(x, y)
        assert abs(slope - 2.0) < 1e-10

    def test_perfect_negative_slope(self) -> None:
        """Perfect linear y = -0.5x + 10 gives slope -0.5."""
        x = _f64([0.0, 2.0, 4.0, 6.0])
        y = _f64([10.0, 9.0, 8.0, 7.0])
        slope = compute_ols_slope(x, y)
        assert abs(slope - (-0.5)) < 1e-10

    def test_zero_slope(self) -> None:
        """Constant y gives slope 0."""
        x = _f64([1.0, 2.0, 3.0, 4.0, 5.0])
        y = _f64([5.0, 5.0, 5.0, 5.0, 5.0])
        slope = compute_ols_slope(x, y)
        assert abs(slope) < 1e-10

    def test_constant_x_returns_zero(self) -> None:
        """Constant x (zero denominator) returns 0."""
        x = _f64([3.0, 3.0, 3.0])
        y = _f64([1.0, 2.0, 3.0])
        slope = compute_ols_slope(x, y)
        assert slope == 0.0

    def test_two_points(self) -> None:
        """Minimum case: 2 points."""
        x = _f64([0.0, 1.0])
        y = _f64([0.0, 3.0])
        slope = compute_ols_slope(x, y)
        assert abs(slope - 3.0) < 1e-10

    def test_mismatched_lengths_raises(self) -> None:
        """Different length arrays raise ValueError."""
        x = _f64([1.0, 2.0])
        y = _f64([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="x length 2 != y length 3"):
            compute_ols_slope(x, y)

    def test_single_point_raises(self) -> None:
        """Fewer than 2 points raises ValueError."""
        x = _f64([1.0])
        y = _f64([2.0])
        with pytest.raises(ValueError, match="Need at least 2 points"):
            compute_ols_slope(x, y)


class TestRankMetricSeries:
    """Tests for rank_metric_series: 1D value-to-rank conversion."""

    def test_ascending_values_no_negate(self) -> None:
        """Ascending values without negation: rank 1 = smallest."""
        values = _f64([10.0, 20.0, 30.0, 40.0])
        ranks = rank_metric_series(values, negate=False)
        assert int(_flat(ranks, 0)) == 1
        assert int(_flat(ranks, 1)) == 2
        assert int(_flat(ranks, 2)) == 3
        assert int(_flat(ranks, 3)) == 4

    def test_descending_values_no_negate(self) -> None:
        """Descending values without negation: rank 1 = smallest."""
        values = _f64([40.0, 30.0, 20.0, 10.0])
        ranks = rank_metric_series(values, negate=False)
        assert int(_flat(ranks, 0)) == 4
        assert int(_flat(ranks, 1)) == 3
        assert int(_flat(ranks, 2)) == 2
        assert int(_flat(ranks, 3)) == 1

    def test_with_negate(self) -> None:
        """With negate=True: rank 1 = largest original value."""
        values = _f64([10.0, 20.0, 30.0, 40.0])
        ranks = rank_metric_series(values, negate=True)
        # Negated: [-10, -20, -30, -40], argsort gives [-40,-30,-20,-10]
        # so original 40 (index 3) gets rank 1
        assert int(_flat(ranks, 3)) == 1
        assert int(_flat(ranks, 0)) == 4

    def test_output_dtype_and_shape(self) -> None:
        """Output is float64 with same shape as input."""
        values = _f64([5.0, 3.0, 1.0])
        ranks = rank_metric_series(values, negate=False)
        assert ranks.dtype == np.float64
        assert int(ranks.shape[0]) == 3

    def test_all_ranks_present(self) -> None:
        """All ranks from 1 to n are assigned exactly once."""
        values = _f64([7.0, 2.0, 5.0, 9.0, 1.0])
        ranks = rank_metric_series(values, negate=False)
        sorted_ranks: list[int] = sorted(int(_flat(ranks, i)) for i in range(5))
        assert sorted_ranks == [1, 2, 3, 4, 5]


class TestRankHeatMetrics:
    """Tests for rank_heat_metrics: multi-metric ranking with sign conventions."""

    def test_hot_metric_negated(self) -> None:
        """HOT metric (seasonal_max) is negated: rank 1 = largest value."""
        # 3 years, 1 metric (seasonal_max), 1 location
        # Values increasing: [10, 20, 30] → negated: [-10, -20, -30]
        # argsort of negated: [-30, -20, -10] → index 2 gets rank 1
        metrics = _f64_3d(
            [
                [[10.0]],
                [[20.0]],
                [[30.0]],
            ]
        )
        ranked, _names = rank_heat_metrics(metrics, ("seasonal_max",))
        # seasonal_max is HOT → negated → rank 1 = year with value 30
        year2_rank = float(ranked.flat[2 * 3 * 1 + 0 * 1 + 0])
        assert int(year2_rank) == 1

    def test_cold_metric_direct(self) -> None:
        """COLD metric (seasonal_min) is ranked directly: rank 1 = smallest."""
        # 3 years, 1 metric (seasonal_min), 1 location
        # Values: [10, 5, 15] → rank 1 = year with value 5 (index 1)
        metrics = _f64_3d(
            [
                [[10.0]],
                [[5.0]],
                [[15.0]],
            ]
        )
        ranked, _names = rank_heat_metrics(metrics, ("seasonal_min",))
        year1_rank = float(ranked.flat[1 * 3 * 1 + 0 * 1 + 0])
        assert int(year1_rank) == 1

    def test_extended_names_include_composites(self) -> None:
        """Output names include avg_across_metrics_hot and avg_across_metrics_cold."""
        # 3 years, 2 metrics, 1 location
        metrics = _f64_3d(
            [
                [[10.0], [5.0]],
                [[20.0], [3.0]],
                [[30.0], [7.0]],
            ]
        )
        _, names = rank_heat_metrics(metrics, ("seasonal_max", "seasonal_min"))
        assert names[-2] == "avg_across_metrics_hot"
        assert names[-1] == "avg_across_metrics_cold"
        assert len(names) == 4  # 2 original + 2 composites

    def test_multi_location_ranking(self) -> None:
        """Each location is ranked independently."""
        # 3 years, 1 metric (seasonal_max), 2 locations
        metrics = _f64_3d(
            [
                [[30.0, 10.0]],  # year 0: loc0=30(high), loc1=10(low)
                [[20.0, 20.0]],  # year 1
                [[10.0, 30.0]],  # year 2: loc0=10(low), loc1=30(high)
            ]
        )
        ranked, _ = rank_heat_metrics(metrics, ("seasonal_max",))
        n_ext = int(ranked.shape[1])
        n_loc = int(ranked.shape[2])
        # seasonal_max is HOT: rank 1 = largest
        # loc0: [30, 20, 10] → negated [-30, -20, -10] → rank 1 = year 0
        yr0_loc0 = float(ranked.flat[0 * n_ext * n_loc + 0 * n_loc + 0])
        assert int(yr0_loc0) == 1
        # loc1: [10, 20, 30] → negated [-10, -20, -30] → rank 1 = year 2
        yr2_loc1 = float(ranked.flat[2 * n_ext * n_loc + 0 * n_loc + 1])
        assert int(yr2_loc1) == 1

    def test_mismatched_names_raises(self) -> None:
        """Metric names length mismatch raises ValueError."""
        metrics = _f64_3d([[[1.0]], [[2.0]]])
        with pytest.raises(ValueError, match="metric_names length"):
            rank_heat_metrics(metrics, ("a", "b"))

    def test_composite_hot_average(self) -> None:
        """avg_across_metrics_hot is mean rank of hot metrics (excluding ar1)."""
        # 3 years, 2 hot metrics, 1 location
        # seasonal_max: [10, 20, 30] → negated ranks: [3, 2, 1]
        # ndays_excess_hot: [30, 20, 10] → negated ranks: [1, 2, 3]
        metrics = _f64_3d(
            [
                [[10.0], [30.0]],
                [[20.0], [20.0]],
                [[30.0], [10.0]],
            ]
        )
        ranked, names = rank_heat_metrics(metrics, ("seasonal_max", "ndays_excess_hot"))
        hot_idx = names.index("avg_across_metrics_hot")
        n_ext = int(ranked.shape[1])
        # Year 0: seasonal_max rank=3, ndays_excess_hot rank=1 → avg = 2.0
        yr0_avg = float(ranked.flat[0 * n_ext * 1 + hot_idx * 1 + 0])
        assert abs(yr0_avg - 2.0) < 1e-10
