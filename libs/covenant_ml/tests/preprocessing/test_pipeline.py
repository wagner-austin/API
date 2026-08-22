"""Tests for the preprocessing pipeline.

Tests cover:
- Outlier detection and capping
- Special code detection and replacement
- Missing value imputation
- Z-score normalization
- Full AutoPreprocessor fit/transform
- Data leakage prevention (fit only on training)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.preprocessing import (
    ImputationSpec,
    OutlierBounds,
    SpecialCodeSpec,
    apply_zscore,
    cap_outliers,
    compute_feature_stats,
    compute_imputation_values,
    detect_outlier_bounds,
    detect_special_codes,
    impute_missing,
    replace_special_codes,
)
from tests.preprocessing._pipeline_fixtures import (
    _any_equal,
    _arr,
    _col,
    _get_1d,
    _get_val,
    _is_nan,
    _make_data_with_nan,
    _make_data_with_special_codes,
    _make_simple_data,
    _max_abs,
    _stack_cols,
    _std,
)


class TestDetectOutlierBounds:
    """Tests for detect_outlier_bounds function."""

    def test_returns_bounds_for_each_feature(self) -> None:
        """Returns one OutlierBounds per feature."""
        x = _make_simple_data()
        bounds = detect_outlier_bounds(x)

        assert len(bounds) == 3
        assert all(isinstance(b, dict) for b in bounds)
        assert all("feature_idx" in b for b in bounds)
        assert all("lower" in b for b in bounds)
        assert all("upper" in b for b in bounds)

    def test_bounds_match_percentiles(self) -> None:
        """Bounds match requested percentiles."""
        # Create 0-99 sequence for testing
        x = _col(*[float(i) for i in range(100)])
        bounds = detect_outlier_bounds(x, percentiles=(5.0, 95.0))

        # 5th percentile of 0-99 is ~4.95, 95th is ~94.05
        assert bounds[0]["lower"] == pytest.approx(4.95, rel=0.1)
        assert bounds[0]["upper"] == pytest.approx(94.05, rel=0.1)

    def test_ignores_nan_values(self) -> None:
        """NaN values are excluded from percentile computation."""
        x = _col(1.0, 2.0, 3.0, np.nan, np.nan)
        bounds = detect_outlier_bounds(x, percentiles=(0.0, 100.0))

        # Min is 1.0, max is 3.0 (NaN excluded)
        assert bounds[0]["lower"] == pytest.approx(1.0)
        assert bounds[0]["upper"] == pytest.approx(3.0)

    def test_handles_all_nan_feature(self) -> None:
        """All-NaN feature gets default bounds of 0.0."""
        x = _col(np.nan, np.nan)
        bounds = detect_outlier_bounds(x)

        assert bounds[0]["lower"] == 0.0
        assert bounds[0]["upper"] == 0.0


class TestDetectSpecialCodes:
    """Tests for detect_special_codes function."""

    def test_detects_known_codes(self) -> None:
        """Detects special codes from known set."""
        x = _make_data_with_special_codes()
        specs = detect_special_codes(x, min_frequency=0.1)

        # Feature 1 has codes 96 and 98
        assert len(specs) == 1
        assert specs[0]["feature_idx"] == 1
        assert 96.0 in specs[0]["codes"]
        assert 98.0 in specs[0]["codes"]

    def test_no_codes_when_frequency_too_low(self) -> None:
        """Codes below frequency threshold are not detected."""
        # Create data with 96 appearing in 20% of rows (100 of 500)
        col0 = np.ones(500, dtype=np.float64)
        col1_base = np.full(400, 20.0, dtype=np.float64)
        col1_codes = np.full(100, 96.0, dtype=np.float64)
        col1: NDArray[np.float64] = np.concatenate([col1_codes, col1_base])
        x = _stack_cols(col0, col1)

        # Require 50% frequency
        specs = detect_special_codes(x, min_frequency=0.5)

        assert len(specs) == 0

    def test_empty_when_no_codes_present(self) -> None:
        """Returns empty when no special codes in data."""
        x = _make_simple_data()
        specs = detect_special_codes(x)

        assert len(specs) == 0

    def test_custom_codes_set(self) -> None:
        """Can use custom set of special codes."""
        col0 = _arr(1.0, 2.0, 3.0)
        col1 = _arr(42.0, 42.0, 30.0)
        x = _stack_cols(col0, col1)
        specs = detect_special_codes(x, known_codes=frozenset({42.0}))

        assert len(specs) == 1
        assert specs[0]["codes"] == (42.0,)


class TestComputeImputationValues:
    """Tests for compute_imputation_values function."""

    def test_median_strategy(self) -> None:
        """Median strategy computes median per feature."""
        col0 = _arr(1.0, 2.0, 3.0, 4.0, 5.0)
        col1 = _arr(10.0, 20.0, 30.0, 40.0, 50.0)
        x = _stack_cols(col0, col1)
        specs = compute_imputation_values(x, special_codes=(), strategy="median")

        assert len(specs) == 2
        assert specs[0]["impute_value"] == pytest.approx(3.0)  # median of 1-5
        assert specs[1]["impute_value"] == pytest.approx(30.0)  # median of 10-50

    def test_mean_strategy(self) -> None:
        """Mean strategy computes mean per feature."""
        col0 = _arr(1.0, 2.0, 3.0, 4.0, 5.0)
        col1 = _arr(10.0, 20.0, 30.0, 40.0, 50.0)
        x = _stack_cols(col0, col1)
        specs = compute_imputation_values(x, special_codes=(), strategy="mean")

        assert specs[0]["impute_value"] == pytest.approx(3.0)  # mean of 1-5
        assert specs[1]["impute_value"] == pytest.approx(30.0)  # mean of 10-50

    def test_zero_strategy(self) -> None:
        """Zero strategy always returns 0.0."""
        x = _make_simple_data()
        specs = compute_imputation_values(x, special_codes=(), strategy="zero")

        assert all(s["impute_value"] == 0.0 for s in specs)

    def test_excludes_special_codes(self) -> None:
        """Special codes are excluded from imputation computation."""
        x = _col(1.0, 2.0, 3.0, 96.0, 98.0)
        special_codes = (SpecialCodeSpec(feature_idx=0, codes=(96.0, 98.0)),)
        specs = compute_imputation_values(x, special_codes, strategy="median")

        # Median of [1, 2, 3] = 2.0 (96, 98 excluded)
        assert specs[0]["impute_value"] == pytest.approx(2.0)

    def test_excludes_nan(self) -> None:
        """NaN values are excluded from imputation computation."""
        x = _col(1.0, 2.0, 3.0, np.nan, np.nan)
        specs = compute_imputation_values(x, special_codes=(), strategy="median")

        # Median of [1, 2, 3] = 2.0
        assert specs[0]["impute_value"] == pytest.approx(2.0)

    def test_all_nan_returns_zero(self) -> None:
        """All-NaN column returns imputation value of 0.0."""
        x = _col(np.nan, np.nan, np.nan)
        specs = compute_imputation_values(x, special_codes=(), strategy="median")

        # No valid values → 0.0
        assert specs[0]["impute_value"] == pytest.approx(0.0)

    def test_all_nan_mean_returns_zero(self) -> None:
        """All-NaN column with mean strategy returns 0.0."""
        x = _col(np.nan, np.nan, np.nan)
        specs = compute_imputation_values(x, special_codes=(), strategy="mean")

        # No valid values → 0.0
        assert specs[0]["impute_value"] == pytest.approx(0.0)


class TestReplaceSpecialCodes:
    """Tests for replace_special_codes function."""

    def test_replaces_codes_with_nan(self) -> None:
        """Special codes are replaced with NaN."""
        x = _make_data_with_special_codes().copy()
        specs = (SpecialCodeSpec(feature_idx=1, codes=(96.0, 98.0)),)

        result = replace_special_codes(x, specs)

        assert _is_nan(_get_val(result, 0, 1))  # Was 96
        assert _is_nan(_get_val(result, 2, 1))  # Was 98
        assert _get_val(result, 1, 1) == 20.0  # Unchanged

    def test_empty_specs_no_change(self) -> None:
        """Empty specs tuple leaves data unchanged."""
        x = _make_simple_data().copy()
        original = x.copy()

        result = replace_special_codes(x, ())

        np.testing.assert_array_equal(result, original)


class TestCapOutliers:
    """Tests for cap_outliers function."""

    def test_caps_to_bounds(self) -> None:
        """Values outside bounds are capped."""
        x = _col(0.0, 50.0, 100.0)
        bounds = (OutlierBounds(feature_idx=0, lower=10.0, upper=90.0),)

        result = cap_outliers(x, bounds)

        assert _get_val(result, 0, 0) == 10.0  # Was 0, capped up
        assert _get_val(result, 1, 0) == 50.0  # Unchanged
        assert _get_val(result, 2, 0) == 90.0  # Was 100, capped down

    def test_equal_bounds_no_change(self) -> None:
        """Equal lower/upper bounds means no capping."""
        x = _col(0.0, 50.0, 100.0)
        bounds = (OutlierBounds(feature_idx=0, lower=50.0, upper=50.0),)

        result = cap_outliers(x, bounds)

        # No change because lower >= upper
        assert _get_val(result, 0, 0) == 0.0
        assert _get_val(result, 2, 0) == 100.0


class TestImputeMissing:
    """Tests for impute_missing function."""

    def test_replaces_nan_with_impute_value(self) -> None:
        """NaN values are replaced with imputation values."""
        x = _make_data_with_nan().copy()
        specs = (
            ImputationSpec(feature_idx=0, impute_value=3.0),
            ImputationSpec(feature_idx=1, impute_value=35.0),
        )

        result = impute_missing(x, specs)

        assert _get_val(result, 2, 0) == 3.0  # Was NaN
        assert _get_val(result, 0, 1) == 35.0  # Was NaN

    def test_no_nan_no_change(self) -> None:
        """Data without NaN is unchanged."""
        x = _make_simple_data().copy()
        specs = (
            ImputationSpec(feature_idx=0, impute_value=99.0),
            ImputationSpec(feature_idx=1, impute_value=99.0),
            ImputationSpec(feature_idx=2, impute_value=99.0),
        )

        result = impute_missing(x, specs)

        # No NaN in original, so no 99.0 should appear
        assert not _any_equal(result, 99.0)


class TestApplyZscore:
    """Tests for apply_zscore function."""

    def test_normalizes_to_zero_mean_unit_std(self) -> None:
        """Z-score normalization produces zero mean and unit std."""
        col0 = _arr(10.0, 20.0, 30.0, 40.0, 50.0)
        col1 = col0 * 10.0
        x = _stack_cols(col0, col1)
        means = _arr(30.0, 300.0)
        stds = _arr(14.142135, 141.42135)

        result = apply_zscore(x.copy(), means, stds)

        # Mean should be close to 0
        result_means: NDArray[np.float64] = result.mean(axis=0)
        max_abs_mean = _max_abs(result_means)
        assert max_abs_mean < 1e-10
        # Std should be close to 1
        result_stds: NDArray[np.float64] = result.std(axis=0)
        np.testing.assert_allclose(result_stds, _arr(1.0, 1.0), rtol=0.01)


class TestComputeFeatureStats:
    """Tests for compute_feature_stats function."""

    def test_computes_mean_and_std(self) -> None:
        """Computes mean and std per feature."""
        col0 = _arr(1.0, 2.0, 3.0)
        col1 = _arr(10.0, 20.0, 30.0)
        x = _stack_cols(col0, col1)
        means, stds = compute_feature_stats(x)

        assert _get_1d(means, 0) == pytest.approx(2.0)
        assert _get_1d(means, 1) == pytest.approx(20.0)
        # Compute expected std manually
        col0_std = _std(col0)
        col1_std = _std(col1)
        assert _get_1d(stds, 0) == pytest.approx(col0_std)
        assert _get_1d(stds, 1) == pytest.approx(col1_std)

    def test_zero_std_replaced_with_one(self) -> None:
        """Zero std is replaced with 1.0 to avoid division by zero."""
        col0 = _arr(5.0, 5.0, 5.0)  # Constant
        col1 = _arr(10.0, 20.0, 30.0)
        x = _stack_cols(col0, col1)
        _means, stds = compute_feature_stats(x)

        # Feature 0 has zero std (constant)
        assert _get_1d(stds, 0) == 1.0
        assert _get_1d(stds, 1) > 0.0

    def test_ignores_nan(self) -> None:
        """NaN values are ignored in mean/std computation."""
        col0 = _arr(1.0, 2.0, 3.0)
        col1 = _arr(np.nan, 20.0, 30.0)
        x = _stack_cols(col0, col1)
        means, _stds = compute_feature_stats(x)

        assert _get_1d(means, 0) == pytest.approx(2.0)
        assert _get_1d(means, 1) == pytest.approx(25.0)  # mean of [20, 30]

    def test_all_nan_column(self) -> None:
        """All-NaN column returns zero mean and one std."""
        col0 = _arr(1.0, 2.0, 3.0)
        col1 = _arr(np.nan, np.nan, np.nan)
        x = _stack_cols(col0, col1)
        means, stds = compute_feature_stats(x)

        # Column 0: normal computation
        assert _get_1d(means, 0) == pytest.approx(2.0)
        # Column 1: all NaN → mean=0.0, std=0.0 → replaced with 1.0
        assert _get_1d(means, 1) == pytest.approx(0.0)
        assert _get_1d(stds, 1) == pytest.approx(1.0)
