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

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.preprocessing import (
    AutoPreprocessor,
    ImputationSpec,
    OutlierBounds,
    SpecialCodeSpec,
    apply_zscore,
    cap_outliers,
    compute_feature_stats,
    compute_imputation_values,
    create_auto_preprocessor,
    detect_outlier_bounds,
    detect_special_codes,
    impute_missing,
    replace_special_codes,
)

# =============================================================================
# Type-safe array helpers
# =============================================================================


def _arr(*values: float) -> NDArray[np.float64]:
    """Create 1D float64 array from values.

    Uses *args to ensure proper typing (no list[Any]).
    """
    result: NDArray[np.float64] = np.array(values, dtype=np.float64)
    return result


def _col(*values: float) -> NDArray[np.float64]:
    """Create single-column 2D array from values.

    Returns array of shape (len(values), 1).
    """
    arr_1d = _arr(*values)
    result: NDArray[np.float64] = arr_1d.reshape(-1, 1)
    return result


def _stack_cols(*cols: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stack 1D arrays as columns."""
    result: NDArray[np.float64] = np.column_stack(cols)
    return result


def _all_finite(arr: NDArray[np.float64]) -> bool:
    """Check if all values in array are finite (not NaN or inf)."""
    finite_mask: NDArray[np.bool_] = np.isfinite(arr)
    all_result: np.bool_ = np.all(finite_mask)
    return bool(all_result)


def _get_val(arr: NDArray[np.float64], row: int, col: int) -> float:
    """Get scalar value from 2D array with proper typing."""
    return float(arr.item((row, col)))


def _get_1d(arr: NDArray[np.float64], idx: int) -> float:
    """Get scalar value from 1D array with proper typing."""
    return float(arr.item(idx))


def _is_nan(value: float) -> bool:
    """Check if a scalar value is NaN."""
    import math

    return math.isnan(value)


def _any_equal(arr: NDArray[np.float64], value: float) -> bool:
    """Check if any element in array equals value."""
    eq_mask: NDArray[np.bool_] = arr == value
    any_result: np.bool_ = np.any(eq_mask)
    return bool(any_result)


def _max_abs(arr: NDArray[np.float64]) -> float:
    """Get max absolute value in array using iteration."""
    abs_arr: NDArray[np.float64] = np.abs(arr)
    max_val = 0.0
    for val in abs_arr.flat:
        v = float(val)
        if v > max_val:
            max_val = v
    return max_val


def _std(arr: NDArray[np.float64]) -> float:
    """Compute standard deviation using iteration."""
    n = int(arr.shape[0])
    if n == 0:
        return 0.0
    # Compute mean
    total = 0.0
    for val in arr.flat:
        total += float(val)
    mean = total / n
    # Compute variance
    var_sum = 0.0
    for val in arr.flat:
        diff = float(val) - mean
        var_sum += diff * diff
    return math.sqrt(var_sum / n)


def _make_test_matrix(n_rows: int, n_cols: int, seed: int = 42) -> NDArray[np.float64]:
    """Create test matrix with random but reproducible values."""
    rng = np.random.default_rng(seed)
    result: NDArray[np.float64] = rng.standard_normal((n_rows, n_cols)).astype(np.float64)
    return result


# =============================================================================
# Test Fixtures
# =============================================================================


def _make_simple_data() -> NDArray[np.float64]:
    """Create simple test data with known properties (5 rows, 3 cols)."""
    col0 = _arr(1.0, 2.0, 3.0, 4.0, 5.0)
    col1 = _arr(11.0, 12.0, 13.0, 14.0, 15.0)
    col2 = _arr(21.0, 22.0, 23.0, 24.0, 25.0)
    return _stack_cols(col0, col1, col2)


def _make_data_with_outliers() -> NDArray[np.float64]:
    """Create test data with extreme outliers."""
    col0 = _arr(1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -50.0)
    col1 = _arr(10.0, 20.0, 30.0, 40.0, 50.0, 1000.0, -500.0)
    return _stack_cols(col0, col1)


def _make_data_with_special_codes() -> NDArray[np.float64]:
    """Create test data with special codes (96, 98)."""
    col0 = _arr(1.0, 2.0, 3.0, 4.0, 5.0)
    col1 = _arr(96.0, 20.0, 98.0, 40.0, 50.0)
    return _stack_cols(col0, col1)


def _make_data_with_nan() -> NDArray[np.float64]:
    """Create test data with NaN values."""
    col0 = _arr(1.0, 2.0, np.nan, 4.0, 5.0)
    col1 = _arr(np.nan, 20.0, 30.0, 40.0, 50.0)
    return _stack_cols(col0, col1)


def _make_labels(n_samples: int) -> NDArray[np.int64]:
    """Create dummy labels for API consistency."""
    return np.zeros(n_samples, dtype=np.int64)


# =============================================================================
# Test: Outlier Detection
# =============================================================================


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


# =============================================================================
# Test: Special Code Detection
# =============================================================================


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


# =============================================================================
# Test: Imputation Value Computation
# =============================================================================


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


# =============================================================================
# Test: Transform Functions
# =============================================================================


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


# =============================================================================
# Test: Feature Stats Computation
# =============================================================================


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


# =============================================================================
# Test: AutoPreprocessor
# =============================================================================


class TestAutoPreprocessor:
    """Tests for AutoPreprocessor class."""

    def test_fit_returns_preprocessing_state(self) -> None:
        """fit() returns a PreprocessingState TypedDict."""
        x = _make_simple_data()
        y = _make_labels(len(x))
        preprocessor = AutoPreprocessor()

        state = preprocessor.fit(x, y)

        # Verify all required fields are present with correct types
        assert state["n_features"] == 3
        assert len(state["outlier_bounds"]) == 3
        assert len(state["special_codes"]) == 0  # No special codes in simple data
        assert len(state["imputation_values"]) == 3
        assert state["feature_means"].shape == (3,)
        assert state["feature_stds"].shape == (3,)

    def test_transform_applies_all_steps(self) -> None:
        """transform() applies special code, outlier, imputation, and z-score."""
        # Data with special code, outlier, and normal values
        col0 = _arr(1.0, 2.0, 3.0, 4.0, 100.0)  # Outlier at end
        col1 = _arr(96.0, 20.0, 30.0, 40.0, 50.0)  # Special code at start
        x_train = _stack_cols(col0, col1)
        y_train = _make_labels(len(x_train))
        preprocessor = AutoPreprocessor()

        state = preprocessor.fit(x_train, y_train)
        result = preprocessor.transform(x_train, state)

        # All values should be finite (no NaN)
        assert _all_finite(result)

        # Feature 0 outlier (100.0) should be capped
        assert _get_val(result, 4, 0) < 100.0

    def test_transform_on_new_data(self) -> None:
        """transform() works on new data using fitted state."""
        x_train = _make_simple_data()
        y_train = _make_labels(len(x_train))
        preprocessor = AutoPreprocessor()

        state = preprocessor.fit(x_train, y_train)

        # New data with same number of features
        x_new = _make_test_matrix(2, 3, seed=123)
        result = preprocessor.transform(x_new, state)

        assert result.shape == x_new.shape
        assert _all_finite(result)

    def test_transform_rejects_wrong_feature_count(self) -> None:
        """transform() raises ValueError on feature count mismatch."""
        x_train = _make_simple_data()  # 3 features
        y_train = _make_labels(len(x_train))
        preprocessor = AutoPreprocessor()

        state = preprocessor.fit(x_train, y_train)

        # Wrong feature count: 2 features instead of 3
        x_wrong = _make_test_matrix(1, 2)

        with pytest.raises(ValueError, match="Feature count mismatch"):
            preprocessor.transform(x_wrong, state)

    def test_does_not_modify_input(self) -> None:
        """transform() creates a copy, does not modify input."""
        x_train = _make_simple_data()
        y_train = _make_labels(len(x_train))
        preprocessor = AutoPreprocessor()

        state = preprocessor.fit(x_train, y_train)

        x_input = x_train.copy()
        original = x_input.copy()

        _ = preprocessor.transform(x_input, state)

        np.testing.assert_array_equal(x_input, original)

    def test_fit_only_uses_training_data(self) -> None:
        """fit() uses only training data, not validation data."""
        # Training data: wide range so validation values aren't outliers
        x_train = _col(0.0, 25.0, 50.0, 75.0, 100.0)
        y_train = _make_labels(len(x_train))

        # Validation data: within training range but different distribution
        x_val = _col(10.0, 20.0, 30.0, 40.0, 60.0)

        preprocessor = AutoPreprocessor()
        state = preprocessor.fit(x_train, y_train)

        # Stats should reflect training data (mean=50), not validation (mean=32)
        train_mean = _get_1d(state["feature_means"], 0)
        assert train_mean == pytest.approx(50.0)

        # Validation data should be normalized using train stats
        result = preprocessor.transform(x_val, state)

        # 10 with mean=50, std≈35.4 → z-score ≈ (10-50)/35.4 ≈ -1.13
        # If validation stats were used (mean=32), z-score would be different
        first_val = _get_val(result, 0, 0)
        assert first_val < 0.0  # Should be negative (below train mean)


class TestCreateAutoPreprocessor:
    """Tests for create_auto_preprocessor factory function."""

    def test_creates_instance_with_defaults(self) -> None:
        """Creates AutoPreprocessor with default settings."""
        preprocessor = create_auto_preprocessor()

        # Verify it works by running fit/transform
        x = _make_simple_data()
        y = _make_labels(len(x))
        state = preprocessor.fit(x, y)
        result = preprocessor.transform(x, state)

        assert result.shape == x.shape
        assert _all_finite(result)

    def test_creates_instance_with_custom_settings(self) -> None:
        """Creates AutoPreprocessor with custom settings."""
        preprocessor = create_auto_preprocessor(
            outlier_percentiles=(5.0, 95.0),
            imputation_strategy="mean",
            special_codes=frozenset({42.0}),
        )

        # Verify settings are used - 42.0 appears twice so it's detected
        x = _col(42.0, 42.0, 1.0, 2.0, 3.0)
        y = _make_labels(len(x))
        state = preprocessor.fit(x, y)

        # Should detect 42.0 as special code
        assert len(state["special_codes"]) == 1
        assert 42.0 in state["special_codes"][0]["codes"]


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample(self) -> None:
        """Handles single sample without error."""
        # 1 sample, 3 features
        x = _make_test_matrix(1, 3)
        y = _make_labels(1)
        preprocessor = AutoPreprocessor()

        state = preprocessor.fit(x, y)
        result = preprocessor.transform(x, state)

        assert result.shape == x.shape

    def test_single_feature(self) -> None:
        """Handles single feature without error."""
        x = _col(1.0, 2.0, 3.0, 4.0, 5.0)
        y = _make_labels(5)
        preprocessor = AutoPreprocessor()

        state = preprocessor.fit(x, y)
        result = preprocessor.transform(x, state)

        assert result.shape == x.shape

    def test_constant_feature(self) -> None:
        """Handles constant feature (zero variance) without division by zero."""
        col0 = _arr(5.0, 5.0, 5.0)  # Constant
        col1 = _arr(1.0, 2.0, 3.0)
        x = _stack_cols(col0, col1)
        y = _make_labels(3)
        preprocessor = AutoPreprocessor()

        state = preprocessor.fit(x, y)
        result = preprocessor.transform(x, state)

        # Should not have inf or nan
        assert _all_finite(result)

    def test_all_nan_feature(self) -> None:
        """Handles all-NaN feature gracefully."""
        col0 = _arr(1.0, 2.0, 3.0)
        col1 = _arr(np.nan, np.nan, np.nan)
        x = _stack_cols(col0, col1)
        y = _make_labels(3)
        preprocessor = AutoPreprocessor()

        state = preprocessor.fit(x, y)
        result = preprocessor.transform(x, state)

        # All-NaN feature should be imputed to 0.0
        assert _all_finite(result)
