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

from covenant_ml.preprocessing import (
    AutoPreprocessor,
    create_auto_preprocessor,
)
from tests.preprocessing._pipeline_fixtures import (
    _all_finite,
    _arr,
    _col,
    _get_1d,
    _get_val,
    _make_labels,
    _make_simple_data,
    _make_test_matrix,
    _stack_cols,
)


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
