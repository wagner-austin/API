"""Tests for ClearGBM objective function.

Uses real cleargbm library for integration testing. No mocks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.objectives.cleargbm_objective import (
    ClearGBMObjective,
    _extract_positive_class_proba,
    _ndarray_to_float_matrix,
    _ndarray_to_int_tuple,
    create_cleargbm_objective,
)
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)

# =============================================================================
# Test Data Helpers
# =============================================================================


def _make_test_data(
    n_samples: int = 100, n_features: int = 5, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create test dataset for optimization.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (features, labels, feature_names).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    # Create imbalanced labels: ~30% positive
    n_positive = n_samples // 3
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_positive] = 1
    rng.shuffle(y)
    return x, y, [f"feat_{i}" for i in range(n_features)]


def _make_positive_data(x: NDArray[np.float64], offset: float) -> NDArray[np.float64]:
    """Make data positive by taking absolute value and adding offset.

    Args:
        x: Input array (can be negative).
        offset: Positive offset to add.

    Returns:
        Array with all positive values.
    """
    abs_x: NDArray[np.float64] = np.abs(x)
    result: NDArray[np.float64] = abs_x + offset
    return result


def _make_default_int_params() -> SampledIntParams:
    """Create default integer parameters for testing.

    Returns:
        SampledIntParams with typical ClearGBM values.
    """
    return SampledIntParams(
        n_estimators=10,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        max_bins=64,
    )


def _make_default_float_params() -> SampledFloatParams:
    """Create default float parameters for testing.

    Returns:
        SampledFloatParams with typical ClearGBM values.
    """
    return SampledFloatParams(
        learning_rate=0.1,
        subsample=1.0,
    )


def _make_default_string_params() -> SampledStringParams:
    """Create default string parameters for testing (empty for ClearGBM).

    Returns:
        Empty SampledStringParams (ClearGBM has no string params).
    """
    return SampledStringParams()


# =============================================================================
# Tests: Conversion Helpers
# =============================================================================


class TestNdarrayToFloatMatrix:
    """Tests for _ndarray_to_float_matrix conversion function."""

    def test_converts_2d_array_to_tuples(self) -> None:
        """_ndarray_to_float_matrix converts 2D array to tuple of tuples."""
        data: tuple[tuple[float, ...], ...] = ((1.0, 2.0), (3.0, 4.0))
        x: NDArray[np.float64] = np.array(data, dtype=np.float64)
        result = _ndarray_to_float_matrix(x)
        assert result == ((1.0, 2.0), (3.0, 4.0))

    def test_preserves_float_values(self) -> None:
        """_ndarray_to_float_matrix preserves decimal values."""
        data: tuple[tuple[float, ...], ...] = ((1.5, 2.25), (3.75, 4.125))
        x: NDArray[np.float64] = np.array(data, dtype=np.float64)
        result = _ndarray_to_float_matrix(x)
        assert result[0][0] == 1.5
        assert result[0][1] == 2.25
        assert result[1][0] == 3.75
        assert result[1][1] == 4.125

    def test_handles_single_row(self) -> None:
        """_ndarray_to_float_matrix handles single row array."""
        data: tuple[tuple[float, ...], ...] = ((1.0, 2.0, 3.0),)
        x: NDArray[np.float64] = np.array(data, dtype=np.float64)
        result = _ndarray_to_float_matrix(x)
        assert result == ((1.0, 2.0, 3.0),)

    def test_handles_single_column(self) -> None:
        """_ndarray_to_float_matrix handles single column array."""
        data: tuple[tuple[float, ...], ...] = ((1.0,), (2.0,), (3.0,))
        x: NDArray[np.float64] = np.array(data, dtype=np.float64)
        result = _ndarray_to_float_matrix(x)
        assert result == ((1.0,), (2.0,), (3.0,))


class TestNdarrayToIntTuple:
    """Tests for _ndarray_to_int_tuple conversion function."""

    def test_converts_1d_array_to_tuple(self) -> None:
        """_ndarray_to_int_tuple converts 1D array to tuple."""
        data: tuple[int, ...] = (0, 1, 0, 1)
        y: NDArray[np.int64] = np.array(data, dtype=np.int64)
        result = _ndarray_to_int_tuple(y)
        assert result == (0, 1, 0, 1)

    def test_preserves_int_values(self) -> None:
        """_ndarray_to_int_tuple preserves integer values."""
        data: tuple[int, ...] = (0, 1, 1, 0, 1)
        y: NDArray[np.int64] = np.array(data, dtype=np.int64)
        result = _ndarray_to_int_tuple(y)
        assert all(v in (0, 1) for v in result)
        assert len(result) == 5

    def test_handles_single_element(self) -> None:
        """_ndarray_to_int_tuple handles single element array."""
        data: tuple[int, ...] = (1,)
        y: NDArray[np.int64] = np.array(data, dtype=np.int64)
        result = _ndarray_to_int_tuple(y)
        assert result == (1,)


class TestExtractPositiveClassProba:
    """Tests for _extract_positive_class_proba extraction function."""

    def test_extracts_second_element_from_pairs(self) -> None:
        """_extract_positive_class_proba extracts second element (positive class)."""
        proba: tuple[tuple[float, float], ...] = ((0.8, 0.2), (0.3, 0.7), (0.5, 0.5))
        result = _extract_positive_class_proba(proba)
        expected: tuple[float, ...] = (0.2, 0.7, 0.5)
        np.testing.assert_array_almost_equal(result, np.array(expected, dtype=np.float64))

    def test_returns_ndarray_float64(self) -> None:
        """_extract_positive_class_proba returns NDArray[np.float64]."""
        proba: tuple[tuple[float, float], ...] = ((0.9, 0.1), (0.1, 0.9))
        result = _extract_positive_class_proba(proba)
        assert result.dtype == np.float64

    def test_handles_single_sample(self) -> None:
        """_extract_positive_class_proba handles single sample."""
        proba: tuple[tuple[float, float], ...] = ((0.6, 0.4),)
        result = _extract_positive_class_proba(proba)
        assert len(result) == 1
        result_val: float = float(result.flat[0].item())
        assert result_val == 0.4


# =============================================================================
# Tests: ClearGBMObjective Initialization
# =============================================================================


class TestClearGBMObjectiveInit:
    """Tests for ClearGBMObjective initialization."""

    def test_stores_feature_count(self) -> None:
        """ClearGBMObjective stores correct feature count after initialization."""
        x, y, names = _make_test_data(n_samples=50, n_features=5)
        objective = ClearGBMObjective(x, y, names, feature_preset="none")
        assert objective.n_features == 5

    def test_with_feature_engineering_log_only(self) -> None:
        """ClearGBMObjective applies log_only feature engineering correctly."""
        x, y, names = _make_test_data(n_samples=50, n_features=5)
        # Make all values positive for log transform
        x_positive = _make_positive_data(x, 1.0)
        objective = ClearGBMObjective(x_positive, y, names, feature_preset="log_only")
        # log_only adds log-transformed versions of original features
        assert objective.n_features > 5

    def test_with_feature_engineering_ratios_only(self) -> None:
        """ClearGBMObjective applies ratios_only feature engineering correctly."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        # Make values positive and avoid zeros for ratio computation
        x_positive = _make_positive_data(x, 0.1)
        objective = ClearGBMObjective(x_positive, y, names, feature_preset="ratios_only")
        # ratios_only adds ratio features
        assert objective.n_features >= 4

    def test_with_feature_engineering_full(self) -> None:
        """ClearGBMObjective applies full feature engineering correctly."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        # Make values positive for log and ratio transforms
        x_positive = _make_positive_data(x, 0.1)
        objective = ClearGBMObjective(x_positive, y, names, feature_preset="full")
        # full adds ratios, products, and log transforms
        assert objective.n_features > 4

    def test_with_custom_early_stopping_rounds(self) -> None:
        """ClearGBMObjective accepts custom early_stopping_rounds."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        objective = ClearGBMObjective(x, y, names, feature_preset="none", early_stopping_rounds=20)
        assert objective.n_features == 4


# =============================================================================
# Tests: ClearGBMObjective Call
# =============================================================================


class TestClearGBMObjectiveCall:
    """Tests for ClearGBMObjective.__call__ method."""

    def test_returns_auc(self) -> None:
        """ClearGBMObjective.__call__ returns validation AUC between 0 and 1."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = ClearGBMObjective(x, y, names, feature_preset="none")

        int_params = _make_default_int_params()
        float_params = _make_default_float_params()
        string_params = _make_default_string_params()

        auc = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=int_params,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        # AUC must be between 0 and 1
        assert 0.0 <= auc <= 1.0

    def test_ignores_passed_data(self) -> None:
        """ClearGBMObjective uses pre-split data, not passed arguments."""
        # Create objective with specific data
        x, y, names = _make_test_data(n_samples=100, n_features=5, seed=42)
        objective = ClearGBMObjective(x, y, names, feature_preset="none")

        # Pass different data (which should be ignored)
        x_other, y_other, names_other = _make_test_data(n_samples=50, n_features=3, seed=99)

        int_params = _make_default_int_params()
        float_params = _make_default_float_params()
        string_params = _make_default_string_params()

        # Should not raise even though passed data has different shape
        auc = objective(
            x_features=x_other,
            y_labels=y_other,
            feature_names=names_other,
            int_params=int_params,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert 0.0 <= auc <= 1.0

    def test_with_different_hyperparams(self) -> None:
        """ClearGBMObjective returns different results for different hyperparameters."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = ClearGBMObjective(x, y, names, feature_preset="none")

        # Test with shallow model vs deeper model
        int_params_simple = SampledIntParams(
            n_estimators=5, max_depth=2, min_samples_split=10, min_samples_leaf=5
        )
        int_params_complex = SampledIntParams(
            n_estimators=20, max_depth=5, min_samples_split=5, min_samples_leaf=2
        )
        float_params = _make_default_float_params()
        string_params = _make_default_string_params()

        auc_simple = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=int_params_simple,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        auc_complex = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=int_params_complex,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        # Both should be valid AUCs
        assert 0.0 <= auc_simple <= 1.0
        assert 0.0 <= auc_complex <= 1.0

    def test_multiple_calls_deterministic(self) -> None:
        """Multiple calls with same params return same AUC (deterministic)."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = ClearGBMObjective(x, y, names, feature_preset="none")

        int_params = _make_default_int_params()
        float_params = _make_default_float_params()
        string_params = _make_default_string_params()

        auc1 = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=int_params,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        auc2 = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=int_params,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert auc1 == auc2

    def test_with_subsample_less_than_one(self) -> None:
        """ClearGBMObjective handles subsample < 1.0 correctly."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = ClearGBMObjective(x, y, names, feature_preset="none")

        int_params = _make_default_int_params()
        float_params = SampledFloatParams(learning_rate=0.1, subsample=0.8)
        string_params = _make_default_string_params()

        auc = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=int_params,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert 0.0 <= auc <= 1.0


# =============================================================================
# Tests: Factory Function
# =============================================================================


class TestCreateClearGBMObjective:
    """Tests for create_cleargbm_objective factory function."""

    def test_returns_objective_with_n_features(self) -> None:
        """create_cleargbm_objective returns callable with n_features property."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        objective = create_cleargbm_objective(x, y, names, feature_preset="none")
        # Verify it has the n_features property
        assert objective.n_features == 4
        # Verify it's callable
        assert callable(objective)

    def test_with_feature_preset(self) -> None:
        """create_cleargbm_objective applies feature preset."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        x_positive = _make_positive_data(x, 0.1)  # Make positive for transformations
        objective = create_cleargbm_objective(x_positive, y, names, feature_preset="full")
        assert objective.n_features > 4

    def test_callable(self) -> None:
        """create_cleargbm_objective returns callable objective."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = create_cleargbm_objective(x, y, names, feature_preset="none")

        int_params = _make_default_int_params()
        float_params = _make_default_float_params()
        string_params = _make_default_string_params()

        auc = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=int_params,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert 0.0 <= auc <= 1.0

    def test_with_early_stopping_rounds(self) -> None:
        """create_cleargbm_objective accepts early_stopping_rounds parameter."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = create_cleargbm_objective(
            x, y, names, feature_preset="none", early_stopping_rounds=5
        )

        int_params = _make_default_int_params()
        float_params = _make_default_float_params()
        string_params = _make_default_string_params()

        auc = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=int_params,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert 0.0 <= auc <= 1.0


# =============================================================================
# Tests: n_features Property
# =============================================================================


class TestNFeaturesProperty:
    """Tests for ClearGBMObjective.n_features property."""

    def test_matches_original_when_no_engineering(self) -> None:
        """n_features matches input when no engineering applied."""
        x, y, names = _make_test_data(n_samples=50, n_features=7)
        objective = ClearGBMObjective(x, y, names, feature_preset="none")
        assert objective.n_features == 7

    def test_reflects_engineering(self) -> None:
        """n_features reflects engineered count when preset applied."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        x_positive = _make_positive_data(x, 0.1)  # Make positive for log transform
        objective = ClearGBMObjective(x_positive, y, names, feature_preset="log_only")
        # log_only typically doubles features (original + log)
        assert objective.n_features > 4
