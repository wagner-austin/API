"""Tests for Random Forest objective function.

Uses real sklearn RandomForestClassifier for integration testing. No mocks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.objectives.random_forest_objective import (
    RandomForestObjective,
    create_random_forest_objective,
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
        SampledIntParams with typical Random Forest values.
    """
    return SampledIntParams(
        n_estimators=10,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
    )


def _make_default_float_params() -> SampledFloatParams:
    """Create default float parameters for testing (empty for RF).

    Returns:
        Empty SampledFloatParams (RF has no float params in search space).
    """
    return SampledFloatParams()


def _make_default_string_params() -> SampledStringParams:
    """Create default string parameters for testing.

    Returns:
        SampledStringParams with typical Random Forest values.
    """
    return SampledStringParams(max_features="sqrt")


# =============================================================================
# Tests: RandomForestObjective Initialization
# =============================================================================


class TestRandomForestObjectiveInit:
    """Tests for RandomForestObjective initialization."""

    def test_stores_feature_count(self) -> None:
        """RandomForestObjective stores correct feature count after initialization."""
        x, y, names = _make_test_data(n_samples=50, n_features=5)
        objective = RandomForestObjective(x, y, names, feature_preset="none")
        assert objective.n_features == 5

    def test_with_feature_engineering_log_only(self) -> None:
        """RandomForestObjective applies log_only feature engineering correctly."""
        x, y, names = _make_test_data(n_samples=50, n_features=5)
        x_positive = _make_positive_data(x, 1.0)
        objective = RandomForestObjective(x_positive, y, names, feature_preset="log_only")
        assert objective.n_features > 5

    def test_with_feature_engineering_full(self) -> None:
        """RandomForestObjective applies full feature engineering correctly."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        x_positive = _make_positive_data(x, 0.1)
        objective = RandomForestObjective(x_positive, y, names, feature_preset="full")
        assert objective.n_features > 4


# =============================================================================
# Tests: RandomForestObjective Call
# =============================================================================


class TestRandomForestObjectiveCall:
    """Tests for RandomForestObjective.__call__ method."""

    def test_returns_auc(self) -> None:
        """RandomForestObjective.__call__ returns validation AUC between 0 and 1."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = RandomForestObjective(x, y, names, feature_preset="none")

        auc = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=_make_default_int_params(),
            float_params=_make_default_float_params(),
            string_params=_make_default_string_params(),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert 0.0 <= auc <= 1.0

    def test_ignores_passed_data(self) -> None:
        """RandomForestObjective uses pre-split data, not passed arguments."""
        x, y, names = _make_test_data(n_samples=100, n_features=5, seed=42)
        objective = RandomForestObjective(x, y, names, feature_preset="none")

        x_other, y_other, names_other = _make_test_data(n_samples=50, n_features=3, seed=99)

        auc = objective(
            x_features=x_other,
            y_labels=y_other,
            feature_names=names_other,
            int_params=_make_default_int_params(),
            float_params=_make_default_float_params(),
            string_params=_make_default_string_params(),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert 0.0 <= auc <= 1.0

    def test_with_log2_max_features(self) -> None:
        """RandomForestObjective works with log2 max_features."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = RandomForestObjective(x, y, names, feature_preset="none")

        auc = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=_make_default_int_params(),
            float_params=_make_default_float_params(),
            string_params=SampledStringParams(max_features="log2"),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert 0.0 <= auc <= 1.0

    def test_multiple_calls_deterministic(self) -> None:
        """Multiple calls with same params return same AUC (deterministic)."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = RandomForestObjective(x, y, names, feature_preset="none")

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

    def test_with_different_depths(self) -> None:
        """RandomForestObjective returns results for different tree depths."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = RandomForestObjective(x, y, names, feature_preset="none")

        auc_shallow = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=SampledIntParams(
                n_estimators=10,
                max_depth=2,
                min_samples_split=5,
                min_samples_leaf=2,
            ),
            float_params=_make_default_float_params(),
            string_params=_make_default_string_params(),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        auc_deep = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=SampledIntParams(
                n_estimators=20,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
            ),
            float_params=_make_default_float_params(),
            string_params=_make_default_string_params(),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert 0.0 <= auc_shallow <= 1.0
        assert 0.0 <= auc_deep <= 1.0


# =============================================================================
# Tests: Factory Function
# =============================================================================


class TestCreateRandomForestObjective:
    """Tests for create_random_forest_objective factory function."""

    def test_returns_objective_with_n_features(self) -> None:
        """create_random_forest_objective returns callable with n_features property."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        objective = create_random_forest_objective(x, y, names, feature_preset="none")
        assert objective.n_features == 4
        assert callable(objective)

    def test_with_feature_preset(self) -> None:
        """create_random_forest_objective applies feature preset."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        x_positive = _make_positive_data(x, 0.1)
        objective = create_random_forest_objective(x_positive, y, names, feature_preset="full")
        assert objective.n_features > 4

    def test_callable(self) -> None:
        """create_random_forest_objective returns callable objective."""
        x, y, names = _make_test_data(n_samples=100, n_features=5)
        objective = create_random_forest_objective(x, y, names, feature_preset="none")

        auc = objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            int_params=_make_default_int_params(),
            float_params=_make_default_float_params(),
            string_params=_make_default_string_params(),
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
    """Tests for RandomForestObjective.n_features property."""

    def test_matches_original_when_no_engineering(self) -> None:
        """n_features matches input when no engineering applied."""
        x, y, names = _make_test_data(n_samples=50, n_features=7)
        objective = RandomForestObjective(x, y, names, feature_preset="none")
        assert objective.n_features == 7

    def test_reflects_engineering(self) -> None:
        """n_features reflects engineered count when preset applied."""
        x, y, names = _make_test_data(n_samples=50, n_features=4)
        x_positive = _make_positive_data(x, 0.1)
        objective = RandomForestObjective(x_positive, y, names, feature_preset="log_only")
        assert objective.n_features > 4
