"""Tests for platform_ml.explainers.regression_permutation module.

Achieves 100% statement and branch coverage by testing all functions,
class methods, validation paths, and edge cases.
Uses real models implementing RegressorPredictorProtocol without mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from platform_ml.explainers.protocol import RegressorPredictorProtocol
from platform_ml.explainers.regression_permutation import (
    REGRESSION_PERMUTATION_CAPABILITIES,
    RegressionPermutationExplainer,
    _compute_baseline_mse,
    _compute_single_feature_importance,
    _get_importance_from_pair,
    _rank_importances,
    _validate_inputs,
    create_regression_permutation_explainer,
)
from platform_ml.explainers.types import PermutationConfig

from .array_helpers import get_float, make_float64_2d


class FeatureSensitiveRegressor:
    """Regressor where predictions depend heavily on feature 0.

    Returns feature[0] * 10.0 + feature[1] * 0.1 so feature 0
    is clearly the most important.
    """

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict based on weighted sum of features.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Predicted values with shape (n_samples,).
        """
        n_samples = int(x.shape[0])
        result: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            result[i] = get_float(x, i, 0) * 10.0 + get_float(x, i, 1) * 0.1
        return result


class ConstantRegressor:
    """Regressor that always returns the same value regardless of input."""

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return constant predictions.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Array of 5.0 values with shape (n_samples,).
        """
        n_samples = int(x.shape[0])
        result: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            result[i] = 5.0
        return result


def test_validate_inputs_valid() -> None:
    """Validate inputs does not raise for matching dimensions."""
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0]])
    _validate_inputs(x, ["a", "b"])


def test_validate_inputs_mismatch() -> None:
    """Validate inputs raises ValueError for mismatched dimensions."""
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="feature_names length"):
        _validate_inputs(x, ["a", "b", "c"])


def test_compute_baseline_mse_returns_zero() -> None:
    """Baseline MSE comparing predictions to themselves is always 0."""
    model: RegressorPredictorProtocol = FeatureSensitiveRegressor()
    x = make_float64_2d([[1.0, 2.0]])
    baseline: NDArray[np.float64] = model.predict(x)
    result = _compute_baseline_mse(model, x, baseline)
    assert result == 0.0


def test_compute_single_feature_importance_positive() -> None:
    """Important feature produces positive importance score."""
    model: RegressorPredictorProtocol = FeatureSensitiveRegressor()
    x = make_float64_2d([[1.0, 2.0], [5.0, 3.0], [10.0, 1.0], [2.0, 8.0]])
    baseline: NDArray[np.float64] = model.predict(x)
    rng = np.random.default_rng(42)

    importance = _compute_single_feature_importance(
        model=model,
        x_data=x,
        baseline_preds=baseline,
        feature_idx=0,
        n_repeats=5,
        rng=rng,
    )
    assert importance > 0.0


def test_compute_single_feature_importance_constant_model() -> None:
    """Constant model gives zero importance for all features."""
    model: RegressorPredictorProtocol = ConstantRegressor()
    x = make_float64_2d([[1.0, 2.0], [5.0, 3.0], [10.0, 1.0], [2.0, 8.0]])
    baseline: NDArray[np.float64] = model.predict(x)
    rng = np.random.default_rng(42)

    importance = _compute_single_feature_importance(
        model=model,
        x_data=x,
        baseline_preds=baseline,
        feature_idx=0,
        n_repeats=3,
        rng=rng,
    )
    assert importance == 0.0


def test_get_importance_from_pair() -> None:
    """Extract importance value from tuple."""
    pair: tuple[str, float] = ("feature_a", 0.75)
    assert _get_importance_from_pair(pair) == 0.75


def test_rank_importances_order() -> None:
    """Rank importances returns correct descending order."""
    names = ["a", "b", "c"]
    importances = [0.1, 0.5, 0.3]
    ranked = _rank_importances(names, importances)

    assert len(ranked) == 3
    assert ranked[0]["name"] == "b"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["importance"] == 0.5
    assert ranked[1]["name"] == "c"
    assert ranked[1]["rank"] == 2
    assert ranked[2]["name"] == "a"
    assert ranked[2]["rank"] == 3


def test_regression_permutation_capabilities() -> None:
    """Check regression permutation capabilities constants."""
    assert REGRESSION_PERMUTATION_CAPABILITIES["requires_gradients"] is False
    assert REGRESSION_PERMUTATION_CAPABILITIES["requires_background_data"] is False
    assert REGRESSION_PERMUTATION_CAPABILITIES["computational_cost"] == "medium"


def test_explainer_name() -> None:
    """Explainer name returns 'permutation'."""
    config: PermutationConfig = {"n_repeats": 3, "random_state": 42}
    explainer = RegressionPermutationExplainer(config)
    assert explainer.explainer_name() == "permutation"


def test_explainer_capabilities() -> None:
    """Capabilities match module-level constant."""
    config: PermutationConfig = {"n_repeats": 3, "random_state": 42}
    explainer = RegressionPermutationExplainer(config)
    caps = explainer.capabilities()
    assert caps == REGRESSION_PERMUTATION_CAPABILITIES


def test_compute_importance_feature_ranking() -> None:
    """Feature 0 (weight=10) should rank higher than feature 1 (weight=0.1)."""
    config: PermutationConfig = {"n_repeats": 10, "random_state": 42}
    explainer = RegressionPermutationExplainer(config)
    model: RegressorPredictorProtocol = FeatureSensitiveRegressor()

    x = make_float64_2d([[1.0, 2.0], [5.0, 3.0], [10.0, 1.0], [2.0, 8.0]])

    importance = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=["feat_heavy", "feat_light"],
    )

    assert len(importance) == 2
    assert importance[0]["name"] == "feat_heavy"
    assert importance[0]["rank"] == 1
    assert importance[1]["name"] == "feat_light"
    assert importance[1]["rank"] == 2
    assert importance[0]["importance"] > importance[1]["importance"]


def test_compute_importance_constant_model() -> None:
    """Constant model gives zero importance for all features."""
    config: PermutationConfig = {"n_repeats": 5, "random_state": 42}
    explainer = RegressionPermutationExplainer(config)
    model: RegressorPredictorProtocol = ConstantRegressor()

    x = make_float64_2d([[1.0, 2.0], [5.0, 3.0], [10.0, 1.0]])

    importance = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=["a", "b"],
    )

    assert len(importance) == 2
    assert importance[0]["importance"] == 0.0
    assert importance[1]["importance"] == 0.0


def test_compute_importance_validation_error() -> None:
    """Mismatched feature names and data columns raises ValueError."""
    config: PermutationConfig = {"n_repeats": 3, "random_state": 42}
    explainer = RegressionPermutationExplainer(config)
    model: RegressorPredictorProtocol = FeatureSensitiveRegressor()

    x = make_float64_2d([[1.0, 2.0]])

    with pytest.raises(ValueError, match="feature_names length"):
        explainer.compute_importance(
            model=model,
            x_data=x,
            feature_names=["a", "b", "c"],
        )


def test_create_factory() -> None:
    """Factory function creates configured explainer."""
    config: PermutationConfig = {"n_repeats": 7, "random_state": 99}
    explainer = create_regression_permutation_explainer(config)
    assert explainer.explainer_name() == "permutation"


def test_deterministic_with_same_seed() -> None:
    """Same random_state produces identical results."""
    config: PermutationConfig = {"n_repeats": 5, "random_state": 42}
    model: RegressorPredictorProtocol = FeatureSensitiveRegressor()
    x = make_float64_2d([[1.0, 2.0], [5.0, 3.0], [10.0, 1.0], [2.0, 8.0]])
    names = ["a", "b"]

    explainer1 = RegressionPermutationExplainer(config)
    result1 = explainer1.compute_importance(model=model, x_data=x, feature_names=names)

    explainer2 = RegressionPermutationExplainer(config)
    result2 = explainer2.compute_importance(model=model, x_data=x, feature_names=names)

    assert result1[0]["importance"] == result2[0]["importance"]
    assert result1[1]["importance"] == result2[1]["importance"]
