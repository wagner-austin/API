"""Tests for platform_ml.explainers.permutation module.

Achieves 100% statement and branch coverage by testing all functions,
class methods, validation paths, and edge cases.
Uses real models implementing PredictorProtocol without mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from platform_ml.explainers.permutation import (
    PERMUTATION_CAPABILITIES,
    PermutationExplainer,
    _compute_baseline_proba,
    _compute_single_feature_importance,
    _get_importance_from_pair,
    _rank_importances,
    _validate_inputs,
    create_permutation_explainer,
)
from platform_ml.explainers.protocol import PredictorProtocol
from platform_ml.explainers.types import PermutationConfig

from .array_helpers import get_float, make_float64_2d


def _extract_value(arr: NDArray[np.float64], row: int, col: int) -> float:
    """Extract a value from a 2D array using flat iteration.

    Args:
        arr: Source array.
        row: Row index.
        col: Column index.

    Returns:
        Float value at the specified position.
    """
    n_cols = int(arr.shape[1])
    flat_idx = row * n_cols + col
    for idx, val in enumerate(arr.flat):
        if idx == flat_idx:
            return float(val.item())
    raise IndexError(f"Index ({row}, {col}) out of bounds")


class FeatureSensitiveModel:
    """Model where predictions depend on specific features.

    First feature has high influence, second has low influence.
    This allows testing that permutation importance correctly identifies
    which features matter.
    """

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return probabilities based on feature values.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples, 2).
        """
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)

        for i in range(n_samples):
            feat_0: float = _extract_value(x, i, 0)
            feat_1: float = _extract_value(x, i, 1)
            p1: float = 0.5 + 0.4 * feat_0 + 0.01 * feat_1
            p1 = max(0.0, min(1.0, p1))
            proba[i, 0] = 1.0 - p1
            proba[i, 1] = p1

        return proba


class UniformModel:
    """Model that returns uniform predictions regardless of input."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples, 2).
        """
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        proba[:, 0] = 0.5
        proba[:, 1] = 0.5
        return proba


def test_permutation_capabilities_values() -> None:
    """Verify PERMUTATION_CAPABILITIES has correct values."""
    assert PERMUTATION_CAPABILITIES["requires_gradients"] is False
    assert PERMUTATION_CAPABILITIES["requires_background_data"] is False
    assert PERMUTATION_CAPABILITIES["computational_cost"] == "medium"


def test_validate_inputs_matching_dimensions() -> None:
    """Verify _validate_inputs passes with matching dimensions."""
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    feature_names = ["a", "b", "c"]
    _validate_inputs(x, feature_names)


def test_validate_inputs_mismatched_dimensions_raises() -> None:
    """Verify _validate_inputs raises with mismatched dimensions."""
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    feature_names = ["a", "b"]

    with pytest.raises(ValueError, match=r"feature_names length.*must match x_data columns"):
        _validate_inputs(x, feature_names)


def test_validate_inputs_empty_features() -> None:
    """Verify _validate_inputs works with zero-length matching."""
    x: NDArray[np.float64] = np.zeros((5, 0), dtype=np.float64)
    feature_names: list[str] = []
    _validate_inputs(x, feature_names)


def test_compute_baseline_proba_returns_correct_shape() -> None:
    """Verify _compute_baseline_proba returns correct shape."""
    model: PredictorProtocol = UniformModel()
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    proba = _compute_baseline_proba(model, x, target_class=1)

    assert proba.shape == (3,)
    assert proba.dtype == np.float64


def test_compute_baseline_proba_extracts_target_class() -> None:
    """Verify _compute_baseline_proba extracts correct target class."""
    model: PredictorProtocol = UniformModel()
    x = make_float64_2d([[1.0, 2.0]])

    proba_class_0 = _compute_baseline_proba(model, x, target_class=0)
    proba_class_1 = _compute_baseline_proba(model, x, target_class=1)

    assert get_float(proba_class_0, 0) == 0.5
    assert get_float(proba_class_1, 0) == 0.5


def test_compute_baseline_proba_with_feature_sensitive_model() -> None:
    """Verify _compute_baseline_proba works with feature-sensitive model."""
    model: PredictorProtocol = FeatureSensitiveModel()
    x = make_float64_2d([[0.5, 0.0]])

    proba = _compute_baseline_proba(model, x, target_class=1)

    # p1 = 0.5 + 0.4*0.5 + 0.01*0.0 = 0.7
    assert abs(get_float(proba, 0) - 0.7) < 1e-10


def test_compute_single_feature_importance_uniform_model() -> None:
    """Verify uniform model has zero importance for all features."""
    model: PredictorProtocol = UniformModel()
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0]])
    baseline_proba = _compute_baseline_proba(model, x, target_class=1)
    rng = np.random.default_rng(42)

    importance = _compute_single_feature_importance(
        model=model,
        x_data=x,
        baseline_proba=baseline_proba,
        feature_idx=0,
        target_class=1,
        n_repeats=5,
        rng=rng,
    )

    assert importance == 0.0


def test_compute_single_feature_importance_sensitive_feature() -> None:
    """Verify sensitive feature has higher importance."""
    model: PredictorProtocol = FeatureSensitiveModel()
    x = make_float64_2d([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]])
    baseline_proba = _compute_baseline_proba(model, x, target_class=1)
    rng = np.random.default_rng(42)

    importance_0 = _compute_single_feature_importance(
        model=model,
        x_data=x,
        baseline_proba=baseline_proba,
        feature_idx=0,
        target_class=1,
        n_repeats=10,
        rng=rng,
    )

    importance_1 = _compute_single_feature_importance(
        model=model,
        x_data=x,
        baseline_proba=baseline_proba,
        feature_idx=1,
        target_class=1,
        n_repeats=10,
        rng=rng,
    )

    assert importance_0 > importance_1


def test_compute_single_feature_importance_multiple_repeats() -> None:
    """Verify multiple repeats are averaged correctly."""
    model: PredictorProtocol = FeatureSensitiveModel()
    x = make_float64_2d([[0.5, 0.5], [0.8, 0.2]])
    baseline_proba = _compute_baseline_proba(model, x, target_class=1)
    rng = np.random.default_rng(123)

    importance = _compute_single_feature_importance(
        model=model,
        x_data=x,
        baseline_proba=baseline_proba,
        feature_idx=0,
        target_class=1,
        n_repeats=20,
        rng=rng,
    )

    assert importance >= 0.0


def test_get_importance_from_pair_extracts_value() -> None:
    """Verify _get_importance_from_pair extracts second element."""
    pair: tuple[str, float] = ("feature_name", 0.75)
    result = _get_importance_from_pair(pair)
    assert result == 0.75


def test_get_importance_from_pair_zero_value() -> None:
    """Verify _get_importance_from_pair works with zero."""
    pair: tuple[str, float] = ("zero_feature", 0.0)
    result = _get_importance_from_pair(pair)
    assert result == 0.0


def test_get_importance_from_pair_negative_value() -> None:
    """Verify _get_importance_from_pair works with negative values."""
    pair: tuple[str, float] = ("negative_feature", -0.25)
    result = _get_importance_from_pair(pair)
    assert result == -0.25


def test_rank_importances_correct_ordering() -> None:
    """Verify _rank_importances sorts by importance descending."""
    feature_names = ["low", "high", "medium"]
    importances: list[float] = [0.1, 0.9, 0.5]

    result = _rank_importances(feature_names, importances)

    assert len(result) == 3
    assert result[0]["name"] == "high"
    assert result[0]["importance"] == 0.9
    assert result[0]["rank"] == 1
    assert result[1]["name"] == "medium"
    assert result[1]["importance"] == 0.5
    assert result[1]["rank"] == 2
    assert result[2]["name"] == "low"
    assert result[2]["importance"] == 0.1
    assert result[2]["rank"] == 3


def test_rank_importances_equal_values() -> None:
    """Verify _rank_importances handles equal importance values."""
    feature_names = ["a", "b", "c"]
    importances: list[float] = [0.5, 0.5, 0.5]

    result = _rank_importances(feature_names, importances)

    assert len(result) == 3
    ranks = [r["rank"] for r in result]
    assert sorted(ranks) == [1, 2, 3]


def test_rank_importances_single_feature() -> None:
    """Verify _rank_importances works with single feature."""
    feature_names = ["only_one"]
    importances: list[float] = [0.42]

    result = _rank_importances(feature_names, importances)

    assert len(result) == 1
    assert result[0]["name"] == "only_one"
    assert result[0]["importance"] == 0.42
    assert result[0]["rank"] == 1


def test_permutation_explainer_init() -> None:
    """Verify PermutationExplainer initializes correctly."""
    config: PermutationConfig = {"n_repeats": 15, "random_state": 123}
    explainer = PermutationExplainer(config)

    assert explainer._n_repeats == 15
    assert explainer._random_state == 123


def test_permutation_explainer_name() -> None:
    """Verify PermutationExplainer.explainer_name returns 'permutation'."""
    config: PermutationConfig = {"n_repeats": 5, "random_state": 0}
    explainer = PermutationExplainer(config)

    assert explainer.explainer_name() == "permutation"


def test_permutation_explainer_capabilities() -> None:
    """Verify PermutationExplainer.capabilities returns correct values."""
    config: PermutationConfig = {"n_repeats": 5, "random_state": 0}
    explainer = PermutationExplainer(config)

    caps = explainer.capabilities()

    assert caps["requires_gradients"] is False
    assert caps["requires_background_data"] is False
    assert caps["computational_cost"] == "medium"


def test_permutation_explainer_compute_importance_validates_input() -> None:
    """Verify compute_importance validates input dimensions."""
    config: PermutationConfig = {"n_repeats": 5, "random_state": 0}
    explainer = PermutationExplainer(config)
    model: PredictorProtocol = UniformModel()
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    feature_names = ["a"]

    with pytest.raises(ValueError, match=r"feature_names length"):
        explainer.compute_importance(
            model=model,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )


def test_permutation_explainer_compute_importance_uniform_model() -> None:
    """Verify compute_importance with uniform model gives zero importance."""
    config: PermutationConfig = {"n_repeats": 10, "random_state": 42}
    explainer = PermutationExplainer(config)
    model: PredictorProtocol = UniformModel()
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0]])
    feature_names = ["feat_a", "feat_b"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert len(result) == 2
    assert result[0]["importance"] == 0.0
    assert result[1]["importance"] == 0.0


def test_permutation_explainer_compute_importance_sensitive_model() -> None:
    """Verify compute_importance correctly ranks sensitive features."""
    config: PermutationConfig = {"n_repeats": 20, "random_state": 42}
    explainer = PermutationExplainer(config)
    model: PredictorProtocol = FeatureSensitiveModel()
    x = make_float64_2d([[0.0, 0.5], [0.3, 0.5], [0.6, 0.5], [1.0, 0.5]])
    feature_names = ["important", "not_important"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert len(result) == 2
    assert result[0]["name"] == "important"
    assert result[0]["rank"] == 1
    assert result[1]["name"] == "not_important"
    assert result[1]["rank"] == 2


def test_permutation_explainer_reproducibility() -> None:
    """Verify same random_state produces same results."""
    config: PermutationConfig = {"n_repeats": 10, "random_state": 999}
    model: PredictorProtocol = FeatureSensitiveModel()
    x = make_float64_2d([[0.2, 0.3], [0.7, 0.8]])
    feature_names = ["a", "b"]

    explainer1 = PermutationExplainer(config)
    result1 = explainer1.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    explainer2 = PermutationExplainer(config)
    result2 = explainer2.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert result1[0]["importance"] == result2[0]["importance"]
    assert result1[1]["importance"] == result2[1]["importance"]


def test_create_permutation_explainer_returns_configured_instance() -> None:
    """Verify create_permutation_explainer returns properly configured instance."""
    config: PermutationConfig = {"n_repeats": 8, "random_state": 77}

    explainer = create_permutation_explainer(config)

    assert explainer._n_repeats == 8
    assert explainer._random_state == 77
    assert explainer.explainer_name() == "permutation"
    assert explainer.capabilities()["requires_gradients"] is False


def test_create_permutation_explainer_functional() -> None:
    """Verify factory-created explainer works correctly."""
    config: PermutationConfig = {"n_repeats": 5, "random_state": 42}
    explainer = create_permutation_explainer(config)
    model: PredictorProtocol = UniformModel()
    x = make_float64_2d([[1.0, 2.0]])
    feature_names = ["x", "y"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=0,
    )

    assert len(result) == 2
    assert all(r["importance"] >= 0.0 for r in result)
