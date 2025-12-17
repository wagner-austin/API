"""Tests for platform_ml.explainers.tree module.

Uses XGBoost for SHAP TreeExplainer tests. Requires SHAP >= 0.50 for XGBoost 3.x compatibility.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

from platform_ml.explainers.tree import (
    LocalExplanation,
    ShapTreeWrapper,
    TreeModelProtocol,
    _extract_expected_value,
    _extract_float_from_array,
    _is_ndarray,
)

from .array_helpers import make_float64_1d, make_float64_2d

# -----------------------------------------------------------------------------
# Typed Protocols for XGBoost (avoids Any from untyped library)
# -----------------------------------------------------------------------------


class XGBClassifierProtocol(Protocol):
    """Protocol for XGBClassifier interface."""

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> XGBClassifierProtocol:
        """Fit the model."""
        ...

    def predict_proba(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict class probabilities."""
        ...


class XGBClassifierConstructor(Protocol):
    """Protocol for XGBClassifier constructor."""

    def __call__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        random_state: int | None = None,
    ) -> XGBClassifierProtocol:
        """Create an XGBClassifier."""
        ...


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _get_float_from_2d(arr: NDArray[np.float64], row: int, col: int) -> float:
    """Extract float from 2D array using flat iteration.

    Args:
        arr: 2D source array.
        row: Row index.
        col: Column index.

    Returns:
        Float value at position.
    """
    n_cols = int(arr.shape[1])
    flat_idx = row * n_cols + col
    for idx, val in enumerate(arr.flat):
        if idx == flat_idx:
            return float(val.item())
    raise IndexError(f"Index ({row}, {col}) out of bounds")


def _make_int64_1d(values: list[int]) -> NDArray[np.int64]:
    """Create a 1D int64 array without list[Any] inference.

    Args:
        values: List of integer values.

    Returns:
        1D numpy array with int64 dtype.
    """
    n = len(values)
    result: NDArray[np.int64] = np.zeros(n, dtype=np.int64)
    for i, v in enumerate(values):
        result[i] = v
    return result


class SimpleTreeModel:
    """Simple model implementing TreeModelProtocol for conformance testing."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return simple probabilities based on first feature.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples, 2).
        """
        n_samples = int(x.shape[0])
        result: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        for i in range(n_samples):
            p1 = _get_float_from_2d(x, i, 0) * 0.1
            p1 = max(0.0, min(1.0, p1))
            result[i, 0] = 1.0 - p1
            result[i, 1] = p1
        return result


def _create_xgboost_model() -> XGBClassifierProtocol:
    """Create a trained XGBoost model for SHAP testing.

    Uses clearly separable data so XGBoost learns meaningful patterns.

    Returns:
        Trained XGBClassifier.
    """
    # Create clearly separable classes: class 0 has low feature values, class 1 has high
    x_train = make_float64_2d(
        [
            # Class 0 samples (low values)
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.15, 0.25, 0.35],
            [0.25, 0.35, 0.45],
            # Class 1 samples (high values)
            [0.7, 0.8, 0.9],
            [0.8, 0.9, 1.0],
            [0.75, 0.85, 0.95],
            [0.65, 0.75, 0.85],
            [0.6, 0.7, 0.8],
        ]
    )
    y_train = _make_int64_1d([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Dynamic import to avoid untyped xgboost import at module level
    xgb = __import__("xgboost")
    xgb_cls: XGBClassifierConstructor = xgb.XGBClassifier

    model = xgb_cls(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_tree_model_protocol_conformance() -> None:
    """Verify SimpleTreeModel conforms to TreeModelProtocol."""
    model: TreeModelProtocol = SimpleTreeModel()
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0]])

    proba = model.predict_proba(x)

    assert proba.shape == (2, 2)
    assert proba.dtype == np.float64


def test_local_explanation_creation() -> None:
    """Verify LocalExplanation TypedDict can be instantiated."""
    explanation: LocalExplanation = {
        "base_value": 0.5,
        "feature_names": ["f1", "f2", "f3"],
        "values": [0.1, -0.2, 0.3],
    }

    assert explanation["base_value"] == 0.5
    assert explanation["feature_names"] == ["f1", "f2", "f3"]
    assert explanation["values"] == [0.1, -0.2, 0.3]


def test_local_explanation_with_empty_features() -> None:
    """Verify LocalExplanation handles empty feature list."""
    explanation: LocalExplanation = {
        "base_value": 0.0,
        "feature_names": [],
        "values": [],
    }

    assert explanation["base_value"] == 0.0
    assert len(explanation["feature_names"]) == 0
    assert len(explanation["values"]) == 0


def test_shap_tree_wrapper_initialization() -> None:
    """Verify ShapTreeWrapper initializes with XGBoost model."""
    model = _create_xgboost_model()

    wrapper = ShapTreeWrapper(model)

    # Verify expected_value is a valid float by performing arithmetic
    _ = wrapper._expected_value + 1.0
    assert wrapper._expected_value == wrapper._expected_value  # Not NaN


def test_shap_tree_wrapper_explain_local_single_sample() -> None:
    """Verify explain_local returns correct structure for single sample."""
    model = _create_xgboost_model()
    wrapper = ShapTreeWrapper(model)
    x = make_float64_2d([[2.0, 3.0, 4.0]])
    feature_names = ["f1", "f2", "f3"]

    explanations = wrapper.explain_local(x, feature_names)

    assert len(explanations) == 1
    assert explanations[0]["feature_names"] == feature_names
    assert len(explanations[0]["values"]) == 3
    # Verify base_value is usable as float by performing arithmetic
    _ = explanations[0]["base_value"] + 1.0


def test_shap_tree_wrapper_explain_local_multiple_samples() -> None:
    """Verify explain_local returns correct structure for multiple samples."""
    model = _create_xgboost_model()
    wrapper = ShapTreeWrapper(model)
    x = make_float64_2d(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    feature_names = ["feat_a", "feat_b", "feat_c"]

    explanations = wrapper.explain_local(x, feature_names)

    assert len(explanations) == 3
    for expl in explanations:
        assert expl["feature_names"] == feature_names
        assert len(expl["values"]) == 3
        # Verify base_value usable as float
        _ = expl["base_value"] + 1.0


def test_shap_tree_wrapper_validation_feature_mismatch() -> None:
    """Verify explain_local raises on feature count mismatch."""
    model = _create_xgboost_model()
    wrapper = ShapTreeWrapper(model)
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    feature_names = ["f1", "f2"]  # Only 2 names for 3 features

    with pytest.raises(ValueError, match=r"Feature count mismatch"):
        wrapper.explain_local(x, feature_names)


def test_shap_tree_wrapper_validation_too_many_names() -> None:
    """Verify explain_local raises when too many feature names."""
    model = _create_xgboost_model()
    wrapper = ShapTreeWrapper(model)
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    feature_names = ["f1", "f2", "f3", "f4"]  # 4 names for 3 features

    with pytest.raises(ValueError, match=r"Feature count mismatch"):
        wrapper.explain_local(x, feature_names)


def test_shap_tree_wrapper_values_are_floats() -> None:
    """Verify all SHAP values are usable as Python floats."""
    model = _create_xgboost_model()
    wrapper = ShapTreeWrapper(model)
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    feature_names = ["f1", "f2", "f3"]

    explanations = wrapper.explain_local(x, feature_names)

    # Verify each value is usable as float by performing arithmetic
    total = 0.0
    for val in explanations[0]["values"]:
        total += val
    # Sum should be computable without error
    assert total == total  # Not NaN


def test_shap_tree_wrapper_expected_value_is_float() -> None:
    """Verify expected_value is stored as Python float."""
    model = _create_xgboost_model()

    wrapper = ShapTreeWrapper(model)

    # Verify expected_value is usable as float by performing arithmetic
    result = wrapper._expected_value * 2.0
    assert result == result  # Not NaN


def test_shap_tree_wrapper_reproducibility() -> None:
    """Verify same model produces same SHAP values."""
    model = _create_xgboost_model()
    x = make_float64_2d([[2.5, 3.5, 4.5]])
    feature_names = ["a", "b", "c"]

    wrapper1 = ShapTreeWrapper(model)
    explanations1 = wrapper1.explain_local(x, feature_names)

    wrapper2 = ShapTreeWrapper(model)
    explanations2 = wrapper2.explain_local(x, feature_names)

    assert explanations1[0]["values"] == explanations2[0]["values"]
    assert explanations1[0]["base_value"] == explanations2[0]["base_value"]


def test_shap_tree_wrapper_different_inputs_different_values() -> None:
    """Verify different inputs produce different SHAP values."""
    model = _create_xgboost_model()
    feature_names = ["f1", "f2", "f3"]

    wrapper = ShapTreeWrapper(model)

    # Use inputs within training data range to get meaningful SHAP differences
    x1 = make_float64_2d([[0.1, 0.2, 0.3]])  # Similar to class 0 sample
    x2 = make_float64_2d([[0.8, 0.9, 1.0]])  # Similar to class 1 sample

    expl1 = wrapper.explain_local(x1, feature_names)
    expl2 = wrapper.explain_local(x2, feature_names)

    # At least one SHAP value should differ
    values_differ = False
    for v1, v2 in zip(expl1[0]["values"], expl2[0]["values"], strict=True):
        if abs(v1 - v2) > 1e-10:
            values_differ = True
            break

    assert values_differ


# -----------------------------------------------------------------------------
# Helper Function Tests (for coverage)
# -----------------------------------------------------------------------------


def test_is_ndarray_with_scalar() -> None:
    """Verify _is_ndarray returns False for scalar float."""
    result = _is_ndarray(0.5)
    assert result is False


def test_is_ndarray_with_array() -> None:
    """Verify _is_ndarray returns True for ndarray."""
    arr = make_float64_1d([0.5])
    result = _is_ndarray(arr)
    assert result is True


def test_extract_expected_value_with_scalar() -> None:
    """Verify _extract_expected_value handles scalar float.

    This covers the else branch where expected_value is a plain float,
    not an ndarray (common with XGBoost binary classification).
    """
    result = _extract_expected_value(0.42)
    assert result == 0.42
    # Verify it's a Python float usable in arithmetic
    _ = result + 1.0


def test_extract_expected_value_with_array() -> None:
    """Verify _extract_expected_value handles 1D array.

    SHAP TreeExplainer can return expected_value as an array with
    [neg_class, pos_class] values. We extract the last (positive class).
    """
    arr = make_float64_1d([0.3, 0.7])
    result = _extract_expected_value(arr)
    # Should return last element (positive class)
    assert result == 0.7


def test_extract_float_from_array_single() -> None:
    """Verify _extract_float_from_array extracts from single-element array."""
    arr = make_float64_1d([0.99])
    result = _extract_float_from_array(arr)
    assert result == 0.99
