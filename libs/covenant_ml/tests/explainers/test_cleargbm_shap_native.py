"""Tests for ClearGBM SHAP adapter.

Comprehensive tests for converting ClearGBM models to SHAP format
and computing local explanations.
"""

from __future__ import annotations

import numpy as np
import pytest
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from covenant_ml.explainers.cleargbm_shap import (
    ClearGBMShapWrapper,
    _convert_decision_tree,
)
from covenant_ml.explainers.cleargbm_shap_decode import (
    _decode_rust_monotonic_constraints,
    _decode_rust_node,
)
from tests.explainers._cleargbm_shap_fixtures import (
    _make_deeper_tree,
    _NativePyGbmModelProto,
)


def _train_native_binary_model(
    n_estimators: int = 5,
    max_depth: int = 3,
    n_samples: int = 32,
    n_features: int = 3,
    random_state: int = 42,
) -> tuple[_NativePyGbmModelProto, list[str], NDArray[np.float64]]:
    """Train a small binary-classification native ClearGBM model for SHAP tests.

    Uses a linearly-separable synthetic dataset so a shallow tree ensemble
    reaches useful splits.

    Args:
        n_estimators: Number of trees in the ensemble.
        max_depth: Maximum tree depth.
        n_samples: Number of training rows.
        n_features: Number of features.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(native_model, feature_names, x_test)`` where ``x_test`` is
        a small holdout matrix suitable for calling ``explain_local``.
    """
    from cleargbm.ensemble import train_gradient_boosting

    rng = np.random.default_rng(random_state)
    x_train: NDArray[np.float64] = rng.random((n_samples, n_features), dtype=np.float64)
    # Linearly separable: label = 1 iff sum of first half > sum of second half.
    half = n_features // 2 if n_features > 1 else 1
    left_sum: NDArray[np.float64] = np.sum(x_train[:, :half], axis=1)
    right_sum: NDArray[np.float64] = np.sum(x_train[:, half:], axis=1)
    score: NDArray[np.float64] = left_sum - right_sum
    y_train: NDArray[np.int64] = (score > 0.0).astype(np.int64)

    feature_names = tuple(f"f{i}" for i in range(n_features))
    cfg: GradientBoostingConfig = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": 0.3,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": None,
        "colsample_bytree": None,
        "max_bins": 8,
        "subsample": 1.0,
        "random_state": random_state,
        "monotonic_constraints": None,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "early_stopping_rounds": None,
        "growth_strategy": "depth_wise",
        "num_leaves": None,
        "objective": "binary_log_loss",
        "scale_pos_weight": 1.0,
    }
    native_model = train_gradient_boosting(
        x_train=x_train,
        y_train=y_train,
        x_val=None,
        y_val=None,
        config=cfg,
        feature_names=feature_names,
    )
    x_test = rng.random((3, n_features), dtype=np.float64)
    return native_model, list(feature_names), x_test


class TestDecodeRustNodeErrors:
    """Coverage for the ``feature_index`` out-of-range error path in `_decode_rust_node`."""

    def test_decode_rust_node_raises_when_feature_index_out_of_range(self) -> None:
        """A Rust-shape internal node with a bogus feature_index must be rejected."""
        # Rust-shape node payload (integer feature_index that references a
        # feature past the end of the model-level feature_names tuple).
        node_json: JSONValue = {
            "node_id": 0,
            "is_leaf": False,
            "feature_index": 5,  # out of range: only 2 features declared below.
            "threshold": 0.5,
            "value": 0.0,
            "n_samples": 10,
            "left_child": 1,
            "right_child": 2,
            "nan_goes_left": True,
        }
        feature_names = ("f0", "f1")
        with pytest.raises(ValueError, match="out of range"):
            _decode_rust_node(node_json, feature_names)


class TestDecodeRustMonotonicConstraints:
    """Coverage for the list-of-labels branch of `_decode_rust_monotonic_constraints`."""

    def test_decode_none_returns_none(self) -> None:
        """A JSON null input decodes to Python ``None``."""
        assert _decode_rust_monotonic_constraints(None) is None

    def test_decode_translates_variants_to_ints(self) -> None:
        """The three known variants translate to the expected integer codes."""
        result = _decode_rust_monotonic_constraints(["Increasing", "None", "Decreasing"])
        assert result == (1, 0, -1)

    def test_decode_empty_list_returns_empty_tuple(self) -> None:
        """An empty JSON list yields an empty tuple (no constraints applied)."""
        result = _decode_rust_monotonic_constraints([])
        assert result == ()

    def test_decode_rejects_unknown_variant(self) -> None:
        """An unrecognized variant label surfaces as ``ValueError``."""
        with pytest.raises(ValueError, match="unknown monotonic constraint variant"):
            _decode_rust_monotonic_constraints(["Bogus"])


class TestClearGBMShapWrapperNativeIntegration:
    """End-to-end integration tests for ``ClearGBMShapWrapper``.

    These replace the earlier fake-model-based unit tests. The wrapper's
    constructor takes a native ``PyGbmModel`` that only ``train_gradient_boosting``
    can produce, so exercising it requires real training.
    """

    def test_wrapper_construction_from_native_model(self) -> None:
        """Wrapping a native model produces a populated SHAP format."""
        native_model, _, _ = _train_native_binary_model()
        wrapper = ClearGBMShapWrapper(native_model)
        assert len(wrapper._shap_format["trees"]) == 5
        assert wrapper._shap_format["num_outputs"] == 1
        assert wrapper._shap_format["objective"] == "binary:logistic"

    def test_wrapper_explain_local_returns_per_sample_values(self) -> None:
        """``explain_local`` returns one explanation per input row."""
        import math

        native_model, feature_names, x = _train_native_binary_model()
        wrapper = ClearGBMShapWrapper(native_model)
        result = wrapper.explain_local(x, feature_names)
        n_rows: int = int(x.shape[0])
        assert len(result) == n_rows
        for exp in result:
            assert exp["feature_names"] == feature_names
            assert len(exp["values"]) == len(feature_names)
            base_value: float = float(exp["base_value"])
            assert math.isfinite(base_value)
            for v in exp["values"]:
                v_f: float = float(v)
                assert math.isfinite(v_f)

    def test_wrapper_explain_local_raises_on_feature_mismatch(self) -> None:
        """``explain_local`` raises when the feature-name count is wrong."""
        native_model, _, x = _train_native_binary_model()
        wrapper = ClearGBMShapWrapper(native_model)
        with pytest.raises(ValueError, match="Feature count mismatch"):
            wrapper.explain_local(x, ["only_one_name"])


class TestIntegration:
    """Integration tests with more realistic models."""

    def test_shap_tree_arrays_structure(self) -> None:
        """Verify ShapTreeArrays has correct SHAP format structure."""
        tree = _make_deeper_tree()

        result = _convert_decision_tree(tree)

        # Deeper tree has 5 nodes - verify all arrays have correct length
        assert len(result["children_left"]) == 5
        assert len(result["children_right"]) == 5
        assert len(result["children_default"]) == 5
        assert len(result["features"]) == 5
        assert len(result["thresholds"]) == 5
        assert len(result["node_sample_weight"]) == 5

        # values must be 2D for SHAP
        values_shape = result["values"].shape
        assert len(values_shape) == 2
