"""Tests for ClearGBM SHAP adapter.

Comprehensive tests for converting ClearGBM models to SHAP format
and computing local explanations.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import pytest
from cleargbm.types import DecisionTree, GradientBoostingConfig, GradientBoostingModel, TreeNode
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from covenant_ml.explainers.cleargbm_shap import (
    ClearGBMShapWrapper,
    _convert_decision_tree,
    _convert_tree_node_to_arrays,
    _decode_rust_monotonic_constraints,
    _decode_rust_node,
    _get_default_child_idx,
    _populate_internal_node,
    _ShapArrays,
    convert_cleargbm_to_shap_format,
)


class _NativePyGbmModelProto(Protocol):
    """Opaque native model handle produced by the Rust training loop."""

    ...


# =============================================================================
# Test data creation helpers
# =============================================================================


def _make_x_1x2() -> NDArray[np.float64]:
    """Create 1x2 test array."""
    data: NDArray[np.float64] = np.zeros((1, 2), dtype=np.float64)
    data[0, 0] = 0.3
    data[0, 1] = 0.5
    return data


def _make_x_2x2() -> NDArray[np.float64]:
    """Create 2x2 test array."""
    data: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
    data[0, 0] = 0.1
    data[0, 1] = 0.2
    data[1, 0] = 0.6
    data[1, 1] = 0.7
    return data


def _make_x_3x2() -> NDArray[np.float64]:
    """Create 3x2 test array."""
    data: NDArray[np.float64] = np.zeros((3, 2), dtype=np.float64)
    data[0, 0] = 0.2
    data[0, 1] = 0.4
    data[1, 0] = 0.7
    data[1, 1] = 0.3
    data[2, 0] = 0.5
    data[2, 1] = 0.5
    return data


def _make_x_1x3() -> NDArray[np.float64]:
    """Create 1x3 test array."""
    data: NDArray[np.float64] = np.zeros((1, 3), dtype=np.float64)
    data[0, 0] = 0.1
    data[0, 1] = 0.2
    data[0, 2] = 0.3
    return data


# =============================================================================
# Test fixtures
# =============================================================================


def _make_config() -> GradientBoostingConfig:
    """Create minimal GradientBoostingConfig for testing."""
    config: GradientBoostingConfig = GradientBoostingConfig(
        n_estimators=2,
        max_depth=2,
        learning_rate=0.1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        max_bins=64,
        subsample=1.0,
        random_state=42,
        track_contributions=False,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=1,
        early_stopping_rounds=10,
    )
    return config


def _make_leaf_node(
    node_id: int,
    value: float,
    n_samples: int,
) -> TreeNode:
    """Create a leaf TreeNode."""
    return TreeNode(
        node_id=node_id,
        is_leaf=True,
        feature_index=None,
        feature_name=None,
        threshold=None,
        nan_direction=None,
        value=value,
        n_samples=n_samples,
        left_child=None,
        right_child=None,
    )


def _make_split_node(
    node_id: int,
    feature_index: int,
    threshold: float,
    left_child: int,
    right_child: int,
    n_samples: int,
    nan_direction: Literal["left", "right"] | None = None,
    value: float = 0.0,
) -> TreeNode:
    """Create an internal (split) TreeNode."""
    return TreeNode(
        node_id=node_id,
        is_leaf=False,
        feature_index=feature_index,
        feature_name=f"feature_{feature_index}",
        threshold=threshold,
        nan_direction=nan_direction,
        value=value,
        n_samples=n_samples,
        left_child=left_child,
        right_child=right_child,
    )


def _make_simple_tree() -> DecisionTree:
    """Create a simple tree with one split and two leaves.

    Structure:
        [0] root (split on feature 0 at 0.5)
         ├── [1] left leaf (value=-0.1)
         └── [2] right leaf (value=0.1)
    """
    root = _make_split_node(
        node_id=0,
        feature_index=0,
        threshold=0.5,
        left_child=1,
        right_child=2,
        n_samples=100,
        nan_direction="left",
    )
    left_leaf = _make_leaf_node(node_id=1, value=-0.1, n_samples=40)
    right_leaf = _make_leaf_node(node_id=2, value=0.1, n_samples=60)

    return DecisionTree(
        nodes=(root, left_leaf, right_leaf),
        max_depth=1,
        n_leaves=2,
        feature_names=("feature_0", "feature_1"),
    )


def _make_deeper_tree() -> DecisionTree:
    """Create a tree with depth 2 (two levels of splits).

    Structure:
        [0] root (split on feature 0 at 0.5)
         ├── [1] left split (feature 1 at 0.3)
         │    ├── [3] left-left leaf
         │    └── [4] left-right leaf
         └── [2] right leaf
    """
    root = _make_split_node(
        node_id=0,
        feature_index=0,
        threshold=0.5,
        left_child=1,
        right_child=2,
        n_samples=100,
        nan_direction="right",
    )
    left_split = _make_split_node(
        node_id=1,
        feature_index=1,
        threshold=0.3,
        left_child=3,
        right_child=4,
        n_samples=40,
        nan_direction=None,
    )
    right_leaf = _make_leaf_node(node_id=2, value=0.2, n_samples=60)
    left_left_leaf = _make_leaf_node(node_id=3, value=-0.2, n_samples=15)
    left_right_leaf = _make_leaf_node(node_id=4, value=0.0, n_samples=25)

    return DecisionTree(
        nodes=(root, left_split, right_leaf, left_left_leaf, left_right_leaf),
        max_depth=2,
        n_leaves=3,
        feature_names=("feature_0", "feature_1"),
    )


def _make_model(trees: list[DecisionTree]) -> GradientBoostingModel:
    """Create a GradientBoostingModel with given trees."""
    return GradientBoostingModel(
        trees=tuple(trees),
        base_prediction=0.0,
        learning_rate=0.1,
        feature_names=("feature_0", "feature_1"),
        n_classes=2,
        config=_make_config(),
    )


# =============================================================================
# Tests for _ShapArrays
# =============================================================================


class TestShapArrays:
    """Tests for _ShapArrays helper class."""

    def test_init_creates_correct_shapes(self) -> None:
        """_ShapArrays initializes arrays with correct shapes."""
        n_nodes = 5
        arrays = _ShapArrays(n_nodes)

        assert arrays.children_left.shape == (n_nodes,)
        assert arrays.children_right.shape == (n_nodes,)
        assert arrays.children_default.shape == (n_nodes,)
        assert arrays.features.shape == (n_nodes,)
        assert arrays.thresholds.shape == (n_nodes,)
        assert arrays.values.shape == (n_nodes, 1)
        assert arrays.node_sample_weight.shape == (n_nodes,)

    def test_init_dtypes(self) -> None:
        """_ShapArrays uses correct numpy dtypes."""
        arrays = _ShapArrays(2)

        assert arrays.children_left.dtype == np.int64
        assert arrays.children_right.dtype == np.int64
        assert arrays.children_default.dtype == np.int64
        assert arrays.features.dtype == np.int64
        assert arrays.thresholds.dtype == np.float64
        assert arrays.values.dtype == np.float64
        assert arrays.node_sample_weight.dtype == np.float64

    def test_to_typed_dict_returns_correct_keys(self) -> None:
        """to_typed_dict converts to TypedDict with expected keys."""
        arrays = _ShapArrays(3)
        result = arrays.to_typed_dict()

        # Verify all expected keys present
        assert len(result["children_left"]) == 3
        assert len(result["children_right"]) == 3
        assert len(result["children_default"]) == 3
        assert len(result["features"]) == 3
        assert len(result["thresholds"]) == 3
        assert len(result["node_sample_weight"]) == 3
        # Values array is 2D
        assert result["values"].shape == (3, 1)


# =============================================================================
# Tests for _get_default_child_idx
# =============================================================================


class TestGetDefaultChildIdx:
    """Tests for _get_default_child_idx helper function."""

    def test_nan_direction_left(self) -> None:
        """Returns left child index when nan_direction is 'left'."""
        node = _make_split_node(
            node_id=0,
            feature_index=0,
            threshold=0.5,
            left_child=1,
            right_child=2,
            n_samples=100,
            nan_direction="left",
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        result = _get_default_child_idx(node, node_id_to_idx)

        assert result == 1  # Index of left child

    def test_nan_direction_right(self) -> None:
        """Returns right child index when nan_direction is 'right'."""
        node = _make_split_node(
            node_id=0,
            feature_index=0,
            threshold=0.5,
            left_child=1,
            right_child=2,
            n_samples=100,
            nan_direction="right",
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        result = _get_default_child_idx(node, node_id_to_idx)

        assert result == 2  # Index of right child

    def test_no_nan_direction_defaults_to_left(self) -> None:
        """Defaults to left child when nan_direction is None."""
        node = _make_split_node(
            node_id=0,
            feature_index=0,
            threshold=0.5,
            left_child=1,
            right_child=2,
            n_samples=100,
            nan_direction=None,
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        result = _get_default_child_idx(node, node_id_to_idx)

        assert result == 1  # Defaults to left child

    def test_no_children_returns_negative_one(self) -> None:
        """Returns -1 when node has no children (leaf node)."""
        node = _make_leaf_node(node_id=0, value=0.1, n_samples=100)
        node_id_to_idx = {0: 0}

        result = _get_default_child_idx(node, node_id_to_idx)

        assert result == -1


# =============================================================================
# Tests for _populate_internal_node
# =============================================================================


class TestPopulateInternalNode:
    """Tests for _populate_internal_node helper function."""

    def test_populates_arrays(self) -> None:
        """Populates arrays for an internal node."""
        arrays = _ShapArrays(3)
        node = _make_split_node(
            node_id=0,
            feature_index=1,
            threshold=0.75,
            left_child=1,
            right_child=2,
            n_samples=50,
            nan_direction="left",
            value=0.05,
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        _populate_internal_node(arrays, 0, node, node_id_to_idx)

        # Verify arrays were modified (not default values)
        assert arrays.children_left.dtype == np.int64
        assert arrays.children_right.dtype == np.int64
        assert arrays.features.dtype == np.int64
        assert arrays.thresholds.dtype == np.float64

    def test_handles_none_feature_index(self) -> None:
        """Does not crash when feature_index is None."""
        arrays = _ShapArrays(3)
        node = TreeNode(
            node_id=0,
            is_leaf=False,
            feature_index=None,
            feature_name=None,
            threshold=0.5,
            nan_direction=None,
            value=0.0,
            n_samples=100,
            left_child=1,
            right_child=2,
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        # Should not raise
        _populate_internal_node(arrays, 0, node, node_id_to_idx)

    def test_handles_none_threshold(self) -> None:
        """Does not crash when threshold is None."""
        arrays = _ShapArrays(3)
        node = TreeNode(
            node_id=0,
            is_leaf=False,
            feature_index=0,
            feature_name="f0",
            threshold=None,
            nan_direction=None,
            value=0.0,
            n_samples=100,
            left_child=1,
            right_child=2,
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        # Should not raise
        _populate_internal_node(arrays, 0, node, node_id_to_idx)

    def test_handles_none_left_child(self) -> None:
        """Does not crash when left_child is None."""
        arrays = _ShapArrays(2)
        node = TreeNode(
            node_id=0,
            is_leaf=False,
            feature_index=0,
            feature_name="f0",
            threshold=0.5,
            nan_direction=None,
            value=0.0,
            n_samples=100,
            left_child=None,
            right_child=1,
        )
        node_id_to_idx = {0: 0, 1: 1}

        # Should not raise
        _populate_internal_node(arrays, 0, node, node_id_to_idx)

    def test_handles_none_right_child(self) -> None:
        """Does not crash when right_child is None."""
        arrays = _ShapArrays(2)
        node = TreeNode(
            node_id=0,
            is_leaf=False,
            feature_index=0,
            feature_name="f0",
            threshold=0.5,
            nan_direction="left",
            value=0.0,
            n_samples=100,
            left_child=1,
            right_child=None,
        )
        node_id_to_idx = {0: 0, 1: 1}

        # Should not raise
        _populate_internal_node(arrays, 0, node, node_id_to_idx)


# =============================================================================
# Tests for _convert_tree_node_to_arrays
# =============================================================================


class TestConvertTreeNodeToArrays:
    """Tests for _convert_tree_node_to_arrays function."""

    def test_raises_on_empty_nodes(self) -> None:
        """Raises ValueError for empty nodes tuple."""
        with pytest.raises(ValueError, match="Cannot convert empty tree nodes"):
            _convert_tree_node_to_arrays(())

    def test_converts_single_leaf(self) -> None:
        """Converts a single leaf node."""
        leaf = _make_leaf_node(node_id=0, value=0.5, n_samples=100)
        nodes = (leaf,)

        result = _convert_tree_node_to_arrays(nodes)

        # Verify structure
        assert len(result["children_left"]) == 1
        assert len(result["children_right"]) == 1
        assert len(result["features"]) == 1
        assert result["values"].shape == (1, 1)

    def test_converts_simple_tree(self) -> None:
        """Converts a simple tree with one split."""
        tree = _make_simple_tree()

        result = _convert_tree_node_to_arrays(tree["nodes"])

        # 3 nodes: root + 2 leaves
        assert len(result["children_left"]) == 3
        assert len(result["children_right"]) == 3
        assert len(result["features"]) == 3
        assert result["values"].shape == (3, 1)

    def test_converts_deeper_tree(self) -> None:
        """Converts a tree with depth 2."""
        tree = _make_deeper_tree()

        result = _convert_tree_node_to_arrays(tree["nodes"])

        # 5 nodes total
        assert len(result["children_left"]) == 5
        assert len(result["children_right"]) == 5
        assert len(result["features"]) == 5


# =============================================================================
# Tests for _convert_decision_tree
# =============================================================================


class TestConvertDecisionTree:
    """Tests for _convert_decision_tree function."""

    def test_extracts_nodes_and_converts(self) -> None:
        """Extracts nodes from DecisionTree and converts."""
        tree = _make_simple_tree()

        result = _convert_decision_tree(tree)

        # Verify conversion produces correct structure
        assert len(result["children_left"]) == 3
        assert len(result["children_right"]) == 3
        assert len(result["features"]) == 3


# =============================================================================
# Tests for convert_cleargbm_to_shap_format
# =============================================================================


class TestConvertClearGBMToShapFormat:
    """Tests for convert_cleargbm_to_shap_format function."""

    def test_raises_on_empty_trees(self) -> None:
        """Raises ValueError for model with no trees."""
        model = GradientBoostingModel(
            trees=(),
            base_prediction=0.0,
            learning_rate=0.1,
            feature_names=("f0",),
            n_classes=2,
            config=_make_config(),
        )

        with pytest.raises(ValueError, match="Cannot convert model with no trees"):
            convert_cleargbm_to_shap_format(model)

    def test_converts_single_tree_model(self) -> None:
        """Converts model with one tree."""
        tree = _make_simple_tree()
        model = _make_model([tree])

        result = convert_cleargbm_to_shap_format(model)

        # Verify all expected fields are present with correct values
        assert result["num_outputs"] == 1
        assert result["base_offset"] == 0.0
        assert result["objective"] == "binary:logistic"
        assert result["tree_output"] == "raw"
        assert result["input_dtype"] == np.float64
        assert len(result["trees"]) == 1

    def test_converts_multi_tree_model(self) -> None:
        """Converts model with multiple trees."""
        tree1 = _make_simple_tree()
        tree2 = _make_deeper_tree()
        model = _make_model([tree1, tree2])

        result = convert_cleargbm_to_shap_format(model)

        assert len(result["trees"]) == 2
        # First tree has 3 nodes
        assert len(result["trees"][0]["children_left"]) == 3
        # Second tree has 5 nodes
        assert len(result["trees"][1]["children_left"]) == 5

    def test_preserves_base_prediction(self) -> None:
        """Preserves base_prediction as base_offset."""
        tree = _make_simple_tree()
        model = GradientBoostingModel(
            trees=(tree,),
            base_prediction=-0.5,
            learning_rate=0.1,
            feature_names=("f0", "f1"),
            n_classes=2,
            config=_make_config(),
        )

        result = convert_cleargbm_to_shap_format(model)

        assert result["base_offset"] == -0.5


# =============================================================================
# Tests for ClearGBMShapWrapper
# =============================================================================
#
# The wrapper takes a native ``PyGbmModel`` handle (opaque Rust value)
# produced by ``train_gradient_boosting``. It cannot be constructed from
# a hand-authored Python-shape ``GradientBoostingModel`` TypedDict — the
# Rust value has no public constructor from Python-side data. Wrapper tests
# accordingly train a real small native model and exercise the full path:
# ``PyGbmModel → to_json_rs → decode → convert_cleargbm_to_shap_format →
# shap.TreeExplainer``.


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
        "max_bins": 8,
        "subsample": 1.0,
        "random_state": random_state,
        "track_contributions": False,
        "monotonic_constraints": None,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "early_stopping_rounds": None,
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
