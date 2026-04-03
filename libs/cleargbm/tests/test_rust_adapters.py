"""Tests for Rust backend adapters.

Verifies that Rust-backed functions produce identical results to the
Python defaults, and that use_rust_backend()/use_python_backend() correctly
wire/restore all hooks.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cleargbm import (
    _hooks_binning,
    _hooks_ensemble,
    _hooks_histogram,
    _hooks_loss,
    _hooks_prediction,
    _hooks_sigmoid,
)
from cleargbm._rust_adapters import (
    _load_native_functions,
    _rust_binary_log_loss,
    _rust_binary_log_loss_gradients,
    _rust_binary_log_loss_hessians,
    _rust_binary_log_loss_initial_prediction,
    _rust_build_histogram,
    _rust_precompute_feature_bins,
    _rust_predict_proba,
    _rust_predict_raw,
    _rust_predict_tree,
    _rust_sigmoid,
    _rust_sigmoid_array,
    _rust_subtract_histogram,
    use_python_backend,
    use_rust_backend,
)
from cleargbm.buffers import HistogramBuffer
from cleargbm.types import DecisionTree, TreeNode

# Load native functions so adapter functions can reference Rust bindings.
# Must run after imports but before any test calls adapter functions.
_load_native_functions()

# =============================================================================
# Test fixtures
# =============================================================================


def _make_simple_tree() -> DecisionTree:
    """Build a simple 3-node decision tree for testing.

    Structure:
        node 0: split on feature 0 at threshold 0.5
        node 1: left leaf, value -0.3 (5 samples)
        node 2: right leaf, value 0.7 (5 samples)

    Returns:
        DecisionTree TypedDict.
    """
    root = TreeNode(
        node_id=0,
        is_leaf=False,
        feature_index=0,
        feature_name="feat_0",
        threshold=0.5,
        nan_direction="left",
        value=0.0,
        n_samples=10,
        left_child=1,
        right_child=2,
    )
    left = TreeNode(
        node_id=1,
        is_leaf=True,
        feature_index=None,
        feature_name=None,
        threshold=None,
        nan_direction=None,
        value=-0.3,
        n_samples=5,
        left_child=None,
        right_child=None,
    )
    right = TreeNode(
        node_id=2,
        is_leaf=True,
        feature_index=None,
        feature_name=None,
        threshold=None,
        nan_direction=None,
        value=0.7,
        n_samples=5,
        left_child=None,
        right_child=None,
    )
    return DecisionTree(
        nodes=(root, left, right),
        max_depth=1,
        n_leaves=2,
        feature_names=("feat_0",),
    )


def _make_histogram_inputs() -> tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], int
]:
    """Create sample inputs for histogram building.

    Returns:
        Tuple of (sample_indices, gradients, hessians, sample_bins, n_bins).
    """
    sample_indices: NDArray[np.int64] = np.array((0, 1, 2, 3), dtype=np.int64)
    gradients: NDArray[np.float64] = np.array((0.1, -0.2, 0.3, -0.1), dtype=np.float64)
    hessians: NDArray[np.float64] = np.array((0.5, 0.5, 0.5, 0.5), dtype=np.float64)
    sample_bins: NDArray[np.int64] = np.array((0, 1, 0, 1), dtype=np.int64)
    n_bins = 3  # 2 regular + 1 NaN bin
    return sample_indices, gradients, hessians, sample_bins, n_bins


# =============================================================================
# Histogram tests
# =============================================================================


class TestRustBuildHistogram:
    """Tests for _rust_build_histogram adapter."""

    def test_matches_python(self) -> None:
        """Rust histogram matches Python histogram for identical inputs."""
        indices, grads, hess, bins, n_bins = _make_histogram_inputs()

        python_hist = _hooks_histogram._default_build_histogram(indices, grads, hess, bins, n_bins)
        rust_hist = _rust_build_histogram(indices, grads, hess, bins, n_bins)

        assert rust_hist.n_bins == python_hist.n_bins
        for i in range(n_bins):
            assert abs(rust_hist.get_gradient_sum(i) - python_hist.get_gradient_sum(i)) < 1e-10
            assert abs(rust_hist.get_hessian_sum(i) - python_hist.get_hessian_sum(i)) < 1e-10
            assert rust_hist.get_count(i) == python_hist.get_count(i)

    def test_returns_histogram_buffer(self) -> None:
        """Rust adapter returns a proper HistogramBuffer instance."""
        indices, grads, hess, bins, n_bins = _make_histogram_inputs()
        result = _rust_build_histogram(indices, grads, hess, bins, n_bins)
        assert result.n_bins == n_bins
        # Bin 0: samples 0,2 → grads 0.1+0.3=0.4
        assert abs(result.get_gradient_sum(0) - 0.4) < 1e-10
        # Bin 1: samples 1,3 → grads -0.2+(-0.1)=-0.3
        assert abs(result.get_gradient_sum(1) - (-0.3)) < 1e-10
        # Bin 2 (NaN bin): no samples → 0
        assert result.get_count(2) == 0


class TestRustSubtractHistogram:
    """Tests for _rust_subtract_histogram adapter."""

    def test_matches_python(self) -> None:
        """Rust subtraction matches Python subtraction for identical inputs."""
        parent = HistogramBuffer.from_tuples(
            gradient_sums=(1.0, 2.0, 0.5),
            hessian_sums=(3.0, 4.0, 1.0),
            counts=(10, 20, 5),
        )
        child = HistogramBuffer.from_tuples(
            gradient_sums=(0.3, 0.8, 0.2),
            hessian_sums=(1.0, 1.5, 0.3),
            counts=(4, 8, 2),
        )

        python_sib = _hooks_histogram._default_subtract_histogram(parent, child)
        rust_sib = _rust_subtract_histogram(parent, child)

        assert rust_sib.n_bins == python_sib.n_bins
        for i in range(rust_sib.n_bins):
            assert abs(rust_sib.get_gradient_sum(i) - python_sib.get_gradient_sum(i)) < 1e-10
            assert abs(rust_sib.get_hessian_sum(i) - python_sib.get_hessian_sum(i)) < 1e-10
            assert rust_sib.get_count(i) == python_sib.get_count(i)

    def test_sibling_values_correct(self) -> None:
        """Subtraction produces correct sibling values."""
        parent = HistogramBuffer.from_tuples(
            gradient_sums=(1.0, 2.0),
            hessian_sums=(3.0, 4.0),
            counts=(10, 20),
        )
        child = HistogramBuffer.from_tuples(
            gradient_sums=(0.4, 0.6),
            hessian_sums=(1.0, 1.5),
            counts=(4, 8),
        )
        sib = _rust_subtract_histogram(parent, child)
        assert abs(sib.get_gradient_sum(0) - 0.6) < 1e-10
        assert abs(sib.get_gradient_sum(1) - 1.4) < 1e-10
        assert sib.get_count(0) == 6
        assert sib.get_count(1) == 12


# =============================================================================
# Prediction tests
# =============================================================================


class TestRustPredictTree:
    """Tests for _rust_predict_tree adapter."""

    def test_matches_python(self) -> None:
        """Rust prediction matches Python prediction for identical tree and data."""
        tree = _make_simple_tree()
        x: NDArray[np.float64] = np.array(
            ((0.2,), (0.8,), (0.5,), (0.1,), (0.9,)),
            dtype=np.float64,
        )

        python_preds = _hooks_prediction._default_predict_tree(tree, x)
        rust_preds = _rust_predict_tree(tree, x)

        np.testing.assert_allclose(rust_preds, python_preds, atol=1e-10)

    def test_correct_leaf_values(self) -> None:
        """Predictions match expected leaf values from the tree."""
        tree = _make_simple_tree()
        x: NDArray[np.float64] = np.array(
            ((0.2,), (0.8,)),
            dtype=np.float64,
        )
        preds = _rust_predict_tree(tree, x)
        # 0.2 <= 0.5 → left leaf → -0.3
        pred_0: float = preds.item(0)
        assert abs(pred_0 - (-0.3)) < 1e-10
        # 0.8 > 0.5 → right leaf → 0.7
        pred_1: float = preds.item(1)
        assert abs(pred_1 - 0.7) < 1e-10

    def test_nan_goes_left(self) -> None:
        """NaN values follow nan_direction in the tree."""
        tree = _make_simple_tree()
        # Root has nan_direction="left" → NaN goes to left leaf (-0.3)
        x: NDArray[np.float64] = np.array(((float("nan"),),), dtype=np.float64)

        python_pred = _hooks_prediction._default_predict_tree(tree, x)
        rust_pred = _rust_predict_tree(tree, x)

        np.testing.assert_allclose(rust_pred, python_pred, atol=1e-10)
        rust_val: float = rust_pred.item(0)
        assert abs(rust_val - (-0.3)) < 1e-10

    def test_nan_goes_right(self) -> None:
        """NaN follows right direction when nan_direction is right."""
        nan_right: Literal["right"] = "right"
        root = TreeNode(
            node_id=0,
            is_leaf=False,
            feature_index=0,
            feature_name="f0",
            threshold=0.5,
            nan_direction=nan_right,
            value=0.0,
            n_samples=10,
            left_child=1,
            right_child=2,
        )
        left = TreeNode(
            node_id=1,
            is_leaf=True,
            feature_index=None,
            feature_name=None,
            threshold=None,
            nan_direction=None,
            value=-1.0,
            n_samples=5,
            left_child=None,
            right_child=None,
        )
        right = TreeNode(
            node_id=2,
            is_leaf=True,
            feature_index=None,
            feature_name=None,
            threshold=None,
            nan_direction=None,
            value=1.0,
            n_samples=5,
            left_child=None,
            right_child=None,
        )
        tree = DecisionTree(
            nodes=(root, left, right),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )
        x: NDArray[np.float64] = np.array(((float("nan"),),), dtype=np.float64)

        python_pred = _hooks_prediction._default_predict_tree(tree, x)
        rust_pred = _rust_predict_tree(tree, x)

        np.testing.assert_allclose(rust_pred, python_pred, atol=1e-10)
        rust_val: float = rust_pred.item(0)
        assert abs(rust_val - 1.0) < 1e-10


# =============================================================================
# Sigmoid tests
# =============================================================================


class TestRustSigmoid:
    """Tests for _rust_sigmoid adapter."""

    def test_matches_python_at_zero(self) -> None:
        """sigmoid(0) = 0.5."""
        assert abs(_rust_sigmoid(0.0) - 0.5) < 1e-10
        assert abs(_rust_sigmoid(0.0) - _hooks_sigmoid._default_sigmoid(0.0)) < 1e-10

    def test_matches_python_positive(self) -> None:
        """Positive inputs produce probabilities > 0.5."""
        rust_val = _rust_sigmoid(2.0)
        python_val = _hooks_sigmoid._default_sigmoid(2.0)
        assert abs(rust_val - python_val) < 1e-10
        assert rust_val > 0.5

    def test_matches_python_negative(self) -> None:
        """Negative inputs produce probabilities < 0.5."""
        rust_val = _rust_sigmoid(-2.0)
        python_val = _hooks_sigmoid._default_sigmoid(-2.0)
        assert abs(rust_val - python_val) < 1e-10
        assert rust_val < 0.5

    def test_extreme_values(self) -> None:
        """Extreme values match Python defaults and stay in [0, 1]."""
        assert _rust_sigmoid(500.0) >= 0.0
        assert _rust_sigmoid(500.0) <= 1.0
        assert _rust_sigmoid(-500.0) >= 0.0
        assert _rust_sigmoid(-500.0) <= 1.0
        assert abs(_rust_sigmoid(500.0) - _hooks_sigmoid._default_sigmoid(500.0)) < 1e-10
        assert abs(_rust_sigmoid(-500.0) - _hooks_sigmoid._default_sigmoid(-500.0)) < 1e-10


class TestRustSigmoidArray:
    """Tests for _rust_sigmoid_array adapter."""

    def test_matches_python(self) -> None:
        """Array sigmoid matches Python element-wise."""
        x: NDArray[np.float64] = np.array((-2.0, -1.0, 0.0, 1.0, 2.0), dtype=np.float64)
        python_result = _hooks_sigmoid._default_sigmoid_array(x)
        rust_result = _rust_sigmoid_array(x)
        np.testing.assert_allclose(rust_result, python_result, atol=1e-10)

    def test_single_element(self) -> None:
        """Single-element array works correctly."""
        x: NDArray[np.float64] = np.array((0.0,), dtype=np.float64)
        result = _rust_sigmoid_array(x)
        val: float = result.item(0)
        assert abs(val - 0.5) < 1e-10

    def test_extreme_array(self) -> None:
        """Extreme values in array match Python defaults."""
        x: NDArray[np.float64] = np.array((-1000.0, 1000.0), dtype=np.float64)
        rust_result = _rust_sigmoid_array(x)
        python_result = _hooks_sigmoid._default_sigmoid_array(x)
        np.testing.assert_allclose(rust_result, python_result, atol=1e-10)


# =============================================================================
# Tree conversion tests
# =============================================================================


class TestDecisionTreeToPyTree:
    """Tests for _decision_tree_to_py_tree conversion."""

    def test_simple_tree_roundtrip(self) -> None:
        """Converting a tree and predicting produces correct values."""
        tree = _make_simple_tree()
        # _rust_predict_tree calls _decision_tree_to_py_tree internally,
        # so correct predictions prove the conversion is correct
        x: NDArray[np.float64] = np.array(((0.2,), (0.8,)), dtype=np.float64)
        preds: NDArray[np.float64] = _rust_predict_tree(tree, x)
        pred_0: float = preds.item(0)
        pred_1: float = preds.item(1)
        assert abs(pred_0 - (-0.3)) < 1e-10
        assert abs(pred_1 - 0.7) < 1e-10

    def test_single_leaf_tree(self) -> None:
        """A tree with only a root leaf node converts correctly."""
        leaf_only = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=0.42,
                    n_samples=100,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=0,
            n_leaves=1,
            feature_names=("f0",),
        )
        x: NDArray[np.float64] = np.array(((1.0,), (2.0,), (3.0,)), dtype=np.float64)
        preds = _rust_predict_tree(leaf_only, x)
        for i in range(3):
            val: float = preds.item(i)
            assert abs(val - 0.42) < 1e-10


# =============================================================================
# Backend wiring tests
# =============================================================================


class TestRustBinaryLogLoss:
    """Tests for _rust_binary_log_loss adapter."""

    def test_matches_python(self) -> None:
        """Rust loss matches Python loss for identical inputs."""
        y_true: NDArray[np.int64] = np.array((0, 1, 1, 0), dtype=np.int64)
        y_pred: NDArray[np.float64] = np.array((0.1, 0.9, 0.8, 0.2), dtype=np.float64)

        python_loss = _hooks_loss._default_binary_log_loss(y_true, y_pred)
        rust_loss = _rust_binary_log_loss(y_true, y_pred)

        assert abs(rust_loss - python_loss) < 1e-10

    def test_perfect_predictions(self) -> None:
        """Loss is near-zero for near-perfect predictions."""
        y_true: NDArray[np.int64] = np.array((0, 1), dtype=np.int64)
        y_pred: NDArray[np.float64] = np.array((0.001, 0.999), dtype=np.float64)

        loss = _rust_binary_log_loss(y_true, y_pred)
        assert loss < 0.01


class TestRustBinaryLogLossGradients:
    """Tests for _rust_binary_log_loss_gradients adapter."""

    def test_matches_python(self) -> None:
        """Rust gradients match Python gradients."""
        y_true: NDArray[np.int64] = np.array((0, 1, 1, 0), dtype=np.int64)
        y_pred: NDArray[np.float64] = np.array((0.3, 0.7, 0.8, 0.2), dtype=np.float64)

        python_grads = _hooks_loss._default_binary_log_loss_gradients(y_true, y_pred)
        rust_grads = _rust_binary_log_loss_gradients(y_true, y_pred)

        np.testing.assert_allclose(rust_grads, python_grads, atol=1e-10)


class TestRustBinaryLogLossHessians:
    """Tests for _rust_binary_log_loss_hessians adapter."""

    def test_matches_python(self) -> None:
        """Rust hessians match Python hessians."""
        y_true: NDArray[np.int64] = np.array((0, 1, 1, 0), dtype=np.int64)
        y_pred: NDArray[np.float64] = np.array((0.3, 0.7, 0.8, 0.2), dtype=np.float64)

        python_hess = _hooks_loss._default_binary_log_loss_hessians(y_true, y_pred)
        rust_hess = _rust_binary_log_loss_hessians(y_true, y_pred)

        np.testing.assert_allclose(rust_hess, python_hess, atol=1e-10)

    def test_always_positive(self) -> None:
        """Hessians are always positive (p * (1-p) > 0)."""
        y_true: NDArray[np.int64] = np.array((0, 1), dtype=np.int64)
        y_pred: NDArray[np.float64] = np.array((0.5, 0.5), dtype=np.float64)

        hess = _rust_binary_log_loss_hessians(y_true, y_pred)
        for i in range(2):
            val: float = hess.item(i)
            assert val > 0.0


class TestRustBinaryLogLossInitialPrediction:
    """Tests for _rust_binary_log_loss_initial_prediction adapter."""

    def test_matches_python(self) -> None:
        """Rust initial prediction matches Python."""
        y_true: NDArray[np.int64] = np.array((0, 0, 1, 1, 1), dtype=np.int64)

        python_pred = _hooks_loss._default_binary_log_loss_initial_prediction(y_true)
        rust_pred = _rust_binary_log_loss_initial_prediction(y_true)

        assert abs(rust_pred - python_pred) < 1e-10

    def test_balanced_labels_near_zero(self) -> None:
        """Balanced labels produce initial prediction near zero."""
        y_true: NDArray[np.int64] = np.array((0, 1, 0, 1), dtype=np.int64)

        pred = _rust_binary_log_loss_initial_prediction(y_true)
        assert abs(pred) < 1e-10


class TestRustPrecomputeFeatureBins:
    """Tests for _rust_precompute_feature_bins adapter."""

    def test_matches_python(self) -> None:
        """Rust binning matches Python binning."""
        x: NDArray[np.float64] = np.array(
            ((1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)),
            dtype=np.float64,
        )
        max_bins = 4

        python_fb = _hooks_binning._default_precompute_feature_bins(x, max_bins)
        rust_fb = _rust_precompute_feature_bins(x, max_bins)

        assert len(rust_fb.bin_edges) == len(python_fb.bin_edges)
        np.testing.assert_array_equal(rust_fb.sample_bins, python_fb.sample_bins)

    def test_returns_feature_bins(self) -> None:
        """Adapter returns a proper FeatureBins NamedTuple."""
        x: NDArray[np.float64] = np.array(
            ((1.0,), (2.0,), (3.0,)),
            dtype=np.float64,
        )
        fb = _rust_precompute_feature_bins(x, 3)
        assert len(fb.bin_edges) == 1
        assert fb.sample_bins.shape == (3, 1)


class TestRustPredictRaw:
    """Tests for _rust_predict_raw adapter."""

    def test_matches_python(self) -> None:
        """Rust ensemble prediction matches Python."""
        tree = _make_simple_tree()
        x: NDArray[np.float64] = np.array(
            ((0.2,), (0.8,), (0.5,)),
            dtype=np.float64,
        )

        python_preds = _hooks_ensemble._default_predict_raw((tree,), x, 0.0, 1.0)
        rust_preds = _rust_predict_raw((tree,), x, 0.0, 1.0)

        np.testing.assert_allclose(rust_preds, python_preds, atol=1e-10)

    def test_base_prediction_added(self) -> None:
        """Base prediction is added to tree predictions."""
        tree = _make_simple_tree()
        x: NDArray[np.float64] = np.array(((0.2,),), dtype=np.float64)

        # With base_prediction=0.0 → just tree contribution
        preds_no_base = _rust_predict_raw((tree,), x, 0.0, 1.0)
        # With base_prediction=1.0 → tree contribution + 1.0
        preds_with_base = _rust_predict_raw((tree,), x, 1.0, 1.0)

        diff: float = preds_with_base.item(0) - preds_no_base.item(0)
        assert abs(diff - 1.0) < 1e-10


class TestRustPredictProba:
    """Tests for _rust_predict_proba adapter."""

    def test_matches_python(self) -> None:
        """Rust proba matches Python proba."""
        raw: NDArray[np.float64] = np.array((-2.0, 0.0, 2.0), dtype=np.float64)

        python_proba = _hooks_ensemble._default_predict_proba(raw)
        rust_proba = _rust_predict_proba(raw)

        assert len(rust_proba) == len(python_proba)
        for i in range(len(rust_proba)):
            assert abs(rust_proba[i][0] - python_proba[i][0]) < 1e-10
            assert abs(rust_proba[i][1] - python_proba[i][1]) < 1e-10

    def test_probabilities_sum_to_one(self) -> None:
        """Each (p0, p1) pair sums to 1.0."""
        raw: NDArray[np.float64] = np.array((-1.0, 0.0, 1.0), dtype=np.float64)

        proba = _rust_predict_proba(raw)
        for p0, p1 in proba:
            assert abs(p0 + p1 - 1.0) < 1e-10

    def test_zero_raw_gives_half(self) -> None:
        """Raw prediction of 0 gives 50/50 probabilities."""
        raw: NDArray[np.float64] = np.array((0.0,), dtype=np.float64)

        proba = _rust_predict_proba(raw)
        assert abs(proba[0][0] - 0.5) < 1e-10
        assert abs(proba[0][1] - 0.5) < 1e-10


class TestUseRustBackend:
    """Tests for use_rust_backend() and use_python_backend()."""

    def test_sets_all_hooks(self) -> None:
        """use_rust_backend() replaces all 12 hooks."""
        # Save originals
        orig_build = _hooks_histogram._build_histogram_backend
        orig_subtract = _hooks_histogram._subtract_histogram_backend
        orig_predict = _hooks_prediction._predict_tree_backend
        orig_sigmoid = _hooks_sigmoid._sigmoid_backend
        orig_sigmoid_arr = _hooks_sigmoid._sigmoid_array_backend
        orig_loss = _hooks_loss._binary_log_loss_backend
        orig_grads = _hooks_loss._binary_log_loss_gradients_backend
        orig_hess = _hooks_loss._binary_log_loss_hessians_backend
        orig_init = _hooks_loss._binary_log_loss_initial_prediction_backend
        orig_bins = _hooks_binning._precompute_feature_bins_backend
        orig_raw = _hooks_ensemble._predict_raw_backend
        orig_proba = _hooks_ensemble._predict_proba_backend

        use_rust_backend()

        assert _hooks_histogram._build_histogram_backend is _rust_build_histogram
        assert _hooks_histogram._subtract_histogram_backend is _rust_subtract_histogram
        assert _hooks_prediction._predict_tree_backend is _rust_predict_tree
        assert _hooks_sigmoid._sigmoid_backend is _rust_sigmoid
        assert _hooks_sigmoid._sigmoid_array_backend is _rust_sigmoid_array
        assert _hooks_loss._binary_log_loss_backend is _rust_binary_log_loss
        assert _hooks_loss._binary_log_loss_gradients_backend is _rust_binary_log_loss_gradients
        assert _hooks_loss._binary_log_loss_hessians_backend is _rust_binary_log_loss_hessians
        assert (
            _hooks_loss._binary_log_loss_initial_prediction_backend
            is _rust_binary_log_loss_initial_prediction
        )
        assert _hooks_binning._precompute_feature_bins_backend is _rust_precompute_feature_bins
        assert _hooks_ensemble._predict_raw_backend is _rust_predict_raw
        assert _hooks_ensemble._predict_proba_backend is _rust_predict_proba

        # Restore
        _hooks_histogram._build_histogram_backend = orig_build
        _hooks_histogram._subtract_histogram_backend = orig_subtract
        _hooks_prediction._predict_tree_backend = orig_predict
        _hooks_sigmoid._sigmoid_backend = orig_sigmoid
        _hooks_sigmoid._sigmoid_array_backend = orig_sigmoid_arr
        _hooks_loss._binary_log_loss_backend = orig_loss
        _hooks_loss._binary_log_loss_gradients_backend = orig_grads
        _hooks_loss._binary_log_loss_hessians_backend = orig_hess
        _hooks_loss._binary_log_loss_initial_prediction_backend = orig_init
        _hooks_binning._precompute_feature_bins_backend = orig_bins
        _hooks_ensemble._predict_raw_backend = orig_raw
        _hooks_ensemble._predict_proba_backend = orig_proba

    def test_restores_python_defaults(self) -> None:
        """use_python_backend() restores all 12 hooks to Python defaults."""
        use_rust_backend()
        use_python_backend()

        assert (
            _hooks_histogram._build_histogram_backend is _hooks_histogram._default_build_histogram
        )
        assert (
            _hooks_histogram._subtract_histogram_backend
            is _hooks_histogram._default_subtract_histogram
        )
        assert _hooks_prediction._predict_tree_backend is _hooks_prediction._default_predict_tree
        assert _hooks_sigmoid._sigmoid_backend is _hooks_sigmoid._default_sigmoid
        assert _hooks_sigmoid._sigmoid_array_backend is _hooks_sigmoid._default_sigmoid_array
        assert _hooks_loss._binary_log_loss_backend is _hooks_loss._default_binary_log_loss
        assert (
            _hooks_loss._binary_log_loss_gradients_backend
            is _hooks_loss._default_binary_log_loss_gradients
        )
        assert (
            _hooks_loss._binary_log_loss_hessians_backend
            is _hooks_loss._default_binary_log_loss_hessians
        )
        assert (
            _hooks_loss._binary_log_loss_initial_prediction_backend
            is _hooks_loss._default_binary_log_loss_initial_prediction
        )
        assert (
            _hooks_binning._precompute_feature_bins_backend
            is _hooks_binning._default_precompute_feature_bins
        )
        assert _hooks_ensemble._predict_raw_backend is _hooks_ensemble._default_predict_raw
        assert _hooks_ensemble._predict_proba_backend is _hooks_ensemble._default_predict_proba

    def test_hooks_work_after_wiring(self) -> None:
        """Public API functions work correctly after Rust wiring."""
        use_rust_backend()
        result = _hooks_sigmoid.sigmoid(0.0)
        assert abs(result - 0.5) < 1e-10
        use_python_backend()
