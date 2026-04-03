"""Tests for cleargbm hook modules (_hooks_infra, _hooks_* sub-modules, _test_hooks re-export)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm._hooks_infra import (
    RandomStateProtocol,
    _PythonRandomStateWrapper,
    create_float_buffer,
    create_histogram_buffer,
    create_int_buffer,
    get_random_state,
)
from cleargbm._hooks_prediction import predict_tree
from cleargbm._hooks_sigmoid import sigmoid, sigmoid_array
from cleargbm.types import DecisionTree, TreeNode


class TestReExportLayer:
    """Tests verifying _test_hooks re-export layer exposes all names."""

    def test_re_export_protocols_are_same_objects(self) -> None:
        """_test_hooks re-exports are identical objects to sub-module originals."""
        from cleargbm import (
            _hooks_guard,
            _hooks_histogram,
            _hooks_infra,
            _hooks_prediction,
            _hooks_sigmoid,
            _test_hooks,
        )

        # Histogram protocols
        assert _test_hooks.BuildHistogramBackend is _hooks_histogram.BuildHistogramBackend

        # Sigmoid protocols
        assert _test_hooks.SigmoidBackend is _hooks_sigmoid.SigmoidBackend

        # Prediction protocols
        assert _test_hooks.PredictTreeBackend is _hooks_prediction.PredictTreeBackend

        # Guard protocols
        assert _test_hooks.FindMonorepoRootProto is _hooks_guard.FindMonorepoRootProto
        assert _test_hooks.RunForProjectProto is _hooks_guard.RunForProjectProto

        # Infra accessors
        assert _test_hooks.get_random_state is _hooks_infra.get_random_state
        assert _test_hooks.create_float_buffer is _hooks_infra.create_float_buffer

        # Sigmoid + prediction accessors
        assert _test_hooks.sigmoid is _hooks_sigmoid.sigmoid
        assert _test_hooks.predict_tree is _hooks_prediction.predict_tree

    def test_hooks_compute_re_export_layer(self) -> None:
        """_hooks_compute re-export layer exposes same objects as sub-modules."""
        from cleargbm import (
            _hooks_binning,
            _hooks_compute,
            _hooks_ensemble,
            _hooks_histogram,
            _hooks_loss,
            _hooks_prediction,
            _hooks_sigmoid,
        )

        assert _hooks_compute.BuildHistogramBackend is _hooks_histogram.BuildHistogramBackend
        assert _hooks_compute.SubtractHistogramBackend is _hooks_histogram.SubtractHistogramBackend
        assert _hooks_compute.PredictTreeBackend is _hooks_prediction.PredictTreeBackend
        assert _hooks_compute.SigmoidBackend is _hooks_sigmoid.SigmoidBackend
        assert _hooks_compute.SigmoidArrayBackend is _hooks_sigmoid.SigmoidArrayBackend
        assert _hooks_compute.BinaryLogLossBackend is _hooks_loss.BinaryLogLossBackend
        assert (
            _hooks_compute.PrecomputeFeatureBinsBackend
            is _hooks_binning.PrecomputeFeatureBinsBackend
        )
        assert _hooks_compute.PredictRawBackend is _hooks_ensemble.PredictRawBackend
        assert _hooks_compute.sigmoid is _hooks_sigmoid.sigmoid
        assert _hooks_compute.predict_tree is _hooks_prediction.predict_tree
        assert _hooks_compute.build_histogram is _hooks_histogram.build_histogram
        assert _hooks_compute.binary_log_loss is _hooks_loss.binary_log_loss

    def test_re_exported_sigmoid_delegates_correctly(self) -> None:
        """Re-exported sigmoid function delegates to the active backend."""
        from cleargbm import _test_hooks

        result = _test_hooks.sigmoid(0.0)
        assert abs(result - 0.5) < 1e-10


class TestPythonRandomStateWrapper:
    """Tests for _PythonRandomStateWrapper."""

    def test_permutation_returns_permuted_tuple(self) -> None:
        """permutation should return a tuple of the correct length."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.permutation(10)

        assert len(result) == 10
        # Should contain all integers 0-9
        assert set(result) == set(range(10))

    def test_choice_returns_correct_size(self) -> None:
        """choice should return tuple of requested size."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.choice(100, size=5, replace=False)

        assert len(result) == 5
        # All values should be in range
        assert all(0 <= v < 100 for v in result)

    def test_choice_with_replacement(self) -> None:
        """choice with replacement can have duplicates."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.choice(3, size=10, replace=True)

        assert len(result) == 10
        # All values should be in range
        assert all(0 <= v < 3 for v in result)

    def test_rand_1d_returns_floats_in_range(self) -> None:
        """rand_1d should return tuple of floats in [0, 1)."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.rand_1d(5)

        assert len(result) == 5
        assert all(0.0 <= v < 1.0 for v in result)

    def test_rand_2d_returns_nested_tuples(self) -> None:
        """rand_2d should return tuple of tuples."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.rand_2d(3, 4)

        assert len(result) == 3
        for row in result:
            assert len(row) == 4
            assert all(0.0 <= v < 1.0 for v in row)

    def test_same_seed_gives_same_results(self) -> None:
        """Same seed should produce identical sequences."""
        wrapper1 = _PythonRandomStateWrapper(123)
        wrapper2 = _PythonRandomStateWrapper(123)

        result1 = wrapper1.permutation(10)
        result2 = wrapper2.permutation(10)

        assert result1 == result2

    def test_different_seeds_give_different_results(self) -> None:
        """Different seeds should (usually) produce different sequences."""
        wrapper1 = _PythonRandomStateWrapper(1)
        wrapper2 = _PythonRandomStateWrapper(2)

        result1 = wrapper1.rand_1d(100)
        result2 = wrapper2.rand_1d(100)

        # Very unlikely to be equal with different seeds
        assert result1 != result2


class TestGetRandomState:
    """Tests for get_random_state function."""

    def test_returns_random_state_protocol(self) -> None:
        """get_random_state should return something conforming to protocol."""
        rng = get_random_state(42)

        # Verify all protocol methods work by actually calling them
        perm = rng.permutation(5)
        assert len(perm) == 5

        choice_result = rng.choice(10, size=3, replace=False)
        assert len(choice_result) == 3

        rand_1d_result = rng.rand_1d(5)
        assert len(rand_1d_result) == 5

        rand_2d_result = rng.rand_2d(2, 3)
        assert len(rand_2d_result) == 2
        assert len(rand_2d_result[0]) == 3

    def test_default_factory_creates_python_wrapper(self) -> None:
        """Default factory should create _PythonRandomStateWrapper."""
        rng = get_random_state(42)

        # Check it works like Python random
        result = rng.rand_1d(10)
        assert len(result) == 10
        assert all(0.0 <= v < 1.0 for v in result)


class TestRandomStateProtocol:
    """Tests to verify protocol is correctly implemented."""

    def test_wrapper_implements_protocol(self) -> None:
        """_PythonRandomStateWrapper should implement RandomStateProtocol."""
        wrapper = _PythonRandomStateWrapper(42)

        # Type check at runtime via duck typing
        def accepts_protocol(rng: RandomStateProtocol) -> int:
            return len(rng.permutation(5))

        result = accepts_protocol(wrapper)
        assert result == 5


class TestCreateFloatBuffer:
    """Tests for create_float_buffer factory hook."""

    def test_creates_float_buffer(self) -> None:
        """create_float_buffer creates FloatBuffer with correct size."""
        buf = create_float_buffer(5)
        assert len(buf) == 5

    def test_buffer_initialized_to_zero(self) -> None:
        """Buffer should be initialized to zero."""
        buf = create_float_buffer(3)
        for i in range(3):
            assert buf.get(i) == 0.0

    def test_buffer_is_functional(self) -> None:
        """Created buffer should be fully functional."""
        buf = create_float_buffer(3)
        buf.set(0, 1.5)
        buf.set(1, 2.5)
        assert buf.get(0) == 1.5
        assert buf.get(1) == 2.5


class TestCreateIntBuffer:
    """Tests for create_int_buffer factory hook."""

    def test_creates_int_buffer(self) -> None:
        """create_int_buffer creates IntBuffer with correct size."""
        buf = create_int_buffer(5)
        assert len(buf) == 5

    def test_buffer_initialized_to_zero(self) -> None:
        """Buffer should be initialized to zero."""
        buf = create_int_buffer(3)
        for i in range(3):
            assert buf.get(i) == 0

    def test_buffer_is_functional(self) -> None:
        """Created buffer should be fully functional."""
        buf = create_int_buffer(3)
        buf.set(0, 10)
        buf.set(1, 20)
        assert buf.get(0) == 10
        assert buf.get(1) == 20


class TestCreateHistogramBuffer:
    """Tests for create_histogram_buffer factory hook."""

    def test_creates_histogram_buffer(self) -> None:
        """create_histogram_buffer creates HistogramBuffer with correct size."""
        buf = create_histogram_buffer(5)
        assert buf.n_bins == 5

    def test_buffer_initialized_to_zero(self) -> None:
        """Buffer should be initialized to zero."""
        buf = create_histogram_buffer(3)
        for i in range(3):
            assert buf.get_gradient_sum(i) == 0.0
            assert buf.get_hessian_sum(i) == 0.0
            assert buf.get_count(i) == 0

    def test_buffer_is_functional(self) -> None:
        """Created buffer should be fully functional."""
        buf = create_histogram_buffer(3)
        buf.accumulate(0, 1.0, 0.5)
        buf.accumulate(1, 2.0, 1.0)
        assert buf.get_gradient_sum(0) == 1.0
        assert buf.get_hessian_sum(0) == 0.5
        assert buf.get_count(0) == 1
        assert buf.get_gradient_sum(1) == 2.0
        assert buf.get_hessian_sum(1) == 1.0
        assert buf.get_count(1) == 1


def _float_matrix(data: list[list[float]]) -> NDArray[np.float64]:
    """Create a 2D float array from nested list."""
    return np.array(data, dtype=np.float64)


def _float_array(data: list[float]) -> NDArray[np.float64]:
    """Create a 1D float array from list."""
    return np.array(data, dtype=np.float64)


class TestPredictTreeHook:
    """Tests for predict_tree backend hook."""

    def test_predicts_single_leaf(self) -> None:
        """Should return leaf value for all samples."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=0.5,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=0,
            n_leaves=1,
            feature_names=("f0",),
        )

        x = _float_matrix([[1.0], [2.0]])
        preds = predict_tree(tree, x)

        assert preds.shape == (2,)
        pred_0: float = preds.item(0)
        pred_1: float = preds.item(1)
        assert pred_0 == 0.5
        assert pred_1 == 0.5

    def test_navigates_left_and_right(self) -> None:
        """Should navigate tree based on feature values."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="left",
                    value=0.0,
                    n_samples=10,
                    left_child=1,
                    right_child=2,
                ),
                TreeNode(
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
                ),
                TreeNode(
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
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        x = _float_matrix([[0.0], [1.0]])
        preds = predict_tree(tree, x)

        pred_0: float = preds.item(0)
        pred_1: float = preds.item(1)
        assert pred_0 == -1.0
        assert pred_1 == 1.0

    def test_handles_missing_feature_info(self) -> None:
        """Should return node value when feature_index or threshold is None."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=0.25,
                    n_samples=10,
                    left_child=1,
                    right_child=2,
                ),
            ),
            max_depth=0,
            n_leaves=0,
            feature_names=("f0",),
        )

        x = _float_matrix([[0.0]])
        preds = predict_tree(tree, x)

        pred_0: float = preds.item(0)
        assert pred_0 == 0.25

    def test_handles_missing_child(self) -> None:
        """Should return node value when next child is None."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="left",
                    value=0.75,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=0,
            n_leaves=0,
            feature_names=("f0",),
        )

        x = _float_matrix([[0.0]])
        preds = predict_tree(tree, x)

        pred_0: float = preds.item(0)
        assert pred_0 == 0.75

    def test_routes_nan_left(self) -> None:
        """Should route NaN to left child when nan_direction is left."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="left",
                    value=0.0,
                    n_samples=10,
                    left_child=1,
                    right_child=2,
                ),
                TreeNode(
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
                ),
                TreeNode(
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
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        x = _float_matrix([[float("nan")]])
        preds = predict_tree(tree, x)

        pred_0: float = preds.item(0)
        assert pred_0 == -1.0

    def test_routes_nan_right(self) -> None:
        """Should route NaN to right child when nan_direction is right."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="right",
                    value=0.0,
                    n_samples=10,
                    left_child=1,
                    right_child=2,
                ),
                TreeNode(
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
                ),
                TreeNode(
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
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        x = _float_matrix([[float("nan")]])
        preds = predict_tree(tree, x)

        pred_0: float = preds.item(0)
        assert pred_0 == 1.0


class TestSigmoidHook:
    """Tests for sigmoid backend hook."""

    def test_sigmoid_at_zero(self) -> None:
        """sigmoid(0) should return 0.5."""
        assert sigmoid(0.0) == 0.5

    def test_sigmoid_positive(self) -> None:
        """Large positive input should produce value near 1."""
        result = sigmoid(100.0)
        assert result > 0.99
        assert result <= 1.0

    def test_sigmoid_negative(self) -> None:
        """Large negative input should produce value near 0."""
        result = sigmoid(-100.0)
        assert result > 0.0
        assert result < 0.01

    def test_sigmoid_extreme_positive(self) -> None:
        """Extreme positive input should not overflow."""
        result = sigmoid(1000.0)
        assert result > 0.0
        assert result <= 1.0

    def test_sigmoid_extreme_negative(self) -> None:
        """Extreme negative input should not overflow."""
        result = sigmoid(-1000.0)
        assert result >= 0.0
        assert result < 1.0


class TestSigmoidArrayHook:
    """Tests for sigmoid_array backend hook."""

    def test_sigmoid_array_basic(self) -> None:
        """Should compute sigmoid for each element."""
        x = _float_array([0.0, 100.0, -100.0])
        result = sigmoid_array(x)

        assert result.shape == (3,)
        val_0: float = result.item(0)
        val_1: float = result.item(1)
        val_2: float = result.item(2)
        assert abs(val_0 - 0.5) < 1e-10
        assert val_1 > 0.99
        assert val_2 < 0.01

    def test_sigmoid_array_extreme_values(self) -> None:
        """Should handle extreme values without overflow."""
        x = _float_array([1000.0, -1000.0])
        result = sigmoid_array(x)

        val_0: float = result.item(0)
        val_1: float = result.item(1)
        assert val_0 <= 1.0
        assert val_0 > 0.0
        assert val_1 >= 0.0
        assert val_1 < 1.0

    def test_sigmoid_array_preserves_shape(self) -> None:
        """Should preserve input array shape."""
        x: NDArray[np.float64] = np.zeros(5, dtype=np.float64)
        result = sigmoid_array(x)

        assert result.shape == (5,)
        for i in range(5):
            val: float = result.item(i)
            assert abs(val - 0.5) < 1e-10
