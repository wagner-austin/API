"""Rust backend adapters for cleargbm hot paths.

Bridges between cleargbm's Python types and cleargbm_rs's native functions.
Each adapter conforms to the corresponding Protocol in the _hooks_* sub-modules.

Call ``use_rust_backend()`` at startup to wire all hooks to Rust.
Call ``use_python_backend()`` to restore Python defaults.

No try/except, no auto-detection, no fallback.
"""

from __future__ import annotations

import types
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm.buffers import HistogramBuffer
from cleargbm.types import BinEdges, DecisionTree, FeatureBins

# =============================================================================
# Protocols for native Rust functions (typed getattr assignments)
# =============================================================================


class _BuildHistogramRs(Protocol):
    """Protocol matching ``cleargbm_rs.build_histogram_rs``."""

    def __call__(
        self,
        sample_indices: NDArray[np.int64],
        gradients: NDArray[np.float64],
        hessians: NDArray[np.float64],
        bins: NDArray[np.int64],
        n_bins: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint64]]:
        """Build histogram from sample data.

        Args:
            sample_indices: Indices of samples at this node.
            gradients: Gradient values for all samples.
            hessians: Hessian values for all samples.
            bins: Pre-computed bin assignments.
            n_bins: Number of histogram bins.

        Returns:
            Tuple of (gradient_sums, hessian_sums, counts) as numpy arrays.
        """
        ...


class _SubtractHistogramRs(Protocol):
    """Protocol matching ``cleargbm_rs.subtract_histogram_rs``."""

    def __call__(
        self,
        parent_grads: NDArray[np.float64],
        parent_hess: NDArray[np.float64],
        parent_counts: NDArray[np.uint64],
        child_grads: NDArray[np.float64],
        child_hess: NDArray[np.float64],
        child_counts: NDArray[np.uint64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint64]]:
        """Compute sibling histogram by subtraction.

        Args:
            parent_grads: Parent gradient sums per bin.
            parent_hess: Parent hessian sums per bin.
            parent_counts: Parent sample counts per bin.
            child_grads: Child gradient sums per bin.
            child_hess: Child hessian sums per bin.
            child_counts: Child sample counts per bin.

        Returns:
            Tuple of (gradient_sums, hessian_sums, counts) for sibling.
        """
        ...


class _PredictTreeRs(Protocol):
    """Protocol matching ``cleargbm_rs.predict_tree_rs``."""

    def __call__(
        self,
        tree: _PyTree,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict leaf values for a batch of samples.

        Args:
            tree: Rust PyTree instance.
            features: 2D feature matrix (n_samples, n_features).

        Returns:
            1D array of predictions.
        """
        ...


class _SigmoidRs(Protocol):
    """Protocol matching ``cleargbm_rs.sigmoid_rs``."""

    def __call__(self, x: float) -> float:
        """Compute sigmoid function.

        Args:
            x: Input value (log-odds).

        Returns:
            Probability in [0, 1].
        """
        ...


class _SigmoidArrayRs(Protocol):
    """Protocol matching ``cleargbm_rs.sigmoid_array_rs``."""

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply sigmoid to each element.

        Args:
            x: Input array (log-odds).

        Returns:
            Array of probabilities.
        """
        ...


class _PyTreeFromJsonRs(Protocol):
    """Protocol matching ``cleargbm_rs.py_tree_from_json_rs``."""

    def __call__(self, json_str: str) -> _PyTree:
        """Deserialize a PyTree from JSON.

        Args:
            json_str: JSON string in Rust Tree serde format.

        Returns:
            PyTree instance.
        """
        ...


class _PyTree(Protocol):
    """Protocol matching ``cleargbm_rs.PyTree`` (opaque)."""

    ...


class _BinaryLogLossRs(Protocol):
    """Protocol matching ``cleargbm_rs.binary_log_loss_rs``."""

    def __call__(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> float:
        """Compute mean binary cross-entropy loss.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities.

        Returns:
            Mean loss value.
        """
        ...


class _BinaryLogLossGradientsRs(Protocol):
    """Protocol matching ``cleargbm_rs.binary_log_loss_gradients_rs``."""

    def __call__(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute gradients of binary log loss.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities.

        Returns:
            Gradient array.
        """
        ...


class _BinaryLogLossHessiansRs(Protocol):
    """Protocol matching ``cleargbm_rs.binary_log_loss_hessians_rs``."""

    def __call__(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute hessians of binary log loss.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities.

        Returns:
            Hessian array.
        """
        ...


class _BinaryLogLossInitialPredictionRs(Protocol):
    """Protocol matching ``cleargbm_rs.binary_log_loss_initial_prediction_rs``."""

    def __call__(
        self,
        y_true: NDArray[np.int64],
    ) -> float:
        """Compute initial prediction (log-odds of positive class rate).

        Args:
            y_true: True labels (0 or 1).

        Returns:
            Initial prediction in log-odds space.
        """
        ...


class _PrecomputeFeatureBinsRs(Protocol):
    """Protocol matching ``cleargbm_rs.precompute_feature_bins_rs``."""

    def __call__(
        self,
        features: NDArray[np.float64],
        max_bins: int,
    ) -> tuple[list[list[float]], NDArray[np.int64], int]:
        """Precompute feature bins from a feature matrix.

        Args:
            features: 2D feature matrix (n_samples, n_features).
            max_bins: Maximum number of bins per feature.

        Returns:
            Tuple of (bin_thresholds, sample_bins, n_regular_bins).
        """
        ...


class _PredictEnsembleRs(Protocol):
    """Protocol matching ``cleargbm_rs.predict_ensemble_rs``."""

    def __call__(
        self,
        trees: list[_PyTree],
        features: NDArray[np.float64],
        base_prediction: float,
        learning_rate: float,
    ) -> NDArray[np.float64]:
        """Predict raw scores from an ensemble of trees.

        Args:
            trees: List of Rust PyTree instances.
            features: 2D feature matrix (n_samples, n_features).
            base_prediction: Initial prediction.
            learning_rate: Shrinkage factor.

        Returns:
            1D array of raw predictions.
        """
        ...


class _PredictProbaRs(Protocol):
    """Protocol matching ``cleargbm_rs.predict_proba_rs``."""

    def __call__(
        self,
        raw_predictions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert raw predictions to class probabilities.

        Args:
            raw_predictions: 1D array of raw predictions (log-odds).

        Returns:
            2D array of shape (n_samples, 2) with (prob_0, prob_1).
        """
        ...


class _JsonDumps(Protocol):
    """Protocol matching ``json.dumps`` for tree serialization."""

    def __call__(
        self,
        obj: dict[str, list[dict[str, int | float | bool | None]] | int],
    ) -> str:
        """Serialize a dict to a JSON string.

        Args:
            obj: Dictionary to serialize (tree structure).

        Returns:
            JSON string.
        """
        ...


# =============================================================================
# Deferred native module loading (called by use_rust_backend / tests)
# =============================================================================

# Typed module-level references — set by _load_native_functions().
# Adapter functions below reference these; they are only called after loading.
_rs_build_histogram: _BuildHistogramRs
_rs_subtract_histogram: _SubtractHistogramRs
_rs_predict_tree: _PredictTreeRs
_rs_sigmoid: _SigmoidRs
_rs_sigmoid_array: _SigmoidArrayRs
_rs_py_tree_from_json: _PyTreeFromJsonRs
_rs_binary_log_loss: _BinaryLogLossRs
_rs_binary_log_loss_gradients: _BinaryLogLossGradientsRs
_rs_binary_log_loss_hessians: _BinaryLogLossHessiansRs
_rs_binary_log_loss_initial_prediction: _BinaryLogLossInitialPredictionRs
_rs_precompute_feature_bins: _PrecomputeFeatureBinsRs
_rs_predict_ensemble: _PredictEnsembleRs
_rs_predict_proba: _PredictProbaRs

_json_dumps: _JsonDumps = __import__("json").dumps


def _load_native_functions() -> None:
    """Load native Rust functions from cleargbm_rs and bind to module globals.

    Must be called before any adapter function is used. Called automatically
    by ``use_rust_backend()``. Tests that call adapter functions directly
    should call this at module level.

    Raises:
        ModuleNotFoundError: If cleargbm_rs native extension is not installed.
    """
    global _rs_build_histogram, _rs_subtract_histogram, _rs_predict_tree
    global _rs_sigmoid, _rs_sigmoid_array, _rs_py_tree_from_json
    global _rs_binary_log_loss, _rs_binary_log_loss_gradients
    global _rs_binary_log_loss_hessians, _rs_binary_log_loss_initial_prediction
    global _rs_precompute_feature_bins, _rs_predict_ensemble, _rs_predict_proba

    mod: types.ModuleType = __import__("cleargbm_rs.cleargbm_rs", fromlist=["cleargbm_rs"])

    # Typed intermediates — Protocol annotations override Any from ModuleType
    bh: _BuildHistogramRs = mod.build_histogram_rs
    sh: _SubtractHistogramRs = mod.subtract_histogram_rs
    pt: _PredictTreeRs = mod.predict_tree_rs
    sig: _SigmoidRs = mod.sigmoid_rs
    siga: _SigmoidArrayRs = mod.sigmoid_array_rs
    ptj: _PyTreeFromJsonRs = mod.py_tree_from_json_rs
    bll: _BinaryLogLossRs = mod.binary_log_loss_rs
    bllg: _BinaryLogLossGradientsRs = mod.binary_log_loss_gradients_rs
    bllh: _BinaryLogLossHessiansRs = mod.binary_log_loss_hessians_rs
    blli: _BinaryLogLossInitialPredictionRs = mod.binary_log_loss_initial_prediction_rs
    pfb: _PrecomputeFeatureBinsRs = mod.precompute_feature_bins_rs
    pe: _PredictEnsembleRs = mod.predict_ensemble_rs
    pp: _PredictProbaRs = mod.predict_proba_rs

    _rs_build_histogram = bh
    _rs_subtract_histogram = sh
    _rs_predict_tree = pt
    _rs_sigmoid = sig
    _rs_sigmoid_array = siga
    _rs_py_tree_from_json = ptj
    _rs_binary_log_loss = bll
    _rs_binary_log_loss_gradients = bllg
    _rs_binary_log_loss_hessians = bllh
    _rs_binary_log_loss_initial_prediction = blli
    _rs_precompute_feature_bins = pfb
    _rs_predict_ensemble = pe
    _rs_predict_proba = pp


# =============================================================================
# Type bridge: DecisionTree → PyTree via JSON
# =============================================================================


def _decision_tree_to_py_tree(tree: DecisionTree) -> _PyTree:
    """Convert a Python DecisionTree TypedDict to a Rust PyTree.

    Serializes the tree to JSON matching Rust's ``Tree`` serde format,
    then deserializes via the native extension.

    Field mapping:
    - Python ``nan_direction: "left"|"right"`` → Rust ``nan_goes_left: bool``
    - Python ``feature_name`` → dropped (Rust ``TreeNode`` has no ``feature_name``)
    - All other fields map 1:1

    Args:
        tree: Python DecisionTree TypedDict.

    Returns:
        Rust PyTree instance.
    """
    rust_nodes: list[dict[str, int | float | bool | None]] = []
    for node in tree["nodes"]:
        nan_dir = node["nan_direction"]
        nan_goes_left: bool = nan_dir is None or nan_dir == "left"
        rust_node: dict[str, int | float | bool | None] = {
            "node_id": node["node_id"],
            "is_leaf": node["is_leaf"],
            "feature_index": node["feature_index"],
            "threshold": node["threshold"],
            "value": node["value"],
            "n_samples": node["n_samples"],
            "left_child": node["left_child"],
            "right_child": node["right_child"],
            "nan_goes_left": nan_goes_left,
        }
        rust_nodes.append(rust_node)

    rust_tree: dict[str, list[dict[str, int | float | bool | None]] | int] = {
        "nodes": rust_nodes,
        "max_depth": tree["max_depth"],
        "n_leaves": tree["n_leaves"],
    }
    json_str: str = _json_dumps(rust_tree)
    return _rs_py_tree_from_json(json_str)


# =============================================================================
# Adapter functions (match _hooks_* sub-module Protocol signatures)
# =============================================================================


def _rust_build_histogram(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    sample_bins: NDArray[np.int64],
    n_bins: int,
) -> HistogramBuffer:
    """Rust-backed histogram building.

    Calls ``cleargbm_rs.build_histogram_rs`` and converts the returned
    numpy tuple to a Python ``HistogramBuffer``.

    Args:
        sample_indices: Indices of samples in this node.
        gradients: Gradient for each sample (full dataset).
        hessians: Hessian for each sample (full dataset).
        sample_bins: Bin ID for each sample on this feature (1D array).
        n_bins: Number of bins.

    Returns:
        HistogramBuffer with gradient/hessian sums per bin.
    """
    grad_sums, hess_sums, counts_u64 = _rs_build_histogram(
        sample_indices,
        gradients,
        hessians,
        sample_bins,
        n_bins,
    )
    counts_i64: NDArray[np.int64] = counts_u64.astype(np.int64)
    return HistogramBuffer.from_arrays(grad_sums, hess_sums, counts_i64)


def _rust_subtract_histogram(
    parent: HistogramBuffer,
    child: HistogramBuffer,
) -> HistogramBuffer:
    """Rust-backed histogram subtraction.

    Extracts numpy arrays from parent/child ``HistogramBuffer`` objects,
    calls ``cleargbm_rs.subtract_histogram_rs``, and converts back.

    Args:
        parent: Parent node histogram buffer.
        child: One child's histogram buffer.

    Returns:
        Sibling histogram buffer (parent - child).
    """
    p_grad: NDArray[np.float64] = parent.gradient_sums_array()
    p_hess: NDArray[np.float64] = parent.hessian_sums_array()
    p_counts: NDArray[np.uint64] = parent.counts_array().astype(np.uint64)
    c_grad: NDArray[np.float64] = child.gradient_sums_array()
    c_hess: NDArray[np.float64] = child.hessian_sums_array()
    c_counts: NDArray[np.uint64] = child.counts_array().astype(np.uint64)

    sib_grad, sib_hess, sib_counts_u64 = _rs_subtract_histogram(
        p_grad,
        p_hess,
        p_counts,
        c_grad,
        c_hess,
        c_counts,
    )
    sib_counts_i64: NDArray[np.int64] = sib_counts_u64.astype(np.int64)
    return HistogramBuffer.from_arrays(sib_grad, sib_hess, sib_counts_i64)


def _rust_predict_tree(
    tree: DecisionTree,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rust-backed tree prediction.

    Converts the Python ``DecisionTree`` TypedDict to a Rust ``PyTree``
    via JSON, then calls ``cleargbm_rs.predict_tree_rs``.

    Args:
        tree: Trained decision tree.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Prediction array for each sample.
    """
    py_tree = _decision_tree_to_py_tree(tree)
    result: NDArray[np.float64] = _rs_predict_tree(py_tree, x)
    return result


def _rust_sigmoid(x: float) -> float:
    """Rust-backed scalar sigmoid.

    Args:
        x: Input value (log-odds).

    Returns:
        Probability in [0, 1].
    """
    return _rs_sigmoid(x)


def _rust_sigmoid_array(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rust-backed vectorized sigmoid.

    Args:
        x: Input array (log-odds).

    Returns:
        Probabilities in [0, 1].
    """
    return _rs_sigmoid_array(x)


def _rust_binary_log_loss(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> float:
    """Rust-backed binary log loss computation.

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted probabilities.

    Returns:
        Mean loss value.
    """
    return _rs_binary_log_loss(y_true, y_pred)


def _rust_binary_log_loss_gradients(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rust-backed binary log loss gradients.

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted probabilities.

    Returns:
        Gradient for each sample.
    """
    result: NDArray[np.float64] = _rs_binary_log_loss_gradients(y_true, y_pred)
    return result


def _rust_binary_log_loss_hessians(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rust-backed binary log loss hessians.

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted probabilities.

    Returns:
        Hessian for each sample.
    """
    result: NDArray[np.float64] = _rs_binary_log_loss_hessians(y_true, y_pred)
    return result


def _rust_binary_log_loss_initial_prediction(
    y_true: NDArray[np.int64],
) -> float:
    """Rust-backed binary log loss initial prediction.

    Args:
        y_true: True labels (0 or 1).

    Returns:
        Initial prediction in log-odds space.
    """
    return _rs_binary_log_loss_initial_prediction(y_true)


def _rust_precompute_feature_bins(
    x: NDArray[np.float64],
    max_bins: int,
) -> FeatureBins:
    """Rust-backed feature binning.

    Calls ``cleargbm_rs.precompute_feature_bins_rs`` and converts
    the returned tuple to a Python ``FeatureBins`` NamedTuple.

    Args:
        x: Feature matrix (n_samples, n_features).
        max_bins: Maximum number of bins per feature.

    Returns:
        FeatureBins containing edges and per-sample bin assignments.
    """
    bin_thresholds, sample_bins_2d, _n_regular_bins = _rs_precompute_feature_bins(x, max_bins)
    edges: tuple[BinEdges, ...] = tuple(
        BinEdges(edges=tuple(thresholds)) for thresholds in bin_thresholds
    )
    return FeatureBins(bin_edges=edges, sample_bins=sample_bins_2d)


def _rust_predict_raw(
    trees: tuple[DecisionTree, ...],
    features: NDArray[np.float64],
    base_prediction: float,
    learning_rate: float,
) -> NDArray[np.float64]:
    """Rust-backed ensemble raw prediction.

    Converts all Python ``DecisionTree`` TypedDicts to Rust ``PyTree``
    objects via JSON, then calls ``cleargbm_rs.predict_ensemble_rs``.

    Args:
        trees: Trained decision trees.
        features: Feature matrix (n_samples, n_features).
        base_prediction: Initial prediction before any tree contributions.
        learning_rate: Shrinkage factor for tree contributions.

    Returns:
        Raw predictions (log-odds) for each sample.
    """
    py_trees: list[_PyTree] = [_decision_tree_to_py_tree(tree) for tree in trees]
    result: NDArray[np.float64] = _rs_predict_ensemble(
        py_trees, features, base_prediction, learning_rate
    )
    return result


def _rust_predict_proba(
    raw_predictions: NDArray[np.float64],
) -> tuple[tuple[float, float], ...]:
    """Rust-backed probability prediction.

    Calls ``cleargbm_rs.predict_proba_rs`` and converts the 2D numpy
    result to a tuple of (prob_class_0, prob_class_1) tuples.

    Args:
        raw_predictions: Raw predictions (log-odds).

    Returns:
        Tuple of (prob_class_0, prob_class_1) per sample.
    """
    proba_2d: NDArray[np.float64] = _rs_predict_proba(raw_predictions)
    n_samples: int = int(proba_2d.shape[0])
    result: list[tuple[float, float]] = []
    for i in range(n_samples):
        p0: float = proba_2d.item(i * 2)
        p1: float = proba_2d.item(i * 2 + 1)
        result.append((p0, p1))
    return tuple(result)


# =============================================================================
# Backend wiring
# =============================================================================


def use_rust_backend() -> None:
    """Set all hooks to Rust implementations.

    Call at production startup. After this call, all per-operation hooks
    (histogram, prediction, sigmoid, loss, binning, ensemble) and native
    training hooks (full-loop train, model predict) use the Rust backend.

    Raises:
        ModuleNotFoundError: If cleargbm_rs native extension is not installed.
    """
    _load_native_functions()

    from cleargbm import (
        _hooks_binning,
        _hooks_ensemble,
        _hooks_histogram,
        _hooks_loss,
        _hooks_prediction,
        _hooks_sigmoid,
    )
    from cleargbm._rust_native_adapters import wire_native_hooks

    _hooks_histogram._build_histogram_backend = _rust_build_histogram
    _hooks_histogram._subtract_histogram_backend = _rust_subtract_histogram
    _hooks_prediction._predict_tree_backend = _rust_predict_tree
    _hooks_sigmoid._sigmoid_backend = _rust_sigmoid
    _hooks_sigmoid._sigmoid_array_backend = _rust_sigmoid_array
    _hooks_loss._binary_log_loss_backend = _rust_binary_log_loss
    _hooks_loss._binary_log_loss_gradients_backend = _rust_binary_log_loss_gradients
    _hooks_loss._binary_log_loss_hessians_backend = _rust_binary_log_loss_hessians
    _hooks_loss._binary_log_loss_initial_prediction_backend = (
        _rust_binary_log_loss_initial_prediction
    )
    _hooks_binning._precompute_feature_bins_backend = _rust_precompute_feature_bins
    _hooks_ensemble._predict_raw_backend = _rust_predict_raw
    _hooks_ensemble._predict_proba_backend = _rust_predict_proba
    wire_native_hooks()


def use_python_backend() -> None:
    """Restore all hooks to Python default implementations.

    Resets per-operation hooks to Python defaults and clears native
    training hooks (sets them to None).
    """
    from cleargbm import (
        _hooks_binning,
        _hooks_ensemble,
        _hooks_histogram,
        _hooks_loss,
        _hooks_prediction,
        _hooks_sigmoid,
    )
    from cleargbm._rust_native_adapters import unwire_native_hooks

    _hooks_histogram._build_histogram_backend = _hooks_histogram._default_build_histogram
    _hooks_histogram._subtract_histogram_backend = _hooks_histogram._default_subtract_histogram
    _hooks_prediction._predict_tree_backend = _hooks_prediction._default_predict_tree
    _hooks_sigmoid._sigmoid_backend = _hooks_sigmoid._default_sigmoid
    _hooks_sigmoid._sigmoid_array_backend = _hooks_sigmoid._default_sigmoid_array
    _hooks_loss._binary_log_loss_backend = _hooks_loss._default_binary_log_loss
    _hooks_loss._binary_log_loss_gradients_backend = _hooks_loss._default_binary_log_loss_gradients
    _hooks_loss._binary_log_loss_hessians_backend = _hooks_loss._default_binary_log_loss_hessians
    _hooks_loss._binary_log_loss_initial_prediction_backend = (
        _hooks_loss._default_binary_log_loss_initial_prediction
    )
    _hooks_binning._precompute_feature_bins_backend = (
        _hooks_binning._default_precompute_feature_bins
    )
    _hooks_ensemble._predict_raw_backend = _hooks_ensemble._default_predict_raw
    _hooks_ensemble._predict_proba_backend = _hooks_ensemble._default_predict_proba
    unwire_native_hooks()


__all__ = [
    "_load_native_functions",
    "use_python_backend",
    "use_rust_backend",
]
