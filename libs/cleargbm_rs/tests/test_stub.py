"""Tests for cleargbm_rs Python stubs.

Verifies that all stub functions in ``__init__.py`` raise ``ImportError``
with a clear message when the Rust extension is not built. These stubs
provide typed signatures for mypy and explicit errors for consumers.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleargbm_rs import (
    PyGbmModel,
    PyTree,
    __version__,
    bin_samples_rs,
    binary_log_loss_gradients_rs,
    binary_log_loss_hessians_rs,
    binary_log_loss_initial_prediction_rs,
    binary_log_loss_rs,
    build_histogram_rs,
    build_tree_rs,
    compute_bin_edges_rs,
    precompute_feature_bins_rs,
    predict_ensemble_rs,
    predict_proba_model_rs,
    predict_proba_rs,
    predict_raw_model_rs,
    predict_single_rs,
    predict_tree_rs,
    py_tree_from_json_rs,
    py_tree_max_depth_rs,
    py_tree_n_leaves_rs,
    py_tree_n_nodes_rs,
    py_tree_repr_rs,
    py_tree_to_json_rs,
    sigmoid_array_rs,
    sigmoid_rs,
    subtract_histogram_rs,
    train_gradient_boosting_rs,
)

_MATCH: str = "Rust extension not built"


def test_version_value() -> None:
    """Version should be the expected value."""
    assert __version__ == "0.1.0"


def test_build_histogram_rs_raises_import_error() -> None:
    """build_histogram_rs raises ImportError when extension not built."""
    sample_indices: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array((0, 1, 2), dtype=np.int64)
    gradients: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
        (0.1, 0.2, 0.3), dtype=np.float64
    )
    hessians: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
        (1.0, 1.0, 1.0), dtype=np.float64
    )
    bins: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array((0, 1, 0), dtype=np.int64)
    n_bins = 3

    with pytest.raises(ImportError, match=_MATCH):
        build_histogram_rs(sample_indices, gradients, hessians, bins, n_bins)


def test_subtract_histogram_rs_raises_import_error() -> None:
    """subtract_histogram_rs raises ImportError when extension not built."""
    grads: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array((1.0, 2.0), dtype=np.float64)
    hess: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array((3.0, 4.0), dtype=np.float64)
    counts: np.ndarray[tuple[int], np.dtype[np.uint64]] = np.array((10, 20), dtype=np.uint64)

    with pytest.raises(ImportError, match=_MATCH):
        subtract_histogram_rs(grads, hess, counts, grads, hess, counts)


def test_predict_tree_rs_raises_import_error() -> None:
    """predict_tree_rs raises ImportError when extension not built."""
    features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        ((0.5,),), dtype=np.float64
    )

    with pytest.raises(ImportError, match=_MATCH):
        predict_tree_rs(PyTree.__new__(PyTree), features)


def test_sigmoid_rs_raises_import_error() -> None:
    """sigmoid_rs raises ImportError when extension not built."""
    with pytest.raises(ImportError, match=_MATCH):
        sigmoid_rs(0.0)


def test_sigmoid_array_rs_raises_import_error() -> None:
    """sigmoid_array_rs raises ImportError when extension not built."""
    x: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array((0.0, 1.0), dtype=np.float64)

    with pytest.raises(ImportError, match=_MATCH):
        sigmoid_array_rs(x)


def test_py_tree_from_json_rs_raises_import_error() -> None:
    """py_tree_from_json_rs raises ImportError when extension not built."""
    with pytest.raises(ImportError, match=_MATCH):
        py_tree_from_json_rs('{"nodes": [], "max_depth": 0, "n_leaves": 0}')


def test_py_tree_init_raises_import_error() -> None:
    """PyTree() raises ImportError when extension not built."""
    with pytest.raises(ImportError, match=_MATCH):
        PyTree()


def test_binary_log_loss_rs_raises_import_error() -> None:
    """binary_log_loss_rs raises ImportError when extension not built."""
    y_true: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array((0, 1, 1), dtype=np.int64)
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
        (0.1, 0.9, 0.8), dtype=np.float64
    )

    with pytest.raises(ImportError, match=_MATCH):
        binary_log_loss_rs(y_true, y_pred)


def test_binary_log_loss_gradients_rs_raises_import_error() -> None:
    """binary_log_loss_gradients_rs raises ImportError when extension not built."""
    y_true: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array((0, 1), dtype=np.int64)
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array((0.3, 0.7), dtype=np.float64)

    with pytest.raises(ImportError, match=_MATCH):
        binary_log_loss_gradients_rs(y_true, y_pred)


def test_binary_log_loss_hessians_rs_raises_import_error() -> None:
    """binary_log_loss_hessians_rs raises ImportError when extension not built."""
    y_true: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array((0, 1), dtype=np.int64)
    y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array((0.3, 0.7), dtype=np.float64)

    with pytest.raises(ImportError, match=_MATCH):
        binary_log_loss_hessians_rs(y_true, y_pred)


def test_binary_log_loss_initial_prediction_rs_raises_import_error() -> None:
    """binary_log_loss_initial_prediction_rs raises ImportError when not built."""
    y_true: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array((0, 1, 1, 0), dtype=np.int64)

    with pytest.raises(ImportError, match=_MATCH):
        binary_log_loss_initial_prediction_rs(y_true)


def test_precompute_feature_bins_rs_raises_import_error() -> None:
    """precompute_feature_bins_rs raises ImportError when not built."""
    features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        ((1.0, 2.0), (3.0, 4.0)), dtype=np.float64
    )

    with pytest.raises(ImportError, match=_MATCH):
        precompute_feature_bins_rs(features, 4)


def test_predict_ensemble_rs_raises_import_error() -> None:
    """predict_ensemble_rs raises ImportError when extension not built."""
    features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        ((0.5,),), dtype=np.float64
    )

    with pytest.raises(ImportError, match=_MATCH):
        predict_ensemble_rs([], features, 0.0, 0.1)


def test_predict_proba_rs_raises_import_error() -> None:
    """predict_proba_rs raises ImportError when extension not built."""
    raw: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array((0.0, 1.0, -1.0), dtype=np.float64)

    with pytest.raises(ImportError, match=_MATCH):
        predict_proba_rs(raw)


def test_build_tree_rs_raises_import_error() -> None:
    """build_tree_rs raises ImportError when extension not built."""
    indices: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array((0,), dtype=np.int64)
    grads: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array((0.1,), dtype=np.float64)
    hess: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array((1.0,), dtype=np.float64)
    bins: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array((0,), dtype=np.int64)

    with pytest.raises(ImportError, match=_MATCH):
        build_tree_rs(indices, grads, hess, bins, 2, [[0.5]], "{}")


def test_py_tree_to_json_rs_raises_import_error() -> None:
    """py_tree_to_json_rs raises ImportError when extension not built."""
    tree = PyTree.__new__(PyTree)

    with pytest.raises(ImportError, match=_MATCH):
        py_tree_to_json_rs(tree)


def test_py_tree_max_depth_rs_raises_import_error() -> None:
    """py_tree_max_depth_rs raises ImportError when extension not built."""
    tree = PyTree.__new__(PyTree)

    with pytest.raises(ImportError, match=_MATCH):
        py_tree_max_depth_rs(tree)


def test_py_tree_n_leaves_rs_raises_import_error() -> None:
    """py_tree_n_leaves_rs raises ImportError when extension not built."""
    tree = PyTree.__new__(PyTree)

    with pytest.raises(ImportError, match=_MATCH):
        py_tree_n_leaves_rs(tree)


def test_py_tree_n_nodes_rs_raises_import_error() -> None:
    """py_tree_n_nodes_rs raises ImportError when extension not built."""
    tree = PyTree.__new__(PyTree)

    with pytest.raises(ImportError, match=_MATCH):
        py_tree_n_nodes_rs(tree)


def test_py_tree_repr_rs_raises_import_error() -> None:
    """py_tree_repr_rs raises ImportError when extension not built."""
    tree = PyTree.__new__(PyTree)

    with pytest.raises(ImportError, match=_MATCH):
        py_tree_repr_rs(tree)


def test_predict_single_rs_raises_import_error() -> None:
    """predict_single_rs raises ImportError when extension not built."""
    features: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array((0.5,), dtype=np.float64)
    tree = PyTree.__new__(PyTree)

    with pytest.raises(ImportError, match=_MATCH):
        predict_single_rs(tree, features)


def test_compute_bin_edges_rs_raises_import_error() -> None:
    """compute_bin_edges_rs raises ImportError when extension not built."""
    features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        ((1.0, 2.0), (3.0, 4.0)), dtype=np.float64
    )

    with pytest.raises(ImportError, match=_MATCH):
        compute_bin_edges_rs(features, 4)


def test_bin_samples_rs_raises_import_error() -> None:
    """bin_samples_rs raises ImportError when extension not built."""
    features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        ((1.0,), (2.0,)), dtype=np.float64
    )

    with pytest.raises(ImportError, match=_MATCH):
        bin_samples_rs(features, [[1.5]], 2)


def test_pygbmmodel_init_raises_import_error() -> None:
    """PyGbmModel() raises ImportError when extension not built."""
    with pytest.raises(ImportError, match=_MATCH):
        PyGbmModel()


def test_train_gradient_boosting_rs_raises_import_error() -> None:
    """train_gradient_boosting_rs raises ImportError when not built."""
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        ((1.0,), (2.0,)), dtype=np.float64
    )
    y: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array((0, 1), dtype=np.int64)
    config: dict[str, int | float | bool | list[int] | None] = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_bins": 32,
        "subsample": 1.0,
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "monotonic_constraints": None,
        "early_stopping_rounds": None,
    }

    with pytest.raises(ImportError, match=_MATCH):
        train_gradient_boosting_rs(x, y, None, None, config, ["f1"])


def test_predict_proba_model_rs_raises_import_error() -> None:
    """predict_proba_model_rs raises ImportError when not built."""
    model = PyGbmModel.__new__(PyGbmModel)
    features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        ((0.5,),), dtype=np.float64
    )

    with pytest.raises(ImportError, match=_MATCH):
        predict_proba_model_rs(model, features)


def test_predict_raw_model_rs_raises_import_error() -> None:
    """predict_raw_model_rs raises ImportError when not built."""
    model = PyGbmModel.__new__(PyGbmModel)
    features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        ((0.5,),), dtype=np.float64
    )

    with pytest.raises(ImportError, match=_MATCH):
        predict_raw_model_rs(model, features)
