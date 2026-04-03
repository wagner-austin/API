"""ClearGBM Rust core Python bindings — re-export layer.

Re-exports all function stubs and classes from the focused sub-modules:

- ``_stubs_histogram`` — build_histogram_rs, subtract_histogram_rs
- ``_stubs_tree`` — PyTree, build_tree_rs, py_tree_*_rs
- ``_stubs_prediction`` — sigmoid_rs, predict_*_rs
- ``_stubs_loss`` — binary_log_loss_*_rs, sigmoid_array_rs
- ``_stubs_binning`` — precompute_feature_bins_rs, compute_bin_edges_rs, bin_samples_rs
- ``_stubs_training`` — PyGbmModel, train_gradient_boosting_rs, predict_*_model_rs

The compiled native extension (``.pyd``) replaces these stubs at runtime.
When the extension is not built, all stubs raise ``ImportError``.
"""

from __future__ import annotations

from cleargbm_rs._stubs_binning import (
    bin_samples_rs,
    compute_bin_edges_rs,
    precompute_feature_bins_rs,
)
from cleargbm_rs._stubs_histogram import (
    build_histogram_rs,
    subtract_histogram_rs,
)
from cleargbm_rs._stubs_loss import (
    binary_log_loss_gradients_rs,
    binary_log_loss_hessians_rs,
    binary_log_loss_initial_prediction_rs,
    binary_log_loss_rs,
    sigmoid_array_rs,
)
from cleargbm_rs._stubs_prediction import (
    predict_ensemble_rs,
    predict_proba_rs,
    predict_single_rs,
    predict_tree_rs,
    sigmoid_rs,
)
from cleargbm_rs._stubs_training import (
    PyGbmModel,
    predict_proba_model_rs,
    predict_raw_model_rs,
    train_gradient_boosting_rs,
)
from cleargbm_rs._stubs_tree import (
    PyTree,
    build_tree_rs,
    py_tree_from_json_rs,
    py_tree_max_depth_rs,
    py_tree_n_leaves_rs,
    py_tree_n_nodes_rs,
    py_tree_repr_rs,
    py_tree_to_json_rs,
)

__version__ = "0.1.0"

__all__ = [
    "PyGbmModel",
    "PyTree",
    "__version__",
    "bin_samples_rs",
    "binary_log_loss_gradients_rs",
    "binary_log_loss_hessians_rs",
    "binary_log_loss_initial_prediction_rs",
    "binary_log_loss_rs",
    "build_histogram_rs",
    "build_tree_rs",
    "compute_bin_edges_rs",
    "precompute_feature_bins_rs",
    "predict_ensemble_rs",
    "predict_proba_model_rs",
    "predict_proba_rs",
    "predict_raw_model_rs",
    "predict_single_rs",
    "predict_tree_rs",
    "py_tree_from_json_rs",
    "py_tree_max_depth_rs",
    "py_tree_n_leaves_rs",
    "py_tree_n_nodes_rs",
    "py_tree_repr_rs",
    "py_tree_to_json_rs",
    "sigmoid_array_rs",
    "sigmoid_rs",
    "subtract_histogram_rs",
    "train_gradient_boosting_rs",
]
