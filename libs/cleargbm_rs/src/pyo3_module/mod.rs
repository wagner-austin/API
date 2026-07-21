//! PyO3 Python bindings for the `ClearGBM` Rust core.
//!
//! This module exposes the Rust API to Python through a native extension module.
//! All functions use the `_rs` suffix to distinguish them from Python-native
//! implementations during the migration period.
//!
//! # Registration Strategy
//!
//! Functions are registered via [`PyCFunction::new_closure`] with manual argument
//! extraction using explicit `match`. This avoids `#[pyfunction]` proc-macro
//! generated code that contains `?` operators, allowing all clippy lints
//! (including `question_mark_used`) to stay at `forbid`.
//!
//! Registration uses `.and_then()` chains so that error propagation happens
//! inside `Result::and_then` (standard library), keeping all match arms in our
//! source testable.
//!
//! # Module Structure
//!
//! - [`error_conversion`] - Maps [`ClearGbmError`](crate::error::ClearGbmError) to Python exceptions
//! - [`array_helpers`] - Numpy ↔ Rust type conversions (no `as` casts)
//! - [`histogram_fns`] - Histogram building and subtraction
//! - [`tree_fns`] - Tree construction and the [`PyTree`](tree_fns::PyTree) class
//! - [`prediction_fns`] - Sigmoid, single/batch/ensemble prediction, probability conversion
//! - [`loss_fns`] - Loss functions (binary log loss, gradients, hessians, initial prediction)
//! - [`binning_fns`] - Feature binning (bin edges, sample assignment, precomputation)
//! - [`training_fns`] - Training loop and the [`PyGbmModel`](training_fns::PyGbmModel) class

pub(crate) mod array_helpers;
pub(crate) mod binning_fns;
mod error_conversion;
pub(crate) mod histogram_fns;
pub(crate) mod loss_fns;
pub(crate) mod prediction_fns;
pub(crate) mod training_fns;
pub(crate) mod tree_fns;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};

use training_fns::PyGbmModel;
use tree_fns::PyTree;

/// Registers all Python-callable functions and classes in the `cleargbm_rs` module.
///
/// Functions are registered via [`PyCFunction::new_closure`] with manual argument
/// extraction. Error propagation uses `.and_then()` chains.
///
/// # Args
///
/// * `m` - The Python module to populate.
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// Returns `PyErr` if any function or class registration fails.
#[pymodule]
fn cleargbm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_all(m.py(), m)
}

/// Registers all functions and classes on the module using `.and_then()` chains.
///
/// Each function is created with [`PyCFunction::new_closure`] and added via
/// [`PyModuleMethods::add_function`]. The chain propagates the first error
/// encountered, if any.
///
/// # Errors
///
/// Returns `PyErr` if any closure creation, function registration, or class
/// registration fails.
fn register_all(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- Histogram functions ---
    PyCFunction::new_closure(
        py,
        Some(c"build_histogram_rs"),
        Some(c"Build histogram from sample gradients and hessians."),
        |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
            histogram_fns::build_histogram_from_args(args)
        },
    )
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"subtract_histogram_rs"),
            Some(c"Compute sibling histogram by subtraction (parent - child)."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                histogram_fns::subtract_histogram_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    // --- Tree functions ---
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"build_tree_rs"),
            Some(c"Build a decision tree using histogram-based split finding."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                tree_fns::build_tree_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_tree_max_depth_rs"),
            Some(c"Return the maximum depth of a PyTree."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                tree_fns::py_tree_max_depth_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_tree_n_leaves_rs"),
            Some(c"Return the number of leaf nodes in a PyTree."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                tree_fns::py_tree_n_leaves_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_tree_n_nodes_rs"),
            Some(c"Return the total number of nodes in a PyTree."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                tree_fns::py_tree_n_nodes_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_tree_to_json_rs"),
            Some(c"Serialize a PyTree to a JSON string."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                tree_fns::py_tree_to_json_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_tree_from_json_rs"),
            Some(c"Deserialize a PyTree from a JSON string."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                tree_fns::py_tree_from_json_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_tree_repr_rs"),
            Some(c"Return a string representation of a PyTree for debugging."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                tree_fns::py_tree_repr_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    // --- Prediction functions ---
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"sigmoid_rs"),
            Some(c"Compute sigmoid (logistic) function with numerical stability."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                prediction_fns::sigmoid_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"predict_single_rs"),
            Some(c"Predict leaf value for a single sample."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                prediction_fns::predict_single_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"predict_tree_rs"),
            Some(c"Predict leaf values for a batch of samples using a single tree."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                prediction_fns::predict_tree_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"predict_ensemble_rs"),
            Some(c"Predict raw scores using an ensemble of trees."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                prediction_fns::predict_ensemble_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"predict_proba_rs"),
            Some(c"Convert raw predictions to binary class probabilities."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                prediction_fns::predict_proba_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    // --- Loss functions ---
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"binary_log_loss_rs"),
            Some(c"Compute mean binary cross-entropy (log loss)."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                loss_fns::binary_log_loss_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"binary_log_loss_gradients_rs"),
            Some(c"Compute gradients of binary log loss (p - y)."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                loss_fns::binary_log_loss_gradients_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"binary_log_loss_hessians_rs"),
            Some(c"Compute hessians of binary log loss (p * (1-p))."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                loss_fns::binary_log_loss_hessians_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"binary_log_loss_initial_prediction_rs"),
            Some(c"Compute initial prediction (log-odds of positive class rate)."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                loss_fns::binary_log_loss_initial_prediction_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"sigmoid_array_rs"),
            Some(c"Apply sigmoid to each element of an array."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                loss_fns::sigmoid_array_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    // --- Binning functions ---
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"precompute_feature_bins_rs"),
            Some(c"Precompute feature bins from a 2D feature matrix."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                binning_fns::precompute_feature_bins_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"compute_bin_edges_rs"),
            Some(c"Compute bin edges for each feature."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                binning_fns::compute_bin_edges_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"bin_samples_rs"),
            Some(c"Assign bin indices to samples given precomputed bin edges."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                binning_fns::bin_samples_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    // --- Training functions ---
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"train_gradient_boosting_rs"),
            Some(c"Train a gradient boosting model on binary classification data."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                training_fns::train_gradient_boosting_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"predict_proba_model_rs"),
            Some(c"Predict class probabilities using a trained model."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                training_fns::predict_proba_model_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"predict_raw_model_rs"),
            Some(c"Predict raw log-odds using a trained model."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                training_fns::predict_raw_model_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_gbm_model_to_json_rs"),
            Some(c"Serialize a PyGbmModel to a JSON string."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                training_fns::py_gbm_model_to_json_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_gbm_model_from_json_rs"),
            Some(c"Deserialize a PyGbmModel from a JSON string."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                training_fns::py_gbm_model_from_json_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_gbm_model_feature_importances_rs"),
            Some(c"Return per-feature split-count importance for a PyGbmModel."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                training_fns::py_gbm_model_feature_importances_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_gbm_model_n_trees_rs"),
            Some(c"Return the number of trees in a PyGbmModel."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                training_fns::py_gbm_model_n_trees_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"py_gbm_model_n_classes_rs"),
            Some(c"Return the number of classes in a PyGbmModel."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                training_fns::py_gbm_model_n_classes_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    // --- Classes ---
    .and_then(|()| m.add_class::<PyTree>())
    .and_then(|()| m.add_class::<PyGbmModel>())
}
