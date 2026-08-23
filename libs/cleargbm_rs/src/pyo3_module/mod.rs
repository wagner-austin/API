//! PyO3 Python bindings for the `ClearGBM` Rust core.
//!
//! Exposes the training + inference surface used by `cleargbm.ensemble`. The
//! entire live Python API is eight functions plus the [`PyGbmModel`] class —
//! everything else runs internally to the two native training entries
//! (`train_gradient_boosting_rs` for binary classification,
//! `train_gradient_boosting_regression_rs` for squared-error regression).
//! Subprimitive-level functions (per-histogram, per-tree, per-loss)
//! from the Python-computed era are gone; the migration to "Rust is the only
//! compute path" (see `libs/cleargbm/src/cleargbm/ensemble.py`) made them
//! unreachable from Python.
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
//! - [`error_conversion`] — Maps [`ClearGbmError`](crate::error::ClearGbmError) to Python exceptions
//! - [`array_helpers`] — Numpy ↔ Rust type conversions (no `as` casts)
//! - [`config_extract`] — Config-dict extraction for the training entries
//! - [`training_fns`] — Training entries (binary + regression) and prediction
//! - [`training_multiclass_fns`] — The multiclass entry and its predict trio
//! - [`training_ranking_fns`] — The LambdaMART ranking entry
//! - [`training_continue_fns`] — The continued-training entries
//! - [`entry_args`] — Positional-argument unpacking for the registrations
//! - [`model_fns`] — The [`PyGbmModel`] class + model serde + importances

pub(crate) mod array_helpers;
pub(crate) mod config_extract;
pub(crate) mod entry_args;
mod error_conversion;
pub(crate) mod model_fns;
pub(crate) mod training_continue_fns;
pub(crate) mod training_fns;
pub(crate) mod training_multiclass_fns;
pub(crate) mod training_ranking_fns;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};

use model_fns::PyGbmModel;

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
    PyCFunction::new_closure(
        py,
        Some(c"train_gradient_boosting_rs"),
        Some(c"Train a gradient boosting model on binary classification data."),
        |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
            entry_args::train_gradient_boosting_from_args(args)
        },
    )
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"train_gradient_boosting_regression_rs"),
            Some(c"Train a gradient boosting model on regression data (squared error)."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                entry_args::train_gradient_boosting_regression_from_args(args)
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
                entry_args::predict_proba_model_from_args(args)
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
                entry_args::predict_raw_model_from_args(args)
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
                model_fns::py_gbm_model_to_json_from_args(args)
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
                model_fns::py_gbm_model_from_json_from_args(args)
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
                model_fns::py_gbm_model_feature_importances_from_args(args)
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
                model_fns::py_gbm_model_n_trees_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"train_gradient_boosting_multiclass_rs"),
            Some(c"Train a gradient boosting model on multiclass data (softmax)."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                entry_args::train_gradient_boosting_multiclass_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"predict_raw_multiclass_model_rs"),
            Some(c"Predict raw per-class scores using a trained multiclass model."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                entry_args::predict_raw_multiclass_model_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"predict_proba_multiclass_model_rs"),
            Some(c"Predict per-class probabilities using a trained multiclass model."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                entry_args::predict_proba_multiclass_model_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"predict_class_model_rs"),
            Some(c"Predict class labels (argmax) using a trained multiclass model."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                entry_args::predict_class_model_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"train_gradient_boosting_ranking_rs"),
            Some(c"Train a LambdaMART ranking model on query-grouped data."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                entry_args::train_gradient_boosting_ranking_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"continue_gradient_boosting_rs"),
            Some(c"Continue a binary-classification model with more boosting rounds."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                entry_args::continue_gradient_boosting_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| {
        PyCFunction::new_closure(
            py,
            Some(c"continue_gradient_boosting_regression_rs"),
            Some(c"Continue a regression model with more boosting rounds."),
            |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                entry_args::continue_gradient_boosting_regression_from_args(args)
            },
        )
    })
    .and_then(|f| m.add_function(f))
    .and_then(|()| m.add_class::<PyGbmModel>())
}
