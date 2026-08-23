//! PyO3 bindings for continued training.
//!
//! Mirrors [`super::training_fns`] for the continuation entries: an
//! existing native model plus continuation data comes in, a NEW
//! self-contained model (old trees plus new trees) comes out. There is no
//! config argument — the model's embedded config drives the run — so the
//! worker-thread policy, which is runtime rather than model state, crosses
//! as an explicit `n_jobs` argument instead.

use pyo3::prelude::*;

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::pyo3_module::model_fns::PyGbmModel;
use crate::pyo3_module::training_fns::{
    extract_labels, extract_rows, extract_targets, missing_val_pair, TrainingArrays,
    ValidationArrays,
};
use crate::training::{continue_gradient_boosting, TrainingLabels, ValidationData};
use crate::training::{Parallelism, TrainingRuntime};

/// Converts a Python `additional_rounds` value to the core's `usize`.
///
/// # Errors
///
/// Returns `PyErr` if the value is negative (zero is rejected later by
/// the core, with its own message).
fn convert_additional_rounds(additional_rounds: i64) -> PyResult<usize> {
    match usize::try_from(additional_rounds) {
        Ok(v) => Ok(v),
        Err(_) => Err(ClearGbmError::InvalidParameter {
            name: "additional_rounds".to_string(),
            reason: format!("must be >= 1, got {additional_rounds}"),
        }
        .into()),
    }
}

/// Continues a binary-classification model with more boosting rounds.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `model` - The existing trained model (objective `binary_log_loss`).
/// * `train` - Continuation arrays: features, i64 binary labels, optional
///   weights.
/// * `val` - Optional validation arrays for the config's early stopping.
/// * `additional_rounds` - New boosting rounds (>= 1).
/// * `n_jobs` - Worker-thread policy for this run.
///
/// # Returns
///
/// A [`PyGbmModel`] wrapping the NEW combined model; the input model is
/// unchanged.
///
/// # Errors
///
/// Returns `PyErr` on argument extraction, validation, or training errors,
/// including a model whose objective is not single-score.
pub(crate) fn continue_gradient_boosting_rs(
    py: Python<'_>,
    model: &PyGbmModel,
    train: &TrainingArrays<'_, '_, i64>,
    val: &ValidationArrays<'_, '_, i64>,
    additional_rounds: i64,
    n_jobs: i64,
) -> PyResult<Py<PyGbmModel>> {
    let train_rows = propagate_into!(extract_rows(train.x));
    let train_slices: Vec<&[f64]> = train_rows.iter().map(Vec::as_slice).collect();
    let y_train = propagate!(extract_labels(train.y));
    let weights: Option<Vec<f64>> = match train.weight {
        Some(w) => Some(propagate!(extract_targets(w))),
        None => None,
    };

    let val_rows: Option<Vec<Vec<f64>>> = match val.x {
        Some(xv) => Some(propagate_into!(extract_rows(xv))),
        None => None,
    };
    let val_slices: Option<Vec<&[f64]>> = val_rows
        .as_ref()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());
    let y_val: Option<Vec<u8>> = match val.y {
        Some(yv) => Some(propagate!(extract_labels(yv))),
        None => None,
    };
    let val_weights: Option<Vec<f64>> = match val.weight {
        Some(w) => Some(propagate!(extract_targets(w))),
        None => None,
    };
    let validation: Option<ValidationData<'_>> = match (&val_slices, &y_val) {
        (Some(xv), Some(yv)) => Some(ValidationData {
            x: xv,
            y: TrainingLabels::Binary(yv),
            weight: val_weights.as_deref(),
        }),
        (None, None) => {
            if val_weights.is_some() {
                return Err(missing_val_pair("y_val", "val_sample_weight"));
            }
            None
        }
        (Some(_), None) => return Err(missing_val_pair("y_val", "x_val")),
        (None, Some(_)) => return Err(missing_val_pair("x_val", "y_val")),
    };

    let rounds = propagate!(convert_additional_rounds(additional_rounds));
    let parallelism = propagate_into!(Parallelism::from_n_jobs(n_jobs));
    let continued = propagate_into!(continue_gradient_boosting(
        &model.inner,
        &train_slices,
        TrainingLabels::Binary(&y_train),
        weights.as_deref(),
        validation,
        rounds,
        &TrainingRuntime {
            parallelism,
            hooks: &Hooks::default(),
        },
    ));
    Py::new(py, PyGbmModel { inner: continued })
}

/// Continues a squared-error regression model with more boosting rounds.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `model` - The existing trained model (objective `squared_error`).
/// * `train` - Continuation arrays: features, f64 targets, optional
///   weights.
/// * `val` - Optional validation arrays for the config's early stopping.
/// * `additional_rounds` - New boosting rounds (>= 1).
/// * `n_jobs` - Worker-thread policy for this run.
///
/// # Returns
///
/// A [`PyGbmModel`] wrapping the NEW combined model; the input model is
/// unchanged.
///
/// # Errors
///
/// Returns `PyErr` on argument extraction, validation, or training errors,
/// including a model whose objective is not single-score.
pub(crate) fn continue_gradient_boosting_regression_rs(
    py: Python<'_>,
    model: &PyGbmModel,
    train: &TrainingArrays<'_, '_, f64>,
    val: &ValidationArrays<'_, '_, f64>,
    additional_rounds: i64,
    n_jobs: i64,
) -> PyResult<Py<PyGbmModel>> {
    let train_rows = propagate_into!(extract_rows(train.x));
    let train_slices: Vec<&[f64]> = train_rows.iter().map(Vec::as_slice).collect();
    let y_train = propagate!(extract_targets(train.y));
    let weights: Option<Vec<f64>> = match train.weight {
        Some(w) => Some(propagate!(extract_targets(w))),
        None => None,
    };

    let val_rows: Option<Vec<Vec<f64>>> = match val.x {
        Some(xv) => Some(propagate_into!(extract_rows(xv))),
        None => None,
    };
    let val_slices: Option<Vec<&[f64]>> = val_rows
        .as_ref()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());
    let y_val: Option<Vec<f64>> = match val.y {
        Some(yv) => Some(propagate!(extract_targets(yv))),
        None => None,
    };
    let val_weights: Option<Vec<f64>> = match val.weight {
        Some(w) => Some(propagate!(extract_targets(w))),
        None => None,
    };
    let validation: Option<ValidationData<'_>> = match (&val_slices, &y_val) {
        (Some(xv), Some(yv)) => Some(ValidationData {
            x: xv,
            y: TrainingLabels::Continuous(yv),
            weight: val_weights.as_deref(),
        }),
        (None, None) => {
            if val_weights.is_some() {
                return Err(missing_val_pair("y_val", "val_sample_weight"));
            }
            None
        }
        (Some(_), None) => return Err(missing_val_pair("y_val", "x_val")),
        (None, Some(_)) => return Err(missing_val_pair("x_val", "y_val")),
    };

    let rounds = propagate!(convert_additional_rounds(additional_rounds));
    let parallelism = propagate_into!(Parallelism::from_n_jobs(n_jobs));
    let continued = propagate_into!(continue_gradient_boosting(
        &model.inner,
        &train_slices,
        TrainingLabels::Continuous(&y_train),
        weights.as_deref(),
        validation,
        rounds,
        &TrainingRuntime {
            parallelism,
            hooks: &Hooks::default(),
        },
    ));
    Py::new(py, PyGbmModel { inner: continued })
}
