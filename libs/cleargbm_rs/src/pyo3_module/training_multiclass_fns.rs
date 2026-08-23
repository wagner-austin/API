//! PyO3 bindings for multiclass training and prediction.
//!
//! Mirrors [`super::training_fns`] for the `multiclass_softmax` objective:
//! labels arrive as numpy i64 class indices and convert to `u32`, and the
//! prediction surface is the multiclass trio (raw score matrix, probability
//! matrix, argmax class vector).

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::hooks::Hooks;
use crate::pyo3_module::array_helpers::try_convert_int;
use crate::pyo3_module::config_extract::{dict_get_i64, extract_config};
use crate::pyo3_module::model_fns::PyGbmModel;
use crate::pyo3_module::training_fns::{
    extract_feature_names, extract_rows, extract_targets, missing_val_pair, TrainingArrays,
    ValidationArrays,
};
use crate::training::{train_gradient_boosting, TrainingLabels, ValidationData};
use crate::training::{Parallelism, TrainingRuntime};

/// Extracts multiclass labels from a numpy i64 array to `Vec<u32>`.
///
/// Range validation against `n_classes` is owned by the core's objective
/// resolution; this rejects only values a `u32` cannot hold.
///
/// # Errors
///
/// Returns `PyErr` if the array is non-contiguous or a value is negative
/// or beyond `u32::MAX`.
pub(super) fn extract_class_labels(labels: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<u32>> {
    let slice = propagate_into!(labels.as_slice());
    let mut result = Vec::with_capacity(slice.len());
    for &val in slice {
        let converted: u32 = propagate_into!(try_convert_int(val, "label"));
        result.push(converted);
    }
    Ok(result)
}

/// Trains a multiclass softmax gradient boosting model from Python data.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `train` - Training arrays: features, i64 class labels, and optional
///   per-row weights (`None` weighs every row 1).
/// * `val` - Optional validation arrays: features, labels, and optional
///   evaluation weights.
/// * `config_dict` - Python dict with training hyperparameters; its
///   `objective` must be `"multiclass_softmax"` with `n_classes` set.
/// * `feature_names` - Python list of feature name strings.
///
/// # Returns
///
/// A [`PyGbmModel`] wrapping the trained model.
///
/// # Errors
///
/// Returns `PyErr` on argument extraction, validation, or training errors.
pub(crate) fn train_gradient_boosting_multiclass_rs(
    py: Python<'_>,
    train: &TrainingArrays<'_, '_, i64>,
    val: &ValidationArrays<'_, '_, i64>,
    config_dict: &Bound<'_, PyDict>,
    feature_names: &Bound<'_, PyList>,
) -> PyResult<Py<PyGbmModel>> {
    let train_rows = propagate_into!(extract_rows(train.x));
    let train_slices: Vec<&[f64]> = train_rows.iter().map(Vec::as_slice).collect();

    let y_train_u32 = propagate!(extract_class_labels(train.y));
    let weights: Option<Vec<f64>> = match train.weight {
        Some(w) => Some(propagate!(extract_targets(w))),
        None => None,
    };

    // Optional validation data: both-or-neither, and a validation weight
    // without a validation split is rejected — the same contract as the
    // other two entries.
    let val_rows: Option<Vec<Vec<f64>>> = match val.x {
        Some(xv) => Some(propagate_into!(extract_rows(xv))),
        None => None,
    };
    let val_slices: Option<Vec<&[f64]>> = val_rows
        .as_ref()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());

    let y_val_u32: Option<Vec<u32>> = match val.y {
        Some(yv) => Some(propagate!(extract_class_labels(yv))),
        None => None,
    };
    let val_weights: Option<Vec<f64>> = match val.weight {
        Some(w) => Some(propagate!(extract_targets(w))),
        None => None,
    };

    let validation: Option<ValidationData<'_>> = match (&val_slices, &y_val_u32) {
        (Some(xv), Some(yv)) => Some(ValidationData {
            x: xv,
            y: TrainingLabels::Multiclass(yv),
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

    let config = propagate!(extract_config(config_dict));
    let names = propagate!(extract_feature_names(feature_names));
    let parallelism = propagate_into!(Parallelism::from_n_jobs(propagate!(dict_get_i64(
        config_dict,
        "n_jobs"
    ))));

    let model = propagate_into!(train_gradient_boosting(
        &train_slices,
        TrainingLabels::Multiclass(&y_train_u32),
        weights.as_deref(),
        validation,
        &config,
        &names,
        &TrainingRuntime {
            parallelism,
            hooks: &Hooks::default(),
        },
    ));

    Py::new(py, PyGbmModel { inner: model })
}

/// Predicts raw per-class scores using a trained multiclass model.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `model` - A trained [`PyGbmModel`].
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
///
/// # Returns
///
/// 2D numpy array (f64) of shape `(n_samples, n_classes)`.
///
/// # Errors
///
/// Returns `PyErr` on validation or prediction errors, including a model
/// not trained under `multiclass_softmax`.
pub(crate) fn predict_raw_multiclass_model_rs<'py>(
    py: Python<'py>,
    model: &PyGbmModel,
    features: &PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = propagate_into!(extract_rows(features));
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let raw = propagate_into!(model.inner.predict_raw_multiclass(&row_slices));
    Ok(propagate_into!(PyArray2::from_vec2(py, &raw)))
}

/// Predicts per-class probabilities using a trained multiclass model.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `model` - A trained [`PyGbmModel`].
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
///
/// # Returns
///
/// 2D numpy array (f64) of shape `(n_samples, n_classes)`; rows sum to 1.
///
/// # Errors
///
/// Returns `PyErr` on validation or prediction errors, including a model
/// not trained under `multiclass_softmax`.
pub(crate) fn predict_proba_multiclass_model_rs<'py>(
    py: Python<'py>,
    model: &PyGbmModel,
    features: &PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = propagate_into!(extract_rows(features));
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let probas = propagate_into!(model.inner.predict_proba_multiclass(&row_slices));
    Ok(propagate_into!(PyArray2::from_vec2(py, &probas)))
}

/// Predicts class labels using a trained multiclass model.
///
/// The argmax over each row's raw scores; ties resolve to the lowest
/// class index.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `model` - A trained [`PyGbmModel`].
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
///
/// # Returns
///
/// 1D numpy array (i64) of class indices.
///
/// # Errors
///
/// Returns `PyErr` on validation or prediction errors, including a model
/// not trained under `multiclass_softmax`.
pub(crate) fn predict_class_model_rs<'py>(
    py: Python<'py>,
    model: &PyGbmModel,
    features: &PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let rows = propagate_into!(extract_rows(features));
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let classes = propagate_into!(model.inner.predict_class(&row_slices));
    // A class index is bounded by the model's per-class base vector length,
    // so it always fits i64 on every supported target; the error arm would
    // be statically dead, hence the saturating conversion (the crate's
    // dead-arm idiom).
    let out: Vec<i64> = classes
        .into_iter()
        .map(|class| i64::try_from(class).unwrap_or(i64::MAX))
        .collect();
    Ok(PyArray1::from_vec(py, out))
}
