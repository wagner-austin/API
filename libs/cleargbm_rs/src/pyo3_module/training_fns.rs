//! PyO3 bindings for gradient boosting training and model inference.
//!
//! Wraps [`crate::training`] functions for calling from Python.
//! The trained model is exposed as a [`PyGbmModel`] opaque class that
//! can be passed back to Rust for prediction without serialization overhead.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::error::ClearGbmError;
use crate::pyo3_module::array_helpers::{i64_to_usize, try_convert_int};
use crate::pyo3_module::prediction_fns::extract_rows;
use crate::split::MonotonicConstraint;
use crate::training::{
    train_gradient_boosting, GradientBoostingConfig, GradientBoostingConfigParams,
    GradientBoostingModel,
};

/// Opaque Python wrapper around a trained [`GradientBoostingModel`].
///
/// Created by [`train_gradient_boosting_rs`] and consumed by
/// [`predict_model_rs`] and [`predict_proba_model_rs`].
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct PyGbmModel {
    /// The underlying trained model.
    inner: GradientBoostingModel,
}

// =============================================================================
// Core wrappers
// =============================================================================

/// Trains a gradient boosting model from Python data.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `x_train` - 2D numpy array (f64) of training features.
/// * `y_train` - 1D numpy array (i64) of binary labels (0/1).
/// * `x_val` - Optional 2D numpy array (f64) of validation features.
/// * `y_val` - Optional 1D numpy array (i64) of validation labels.
/// * `config_dict` - Python dict with training hyperparameters.
/// * `feature_names` - Python list of feature name strings.
///
/// # Returns
///
/// A [`PyGbmModel`] wrapping the trained model.
///
/// # Errors
///
/// Returns `PyErr` on argument extraction, validation, or training errors.
pub(crate) fn train_gradient_boosting_rs(
    py: Python<'_>,
    x_train: &PyReadonlyArray2<'_, f64>,
    y_train: &PyReadonlyArray1<'_, i64>,
    x_val: Option<&PyReadonlyArray2<'_, f64>>,
    y_val: Option<&PyReadonlyArray1<'_, i64>>,
    config_dict: &Bound<'_, PyDict>,
    feature_names: &Bound<'_, PyList>,
) -> PyResult<Py<PyGbmModel>> {
    // Extract training features
    let train_rows = propagate_into!(extract_rows(x_train));
    let train_slices: Vec<&[f64]> = train_rows.iter().map(Vec::as_slice).collect();

    // Extract training labels
    let y_train_u8 = propagate!(extract_labels(y_train));

    // Extract optional validation data
    let val_rows: Option<Vec<Vec<f64>>> = match x_val {
        Some(xv) => Some(propagate_into!(extract_rows(xv))),
        None => None,
    };
    let val_slices: Option<Vec<&[f64]>> = val_rows
        .as_ref()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());

    let y_val_u8: Option<Vec<u8>> = match y_val {
        Some(yv) => Some(propagate!(extract_labels(yv))),
        None => None,
    };

    // Extract config
    let config = propagate!(extract_config(config_dict));

    // Extract feature names
    let names = propagate!(extract_feature_names(feature_names));

    // Call training
    let model = propagate_into!(train_gradient_boosting(
        &train_slices,
        &y_train_u8,
        val_slices.as_deref(),
        y_val_u8.as_deref(),
        &config,
        &names,
    ));

    Py::new(py, PyGbmModel { inner: model })
}

/// Predicts class probabilities using a trained model.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `model` - A trained [`PyGbmModel`].
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
///
/// # Returns
///
/// 2D numpy array (f64) of shape `(n_samples, 2)` with columns
/// `[prob_class_0, prob_class_1]`.
///
/// # Errors
///
/// Returns `PyErr` on validation or prediction errors.
pub(crate) fn predict_proba_model_rs<'py>(
    py: Python<'py>,
    model: &PyGbmModel,
    features: &PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = propagate_into!(extract_rows(features));
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let probas = propagate_into!(model.inner.predict_proba(&row_slices));

    // Convert Vec<(f64, f64)> to 2D array — rows always uniform (length 2)
    let rows_2d: Vec<Vec<f64>> = probas.iter().map(|&(p0, p1)| vec![p0, p1]).collect();
    Ok(propagate_into!(PyArray2::from_vec2(py, &rows_2d)))
}

/// Predicts raw log-odds using a trained model.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `model` - A trained [`PyGbmModel`].
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
///
/// # Returns
///
/// 1D numpy array (f64) of raw predictions (log-odds).
///
/// # Errors
///
/// Returns `PyErr` on validation or prediction errors.
pub(crate) fn predict_raw_model_rs<'py>(
    py: Python<'py>,
    model: &PyGbmModel,
    features: &PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let rows = propagate_into!(extract_rows(features));
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let raw = propagate_into!(model.inner.predict_raw(&row_slices));

    Ok(PyArray1::from_vec(py, raw))
}

// =============================================================================
// Helpers
// =============================================================================

/// Extracts binary labels from a numpy i64 array to `Vec<u8>`.
///
/// # Errors
///
/// Returns `PyErr` if the array is non-contiguous or contains out-of-range values.
fn extract_labels(labels: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<u8>> {
    let slice = propagate_into!(labels.as_slice());
    let mut result = Vec::with_capacity(slice.len());
    for &val in slice {
        let converted: u8 = propagate_into!(try_convert_int(val, "label"));
        result.push(converted);
    }
    Ok(result)
}

/// Extracts a required i64 value from a Python dict.
///
/// # Errors
///
/// Returns `PyErr` if the key is missing or the value is not i64.
fn dict_get_i64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    let opt = propagate!(dict.get_item(key));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: key.to_string(),
                reason: format!("missing required key '{key}'"),
            }
            .into())
        }
    };
    item.extract::<i64>()
}

/// Extracts a required f64 value from a Python dict.
///
/// # Errors
///
/// Returns `PyErr` if the key is missing or the value is not f64.
fn dict_get_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    let opt = propagate!(dict.get_item(key));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: key.to_string(),
                reason: format!("missing required key '{key}'"),
            }
            .into())
        }
    };
    item.extract::<f64>()
}

/// Extracts a `GradientBoostingConfig` from a Python dict.
///
/// # Errors
///
/// Returns `PyErr` if any required key is missing, has wrong type, or validation fails.
fn extract_config(dict: &Bound<'_, PyDict>) -> PyResult<GradientBoostingConfig> {
    let n_estimators = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "n_estimators")),
        "n_estimators"
    ));
    let max_depth = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "max_depth")),
        "max_depth"
    ));
    let learning_rate = propagate!(dict_get_f64(dict, "learning_rate"));
    let min_samples_split = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "min_samples_split")),
        "min_samples_split"
    ));
    let min_samples_leaf = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "min_samples_leaf")),
        "min_samples_leaf"
    ));
    let max_bins = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "max_bins")),
        "max_bins"
    ));
    let subsample = propagate!(dict_get_f64(dict, "subsample"));
    let random_state: u64 = propagate_into!(try_convert_int(
        propagate!(dict_get_i64(dict, "random_state")),
        "random_state"
    ));
    let reg_alpha = propagate!(dict_get_f64(dict, "reg_alpha"));
    let reg_lambda = propagate!(dict_get_f64(dict, "reg_lambda"));
    let monotonic_constraints = propagate!(extract_monotonic_constraints(dict));
    let early_stopping_rounds = propagate!(extract_early_stopping_rounds(dict));

    let params = GradientBoostingConfigParams {
        n_estimators,
        max_depth,
        learning_rate,
        min_samples_split,
        min_samples_leaf,
        max_bins,
        subsample,
        random_state,
        monotonic_constraints,
        reg_alpha,
        reg_lambda,
        early_stopping_rounds,
    };

    Ok(propagate_into!(GradientBoostingConfig::new(params)))
}

/// Extracts optional monotonic constraints from a config dict.
///
/// The key `"monotonic_constraints"` should be `None` or a list of ints
/// where -1 = decreasing, 0 = none, 1 = increasing.
///
/// # Errors
///
/// Returns `PyErr` if the value is present but not a valid list of ints.
fn extract_monotonic_constraints(
    dict: &Bound<'_, PyDict>,
) -> PyResult<Option<Vec<MonotonicConstraint>>> {
    let opt = propagate!(dict.get_item("monotonic_constraints"));
    let item = match opt {
        Some(v) => v,
        None => return Ok(None),
    };

    if item.is_none() {
        return Ok(None);
    }

    let py_list: Bound<'_, PyList> = propagate_into!(item.extract());

    let mut constraints = Vec::with_capacity(py_list.len());
    for i in 0_usize..py_list.len() {
        let val = propagate!(py_list.get_item(i));
        let int_val: i64 = propagate!(val.extract());
        let constraint = match int_val {
            -1_i64 => MonotonicConstraint::Decreasing,
            0_i64 => MonotonicConstraint::None,
            1_i64 => MonotonicConstraint::Increasing,
            other => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "monotonic_constraints".to_string(),
                    reason: format!("invalid value {other}, expected -1, 0, or 1"),
                }
                .into())
            }
        };
        constraints.push(constraint);
    }

    Ok(Some(constraints))
}

/// Extracts optional early stopping rounds from a config dict.
///
/// # Errors
///
/// Returns `PyErr` if the value is present but not a valid int.
fn extract_early_stopping_rounds(dict: &Bound<'_, PyDict>) -> PyResult<Option<usize>> {
    let opt = propagate!(dict.get_item("early_stopping_rounds"));
    let item = match opt {
        Some(v) => v,
        None => return Ok(None),
    };

    if item.is_none() {
        return Ok(None);
    }

    let val: i64 = propagate!(item.extract());
    Ok(Some(propagate_into!(i64_to_usize(
        val,
        "early_stopping_rounds"
    ))))
}

/// Extracts feature names from a Python list of strings.
///
/// # Errors
///
/// Returns `PyErr` if extraction fails.
fn extract_feature_names(names: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
    let mut result = Vec::with_capacity(names.len());
    for i in 0_usize..names.len() {
        let item = propagate!(names.get_item(i));
        let name: String = propagate!(item.extract());
        result.push(name);
    }
    Ok(result)
}

// =============================================================================
// Argument extraction wrappers for PyCFunction::new_closure registration
// =============================================================================

/// Extracts arguments and delegates to [`train_gradient_boosting_rs`].
///
/// # Args (positional)
///
/// 0. `x_train` (numpy f64 2D array) - Training features.
/// 1. `y_train` (numpy i64 1D array) - Training labels.
/// 2. `x_val` (numpy f64 2D array or None) - Optional validation features.
/// 3. `y_val` (numpy i64 1D array or None) - Optional validation labels.
/// 4. `config` (dict) - Training hyperparameters.
/// 5. `feature_names` (list of str) - Feature names.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or training fails.
pub(crate) fn train_gradient_boosting_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let x_train: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let y_train: PyReadonlyArray1<'_, i64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());

    let arg2 = propagate!(args.get_item(2_usize));
    let x_val: Option<PyReadonlyArray2<'_, f64>> = if arg2.is_none() {
        None
    } else {
        Some(propagate_into!(arg2.extract()))
    };

    let arg3 = propagate!(args.get_item(3_usize));
    let y_val: Option<PyReadonlyArray1<'_, i64>> = if arg3.is_none() {
        None
    } else {
        Some(propagate_into!(arg3.extract()))
    };

    let config_dict: Bound<'_, PyDict> =
        propagate_into!(propagate!(args.get_item(4_usize)).extract());
    let feature_names: Bound<'_, PyList> =
        propagate_into!(propagate!(args.get_item(5_usize)).extract());

    let model = propagate!(train_gradient_boosting_rs(
        py,
        &x_train,
        &y_train,
        x_val.as_ref(),
        y_val.as_ref(),
        &config_dict,
        &feature_names,
    ));

    Ok(model.into_any())
}

/// Extracts arguments and delegates to [`predict_proba_model_rs`].
///
/// # Args (positional)
///
/// 0. `model` (PyGbmModel) - Trained model.
/// 1. `features` (numpy f64 2D array) - Feature matrix.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_proba_model_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let model: PyRef<'_, PyGbmModel> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let features: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());

    let result = propagate!(predict_proba_model_rs(py, &model, &features));
    Ok(result.unbind().into_any())
}

/// Extracts arguments and delegates to [`predict_raw_model_rs`].
///
/// # Args (positional)
///
/// 0. `model` (PyGbmModel) - Trained model.
/// 1. `features` (numpy f64 2D array) - Feature matrix.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_raw_model_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let model: PyRef<'_, PyGbmModel> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let features: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());

    let result = propagate!(predict_raw_model_rs(py, &model, &features));
    Ok(result.unbind().into_any())
}
