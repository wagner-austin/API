//! PyO3 bindings for gradient boosting training and prediction.
//!
//! Wraps [`crate::training`] functions for calling from Python. Two training
//! entry points exist, one per label kind: `train_gradient_boosting_rs`
//! takes integer 0/1 labels for the `binary_log_loss` objective, and
//! `train_gradient_boosting_regression_rs` takes float targets for
//! `squared_error`. The config's `objective` must agree with the entry
//! called — the core's objective resolution rejects the mismatch.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::pyo3_module::array_helpers::try_convert_int;
use crate::pyo3_module::config_extract::{dict_get_i64, extract_config};
use crate::pyo3_module::model_fns::PyGbmModel;
use crate::training::{train_gradient_boosting, TrainingLabels, ValidationData};
use crate::training::{Parallelism, TrainingRuntime};

/// Extracts a 2D numpy feature matrix into a `Vec<Vec<f64>>` for row-wise access.
///
/// # Args
///
/// * `features` - 2D numpy readonly view of the feature matrix.
///
/// # Returns
///
/// A vector of row-vectors, each `n_cols` long.
///
/// # Errors
///
/// Returns [`ClearGbmError::EmptyInput`] if the matrix has zero rows.
fn extract_rows(features: &PyReadonlyArray2<'_, f64>) -> Result<Vec<Vec<f64>>, ClearGbmError> {
    let shape = features.shape();
    let n_rows = shape[0_usize];
    let n_cols = shape[1_usize];

    if n_rows == 0_usize {
        return Err(ClearGbmError::EmptyInput {
            context: "features matrix has zero rows".to_string(),
        });
    }

    let array = features.as_array();
    let mut rows = Vec::with_capacity(n_rows);

    for row_idx in 0_usize..n_rows {
        let mut row = Vec::with_capacity(n_cols);
        for col_idx in 0_usize..n_cols {
            row.push(array[[row_idx, col_idx]]);
        }
        rows.push(row);
    }

    Ok(rows)
}

// =============================================================================
// Core wrappers
// =============================================================================

/// Trains a binary-classification gradient boosting model from Python data.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `x_train` - 2D numpy array (f64) of training features.
/// * `y_train` - 1D numpy array (i64) of binary labels (0/1).
/// * `x_val` - Optional 2D numpy array (f64) of validation features.
/// * `y_val` - Optional 1D numpy array (i64) of validation labels.
/// * `config_dict` - Python dict with training hyperparameters; its
///   `objective` must be `"binary_log_loss"`.
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

    // Extract optional validation data (both-or-neither: the typed core
    // takes features and labels as one value, so the pairing is checked
    // here, where they arrive as separate arguments).
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

    let validation: Option<ValidationData<'_>> = match (&val_slices, &y_val_u8) {
        (Some(xv), Some(yv)) => Some(ValidationData {
            x: xv,
            y: TrainingLabels::Binary(yv),
        }),
        (None, None) => None,
        (Some(_), None) => return Err(missing_val_pair("y_val", "x_val")),
        (None, Some(_)) => return Err(missing_val_pair("x_val", "y_val")),
    };

    // Extract config
    let config = propagate!(extract_config(config_dict));

    // Extract feature names
    let names = propagate!(extract_feature_names(feature_names));

    // Extract the worker-thread policy. Read from the same dict as the
    // hyperparameters but kept out of `GradientBoostingConfig`: it does not
    // affect the fitted model and must not be persisted with it.
    let parallelism = propagate_into!(Parallelism::from_n_jobs(propagate!(dict_get_i64(
        config_dict,
        "n_jobs"
    ))));

    // Call training
    let model = propagate_into!(train_gradient_boosting(
        &train_slices,
        TrainingLabels::Binary(&y_train_u8),
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

/// Trains a squared-error regression gradient boosting model from Python data.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `x_train` - 2D numpy array (f64) of training features.
/// * `y_train` - 1D numpy array (f64) of continuous targets.
/// * `x_val` - Optional 2D numpy array (f64) of validation features.
/// * `y_val` - Optional 1D numpy array (f64) of validation targets.
/// * `config_dict` - Python dict with training hyperparameters; its
///   `objective` must be `"squared_error"`.
/// * `feature_names` - Python list of feature name strings.
///
/// # Returns
///
/// A [`PyGbmModel`] wrapping the trained model.
///
/// # Errors
///
/// Returns `PyErr` on argument extraction, validation, or training errors.
pub(crate) fn train_gradient_boosting_regression_rs(
    py: Python<'_>,
    x_train: &PyReadonlyArray2<'_, f64>,
    y_train: &PyReadonlyArray1<'_, f64>,
    x_val: Option<&PyReadonlyArray2<'_, f64>>,
    y_val: Option<&PyReadonlyArray1<'_, f64>>,
    config_dict: &Bound<'_, PyDict>,
    feature_names: &Bound<'_, PyList>,
) -> PyResult<Py<PyGbmModel>> {
    // Extract training features
    let train_rows = propagate_into!(extract_rows(x_train));
    let train_slices: Vec<&[f64]> = train_rows.iter().map(Vec::as_slice).collect();

    // Extract training targets
    let y_train_f64 = propagate!(extract_targets(y_train));

    // Extract optional validation data (both-or-neither, as in the binary
    // entry).
    let val_rows: Option<Vec<Vec<f64>>> = match x_val {
        Some(xv) => Some(propagate_into!(extract_rows(xv))),
        None => None,
    };
    let val_slices: Option<Vec<&[f64]>> = val_rows
        .as_ref()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());

    let y_val_f64: Option<Vec<f64>> = match y_val {
        Some(yv) => Some(propagate!(extract_targets(yv))),
        None => None,
    };

    let validation: Option<ValidationData<'_>> = match (&val_slices, &y_val_f64) {
        (Some(xv), Some(yv)) => Some(ValidationData {
            x: xv,
            y: TrainingLabels::Continuous(yv),
        }),
        (None, None) => None,
        (Some(_), None) => return Err(missing_val_pair("y_val", "x_val")),
        (None, Some(_)) => return Err(missing_val_pair("x_val", "y_val")),
    };

    // Extract config
    let config = propagate!(extract_config(config_dict));

    // Extract feature names
    let names = propagate!(extract_feature_names(feature_names));

    // Extract the worker-thread policy (same contract as the binary entry).
    let parallelism = propagate_into!(Parallelism::from_n_jobs(propagate!(dict_get_i64(
        config_dict,
        "n_jobs"
    ))));

    // Call training
    let model = propagate_into!(train_gradient_boosting(
        &train_slices,
        TrainingLabels::Continuous(&y_train_f64),
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
/// Returns `PyErr` on validation or prediction errors, including a model
/// trained under `squared_error` (probabilities do not exist for it).
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

/// Predicts raw scores using a trained model.
///
/// Under `binary_log_loss` the raw score is a log-odds; under
/// `squared_error` it is the prediction itself — this is the regression
/// inference function.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `model` - A trained [`PyGbmModel`].
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
///
/// # Returns
///
/// 1D numpy array (f64) of raw predictions.
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

/// Builds the both-or-neither validation-pairing error.
///
/// # Args
///
/// * `missing` - The absent argument.
/// * `present` - The provided argument that requires it.
fn missing_val_pair(missing: &str, present: &str) -> PyErr {
    ClearGbmError::InvalidParameter {
        name: missing.to_string(),
        reason: format!("{missing} must be provided when {present} is provided"),
    }
    .into()
}

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

/// Extracts continuous regression targets from a numpy f64 array.
///
/// Finiteness is validated by the core's objective resolution, which owns
/// label semantics; this only materializes the values.
///
/// # Errors
///
/// Returns `PyErr` if the array is non-contiguous.
fn extract_targets(targets: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    let slice = propagate_into!(targets.as_slice());
    Ok(slice.to_vec())
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

/// Extracts arguments and delegates to
/// [`train_gradient_boosting_regression_rs`].
///
/// # Args (positional)
///
/// 0. `x_train` (numpy f64 2D array) - Training features.
/// 1. `y_train` (numpy f64 1D array) - Continuous training targets.
/// 2. `x_val` (numpy f64 2D array or None) - Optional validation features.
/// 3. `y_val` (numpy f64 1D array or None) - Optional validation targets.
/// 4. `config` (dict) - Training hyperparameters.
/// 5. `feature_names` (list of str) - Feature names.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or training fails.
pub(crate) fn train_gradient_boosting_regression_from_args(
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let x_train: PyReadonlyArray2<'_, f64> =
        propagate_into!(propagate!(args.get_item(0_usize)).extract());
    let y_train: PyReadonlyArray1<'_, f64> =
        propagate_into!(propagate!(args.get_item(1_usize)).extract());

    let arg2 = propagate!(args.get_item(2_usize));
    let x_val: Option<PyReadonlyArray2<'_, f64>> = if arg2.is_none() {
        None
    } else {
        Some(propagate_into!(arg2.extract()))
    };

    let arg3 = propagate!(args.get_item(3_usize));
    let y_val: Option<PyReadonlyArray1<'_, f64>> = if arg3.is_none() {
        None
    } else {
        Some(propagate_into!(arg3.extract()))
    };

    let config_dict: Bound<'_, PyDict> =
        propagate_into!(propagate!(args.get_item(4_usize)).extract());
    let feature_names: Bound<'_, PyList> =
        propagate_into!(propagate!(args.get_item(5_usize)).extract());

    let model = propagate!(train_gradient_boosting_regression_rs(
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
