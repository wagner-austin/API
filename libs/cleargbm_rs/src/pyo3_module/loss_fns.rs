//! PyO3 bindings for loss functions.
//!
//! Wraps [`crate::losses`] functions for calling from Python with numpy arrays.
//! Python passes labels as `i64` numpy arrays; the bindings convert to `u8`
//! with validation before delegating to the Rust core.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::ClearGbmError;
use crate::losses;
use crate::pyo3_module::array_helpers::convert_int_slice;

/// Converts a Python `i64` labels array to `Vec<u8>`.
///
/// # Args
///
/// * `labels` - Readonly numpy array of `i64` labels from Python.
///
/// # Errors
///
/// Returns `PyErr` if any label is outside `u8` range (0..=255).
/// Downstream loss functions further validate that labels are exactly 0 or 1.
fn extract_labels_u8(labels: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<u8>> {
    let slice = match labels.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("y_true: {e}"),
            }
            .into())
        }
    };
    match convert_int_slice::<i64, u8>(slice, "y_true label") {
        Ok(v) => Ok(v),
        Err(e) => Err(e.into()),
    }
}

/// Extracts a contiguous `f64` slice from a numpy array.
///
/// # Args
///
/// * `arr` - Readonly numpy array of `f64` values.
/// * `name` - Parameter name for error messages.
///
/// # Errors
///
/// Returns `PyErr` if the array is non-contiguous.
fn extract_f64_slice<'a>(arr: &'a PyReadonlyArray1<'a, f64>, name: &str) -> PyResult<&'a [f64]> {
    match arr.as_slice() {
        Ok(s) => Ok(s),
        Err(e) => Err(ClearGbmError::EmptyInput {
            context: format!("{name}: {e}"),
        }
        .into()),
    }
}

// =============================================================================
// Core wrappers
// =============================================================================

/// Computes mean binary cross-entropy (log loss).
///
/// # Args
///
/// * `y_true` - Labels as `u8` (0 or 1).
/// * `y_pred` - Predicted probabilities.
///
/// # Errors
///
/// Returns `PyErr` on validation failure.
pub(crate) fn binary_log_loss_rs(y_true: &[u8], y_pred: &[f64]) -> PyResult<f64> {
    match losses::binary_log_loss(y_true, y_pred) {
        Ok(v) => Ok(v),
        Err(e) => Err(e.into()),
    }
}

/// Computes gradients of binary log loss (p - y).
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `y_true` - Labels as `u8` (0 or 1).
/// * `y_pred` - Predicted probabilities.
///
/// # Errors
///
/// Returns `PyErr` on validation failure.
pub(crate) fn binary_log_loss_gradients_rs<'py>(
    py: Python<'py>,
    y_true: &[u8],
    y_pred: &[f64],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    match losses::binary_log_loss_gradients(y_true, y_pred) {
        Ok(v) => Ok(PyArray1::from_vec(py, v)),
        Err(e) => Err(e.into()),
    }
}

/// Computes hessians of binary log loss (p * (1-p)).
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `y_true` - Labels as `u8` (0 or 1).
/// * `y_pred` - Predicted probabilities.
///
/// # Errors
///
/// Returns `PyErr` on validation failure.
pub(crate) fn binary_log_loss_hessians_rs<'py>(
    py: Python<'py>,
    y_true: &[u8],
    y_pred: &[f64],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    match losses::binary_log_loss_hessians(y_true, y_pred) {
        Ok(v) => Ok(PyArray1::from_vec(py, v)),
        Err(e) => Err(e.into()),
    }
}

/// Computes initial prediction (log-odds of positive class rate).
///
/// # Args
///
/// * `y_true` - Labels as `u8` (0 or 1).
///
/// # Errors
///
/// Returns `PyErr` on validation failure.
pub(crate) fn binary_log_loss_initial_prediction_rs(y_true: &[u8]) -> PyResult<f64> {
    match losses::binary_log_loss_initial_prediction(y_true) {
        Ok(v) => Ok(v),
        Err(e) => Err(e.into()),
    }
}

/// Applies sigmoid to each element of a numpy array.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `x` - Input values (typically log-odds).
///
/// # Returns
///
/// Numpy array of probabilities.
pub(crate) fn sigmoid_array_rs<'py>(py: Python<'py>, x: &[f64]) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, losses::sigmoid_array(x))
}

// =============================================================================
// Argument extraction wrappers for PyCFunction::new_closure registration
// =============================================================================

/// Extracts arguments and delegates to [`binary_log_loss_rs`].
///
/// # Args (positional)
///
/// 0. `y_true` (numpy i64 array) - Binary labels.
/// 1. `y_pred` (numpy f64 array) - Predicted probabilities.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or computation fails.
pub(crate) fn binary_log_loss_from_args(args: &Bound<'_, PyTuple>) -> PyResult<f64> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let y_true_np: PyReadonlyArray1<'_, i64> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let y_true = match extract_labels_u8(&y_true_np) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg1 = match args.get_item(1_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let y_pred_np: PyReadonlyArray1<'_, f64> = match arg1.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let y_pred = match extract_f64_slice(&y_pred_np, "y_pred") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    binary_log_loss_rs(&y_true, y_pred)
}

/// Extracts arguments and delegates to [`binary_log_loss_gradients_rs`].
///
/// # Args (positional)
///
/// 0. `y_true` (numpy i64 array) - Binary labels.
/// 1. `y_pred` (numpy f64 array) - Predicted probabilities.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or computation fails.
pub(crate) fn binary_log_loss_gradients_from_args(
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let y_true_np: PyReadonlyArray1<'_, i64> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let y_true = match extract_labels_u8(&y_true_np) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg1 = match args.get_item(1_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let y_pred_np: PyReadonlyArray1<'_, f64> = match arg1.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let y_pred = match extract_f64_slice(&y_pred_np, "y_pred") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let result = match binary_log_loss_gradients_rs(py, &y_true, y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok(result.unbind().into_any())
}

/// Extracts arguments and delegates to [`binary_log_loss_hessians_rs`].
///
/// # Args (positional)
///
/// 0. `y_true` (numpy i64 array) - Binary labels.
/// 1. `y_pred` (numpy f64 array) - Predicted probabilities.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or computation fails.
pub(crate) fn binary_log_loss_hessians_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let y_true_np: PyReadonlyArray1<'_, i64> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let y_true = match extract_labels_u8(&y_true_np) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg1 = match args.get_item(1_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let y_pred_np: PyReadonlyArray1<'_, f64> = match arg1.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let y_pred = match extract_f64_slice(&y_pred_np, "y_pred") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let result = match binary_log_loss_hessians_rs(py, &y_true, y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok(result.unbind().into_any())
}

/// Extracts arguments and delegates to [`binary_log_loss_initial_prediction_rs`].
///
/// # Args (positional)
///
/// 0. `y_true` (numpy i64 array) - Binary labels.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or computation fails.
pub(crate) fn binary_log_loss_initial_prediction_from_args(
    args: &Bound<'_, PyTuple>,
) -> PyResult<f64> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let y_true_np: PyReadonlyArray1<'_, i64> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let y_true = match extract_labels_u8(&y_true_np) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    binary_log_loss_initial_prediction_rs(&y_true)
}

/// Extracts arguments and delegates to [`sigmoid_array_rs`].
///
/// # Args (positional)
///
/// 0. `x` (numpy f64 array) - Input values.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction fails.
pub(crate) fn sigmoid_array_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let x_np: PyReadonlyArray1<'_, f64> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let x = match extract_f64_slice(&x_np, "x") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    Ok(sigmoid_array_rs(py, x).unbind().into_any())
}
