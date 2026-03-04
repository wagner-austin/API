//! PyO3 bindings for prediction and inference functions.
//!
//! Wraps [`crate::predict`] functions for calling from Python with numpy arrays.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::ClearGbmError;
use crate::predict;
use crate::pyo3_module::tree_fns::{validate_py_tree, PyTree};

/// Computes the sigmoid (logistic) function with numerical stability.
///
/// Clips the input to `[-500.0, 500.0]` to prevent overflow.
///
/// # Args
///
/// * `x` - Input value (typically log-odds).
///
/// # Returns
///
/// A probability in the range `(0.0, 1.0)`.
/// Core sigmoid computation (called by closure registration).
#[must_use]
pub(crate) fn sigmoid_rs(x: f64) -> f64 {
    predict::sigmoid(x)
}

/// Predicts the leaf value for a single sample by traversing a decision tree.
///
/// # Args
///
/// * `tree` - The decision tree (PyTree).
/// * `features` - 1D numpy array (f64) of feature values for a single sample.
///
/// # Returns
///
/// The leaf prediction value.
///
/// # Errors
///
/// Returns `IndexError` for missing nodes or out-of-bounds features.
/// Returns `RuntimeError` for malformed trees.
pub(crate) fn predict_single_rs(
    tree: &PyTree,
    features: &PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let inner_tree = match validate_py_tree(tree) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    let feat_slice = match features.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("features: {e}"),
            }
            .into())
        }
    };

    match predict::predict_single(inner_tree, feat_slice) {
        Ok(v) => Ok(v),
        Err(e) => Err(e.into()),
    }
}

/// Predicts leaf values for a batch of samples using a single tree.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `tree` - The decision tree (PyTree).
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
///
/// # Returns
///
/// 1D numpy array (f64) of predictions, one per sample.
///
/// # Errors
///
/// Returns `ValueError` for empty inputs.
/// Returns `IndexError` for out-of-bounds features.
/// Returns `RuntimeError` for malformed trees.
pub(crate) fn predict_tree_rs<'py>(
    py: Python<'py>,
    tree: &PyTree,
    features: &PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let inner_tree = match validate_py_tree(tree) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    let rows = match extract_rows(features) {
        Ok(r) => r,
        Err(e) => return Err(e.into()),
    };

    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let predictions = match predict::predict_tree(inner_tree, &row_slices) {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    Ok(PyArray1::from_vec(py, predictions))
}

/// Predicts raw scores for a batch of samples using an ensemble of trees.
///
/// Computes: `raw_pred[i] = base_prediction + learning_rate * sum(tree_j.predict(features[i]))`
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `trees` - List of `PyTree` objects.
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
/// * `base_prediction` - Initial prediction before any tree contributions.
/// * `learning_rate` - Shrinkage factor in `(0.0, 1.0]`.
///
/// # Returns
///
/// 1D numpy array (f64) of raw predictions (log-odds for classification).
///
/// # Errors
///
/// Returns `ValueError` for empty inputs or invalid parameters.
/// Returns `IndexError` for out-of-bounds features.
/// Returns `RuntimeError` for malformed trees.
pub(crate) fn predict_ensemble_rs<'py>(
    py: Python<'py>,
    trees: &[PyRef<'py, PyTree>],
    features: &PyReadonlyArray2<'py, f64>,
    base_prediction: f64,
    learning_rate: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let config = match predict::PredictEnsembleConfig::new(base_prediction, learning_rate) {
        Ok(c) => c,
        Err(e) => return Err(e.into()),
    };

    // Collect inner trees
    let inner_trees: Vec<&crate::tree::Tree> = trees.iter().map(|t| &t.inner).collect();

    let rows = match extract_rows(features) {
        Ok(r) => r,
        Err(e) => return Err(e.into()),
    };

    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Build a slice of Trees (predict_ensemble takes &[Tree])
    // We need owned Trees since predict_ensemble takes &[Tree]
    let owned_trees: Vec<crate::tree::Tree> = inner_trees.iter().map(|t| (*t).clone()).collect();

    let predictions = match predict::predict_ensemble(&owned_trees, &row_slices, &config) {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    Ok(PyArray1::from_vec(py, predictions))
}

/// Converts raw predictions to binary class probabilities using sigmoid.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `raw_predictions` - 1D numpy array (f64) of raw predictions (log-odds).
///
/// # Returns
///
/// 2D numpy array (f64) of shape `(n_samples, 2)` with columns
/// `[prob_class_0, prob_class_1]`.
///
/// # Errors
///
/// Returns `ValueError` if the input array is non-contiguous.
pub(crate) fn predict_proba_rs<'py>(
    py: Python<'py>,
    raw_predictions: &PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raw_slice = match raw_predictions.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("raw_predictions: {e}"),
            }
            .into())
        }
    };

    let proba = predict::predict_proba(raw_slice);

    // Convert Vec<(f64, f64)> to Vec<Vec<f64>> for PyArray2
    let rows: Vec<Vec<f64>> = proba.iter().map(|&(p0, p1)| vec![p0, p1]).collect();

    PyArray2::from_vec2(py, &rows).map_err(|e| shape_err(e.to_string()))
}

/// Converts a shape mismatch description into a [`PyErr`].
///
/// # Args
///
/// * `got` - Human-readable description of the actual shape.
///
/// # Returns
///
/// A Python `ValueError` wrapping the shape mismatch.
pub(crate) fn shape_err(got: String) -> PyErr {
    ClearGbmError::ShapeMismatch {
        expected: "uniform row lengths".to_string(),
        got,
    }
    .into()
}

/// Extracts rows from a 2D numpy array into `Vec<Vec<f64>>`.
///
/// # Args
///
/// * `features` - 2D numpy array of shape `(n_samples, n_features)`.
///
/// # Returns
///
/// A vector of feature vectors, one per sample.
///
/// # Errors
///
/// Returns [`ClearGbmError::EmptyInput`] if the array has zero rows.
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
// Argument extraction wrappers for PyCFunction::new_closure registration
// =============================================================================

/// Extracts arguments from a Python tuple and delegates to [`sigmoid_rs`].
///
/// # Args (positional)
///
/// 0. `x` (f64) - Input value.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction fails.
pub(crate) fn sigmoid_from_args(args: &Bound<'_, PyTuple>) -> PyResult<f64> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let x: f64 = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok(sigmoid_rs(x))
}

/// Extracts arguments from a Python tuple and delegates to [`predict_single_rs`].
///
/// # Args (positional)
///
/// 0. `tree` (PyTree) - The decision tree.
/// 1. `features` (numpy array f64) - Feature values for a single sample.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_single_from_args(args: &Bound<'_, PyTuple>) -> PyResult<f64> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let tree: PyRef<'_, PyTree> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg1 = match args.get_item(1_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let features: PyReadonlyArray1<'_, f64> = match arg1.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    predict_single_rs(&tree, &features)
}

/// Extracts arguments from a Python tuple and delegates to [`predict_tree_rs`].
///
/// Returns a [`PyObject`] (unbounded) so the closure in `mod.rs` can return
/// a lifetime-free type as required by [`PyCFunction::new_closure`].
///
/// # Args (positional)
///
/// 0. `tree` (PyTree) - The decision tree.
/// 1. `features` (numpy 2D array f64) - Feature matrix.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_tree_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let tree: PyRef<'_, PyTree> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg1 = match args.get_item(1_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let features: PyReadonlyArray2<'_, f64> = match arg1.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let result = match predict_tree_rs(py, &tree, &features) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok(result.unbind().into_any())
}

/// Extracts arguments from a Python tuple and delegates to [`predict_ensemble_rs`].
///
/// Returns a [`PyObject`] (unbounded) for closure compatibility.
///
/// # Args (positional)
///
/// 0. `trees` (list of PyTree) - Ensemble of decision trees.
/// 1. `features` (numpy 2D array f64) - Feature matrix.
/// 2. `base_prediction` (f64) - Initial prediction.
/// 3. `learning_rate` (f64) - Shrinkage factor.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or prediction fails.
pub(crate) fn predict_ensemble_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let trees: Vec<PyRef<'_, PyTree>> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg1 = match args.get_item(1_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let features: PyReadonlyArray2<'_, f64> = match arg1.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg2 = match args.get_item(2_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let base_prediction: f64 = match arg2.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg3 = match args.get_item(3_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let learning_rate: f64 = match arg3.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let result = match predict_ensemble_rs(py, &trees, &features, base_prediction, learning_rate) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok(result.unbind().into_any())
}

/// Extracts arguments from a Python tuple and delegates to [`predict_proba_rs`].
///
/// Returns a [`PyObject`] (unbounded) for closure compatibility.
///
/// # Args (positional)
///
/// 0. `raw_predictions` (numpy array f64) - Raw prediction scores.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or conversion fails.
pub(crate) fn predict_proba_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let raw_predictions: PyReadonlyArray1<'_, f64> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let result = match predict_proba_rs(py, &raw_predictions) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok(result.unbind().into_any())
}
