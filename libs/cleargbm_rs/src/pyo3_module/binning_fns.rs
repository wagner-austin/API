//! PyO3 bindings for feature binning functions.
//!
//! Wraps [`crate::binning`] functions for calling from Python with numpy arrays.
//! Python passes features as 2D f64 numpy arrays; the bindings convert to
//! row-major `Vec<Vec<f64>>` before delegating to the Rust core.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::binning;
use crate::error::ClearGbmError;
use crate::pyo3_module::array_helpers::{i64_to_usize, try_convert_int};
use crate::pyo3_module::prediction_fns::extract_rows;

// =============================================================================
// Core wrappers
// =============================================================================

/// Precomputes feature bins from a 2D feature matrix.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
/// * `max_bins` - Maximum number of bins per feature.
///
/// # Returns
///
/// A tuple of (bin_thresholds, sample_bins, n_regular_bins):
/// - `bin_thresholds`: list of lists of f64 (one per feature)
/// - `sample_bins`: 2D i64 numpy array `[n_samples][n_features]`
/// - `n_regular_bins`: int
///
/// # Errors
///
/// Returns `PyErr` on shape validation or binning errors.
pub(crate) fn precompute_feature_bins_rs<'py>(
    py: Python<'py>,
    features: &PyReadonlyArray2<'py, f64>,
    max_bins: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let rows = propagate_into!(extract_rows(features));
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let fb = propagate_into!(binning::precompute_feature_bins(&row_slices, max_bins));

    // Convert bin_thresholds to Python list of lists
    let thresholds = fb.bin_thresholds();
    let py_thresholds = propagate!(build_thresholds_list(py, &thresholds));

    // Convert sample_bins to 2D i64 numpy array
    let py_bins = propagate!(build_bins_array(py, fb.sample_bins()));

    let n_reg = fb.n_regular_bins();
    let n_reg_py = propagate!(n_reg.into_pyobject(py).map_err(Into::<PyErr>::into)).into_any();

    PyTuple::new(py, [py_thresholds.into_any(), py_bins.into_any(), n_reg_py])
}

/// Computes bin edges for each feature.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
/// * `max_bins` - Maximum number of bins per feature.
///
/// # Returns
///
/// A list of lists of f64 (edge thresholds per feature).
///
/// # Errors
///
/// Returns `PyErr` on shape validation or computation errors.
pub(crate) fn compute_bin_edges_rs<'py>(
    py: Python<'py>,
    features: &PyReadonlyArray2<'py, f64>,
    max_bins: usize,
) -> PyResult<Bound<'py, PyList>> {
    let rows = propagate_into!(extract_rows(features));
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let edges = propagate_into!(binning::compute_bin_edges(&row_slices, max_bins));

    // Convert Vec<BinEdges> to Python list of lists
    let items: Vec<Bound<'py, PyList>> = propagate!(edges
        .iter()
        .map(|be| PyList::new(py, be.edges()))
        .collect::<Result<Vec<_>, _>>());

    PyList::new(py, items)
}

/// Assigns bin indices to samples given precomputed bin edges.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `features` - 2D numpy array (f64) of shape `(n_samples, n_features)`.
/// * `bin_edges` - List of lists of f64 (edge thresholds per feature).
/// * `n_regular_bins` - Number of regular bins (NaN bin is at this index).
///
/// # Returns
///
/// 2D i64 numpy array of shape `(n_samples, n_features)` with bin indices.
///
/// # Errors
///
/// Returns `PyErr` on shape validation or assignment errors.
pub(crate) fn bin_samples_rs<'py>(
    py: Python<'py>,
    features: &PyReadonlyArray2<'py, f64>,
    bin_edges: &Bound<'py, PyList>,
    n_regular_bins: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let rows = propagate_into!(extract_rows(features));
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Extract bin edges from Python list of lists
    let edges_vec = propagate!(extract_bin_edges(bin_edges));

    let sample_bins = propagate_into!(binning::bin_samples(
        &row_slices,
        &edges_vec,
        n_regular_bins
    ));

    build_bins_array(py, &sample_bins)
}

// =============================================================================
// Helpers
// =============================================================================

/// Builds a Python list of lists from bin thresholds.
///
/// # Errors
///
/// Returns `PyErr` if list creation fails.
pub(crate) fn build_thresholds_list<'py>(
    py: Python<'py>,
    thresholds: &[Vec<f64>],
) -> PyResult<Bound<'py, PyList>> {
    let items: Vec<Bound<'py, PyList>> = propagate!(thresholds
        .iter()
        .map(|t| PyList::new(py, t))
        .collect::<Result<Vec<_>, _>>());
    PyList::new(py, items)
}

/// Converts sample bins `[n_samples][n_features]` (usize) to a 2D i64 numpy array.
///
/// # Errors
///
/// Returns `PyErr` if integer conversion or array creation fails.
pub(crate) fn build_bins_array<'py>(
    py: Python<'py>,
    sample_bins: &[Vec<usize>],
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let n_rows = sample_bins.len();
    if n_rows == 0_usize {
        return Ok(propagate_into!(PyArray2::from_vec2(
            py,
            &Vec::<Vec<i64>>::new()
        )));
    }
    let n_cols = sample_bins[0_usize].len();

    let mut rows_i64: Vec<Vec<i64>> = Vec::with_capacity(n_rows);
    for row in sample_bins {
        let mut row_i64 = Vec::with_capacity(n_cols);
        for &val in row {
            let converted: i64 = propagate_into!(try_convert_int(val, "sample bin index"));
            row_i64.push(converted);
        }
        rows_i64.push(row_i64);
    }

    match PyArray2::from_vec2(py, &rows_i64) {
        Ok(a) => Ok(a),
        Err(e) => Err(ClearGbmError::ShapeMismatch {
            expected: "uniform row lengths".to_string(),
            got: format!("{e}"),
        }
        .into()),
    }
}

/// Extracts bin edges from a Python list of lists of f64.
///
/// # Errors
///
/// Returns `PyErr` if extraction fails.
fn extract_bin_edges(edges_list: &Bound<'_, PyList>) -> PyResult<Vec<binning::BinEdges>> {
    let mut result = Vec::with_capacity(edges_list.len());
    for i in 0_usize..edges_list.len() {
        let item = propagate!(edges_list.get_item(i));
        let edge_vec: Vec<f64> = propagate!(item.extract());
        let bin_edge = propagate_into!(binning::BinEdges::new(edge_vec));
        result.push(bin_edge);
    }
    Ok(result)
}

// =============================================================================
// Argument extraction wrappers for PyCFunction::new_closure registration
// =============================================================================

/// Extracts arguments and delegates to [`precompute_feature_bins_rs`].
///
/// # Args (positional)
///
/// 0. `features` (numpy f64 2D array) - Feature matrix.
/// 1. `max_bins` (int) - Maximum bins per feature.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or computation fails.
pub(crate) fn precompute_feature_bins_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = propagate!(args.get_item(0_usize));
    let features: PyReadonlyArray2<'_, f64> = propagate_into!(arg0.extract());

    let arg1 = propagate!(args.get_item(1_usize));
    let max_bins_i64: i64 = propagate!(arg1.extract());
    let max_bins = propagate_into!(i64_to_usize(max_bins_i64, "max_bins"));

    let result = propagate!(precompute_feature_bins_rs(py, &features, max_bins));
    Ok(result.unbind().into_any())
}

/// Extracts arguments and delegates to [`compute_bin_edges_rs`].
///
/// # Args (positional)
///
/// 0. `features` (numpy f64 2D array) - Feature matrix.
/// 1. `max_bins` (int) - Maximum bins per feature.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or computation fails.
pub(crate) fn compute_bin_edges_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = propagate!(args.get_item(0_usize));
    let features: PyReadonlyArray2<'_, f64> = propagate_into!(arg0.extract());

    let arg1 = propagate!(args.get_item(1_usize));
    let max_bins_i64: i64 = propagate!(arg1.extract());
    let max_bins = propagate_into!(i64_to_usize(max_bins_i64, "max_bins"));

    let result = propagate!(compute_bin_edges_rs(py, &features, max_bins));
    Ok(result.unbind().into_any())
}

/// Extracts arguments and delegates to [`bin_samples_rs`].
///
/// # Args (positional)
///
/// 0. `features` (numpy f64 2D array) - Feature matrix.
/// 1. `bin_edges` (list of lists of f64) - Edge thresholds per feature.
/// 2. `n_regular_bins` (int) - Number of regular bins.
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or computation fails.
pub(crate) fn bin_samples_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = propagate!(args.get_item(0_usize));
    let features: PyReadonlyArray2<'_, f64> = propagate_into!(arg0.extract());

    let arg1 = propagate!(args.get_item(1_usize));
    let bin_edges: Bound<'_, PyList> = propagate_into!(arg1.extract());

    let arg2 = propagate!(args.get_item(2_usize));
    let n_regular_bins_i64: i64 = propagate!(arg2.extract());
    let n_regular_bins = propagate_into!(i64_to_usize(n_regular_bins_i64, "n_regular_bins"));

    let result = propagate!(bin_samples_rs(py, &features, &bin_edges, n_regular_bins));
    Ok(result.unbind().into_any())
}
