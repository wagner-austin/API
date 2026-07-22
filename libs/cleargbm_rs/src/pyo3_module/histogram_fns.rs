//! PyO3 bindings for histogram operations.
//!
//! Wraps [`crate::histogram::build_histogram`] and [`crate::histogram::subtract_histogram`]
//! for calling from Python with numpy arrays.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::ClearGbmError;
use crate::histogram;
use crate::pyo3_module::array_helpers::{
    i64_slice_to_u8_vec, i64_slice_to_usize_vec, i64_to_usize, u64_slice_to_usize_vec,
    usize_slice_to_u64_vec,
};
use crate::types::HistogramBuffer;

/// Return type for histogram functions: (gradient_sums, hessian_sums, counts).
type HistogramArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u64>>,
);

/// Builds a histogram from sample gradients and hessians.
///
/// This is the core O(n) operation that accumulates gradient statistics
/// into bins for split finding.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `sample_indices` - 1D numpy array (i64) of sample indices at this node.
/// * `gradients` - 1D numpy array (f64) of gradient values for all samples.
/// * `hessians` - 1D numpy array (f64) of hessian values for all samples.
/// * `bins` - 1D numpy array (i64) of pre-computed bin assignments for this feature.
/// * `n_bins` - Number of bins (including NaN bin).
///
/// # Returns
///
/// Tuple of (gradient_sums, hessian_sums, counts) as numpy arrays.
///
/// # Errors
///
/// Returns `ValueError` for empty/mismatched inputs.
/// Returns `IndexError` for out-of-bounds indices.
pub(crate) fn build_histogram_rs<'py>(
    py: Python<'py>,
    sample_indices: &PyReadonlyArray1<'py, i64>,
    gradients: &PyReadonlyArray1<'py, f64>,
    hessians: &PyReadonlyArray1<'py, f64>,
    bins: &PyReadonlyArray1<'py, i64>,
    n_bins: i64,
) -> PyResult<HistogramArrays<'py>> {
    // Convert i64 inputs to usize
    let idx_slice = match sample_indices.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("sample_indices: {e}"),
            }
            .into())
        }
    };
    let sample_idx = match i64_slice_to_usize_vec(idx_slice, "sample_indices") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let grad_slice = match gradients.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("gradients: {e}"),
            }
            .into())
        }
    };

    let hess_slice = match hessians.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("hessians: {e}"),
            }
            .into())
        }
    };

    let bins_i64_slice = match bins.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("bins: {e}"),
            }
            .into())
        }
    };
    let bins_u8 = match i64_slice_to_u8_vec(bins_i64_slice, "bins") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let n_bins_usize = match i64_to_usize(n_bins, "n_bins") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    // Call the Rust core function
    let result = match histogram::build_histogram(
        &sample_idx,
        grad_slice,
        hess_slice,
        &bins_u8,
        n_bins_usize,
    ) {
        Ok(h) => h,
        Err(e) => return Err(e.into()),
    };

    // Convert output to numpy arrays
    histogram_buffer_to_numpy(py, &result)
}

/// Computes sibling histogram by subtraction (parent - child).
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `parent_grads` - 1D numpy array (f64) of parent gradient sums per bin.
/// * `parent_hess` - 1D numpy array (f64) of parent hessian sums per bin.
/// * `parent_counts` - 1D numpy array (u64) of parent sample counts per bin.
/// * `child_grads` - 1D numpy array (f64) of child gradient sums per bin.
/// * `child_hess` - 1D numpy array (f64) of child hessian sums per bin.
/// * `child_counts` - 1D numpy array (u64) of child sample counts per bin.
///
/// # Returns
///
/// Tuple of (gradient_sums, hessian_sums, counts) for the sibling as numpy arrays.
///
/// # Errors
///
/// Returns `ValueError` if parent and child histogram sizes don't match.
pub(crate) fn subtract_histogram_rs<'py>(
    py: Python<'py>,
    parent_grads: &PyReadonlyArray1<'py, f64>,
    parent_hess: &PyReadonlyArray1<'py, f64>,
    parent_counts: &PyReadonlyArray1<'py, u64>,
    child_grads: &PyReadonlyArray1<'py, f64>,
    child_hess: &PyReadonlyArray1<'py, f64>,
    child_counts: &PyReadonlyArray1<'py, u64>,
) -> PyResult<HistogramArrays<'py>> {
    // Build parent HistogramBuffer from numpy arrays
    let parent = match build_histogram_buffer_from_arrays(
        parent_grads,
        parent_hess,
        parent_counts,
        "parent",
    ) {
        Ok(h) => h,
        Err(e) => return Err(e.into()),
    };

    // Build child HistogramBuffer from numpy arrays
    let child =
        match build_histogram_buffer_from_arrays(child_grads, child_hess, child_counts, "child") {
            Ok(h) => h,
            Err(e) => return Err(e.into()),
        };

    // Call the Rust core function
    let sibling = match histogram::subtract_histogram(&parent, &child) {
        Ok(h) => h,
        Err(e) => return Err(e.into()),
    };

    // Convert output to numpy arrays
    histogram_buffer_to_numpy(py, &sibling)
}

/// Reconstructs a [`HistogramBuffer`] from numpy arrays.
///
/// # Args
///
/// * `grads` - Gradient sums per bin.
/// * `hess` - Hessian sums per bin.
/// * `counts` - Sample counts per bin.
/// * `context` - Description for error messages.
///
/// # Returns
///
/// A populated `HistogramBuffer`.
///
/// # Errors
///
/// Returns [`ClearGbmError`] if array lengths don't match or slices are non-contiguous.
fn build_histogram_buffer_from_arrays(
    grads: &PyReadonlyArray1<'_, f64>,
    hess: &PyReadonlyArray1<'_, f64>,
    counts: &PyReadonlyArray1<'_, u64>,
    context: &str,
) -> Result<HistogramBuffer, ClearGbmError> {
    let g_slice = match grads.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("{context}_grads: {e}"),
            })
        }
    };
    let h_slice = match hess.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("{context}_hess: {e}"),
            })
        }
    };
    let c_slice = match counts.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("{context}_counts: {e}"),
            })
        }
    };

    let n_bins = g_slice.len();
    if h_slice.len() != n_bins {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("{context}_hess length {n_bins}"),
            got: format!("{context}_hess length {}", h_slice.len()),
        });
    }
    if c_slice.len() != n_bins {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("{context}_counts length {n_bins}"),
            got: format!("{context}_counts length {}", c_slice.len()),
        });
    }

    u64_slice_to_usize_vec(c_slice, &format!("{context}_counts")).map(|counts_usize| {
        HistogramBuffer {
            gradient_sums: g_slice.to_vec(),
            hessian_sums: h_slice.to_vec(),
            counts: counts_usize,
            n_bins,
        }
    })
}

/// Converts a [`HistogramBuffer`] to a tuple of numpy arrays.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `buffer` - The histogram buffer to convert.
///
/// # Returns
///
/// Tuple of (gradient_sums, hessian_sums, counts) as numpy arrays.
///
/// # Errors
///
/// Returns `ValueError` if usize-to-u64 conversion fails.
fn histogram_buffer_to_numpy<'py>(
    py: Python<'py>,
    buffer: &HistogramBuffer,
) -> PyResult<HistogramArrays<'py>> {
    let counts_u64 = usize_slice_to_u64_vec(buffer.counts(), "histogram_counts")
        .map_err(|e| -> PyErr { e.into() });
    counts_u64.map(|cu64| {
        let grad_array = PyArray1::from_vec(py, buffer.gradient_sums().to_vec());
        let hess_array = PyArray1::from_vec(py, buffer.hessian_sums().to_vec());
        let count_array = PyArray1::from_vec(py, cu64);
        (grad_array, hess_array, count_array)
    })
}

// =============================================================================
// Argument extraction wrappers for PyCFunction::new_closure registration
// =============================================================================

/// Extracts arguments from a Python tuple and delegates to [`build_histogram_rs`].
///
/// Returns a [`PyObject`] (Python tuple of 3 arrays) for closure compatibility.
///
/// # Args (positional)
///
/// 0. `sample_indices` (numpy array i64)
/// 1. `gradients` (numpy array f64)
/// 2. `hessians` (numpy array f64)
/// 3. `bins` (numpy array i64)
/// 4. `n_bins` (i64)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or histogram building fails.
pub(crate) fn build_histogram_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let sample_indices: PyReadonlyArray1<'_, i64> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg1 = match args.get_item(1_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let gradients: PyReadonlyArray1<'_, f64> = match arg1.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg2 = match args.get_item(2_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let hessians: PyReadonlyArray1<'_, f64> = match arg2.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg3 = match args.get_item(3_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let bins: PyReadonlyArray1<'_, i64> = match arg3.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg4 = match args.get_item(4_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let n_bins: i64 = match arg4.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let (g, h, c) =
        match build_histogram_rs(py, &sample_indices, &gradients, &hessians, &bins, n_bins) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
    let elements = [
        g.unbind().into_any(),
        h.unbind().into_any(),
        c.unbind().into_any(),
    ];
    PyTuple::new(py, elements).map(|t| t.unbind().into_any())
}

/// Extracts arguments from a Python tuple and delegates to [`subtract_histogram_rs`].
///
/// Returns a [`PyObject`] (Python tuple of 3 arrays) for closure compatibility.
///
/// # Args (positional)
///
/// 0. `parent_grads` (numpy array f64)
/// 1. `parent_hess` (numpy array f64)
/// 2. `parent_counts` (numpy array u64)
/// 3. `child_grads` (numpy array f64)
/// 4. `child_hess` (numpy array f64)
/// 5. `child_counts` (numpy array u64)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or subtraction fails.
pub(crate) fn subtract_histogram_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let py = args.py();

    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let parent_grads: PyReadonlyArray1<'_, f64> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg1 = match args.get_item(1_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let parent_hess: PyReadonlyArray1<'_, f64> = match arg1.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg2 = match args.get_item(2_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let parent_counts: PyReadonlyArray1<'_, u64> = match arg2.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg3 = match args.get_item(3_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let child_grads: PyReadonlyArray1<'_, f64> = match arg3.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg4 = match args.get_item(4_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let child_hess: PyReadonlyArray1<'_, f64> = match arg4.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg5 = match args.get_item(5_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let child_counts: PyReadonlyArray1<'_, u64> = match arg5.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let (g, h, c) = match subtract_histogram_rs(
        py,
        &parent_grads,
        &parent_hess,
        &parent_counts,
        &child_grads,
        &child_hess,
        &child_counts,
    ) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let elements = [
        g.unbind().into_any(),
        h.unbind().into_any(),
        c.unbind().into_any(),
    ];
    PyTuple::new(py, elements).map(|t| t.unbind().into_any())
}
