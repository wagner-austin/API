//! PyO3 bindings for tree construction.
//!
//! Wraps [`crate::tree::build_tree`] and provides the [`PyTree`] class
//! as an opaque wrapper around [`crate::tree::Tree`].

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::pyo3_module::array_helpers::{
    i64_slice_to_u32_vec, i64_slice_to_usize_vec, i64_to_i32, i64_to_usize,
};
use crate::split::MonotonicConstraint;
use crate::tree::{build_tree, BuildTreeInput, Tree, TreeBuildConfig};
use crate::types::SplitConfig;

/// Opaque Python wrapper around a Rust [`Tree`].
///
/// Avoids JSON serialization overhead by keeping the tree in Rust memory.
/// Created by [`build_tree_rs`] and consumed by prediction functions.
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct PyTree {
    /// The underlying Rust tree.
    pub(crate) inner: Tree,
}

/// Returns the maximum depth of a [`PyTree`].
///
/// # Args
///
/// * `tree` - The `PyTree` reference.
///
/// # Returns
///
/// The maximum depth of the tree.
pub(crate) fn py_tree_max_depth_rs(tree: &PyTree) -> usize {
    tree.inner.max_depth()
}

/// Returns the number of leaf nodes in a [`PyTree`].
///
/// # Args
///
/// * `tree` - The `PyTree` reference.
///
/// # Returns
///
/// The number of leaf nodes.
pub(crate) fn py_tree_n_leaves_rs(tree: &PyTree) -> usize {
    tree.inner.n_leaves()
}

/// Returns the total number of nodes in a [`PyTree`].
///
/// # Args
///
/// * `tree` - The `PyTree` reference.
///
/// # Returns
///
/// The total number of nodes.
pub(crate) fn py_tree_n_nodes_rs(tree: &PyTree) -> usize {
    tree.inner.n_nodes()
}

/// Serializes a [`PyTree`] to a JSON string.
///
/// # Args
///
/// * `tree` - The `PyTree` reference.
///
/// # Returns
///
/// JSON string representation of the tree.
///
/// # Errors
///
/// Returns `RuntimeError` if serialization fails.
pub(crate) fn py_tree_to_json_rs(tree: &PyTree) -> PyResult<String> {
    serde_json::to_string(&tree.inner).map_err(|e| ser_err(e.to_string()))
}

/// Converts a serialization failure description into a [`PyErr`].
///
/// # Args
///
/// * `reason` - Human-readable description of the serialization failure.
///
/// # Returns
///
/// A Python `RuntimeError` wrapping the serialization error.
pub(crate) fn ser_err(reason: String) -> PyErr {
    ClearGbmError::SerializationFailed { reason }.into()
}

/// Deserializes a [`PyTree`] from a JSON string.
///
/// # Args
///
/// * `json_str` - JSON string previously produced by [`py_tree_to_json_rs`].
///
/// # Returns
///
/// A new `PyTree` instance.
///
/// # Errors
///
/// Returns `RuntimeError` if deserialization fails.
pub(crate) fn py_tree_from_json_rs(json_str: &str) -> PyResult<PyTree> {
    let tree: Tree = match serde_json::from_str(json_str) {
        Ok(t) => t,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            }
            .into())
        }
    };
    Ok(PyTree { inner: tree })
}

/// Returns a string representation of a [`PyTree`] for debugging.
///
/// # Args
///
/// * `tree` - The `PyTree` reference.
///
/// # Returns
///
/// A formatted string like `PyTree(n_nodes=3, max_depth=1, n_leaves=2)`.
pub(crate) fn py_tree_repr_rs(tree: &PyTree) -> String {
    format!(
        "PyTree(n_nodes={}, max_depth={}, n_leaves={})",
        tree.inner.n_nodes(),
        tree.inner.max_depth(),
        tree.inner.n_leaves(),
    )
}

// build_tree_rs is merged into build_tree_from_args below to avoid
// clippy::too_many_arguments (16 params). The tree building logic
// lives directly in the argument extraction wrapper.

/// Transposes a row-major i64 bin-index slice into a column-major flat `Vec<u8>`
/// with layout `bins[feat_idx * n_samples + sample_idx]`.
///
/// The Python binding accepts bins as an `i64` numpy array with shape
/// `(n_samples, n_features)` in row-major order. The Rust tree builder wants
/// the same data in column-major u8 (see `FeatureBins` for the rationale).
/// This helper performs both conversions at the boundary: `i64 → u8` with
/// range-checked `TryFrom` for each element, and row-major → column-major
/// index remapping.
///
/// # Args
///
/// * `flat` - Row-major `[sample_idx * n_features + feat_idx]` i64 data.
/// * `n_features` - Number of columns (features).
///
/// # Returns
///
/// A tuple `(column_major_bins, n_samples)` where `column_major_bins.len() ==
/// n_samples * n_features`.
///
/// # Errors
///
/// * `ClearGbmError::InvalidParameter` if `n_features == 0`.
/// * `ClearGbmError::ShapeMismatch` if `flat.len()` is not divisible by
///   `n_features`.
/// * `ClearGbmError::IntegerConversion` if any entry is out of `u8` range.
fn bins_to_column_major_u8(
    flat: &[i64],
    n_features: usize,
) -> Result<(Vec<u8>, usize), ClearGbmError> {
    if n_features == 0_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "n_features".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    if !flat.len().is_multiple_of(n_features) {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("bins_flat length divisible by {n_features}"),
            got: format!("length {}", flat.len()),
        });
    }

    let n_samples = flat.len() / n_features;
    let mut column_major = vec![0_u8; n_samples * n_features];

    for sample_idx in 0_usize..n_samples {
        let row_start = sample_idx * n_features;
        for feat_idx in 0_usize..n_features {
            let src_val = flat[row_start + feat_idx];
            let dst_val: u8 = match u8::try_from(src_val) {
                Ok(v) => v,
                Err(_) => {
                    return Err(ClearGbmError::IntegerConversion {
                        context: format!(
                            "bins: {src_val} at (sample {sample_idx}, feat {feat_idx}) does not fit in u8"
                        ),
                    })
                }
            };
            column_major[feat_idx * n_samples + sample_idx] = dst_val;
        }
    }

    Ok((column_major, n_samples))
}

/// Unflattens a row-major f64 slice into `Vec<Vec<f64>>` with shape
/// `(n_features, n_bins_per_feature)`.
///
/// # Args
///
/// * `flat` - Flattened f64 data.
/// * `n_features` - Number of rows (features).
/// * `n_bins` - Number of columns (bins per feature).
///
/// # Returns
///
/// Nested vector indexed as `[feature][bin]`.
///
/// # Errors
///
/// Returns error if length doesn't match `n_features * n_bins`.
fn unflatten_thresholds(
    flat: &[f64],
    n_features: usize,
    n_bins: usize,
) -> Result<Vec<Vec<f64>>, ClearGbmError> {
    let expected_len = n_features * n_bins;
    if flat.len() != expected_len {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("bin_thresholds_flat length {expected_len} (n_features={n_features} * n_bins={n_bins})"),
            got: format!("length {}", flat.len()),
        });
    }

    let mut result = Vec::with_capacity(n_features);
    for feat_idx in 0_usize..n_features {
        let start = feat_idx * n_bins;
        let end = start + n_bins;
        result.push(flat[start..end].to_vec());
    }

    Ok(result)
}

/// Converts an optional numpy array of i64 constraints to `Vec<MonotonicConstraint>`.
///
/// # Args
///
/// * `constraints` - Optional numpy array where -1=decreasing, 0=none, 1=increasing.
///
/// # Returns
///
/// `Some(Vec<MonotonicConstraint>)` if input is provided, `None` otherwise.
///
/// # Errors
///
/// Returns error if any constraint value is not -1, 0, or 1.
fn convert_monotonic_constraints(
    constraints: &Option<PyReadonlyArray1<'_, i64>>,
) -> Result<Option<Vec<MonotonicConstraint>>, ClearGbmError> {
    let arr = match constraints {
        Some(a) => a,
        None => return Ok(None),
    };

    let slice = match arr.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("monotonic_constraints: {e}"),
            })
        }
    };

    let mut result = Vec::with_capacity(slice.len());
    for &val in slice {
        let val_i32 = match i64_to_i32(val, "monotonic_constraint") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let constraint = match MonotonicConstraint::from_int(val_i32) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        result.push(constraint);
    }

    Ok(Some(result))
}

/// Validates that the given value is a [`PyTree`] and returns a reference
/// to the inner [`Tree`].
///
/// This is used by prediction functions that accept `PyTree` arguments.
///
/// # Args
///
/// * `py_tree` - The `PyTree` reference.
///
/// # Returns
///
/// Reference to the inner `Tree`.
///
/// # Errors
///
/// Returns `ValueError` if the tree has no nodes.
pub(crate) fn validate_py_tree(py_tree: &PyTree) -> PyResult<&Tree> {
    if py_tree.inner.n_nodes() == 0_usize {
        return Err(PyValueError::new_err("tree has no nodes"));
    }
    Ok(&py_tree.inner)
}

// =============================================================================
// Argument extraction wrapper for PyCFunction::new_closure registration
// =============================================================================

/// Extracts arguments from a Python tuple and delegates to [`build_tree_rs`].
///
/// # Args (positional)
///
/// 0. `sample_indices` (numpy array i64)
/// 1. `gradients` (numpy array f64)
/// 2. `hessians` (numpy array f64)
/// 3. `bins_flat` (numpy array i64)
/// 4. `n_features` (i64)
/// 5. `n_regular_bins` (i64)
/// 6. `bin_thresholds_flat` (numpy array f64)
/// 7. `max_depth` (i64)
/// 8. `max_leaves` (i64)
/// 9. `reg_alpha` (f64)
/// 10. `reg_lambda` (f64)
/// 11. `min_samples_split` (i64)
/// 12. `min_samples_leaf` (i64)
/// 13. `max_bins` (i64)
/// 14. `min_gain` (f64)
/// 15. `monotonic_constraints` (optional numpy array i64, or None)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or tree building fails.
pub(crate) fn build_tree_from_args(args: &Bound<'_, PyTuple>) -> PyResult<PyTree> {
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
    let bins_flat: PyReadonlyArray1<'_, i64> = match arg3.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg4 = match args.get_item(4_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let n_features: i64 = match arg4.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg5 = match args.get_item(5_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let n_regular_bins: i64 = match arg5.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg6 = match args.get_item(6_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let bin_thresholds_flat: PyReadonlyArray1<'_, f64> = match arg6.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let arg7 = match args.get_item(7_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let max_depth: i64 = match arg7.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg8 = match args.get_item(8_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let max_leaves: i64 = match arg8.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg9 = match args.get_item(9_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let reg_alpha: f64 = match arg9.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg10 = match args.get_item(10_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let reg_lambda: f64 = match arg10.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg11 = match args.get_item(11_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let min_samples_split: i64 = match arg11.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg12 = match args.get_item(12_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let min_samples_leaf: i64 = match arg12.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg13 = match args.get_item(13_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let max_bins: i64 = match arg13.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let arg14 = match args.get_item(14_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let min_gain: f64 = match arg14.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // Arg 15 is optional: monotonic_constraints (may be None or absent)
    let monotonic_constraints: Option<PyReadonlyArray1<'_, i64>> = match args.get_item(15_usize) {
        Ok(obj) => match obj.extract() {
            Ok(v) => v,
            Err(e) => return Err(e.into()),
        },
        Err(_) => None,
    };

    // --- Convert integer parameters to usize ---
    let n_features_usize = match i64_to_usize(n_features, "n_features") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let n_regular_bins_usize = match i64_to_usize(n_regular_bins, "n_regular_bins") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let max_depth_usize = match i64_to_usize(max_depth, "max_depth") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let max_leaves_usize = match i64_to_usize(max_leaves, "max_leaves") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let min_samples_split_usize = match i64_to_usize(min_samples_split, "min_samples_split") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let min_samples_leaf_usize = match i64_to_usize(min_samples_leaf, "min_samples_leaf") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    let max_bins_usize = match i64_to_usize(max_bins, "max_bins") {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    // --- Convert numpy arrays to Rust slices/vecs ---
    let idx_slice = match sample_indices.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("sample_indices: {e}"),
            }
            .into())
        }
    };
    let sample_idx = match i64_slice_to_u32_vec(idx_slice, "sample_indices") {
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

    // Python numpy sends float64; the histogram hot loop consumes f32 (see
    // `crate::narrow::score_narrow`). Narrow at the pyo3 boundary so the rest
    // of the Rust core is a consistent f32-input world.
    let grad_f32: Vec<f32> = grad_slice.iter().copied().map(crate::narrow::score_narrow).collect();
    let hess_f32: Vec<f32> = hess_slice.iter().copied().map(crate::narrow::score_narrow).collect();

    let bins_i64 = match bins_flat.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("bins_flat: {e}"),
            }
            .into())
        }
    };
    let (bins, n_samples_usize) = match bins_to_column_major_u8(bins_i64, n_features_usize) {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let thresholds_f64 = match bin_thresholds_flat.as_slice() {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::EmptyInput {
                context: format!("bin_thresholds_flat: {e}"),
            }
            .into())
        }
    };
    let bin_thresholds =
        match unflatten_thresholds(thresholds_f64, n_features_usize, n_regular_bins_usize) {
            Ok(v) => v,
            Err(e) => return Err(e.into()),
        };

    let constraints = match convert_monotonic_constraints(&monotonic_constraints) {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    // --- Build configs and tree ---
    let split_config = match SplitConfig::new(
        min_samples_split_usize,
        min_samples_leaf_usize,
        max_bins_usize,
        reg_lambda,
        min_gain,
    ) {
        Ok(c) => c,
        Err(e) => return Err(e.into()),
    };

    let tree_config = match TreeBuildConfig::new(
        max_depth_usize,
        max_leaves_usize,
        reg_alpha,
        reg_lambda,
        split_config,
    ) {
        Ok(c) => c,
        Err(e) => return Err(e.into()),
    };

    let input = BuildTreeInput {
        sample_indices: &sample_idx,
        gradients: &grad_f32,
        hessians: &hess_f32,
        bins: &bins,
        n_samples: n_samples_usize,
        n_features: n_features_usize,
        n_regular_bins: n_regular_bins_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: constraints.as_deref(),
    };

    let hooks = Hooks::default();
    let tree = match build_tree(&input, &hooks) {
        Ok(t) => t,
        Err(e) => return Err(e.into()),
    };

    Ok(PyTree { inner: tree })
}

/// Extracts a [`PyTree`] from arg 0 and returns its max depth.
///
/// # Args (positional)
///
/// 0. `tree` (PyTree)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction fails.
pub(crate) fn py_tree_max_depth_from_args(args: &Bound<'_, PyTuple>) -> PyResult<usize> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let tree: PyRef<'_, PyTree> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    Ok(py_tree_max_depth_rs(&tree))
}

/// Extracts a [`PyTree`] from arg 0 and returns its leaf count.
///
/// # Args (positional)
///
/// 0. `tree` (PyTree)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction fails.
pub(crate) fn py_tree_n_leaves_from_args(args: &Bound<'_, PyTuple>) -> PyResult<usize> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let tree: PyRef<'_, PyTree> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    Ok(py_tree_n_leaves_rs(&tree))
}

/// Extracts a [`PyTree`] from arg 0 and returns its node count.
///
/// # Args (positional)
///
/// 0. `tree` (PyTree)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction fails.
pub(crate) fn py_tree_n_nodes_from_args(args: &Bound<'_, PyTuple>) -> PyResult<usize> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let tree: PyRef<'_, PyTree> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    Ok(py_tree_n_nodes_rs(&tree))
}

/// Extracts a [`PyTree`] from arg 0 and serializes it to JSON.
///
/// # Args (positional)
///
/// 0. `tree` (PyTree)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or serialization fails.
pub(crate) fn py_tree_to_json_from_args(args: &Bound<'_, PyTuple>) -> PyResult<String> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let tree: PyRef<'_, PyTree> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    py_tree_to_json_rs(&tree)
}

/// Extracts a JSON string from arg 0 and deserializes it into a [`PyTree`].
///
/// # Args (positional)
///
/// 0. `json_str` (str)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or deserialization fails.
pub(crate) fn py_tree_from_json_from_args(args: &Bound<'_, PyTuple>) -> PyResult<PyTree> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let json_str: String = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    py_tree_from_json_rs(&json_str)
}

/// Extracts a [`PyTree`] from arg 0 and returns its repr string.
///
/// # Args (positional)
///
/// 0. `tree` (PyTree)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction fails.
pub(crate) fn py_tree_repr_from_args(args: &Bound<'_, PyTuple>) -> PyResult<String> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let tree: PyRef<'_, PyTree> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    Ok(py_tree_repr_rs(&tree))
}
