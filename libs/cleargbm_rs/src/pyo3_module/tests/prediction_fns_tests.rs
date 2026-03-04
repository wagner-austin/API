//! Tests for PyO3 prediction binding functions.
//!
//! Tests [`super::super::prediction_fns`] functions through the PyO3 runtime.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::error::ClearGbmError;
use crate::pyo3_module::prediction_fns::{
    predict_ensemble_from_args, predict_ensemble_rs, predict_proba_from_args, predict_proba_rs,
    predict_single_from_args, predict_single_rs, predict_tree_from_args, predict_tree_rs,
    shape_err, sigmoid_from_args, sigmoid_rs,
};
use crate::pyo3_module::tree_fns::PyTree;
use crate::tree::Tree;
use crate::types::{TreeNode, TreeNodeConfig};

/// Helper: wraps a PyErr into ClearGbmError for test return types.
fn wrap_py_err(e: &PyErr) -> ClearGbmError {
    ClearGbmError::TreeConstructionFailed {
        reason: format!("PyErr: {e}"),
    }
}

/// Helper: creates a single-leaf PyTree with the given value.
fn make_leaf_tree(value: f64) -> PyTree {
    let tree = Tree::new(
        vec![TreeNode::new_leaf(0_usize, value, 10_usize)],
        0_usize,
        1_usize,
    );
    PyTree { inner: tree }
}

/// Helper: creates a PyTree with an internal node whose feature_index=100,
/// left_child=1, right_child=2, but only 2 leaf children.
/// Passing features shorter than 101 elements will trigger an IndexError
/// in predict_single / predict_tree.
fn make_oob_feature_tree() -> PyTree {
    let tree = Tree::new(
        vec![
            TreeNode::new_internal(TreeNodeConfig {
                node_id: 0_usize,
                feature_index: 100_usize,
                threshold: 0.5_f64,
                value: 0.0_f64,
                n_samples: 10_usize,
                left_child: 1_usize,
                right_child: 2_usize,
                nan_goes_left: true,
            }),
            TreeNode::new_leaf(1_usize, 1.0_f64, 5_usize),
            TreeNode::new_leaf(2_usize, 2.0_f64, 5_usize),
        ],
        0_usize,
        2_usize,
    );
    PyTree { inner: tree }
}

/// Helper: creates a non-contiguous 1D f64 numpy array by slicing with stride 2.
fn make_non_contiguous_f64<'py>(py: Python<'py>) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    let numpy = match py.import("numpy") {
        Ok(m) => m,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let arr = match numpy.call_method1(
        "array",
        (vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64],),
    ) {
        Ok(a) => a,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let slice = pyo3::types::PySlice::new(py, 0_isize, 6_isize, 2_isize);
    match arr.get_item(slice) {
        Ok(sliced) => Ok(sliced),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Helper: wraps a PyTree into a Bound<'py, PyAny> for tuple construction.
fn pytree_to_py<'py>(py: Python<'py>, tree: PyTree) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    match Py::new(py, tree) {
        Ok(obj) => Ok(obj.into_bound(py).into_any()),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Helper: converts f64 to Python object.
fn f64_to_py<'py>(py: Python<'py>, v: f64) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    match v.into_pyobject(py) {
        Ok(obj) => Ok(obj.into_any()),
        Err(e) => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("f64 to PyFloat failed: {e}"),
        }),
    }
}

/// Helper: converts a &str to a Python object.
fn str_to_py<'py>(py: Python<'py>, v: &str) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    match v.into_pyobject(py) {
        Ok(obj) => Ok(obj.into_any()),
        Err(e) => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("str to PyString failed: {e}"),
        }),
    }
}

// =============================================================================
// sigmoid_rs
// =============================================================================

#[test]
fn test_sigmoid_rs_zero_returns_half() -> Result<(), ClearGbmError> {
    let result = sigmoid_rs(0.0_f64);
    assert!((result - 0.5_f64).abs() < 1e-10_f64);
    Ok(())
}

#[test]
fn test_sigmoid_rs_large_positive_approaches_one() -> Result<(), ClearGbmError> {
    let result = sigmoid_rs(100.0_f64);
    assert!(result > 0.999_f64);
    assert!(result <= 1.0_f64);
    Ok(())
}

#[test]
fn test_sigmoid_rs_large_negative_approaches_zero() -> Result<(), ClearGbmError> {
    let result = sigmoid_rs(-100.0_f64);
    assert!(result < 0.001_f64);
    assert!(result >= 0.0_f64);
    Ok(())
}

#[test]
fn test_sigmoid_rs_symmetry() -> Result<(), ClearGbmError> {
    let pos = sigmoid_rs(2.0_f64);
    let neg = sigmoid_rs(-2.0_f64);
    assert!((pos + neg - 1.0_f64).abs() < 1e-10_f64);
    Ok(())
}

// =============================================================================
// sigmoid_from_args
// =============================================================================

#[test]
fn test_sigmoid_from_args_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let val = match f64_to_py(py, 0.0_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [val]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match sigmoid_from_args(&args) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!((result - 0.5_f64).abs() < 1e-10_f64);
        Ok(())
    })
}

#[test]
fn test_sigmoid_from_args_wrong_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let val = match str_to_py(py, "not_a_number") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [val]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = sigmoid_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// predict_single_rs
// =============================================================================

#[test]
fn test_predict_single_rs_leaf_tree() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let py_tree = make_leaf_tree(0.5_f64);
        let features = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64, 3.0_f64]);

        let result = predict_single_rs(&py_tree, &features.readonly());

        let value = match result {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!((value - 0.5_f64).abs() < 1e-10_f64);
        Ok(())
    })
}

#[test]
fn test_predict_single_rs_empty_tree_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let py_tree = PyTree {
            inner: Tree::new(vec![], 0_usize, 0_usize),
        };
        let features = PyArray1::from_vec(py, vec![1.0_f64]);

        let result = predict_single_rs(&py_tree, &features.readonly());

        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// predict_single_from_args
// =============================================================================

#[test]
fn test_predict_single_from_args_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree_obj = match pytree_to_py(py, make_leaf_tree(0.5_f64)) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let features = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);
        let args = match PyTuple::new(py, [tree_obj, features.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match predict_single_from_args(&args) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!((result - 0.5_f64).abs() < 1e-10_f64);
        Ok(())
    })
}

#[test]
fn test_predict_single_from_args_wrong_tree_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let not_a_tree = match str_to_py(py, "not_a_tree") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let features = PyArray1::from_vec(py, vec![1.0_f64]);
        let args = match PyTuple::new(py, [not_a_tree, features.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_single_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_predict_single_from_args_wrong_features_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree_obj = match pytree_to_py(py, make_leaf_tree(0.5_f64)) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [tree_obj, not_array]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_single_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// predict_tree_rs
// =============================================================================

#[test]
fn test_predict_tree_rs_leaf_tree_batch() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let py_tree = make_leaf_tree(0.75_f64);
        let data = vec![vec![1.0_f64, 2.0_f64], vec![3.0_f64, 4.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };

        let result = predict_tree_rs(py, &py_tree, &features.readonly());

        let predictions = match result {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        assert_eq!(predictions.len(), 2_usize);
        let pred_ro = predictions.readonly();
        let pred_slice = match pred_ro.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::EmptyInput {
                    context: format!("predictions: {e}"),
                })
            }
        };
        assert!((pred_slice[0_usize] - 0.75_f64).abs() < 1e-10_f64);
        assert!((pred_slice[1_usize] - 0.75_f64).abs() < 1e-10_f64);

        Ok(())
    })
}

#[test]
fn test_predict_tree_rs_empty_tree_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let py_tree = PyTree {
            inner: Tree::new(vec![], 0_usize, 0_usize),
        };
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };

        let result = predict_tree_rs(py, &py_tree, &features.readonly());
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// predict_tree_from_args
// =============================================================================

#[test]
fn test_predict_tree_from_args_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree_obj = match pytree_to_py(py, make_leaf_tree(0.75_f64)) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let data = vec![vec![1.0_f64, 2.0_f64], vec![3.0_f64, 4.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let args = match PyTuple::new(py, [tree_obj, features.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_tree_from_args(&args);
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_predict_tree_from_args_wrong_tree_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let not_tree = match str_to_py(py, "not_a_tree") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let data = vec![vec![1.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let args = match PyTuple::new(py, [not_tree, features.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_predict_tree_from_args_wrong_features_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree_obj = match pytree_to_py(py, make_leaf_tree(0.75_f64)) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [tree_obj, not_array]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// predict_ensemble_rs
// =============================================================================

#[test]
fn test_predict_ensemble_rs_single_tree() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(1.0_f64);
        let bound_tree = match Py::new(py, tree) {
            Ok(p) => p.into_bound(py),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let py_ref: PyRef<'_, PyTree> = bound_tree.borrow();
        let trees = vec![py_ref];

        let data = vec![vec![1.0_f64, 2.0_f64], vec![3.0_f64, 4.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };

        let result = predict_ensemble_rs(py, &trees, &features.readonly(), 0.0_f64, 0.1_f64);

        let predictions = match result {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        assert_eq!(predictions.len(), 2_usize);
        // raw = base_prediction + learning_rate * leaf_value = 0.0 + 0.1 * 1.0 = 0.1
        let pred_ro = predictions.readonly();
        let pred_slice = match pred_ro.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::EmptyInput {
                    context: format!("predictions: {e}"),
                })
            }
        };
        assert!((pred_slice[0_usize] - 0.1_f64).abs() < 1e-10_f64);
        assert!((pred_slice[1_usize] - 0.1_f64).abs() < 1e-10_f64);

        Ok(())
    })
}

#[test]
fn test_predict_ensemble_rs_multiple_trees() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree1 = make_leaf_tree(1.0_f64);
        let tree2 = make_leaf_tree(0.5_f64);
        let bound1 = match Py::new(py, tree1) {
            Ok(p) => p.into_bound(py),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let bound2 = match Py::new(py, tree2) {
            Ok(p) => p.into_bound(py),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let ref1: PyRef<'_, PyTree> = bound1.borrow();
        let ref2: PyRef<'_, PyTree> = bound2.borrow();
        let trees = vec![ref1, ref2];

        let data = vec![vec![1.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };

        let result = predict_ensemble_rs(py, &trees, &features.readonly(), 0.5_f64, 0.1_f64);

        let predictions = match result {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // raw = 0.5 + 0.1 * (1.0 + 0.5) = 0.5 + 0.15 = 0.65
        let pred_ro = predictions.readonly();
        let pred_slice = match pred_ro.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::EmptyInput {
                    context: format!("predictions: {e}"),
                })
            }
        };
        assert!((pred_slice[0_usize] - 0.65_f64).abs() < 1e-10_f64);

        Ok(())
    })
}

#[test]
fn test_predict_ensemble_rs_invalid_learning_rate_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(1.0_f64);
        let bound_tree = match Py::new(py, tree) {
            Ok(p) => p.into_bound(py),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let py_ref: PyRef<'_, PyTree> = bound_tree.borrow();
        let trees = vec![py_ref];

        let data = vec![vec![1.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };

        // learning_rate=0.0 is invalid (must be > 0)
        let result = predict_ensemble_rs(py, &trees, &features.readonly(), 0.0_f64, 0.0_f64);
        assert!(result.is_err());

        Ok(())
    })
}

// =============================================================================
// predict_ensemble_from_args
// =============================================================================

#[test]
fn test_predict_ensemble_from_args_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(1.0_f64);
        let tree_py = match Py::new(py, tree) {
            Ok(p) => p.into_bound(py).into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree_list = match PyList::new(py, [tree_py]) {
            Ok(l) => l.into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };

        let base_pred = match f64_to_py(py, 0.0_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let lr = match f64_to_py(py, 0.1_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(py, [tree_list, features.into_any(), base_pred, lr]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_ensemble_from_args(&args);
        assert!(result.is_ok());

        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_wrong_trees_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let not_list = match str_to_py(py, "not_a_list") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let data = vec![vec![1.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let base_pred = match f64_to_py(py, 0.0_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let lr = match f64_to_py(py, 0.1_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(py, [not_list, features.into_any(), base_pred, lr]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_ensemble_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_wrong_features_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(1.0_f64);
        let tree_py = match Py::new(py, tree) {
            Ok(p) => p.into_bound(py).into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree_list = match PyList::new(py, [tree_py]) {
            Ok(l) => l.into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let base_pred = match f64_to_py(py, 0.0_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let lr = match f64_to_py(py, 0.1_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(py, [tree_list, not_array, base_pred, lr]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_ensemble_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_wrong_base_pred_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(1.0_f64);
        let tree_py = match Py::new(py, tree) {
            Ok(p) => p.into_bound(py).into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree_list = match PyList::new(py, [tree_py]) {
            Ok(l) => l.into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let data = vec![vec![1.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let not_float = match str_to_py(py, "not_a_float") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let lr = match f64_to_py(py, 0.1_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(py, [tree_list, features.into_any(), not_float, lr]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_ensemble_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_wrong_lr_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(1.0_f64);
        let tree_py = match Py::new(py, tree) {
            Ok(p) => p.into_bound(py).into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree_list = match PyList::new(py, [tree_py]) {
            Ok(l) => l.into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let data = vec![vec![1.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let base_pred = match f64_to_py(py, 0.0_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let not_float = match str_to_py(py, "not_a_float") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(py, [tree_list, features.into_any(), base_pred, not_float]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_ensemble_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

// =============================================================================
// predict_proba_rs
// =============================================================================

#[test]
fn test_predict_proba_rs_zero_raw_prediction() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let raw = PyArray1::from_vec(py, vec![0.0_f64]);

        let result = predict_proba_rs(py, &raw.readonly());

        let proba = match result {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // sigmoid(0) = 0.5, so prob_class_0 = 0.5, prob_class_1 = 0.5
        assert_eq!(proba.shape(), [1_usize, 2_usize]);
        let proba_ro = proba.readonly();
        let arr = proba_ro.as_array();
        assert!((arr[[0_usize, 0_usize]] - 0.5_f64).abs() < 1e-10_f64);
        assert!((arr[[0_usize, 1_usize]] - 0.5_f64).abs() < 1e-10_f64);

        Ok(())
    })
}

#[test]
fn test_predict_proba_rs_multiple_samples() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let raw = PyArray1::from_vec(py, vec![0.0_f64, 100.0_f64, -100.0_f64]);

        let result = predict_proba_rs(py, &raw.readonly());

        let proba = match result {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        assert_eq!(proba.shape(), [3_usize, 2_usize]);
        let proba_ro = proba.readonly();
        let arr = proba_ro.as_array();

        // Sample 0: sigmoid(0) = 0.5
        assert!((arr[[0_usize, 0_usize]] - 0.5_f64).abs() < 1e-10_f64);

        // Sample 1: sigmoid(100) ~ 1.0, so class_0 ~ 0.0, class_1 ~ 1.0
        assert!(arr[[1_usize, 0_usize]] < 0.001_f64);
        assert!(arr[[1_usize, 1_usize]] > 0.999_f64);

        // Sample 2: sigmoid(-100) ~ 0.0, so class_0 ~ 1.0, class_1 ~ 0.0
        assert!(arr[[2_usize, 0_usize]] > 0.999_f64);
        assert!(arr[[2_usize, 1_usize]] < 0.001_f64);

        Ok(())
    })
}

// =============================================================================
// predict_proba_from_args
// =============================================================================

#[test]
fn test_predict_proba_from_args_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let raw = PyArray1::from_vec(py, vec![0.0_f64, 2.0_f64]);
        let args = match PyTuple::new(py, [raw.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_proba_from_args(&args);
        assert!(result.is_ok());

        Ok(())
    })
}

#[test]
fn test_predict_proba_from_args_wrong_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [not_array]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = predict_proba_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

// =============================================================================
// Empty tuple tests (cover get_item(0) Err paths in _from_args)
// =============================================================================

#[test]
fn test_sigmoid_from_args_empty_tuple_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(sigmoid_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_single_from_args_empty_tuple_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(predict_single_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_tree_from_args_empty_tuple_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(predict_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_empty_tuple_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(predict_ensemble_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_proba_from_args_empty_tuple_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(predict_proba_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// Empty features (0 rows) tests for extract_rows Err path
// =============================================================================

#[test]
fn test_predict_tree_rs_empty_features_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let py_tree = make_leaf_tree(0.5_f64);
        let data: Vec<Vec<f64>> = vec![];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let result = predict_tree_rs(py, &py_tree, &features.readonly());
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_predict_ensemble_rs_empty_features_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(1.0_f64);
        let bound_tree = match Py::new(py, tree) {
            Ok(p) => p.into_bound(py),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let py_ref: PyRef<'_, PyTree> = bound_tree.borrow();
        let trees = vec![py_ref];

        let data: Vec<Vec<f64>> = vec![];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let result = predict_ensemble_rs(py, &trees, &features.readonly(), 0.0_f64, 0.1_f64);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// Non-contiguous array tests (cover as_slice() Err paths in _rs functions)
// =============================================================================

#[test]
fn test_predict_single_rs_non_contiguous_features_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(0.5_f64);
        let nc = match make_non_contiguous_f64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let features: numpy::PyReadonlyArray1<'_, f64> = match nc.extract() {
            Ok(v) => v,
            Err(e) => {
                let py_err: PyErr = e.into();
                return Err(wrap_py_err(&py_err));
            }
        };
        let result = predict_single_rs(&tree, &features);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_predict_proba_rs_non_contiguous_raw_predictions_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let nc = match make_non_contiguous_f64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let raw_preds: numpy::PyReadonlyArray1<'_, f64> = match nc.extract() {
            Ok(v) => v,
            Err(e) => {
                let py_err: PyErr = e.into();
                return Err(wrap_py_err(&py_err));
            }
        };
        let result = predict_proba_rs(py, &raw_preds);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// Core error propagation tests (cover Err paths in _rs functions)
// =============================================================================

#[test]
fn test_predict_single_rs_oob_feature_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_oob_feature_tree();
        let features = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64, 3.0_f64]);
        let result = predict_single_rs(&tree, &features.readonly());
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_predict_tree_rs_oob_feature_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_oob_feature_tree();
        let data = vec![vec![1.0_f64, 2.0_f64, 3.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let result = predict_tree_rs(py, &tree, &features.readonly());
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_predict_ensemble_rs_oob_feature_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_oob_feature_tree();
        let bound_tree = match Py::new(py, tree) {
            Ok(p) => p.into_bound(py),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let py_ref: PyRef<'_, PyTree> = bound_tree.borrow();
        let trees = vec![py_ref];

        let data = vec![vec![1.0_f64, 2.0_f64, 3.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let result = predict_ensemble_rs(py, &trees, &features.readonly(), 0.0_f64, 0.1_f64);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// Short-tuple tests (cover get_item(N) Err for N>0 in _from_args)
// =============================================================================

#[test]
fn test_predict_single_from_args_one_arg_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(0.5_f64);
        let py_tree = match pytree_to_py(py, tree) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [py_tree]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_single_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_tree_from_args_one_arg_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(0.5_f64);
        let py_tree = match pytree_to_py(py, tree) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [py_tree]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_tree_from_args_wrong_type_features_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_leaf_tree(0.5_f64);
        let py_tree = match pytree_to_py(py, tree) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let bad = match str_to_py(py, "not_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [py_tree, bad]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_one_arg_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let list = PyList::empty(py).into_any();
        let args = match PyTuple::new(py, [list]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_ensemble_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_two_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let list = PyList::empty(py).into_any();
        let bad = match str_to_py(py, "not_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [list, bad]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_ensemble_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_two_valid_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let list = PyList::empty(py).into_any();
        let features_2d = match PyArray2::from_vec2(py, &[vec![1.0_f64]]) {
            Ok(f) => f.into_any(),
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let args = match PyTuple::new(py, [list, features_2d]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_ensemble_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_three_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let list = PyList::empty(py).into_any();
        let features_2d = match PyArray2::from_vec2(py, &[vec![1.0_f64]]) {
            Ok(f) => f.into_any(),
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let base = match f64_to_py(py, 0.0_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [list, features_2d, base]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_ensemble_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// _from_args core error propagation (cover _rs Err through _from_args)
// =============================================================================

#[test]
fn test_predict_tree_from_args_core_error_propagates() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_oob_feature_tree();
        let py_tree = match pytree_to_py(py, tree) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let features_2d = match PyArray2::from_vec2(py, &[vec![1.0_f64, 2.0_f64, 3.0_f64]]) {
            Ok(f) => f.into_any(),
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let args = match PyTuple::new(py, [py_tree, features_2d]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_ensemble_from_args_core_error_propagates() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree = make_oob_feature_tree();
        let bound_tree = match Py::new(py, tree) {
            Ok(p) => p.into_bound(py),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let list = match PyList::new(py, [bound_tree.clone()]) {
            Ok(l) => l.into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let features_2d = match PyArray2::from_vec2(py, &[vec![1.0_f64, 2.0_f64, 3.0_f64]]) {
            Ok(f) => f.into_any(),
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let base = match f64_to_py(py, 0.0_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let lr = match f64_to_py(py, 0.1_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [list, features_2d, base, lr]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_ensemble_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_predict_proba_from_args_non_contiguous_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let nc = match make_non_contiguous_f64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [nc]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(predict_proba_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// Error helper: shape_err
// =============================================================================

#[test]
fn test_shape_err_produces_pyerr() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let py_err = shape_err("ragged rows".to_string());
        let err_str = py_err.to_string();
        assert!(
            err_str.contains("ragged rows"),
            "expected error message, got: {err_str}"
        );
        let _ = py;
        Ok(())
    })
}
