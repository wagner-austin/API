//! Tests for PyO3 module initialization and registered function invocation.
//!
//! Tests that [`super::super::cleargbm_rs`] registers all functions and classes,
//! and that each registered function is callable through the module.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};

use crate::error::ClearGbmError;
use crate::pyo3_module::tree_fns::PyTree;
use crate::tree::Tree;
use crate::types::TreeNode;

/// Helper: wraps a PyErr into ClearGbmError for test return types.
fn wrap_py_err(e: &PyErr) -> ClearGbmError {
    ClearGbmError::TreeConstructionFailed {
        reason: format!("PyErr: {e}"),
    }
}

/// Helper: creates and initializes the cleargbm_rs module.
fn init_module<'py>(py: Python<'py>) -> Result<Bound<'py, PyModule>, ClearGbmError> {
    let module = match pyo3::types::PyModule::new(py, "cleargbm_rs") {
        Ok(m) => m,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match super::super::cleargbm_rs(&module) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    Ok(module)
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

// =============================================================================
// Module initialization
// =============================================================================

#[test]
fn test_module_init_succeeds() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let _module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        Ok(())
    })
}

#[test]
fn test_module_has_all_expected_functions() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };

        let expected_names = [
            "sigmoid_rs",
            "predict_single_rs",
            "predict_tree_rs",
            "predict_ensemble_rs",
            "predict_proba_rs",
            "build_histogram_rs",
            "subtract_histogram_rs",
            "build_tree_rs",
            "py_tree_max_depth_rs",
            "py_tree_n_leaves_rs",
            "py_tree_n_nodes_rs",
            "py_tree_to_json_rs",
            "py_tree_from_json_rs",
            "py_tree_repr_rs",
        ];

        for name in &expected_names {
            let has_attr: bool = module.hasattr(*name).unwrap_or_default();
            assert!(has_attr, "module missing function: {name}");
        }

        Ok(())
    })
}

#[test]
fn test_module_has_pytree_class() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let has_pytree: bool = module.hasattr("PyTree").unwrap_or_default();
        assert!(has_pytree);
        Ok(())
    })
}

// =============================================================================
// Call registered functions through module (covers mod.rs closure bodies)
// =============================================================================

#[test]
fn test_module_call_sigmoid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("sigmoid_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match func.call1((0.0_f64,)) {
            Ok(r) => r,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let value: f64 = match result.extract() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!((value - 0.5_f64).abs() < 1e-10_f64);
        Ok(())
    })
}

#[test]
fn test_module_call_build_histogram() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("build_histogram_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let indices = PyArray1::from_vec(py, vec![0_i64, 1_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64, 0.2_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64, 1_i64]);

        let result = func.call1((indices, grads, hess, bins, 2_i64));
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_module_call_subtract_histogram() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("subtract_histogram_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let p_grads = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64, 4.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64, 6_u64]);
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64, 0.7_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.5_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64, 3_u64]);

        let result = func.call1((p_grads, p_hess, p_counts, c_grads, c_hess, c_counts));
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_module_call_predict_single() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("predict_single_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let tree = match Py::new(py, make_leaf_tree(0.5_f64)) {
            Ok(p) => p,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let features = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);

        let result = match func.call1((tree, features)) {
            Ok(r) => r,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let value: f64 = match result.extract() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!((value - 0.5_f64).abs() < 1e-10_f64);
        Ok(())
    })
}

#[test]
fn test_module_call_predict_tree() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("predict_tree_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let tree = match Py::new(py, make_leaf_tree(0.75_f64)) {
            Ok(p) => p,
            Err(e) => return Err(wrap_py_err(&e)),
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

        let result = func.call1((tree, features));
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_module_call_predict_ensemble() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("predict_ensemble_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let tree = match Py::new(py, make_leaf_tree(1.0_f64)) {
            Ok(p) => p.into_bound(py).into_any(),
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree_list = match PyList::new(py, [tree]) {
            Ok(l) => l,
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

        let result = func.call1((tree_list, features, 0.0_f64, 0.1_f64));
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_module_call_predict_proba() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("predict_proba_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let raw = PyArray1::from_vec(py, vec![0.0_f64, 2.0_f64]);
        let result = func.call1((raw,));
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_module_call_build_tree() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("build_tree_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // 4 samples, 2 features, 3 bins per feature
        let sample_indices = PyArray1::from_vec(py, vec![0_i64, 1_i64, 2_i64, 3_i64]);
        let gradients = PyArray1::from_vec(py, vec![0.1_f64, -0.2_f64, 0.3_f64, -0.1_f64]);
        let hessians = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64]);
        let bins_flat = PyArray1::from_vec(
            py,
            vec![0_i64, 1_i64, 1_i64, 2_i64, 0_i64, 0_i64, 2_i64, 1_i64],
        );
        let bin_thresholds_flat = PyArray1::from_vec(
            py,
            vec![0.0_f64, 0.5_f64, 1.0_f64, 0.0_f64, 0.5_f64, 1.0_f64],
        );

        let n_feat = match 2_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let n_reg = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let max_d = match 1_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let max_l = match 2_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let r_alpha = match 0.0_f64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let r_lambda = match 1.0_f64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let min_split = match 2_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let min_leaf = match 1_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let max_b = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let min_g = match 0.0_f64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };

        let args_tuple = match pyo3::types::PyTuple::new(
            py,
            [
                sample_indices.into_any(),
                gradients.into_any(),
                hessians.into_any(),
                bins_flat.into_any(),
                n_feat,
                n_reg,
                bin_thresholds_flat.into_any(),
                max_d,
                max_l,
                r_alpha,
                r_lambda,
                min_split,
                min_leaf,
                max_b,
                min_g,
                py.None().into_bound(py).into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = func.call1(args_tuple);
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_module_call_py_tree_max_depth() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("py_tree_max_depth_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree = match Py::new(py, make_leaf_tree(0.5_f64)) {
            Ok(p) => p,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match func.call1((tree,)) {
            Ok(r) => r,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let depth: usize = match result.extract() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert_eq!(depth, 0_usize);
        Ok(())
    })
}

#[test]
fn test_module_call_py_tree_n_leaves() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("py_tree_n_leaves_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree = match Py::new(py, make_leaf_tree(0.5_f64)) {
            Ok(p) => p,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match func.call1((tree,)) {
            Ok(r) => r,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let leaves: usize = match result.extract() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert_eq!(leaves, 1_usize);
        Ok(())
    })
}

#[test]
fn test_module_call_py_tree_n_nodes() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("py_tree_n_nodes_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree = match Py::new(py, make_leaf_tree(0.5_f64)) {
            Ok(p) => p,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match func.call1((tree,)) {
            Ok(r) => r,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let nodes: usize = match result.extract() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert_eq!(nodes, 1_usize);
        Ok(())
    })
}

#[test]
fn test_module_call_py_tree_to_json() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("py_tree_to_json_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree = match Py::new(py, make_leaf_tree(0.5_f64)) {
            Ok(p) => p,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match func.call1((tree,)) {
            Ok(r) => r,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let json: String = match result.extract() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(json.contains("node_id"));
        Ok(())
    })
}

#[test]
fn test_module_call_py_tree_from_json() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };

        // First serialize a tree
        let to_json_fn = match module.getattr("py_tree_to_json_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree = match Py::new(py, make_leaf_tree(0.5_f64)) {
            Ok(p) => p,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let json_result = match to_json_fn.call1((tree,)) {
            Ok(r) => r,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let json: String = match json_result.extract() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Then deserialize it
        let from_json_fn = match module.getattr("py_tree_from_json_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = from_json_fn.call1((json,));
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_module_call_py_tree_repr() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("py_tree_repr_rs") {
            Ok(f) => f,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tree = match Py::new(py, make_leaf_tree(0.5_f64)) {
            Ok(p) => p,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match func.call1((tree,)) {
            Ok(r) => r,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let repr: String = match result.extract() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(repr.contains("PyTree"));
        assert!(repr.contains("n_nodes=1"));
        Ok(())
    })
}
