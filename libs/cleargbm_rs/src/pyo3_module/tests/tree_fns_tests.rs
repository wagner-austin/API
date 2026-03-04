//! Tests for PyO3 tree binding functions.
//!
//! Tests [`PyTree`] standalone functions, `_from_args` wrappers, and
//! [`validate_py_tree`] through the PyO3 runtime.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::ClearGbmError;
use crate::pyo3_module::tree_fns::{
    build_tree_from_args, py_tree_from_json_from_args, py_tree_from_json_rs,
    py_tree_max_depth_from_args, py_tree_max_depth_rs, py_tree_n_leaves_from_args,
    py_tree_n_leaves_rs, py_tree_n_nodes_from_args, py_tree_n_nodes_rs, py_tree_repr_from_args,
    py_tree_repr_rs, py_tree_to_json_from_args, py_tree_to_json_rs, ser_err, validate_py_tree,
    PyTree,
};
use crate::tree::Tree;
use crate::types::{TreeNode, TreeNodeConfig};

/// Helper: wraps a PyErr into ClearGbmError for test return types.
fn wrap_py_err(e: &PyErr) -> ClearGbmError {
    ClearGbmError::TreeConstructionFailed {
        reason: format!("PyErr: {e}"),
    }
}

/// Helper: converts an i64 to a Python object for tuple construction.
fn i64_to_py<'py>(py: Python<'py>, v: i64) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    match v.into_pyobject(py) {
        Ok(obj) => Ok(obj.into_any()),
        Err(e) => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("i64 to PyInt failed: {e}"),
        }),
    }
}

/// Helper: converts an f64 to a Python object for tuple construction.
fn f64_to_py<'py>(py: Python<'py>, v: f64) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    match v.into_pyobject(py) {
        Ok(obj) => Ok(obj.into_any()),
        Err(e) => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("f64 to PyFloat failed: {e}"),
        }),
    }
}

/// Helper: converts a &str to a Python object for tuple construction.
fn str_to_py<'py>(py: Python<'py>, v: &str) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    match v.into_pyobject(py) {
        Ok(obj) => Ok(obj.into_any()),
        Err(e) => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("str to PyString failed: {e}"),
        }),
    }
}

/// Helper: wraps a PyTree value in a `Py<PyAny>` for tuple construction.
fn pytree_to_py<'py>(py: Python<'py>, tree: PyTree) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    match Py::new(py, tree) {
        Ok(obj) => Ok(obj.into_bound(py).into_any()),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Builds a `build_tree_from_args` argument tuple with specified `n_features` override.
///
/// Creates a valid 16-arg tuple for tree building. Set `n_features_override`
/// to inject a specific (possibly invalid) value for the `n_features` arg.
fn make_build_tree_args<'py>(
    py: Python<'py>,
    n_features_override: i64,
) -> Result<Bound<'py, PyTuple>, ClearGbmError> {
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

    let n_feat = match i64_to_py(py, n_features_override) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let n_reg_bins = match i64_to_py(py, 3_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let max_d = match i64_to_py(py, 1_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let max_l = match i64_to_py(py, 2_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let r_alpha = match f64_to_py(py, 0.0_f64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let r_lambda = match f64_to_py(py, 1.0_f64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let min_split = match i64_to_py(py, 2_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let min_leaf = match i64_to_py(py, 1_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let max_b = match i64_to_py(py, 3_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let min_g = match f64_to_py(py, 0.0_f64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    match PyTuple::new(
        py,
        [
            sample_indices.into_any(),
            gradients.into_any(),
            hessians.into_any(),
            bins_flat.into_any(),
            n_feat,
            n_reg_bins,
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
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Helper: creates a single-leaf PyTree.
fn make_leaf_tree() -> PyTree {
    let tree = Tree::new(
        vec![TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize)],
        0_usize,
        1_usize,
    );
    PyTree { inner: tree }
}

/// Helper: creates a PyTree with one split and two leaves.
fn make_split_tree() -> PyTree {
    let root = TreeNode::new_internal(TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 0_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 20_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    });
    let left = TreeNode::new_leaf(1_usize, -0.3_f64, 12_usize);
    let right = TreeNode::new_leaf(2_usize, 0.4_f64, 8_usize);

    let tree = Tree::new(vec![root, left, right], 1_usize, 2_usize);
    PyTree { inner: tree }
}

// =============================================================================
// Standalone _rs functions
// =============================================================================

#[test]
fn test_py_tree_max_depth_rs_leaf() -> Result<(), ClearGbmError> {
    let py_tree = make_leaf_tree();
    assert_eq!(py_tree_max_depth_rs(&py_tree), 0_usize);
    Ok(())
}

#[test]
fn test_py_tree_max_depth_rs_split() -> Result<(), ClearGbmError> {
    let py_tree = make_split_tree();
    assert_eq!(py_tree_max_depth_rs(&py_tree), 1_usize);
    Ok(())
}

#[test]
fn test_py_tree_n_leaves_rs_leaf() -> Result<(), ClearGbmError> {
    let py_tree = make_leaf_tree();
    assert_eq!(py_tree_n_leaves_rs(&py_tree), 1_usize);
    Ok(())
}

#[test]
fn test_py_tree_n_leaves_rs_split() -> Result<(), ClearGbmError> {
    let py_tree = make_split_tree();
    assert_eq!(py_tree_n_leaves_rs(&py_tree), 2_usize);
    Ok(())
}

#[test]
fn test_py_tree_n_nodes_rs_leaf() -> Result<(), ClearGbmError> {
    let py_tree = make_leaf_tree();
    assert_eq!(py_tree_n_nodes_rs(&py_tree), 1_usize);
    Ok(())
}

#[test]
fn test_py_tree_n_nodes_rs_split() -> Result<(), ClearGbmError> {
    let py_tree = make_split_tree();
    assert_eq!(py_tree_n_nodes_rs(&py_tree), 3_usize);
    Ok(())
}

#[test]
fn test_py_tree_to_json_rs_roundtrip() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let py_tree = make_split_tree();
        let json_str = match py_tree_to_json_rs(&py_tree) {
            Ok(s) => s,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(!json_str.is_empty());

        let roundtripped = match py_tree_from_json_rs(&json_str) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert_eq!(roundtripped.inner.max_depth(), 1_usize);
        assert_eq!(roundtripped.inner.n_leaves(), 2_usize);
        assert_eq!(roundtripped.inner.n_nodes(), 3_usize);
        Ok(())
    })
}

#[test]
fn test_py_tree_from_json_rs_invalid_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let result = py_tree_from_json_rs("not valid json");
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_from_json_rs_empty_object_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let result = py_tree_from_json_rs("{}");
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_repr_rs_leaf() -> Result<(), ClearGbmError> {
    let py_tree = make_leaf_tree();
    let repr = py_tree_repr_rs(&py_tree);
    assert!(repr.contains("PyTree"));
    assert!(repr.contains("n_nodes=1"));
    assert!(repr.contains("max_depth=0"));
    assert!(repr.contains("n_leaves=1"));
    Ok(())
}

#[test]
fn test_py_tree_repr_rs_split() -> Result<(), ClearGbmError> {
    let py_tree = make_split_tree();
    let repr = py_tree_repr_rs(&py_tree);
    assert!(repr.contains("n_nodes=3"));
    assert!(repr.contains("max_depth=1"));
    assert!(repr.contains("n_leaves=2"));
    Ok(())
}

// =============================================================================
// _from_args wrappers (construct PyTuple, call wrapper)
// =============================================================================

#[test]
fn test_py_tree_max_depth_from_args_leaf() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree_obj = pytree_to_py(py, make_leaf_tree());
        let tree_bound = match tree_obj {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [tree_bound]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match py_tree_max_depth_from_args(&args) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert_eq!(result, 0_usize);
        Ok(())
    })
}

#[test]
fn test_py_tree_n_leaves_from_args_split() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree_bound = match pytree_to_py(py, make_split_tree()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [tree_bound]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match py_tree_n_leaves_from_args(&args) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert_eq!(result, 2_usize);
        Ok(())
    })
}

#[test]
fn test_py_tree_n_nodes_from_args_split() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree_bound = match pytree_to_py(py, make_split_tree()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [tree_bound]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match py_tree_n_nodes_from_args(&args) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert_eq!(result, 3_usize);
        Ok(())
    })
}

#[test]
fn test_py_tree_to_json_from_args_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree_bound = match pytree_to_py(py, make_split_tree()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [tree_bound]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let json_str = match py_tree_to_json_from_args(&args) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(json_str.contains("feature_index"));
        Ok(())
    })
}

#[test]
fn test_py_tree_from_json_from_args_roundtrip() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let original = make_split_tree();
        let json_str = match py_tree_to_json_rs(&original) {
            Ok(s) => s,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let str_obj = match str_to_py(py, &json_str) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [str_obj]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = match py_tree_from_json_from_args(&args) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert_eq!(result.inner.n_nodes(), 3_usize);
        Ok(())
    })
}

#[test]
fn test_py_tree_from_json_from_args_invalid_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let str_obj = match str_to_py(py, "bad json") {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [str_obj]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = py_tree_from_json_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_repr_from_args_split() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tree_bound = match pytree_to_py(py, make_split_tree()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [tree_bound]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let repr = match py_tree_repr_from_args(&args) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(repr.contains("n_nodes=3"));
        Ok(())
    })
}

// --- _from_args error paths: wrong arg type ---

#[test]
fn test_py_tree_max_depth_from_args_wrong_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let int_obj = match i64_to_py(py, 42_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [int_obj]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = py_tree_max_depth_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_n_leaves_from_args_wrong_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let int_obj = match i64_to_py(py, 42_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [int_obj]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = py_tree_n_leaves_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_n_nodes_from_args_wrong_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let int_obj = match i64_to_py(py, 42_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [int_obj]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = py_tree_n_nodes_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_to_json_from_args_wrong_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let int_obj = match i64_to_py(py, 42_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [int_obj]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = py_tree_to_json_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_repr_from_args_wrong_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let int_obj = match i64_to_py(py, 42_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [int_obj]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = py_tree_repr_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// validate_py_tree
// =============================================================================

#[test]
fn test_validate_py_tree_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let py_tree = make_leaf_tree();
        let result = validate_py_tree(&py_tree);
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_validate_py_tree_empty_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let py_tree = PyTree {
            inner: Tree::new(vec![], 0_usize, 0_usize),
        };
        let result = validate_py_tree(&py_tree);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// PyTree clone
// =============================================================================

#[test]
fn test_pytree_clone() -> Result<(), ClearGbmError> {
    let py_tree = make_split_tree();
    let cloned = py_tree.clone();
    assert_eq!(py_tree_n_nodes_rs(&cloned), py_tree_n_nodes_rs(&py_tree));
    assert_eq!(
        py_tree_max_depth_rs(&cloned),
        py_tree_max_depth_rs(&py_tree)
    );
    assert_eq!(py_tree_n_leaves_rs(&cloned), py_tree_n_leaves_rs(&py_tree));
    Ok(())
}

// =============================================================================
// build_tree_from_args
// =============================================================================

#[test]
fn test_build_tree_from_args_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args(py, 2_i64) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };

        let result = match build_tree_from_args(&args) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        assert!(py_tree_n_nodes_rs(&result) >= 1_usize);
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_negative_n_features_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args(py, -1_i64) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };

        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Pass a string where numpy array is expected for arg0
        let bad = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [bad]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// build_tree_from_args: error path tests for unflatten_bins/thresholds
// =============================================================================

/// Configuration for building a custom build_tree_from_args test tuple.
struct CustomBuildTreeConfig<'a, 'py> {
    bins_flat: &'a [i64],
    n_features: i64,
    n_regular_bins: i64,
    bin_thresholds_flat: &'a [f64],
    monotonic_constraints: Bound<'py, PyAny>,
}

/// Helper: builds a custom 16-arg tuple for build_tree_from_args.
///
/// Uses 4 fixed samples with variable bins, features, thresholds, and constraints.
fn make_build_tree_args_custom<'py>(
    py: Python<'py>,
    config: CustomBuildTreeConfig<'_, 'py>,
) -> Result<Bound<'py, PyTuple>, ClearGbmError> {
    let si = PyArray1::from_vec(py, vec![0_i64, 1_i64, 2_i64, 3_i64]);
    let gr = PyArray1::from_vec(py, vec![0.1_f64, -0.2_f64, 0.3_f64, -0.1_f64]);
    let he = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64]);
    let bf = PyArray1::from_vec(py, config.bins_flat.to_vec());
    let bt = PyArray1::from_vec(py, config.bin_thresholds_flat.to_vec());

    let n_feat = match i64_to_py(py, config.n_features) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let n_reg = match i64_to_py(py, config.n_regular_bins) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let max_d = match i64_to_py(py, 3_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let max_l = match i64_to_py(py, 8_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let r_alpha = match f64_to_py(py, 0.0_f64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let r_lambda = match f64_to_py(py, 1.0_f64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let min_split = match i64_to_py(py, 2_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let min_leaf = match i64_to_py(py, 1_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let max_b = match i64_to_py(py, 3_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let min_g = match f64_to_py(py, 0.0_f64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    match PyTuple::new(
        py,
        [
            si.into_any(),
            gr.into_any(),
            he.into_any(),
            bf.into_any(),
            n_feat,
            n_reg,
            bt.into_any(),
            max_d,
            max_l,
            r_alpha,
            r_lambda,
            min_split,
            min_leaf,
            max_b,
            min_g,
            config.monotonic_constraints,
        ],
    ) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

#[test]
fn test_build_tree_from_args_zero_n_features_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // n_features=0 triggers unflatten_bins n_features=0 error
        let args = match make_build_tree_args(py, 0_i64) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_bins_flat_shape_mismatch_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // bins_flat has 5 elements, not divisible by n_features=2
        let args = match make_build_tree_args_custom(
            py,
            CustomBuildTreeConfig {
                bins_flat: &[0_i64, 1_i64, 1_i64, 2_i64, 0_i64], // length 5, not divisible by 2
                n_features: 2_i64,
                n_regular_bins: 3_i64,
                bin_thresholds_flat: &[0.0_f64, 0.5_f64, 1.0_f64, 0.0_f64, 0.5_f64, 1.0_f64],
                monotonic_constraints: py.None().into_bound(py).into_any(),
            },
        ) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_thresholds_flat_shape_mismatch_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // thresholds has wrong length: n_features=2, n_bins=3, so expected 6, but give 5
        let args = match make_build_tree_args_custom(
            py,
            CustomBuildTreeConfig {
                bins_flat: &[0_i64, 1_i64, 1_i64, 2_i64, 0_i64, 0_i64, 2_i64, 1_i64],
                n_features: 2_i64,
                n_regular_bins: 3_i64,
                bin_thresholds_flat: &[0.0_f64, 0.5_f64, 1.0_f64, 0.0_f64, 0.5_f64], // 5, expect 6
                monotonic_constraints: py.None().into_bound(py).into_any(),
            },
        ) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_negative_bin_value_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // bins_flat contains a negative value
        let args = match make_build_tree_args_custom(
            py,
            CustomBuildTreeConfig {
                bins_flat: &[0_i64, -1_i64, 1_i64, 2_i64, 0_i64, 0_i64, 2_i64, 1_i64],
                n_features: 2_i64,
                n_regular_bins: 3_i64,
                bin_thresholds_flat: &[0.0_f64, 0.5_f64, 1.0_f64, 0.0_f64, 0.5_f64, 1.0_f64],
                monotonic_constraints: py.None().into_bound(py).into_any(),
            },
        ) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_with_valid_monotonic_constraints() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // 2 features: constraint [0, 1] = [none, increasing]
        let constraints = PyArray1::from_vec(py, vec![0_i64, 1_i64]);
        let args = match make_build_tree_args_custom(
            py,
            CustomBuildTreeConfig {
                bins_flat: &[0_i64, 1_i64, 1_i64, 2_i64, 0_i64, 0_i64, 2_i64, 1_i64],
                n_features: 2_i64,
                n_regular_bins: 3_i64,
                bin_thresholds_flat: &[0.0_f64, 0.5_f64, 1.0_f64, 0.0_f64, 0.5_f64, 1.0_f64],
                monotonic_constraints: constraints.into_any(),
            },
        ) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_invalid_constraint_value_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Constraint value 2 is invalid (only -1, 0, 1 are valid)
        let constraints = PyArray1::from_vec(py, vec![0_i64, 2_i64]);
        let args = match make_build_tree_args_custom(
            py,
            CustomBuildTreeConfig {
                bins_flat: &[0_i64, 1_i64, 1_i64, 2_i64, 0_i64, 0_i64, 2_i64, 1_i64],
                n_features: 2_i64,
                n_regular_bins: 3_i64,
                bin_thresholds_flat: &[0.0_f64, 0.5_f64, 1.0_f64, 0.0_f64, 0.5_f64, 1.0_f64],
                monotonic_constraints: constraints.into_any(),
            },
        ) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_monotonic_constraint_overflow_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // i64::MAX exceeds i32 range, triggers i64_to_i32 error
        let constraints = PyArray1::from_vec(py, vec![0_i64, i64::MAX]);
        let args = match make_build_tree_args_custom(
            py,
            CustomBuildTreeConfig {
                bins_flat: &[0_i64, 1_i64, 1_i64, 2_i64, 0_i64, 0_i64, 2_i64, 1_i64],
                n_features: 2_i64,
                n_regular_bins: 3_i64,
                bin_thresholds_flat: &[0.0_f64, 0.5_f64, 1.0_f64, 0.0_f64, 0.5_f64, 1.0_f64],
                monotonic_constraints: constraints.into_any(),
            },
        ) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_gradients_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let si = PyArray1::from_vec(py, vec![0_i64, 1_i64]);
        let bad_grads = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        // Only need 2 args to trigger error on arg1 extraction
        let args = match PyTuple::new(py, [si.into_any(), bad_grads]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_hessians_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let si = PyArray1::from_vec(py, vec![0_i64]);
        let gr = PyArray1::from_vec(py, vec![0.1_f64]);
        let bad = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [si.into_any(), gr.into_any(), bad]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_bins_flat_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let si = PyArray1::from_vec(py, vec![0_i64]);
        let gr = PyArray1::from_vec(py, vec![0.1_f64]);
        let he = PyArray1::from_vec(py, vec![1.0_f64]);
        let bad = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [si.into_any(), gr.into_any(), he.into_any(), bad]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_n_features_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let si = PyArray1::from_vec(py, vec![0_i64]);
        let gr = PyArray1::from_vec(py, vec![0.1_f64]);
        let he = PyArray1::from_vec(py, vec![1.0_f64]);
        let bf = PyArray1::from_vec(py, vec![0_i64]);
        let bad = match str_to_py(py, "not_int") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(
            py,
            [
                si.into_any(),
                gr.into_any(),
                he.into_any(),
                bf.into_any(),
                bad,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_bin_thresholds_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let si = PyArray1::from_vec(py, vec![0_i64]);
        let gr = PyArray1::from_vec(py, vec![0.1_f64]);
        let he = PyArray1::from_vec(py, vec![1.0_f64]);
        let bf = PyArray1::from_vec(py, vec![0_i64]);
        let nf = match i64_to_py(py, 1_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let nb = match i64_to_py(py, 1_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let bad = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(
            py,
            [
                si.into_any(),
                gr.into_any(),
                he.into_any(),
                bf.into_any(),
                nf,
                nb,
                bad,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_max_depth_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let si = PyArray1::from_vec(py, vec![0_i64]);
        let gr = PyArray1::from_vec(py, vec![0.1_f64]);
        let he = PyArray1::from_vec(py, vec![1.0_f64]);
        let bf = PyArray1::from_vec(py, vec![0_i64]);
        let nf = match i64_to_py(py, 1_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let nb = match i64_to_py(py, 1_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let bt = PyArray1::from_vec(py, vec![0.5_f64]);
        let bad = match str_to_py(py, "not_int") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(
            py,
            [
                si.into_any(),
                gr.into_any(),
                he.into_any(),
                bf.into_any(),
                nf,
                nb,
                bt.into_any(),
                bad,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = build_tree_from_args(&args);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// build_tree_from_args: full-tuple wrong-type tests for remaining positions
// =============================================================================

/// Helper: builds all 16 valid build_tree args as a Vec<Bound<PyAny>>.
///
/// Allows callers to swap out a single element at a specific index
/// to test wrong-type error paths for that position.
fn make_valid_build_tree_arg_vec<'py>(
    py: Python<'py>,
) -> Result<Vec<Bound<'py, PyAny>>, ClearGbmError> {
    let si = PyArray1::from_vec(py, vec![0_i64, 1_i64, 2_i64, 3_i64]);
    let gr = PyArray1::from_vec(py, vec![0.1_f64, -0.2_f64, 0.3_f64, -0.1_f64]);
    let he = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64]);
    let bf = PyArray1::from_vec(
        py,
        vec![0_i64, 1_i64, 1_i64, 2_i64, 0_i64, 0_i64, 2_i64, 1_i64],
    );
    let bt = PyArray1::from_vec(
        py,
        vec![0.0_f64, 0.5_f64, 1.0_f64, 0.0_f64, 0.5_f64, 1.0_f64],
    );

    let nf = match i64_to_py(py, 2_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let nr = match i64_to_py(py, 3_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let md = match i64_to_py(py, 1_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let ml = match i64_to_py(py, 2_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let ra = match f64_to_py(py, 0.0_f64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let rl = match f64_to_py(py, 1.0_f64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let ms = match i64_to_py(py, 2_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let mf = match i64_to_py(py, 1_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let mb = match i64_to_py(py, 3_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let mg = match f64_to_py(py, 0.0_f64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    Ok(vec![
        si.into_any(),                       // 0: sample_indices
        gr.into_any(),                       // 1: gradients
        he.into_any(),                       // 2: hessians
        bf.into_any(),                       // 3: bins_flat
        nf,                                  // 4: n_features
        nr,                                  // 5: n_regular_bins
        bt.into_any(),                       // 6: bin_thresholds_flat
        md,                                  // 7: max_depth
        ml,                                  // 8: max_leaves
        ra,                                  // 9: reg_alpha
        rl,                                  // 10: reg_lambda
        ms,                                  // 11: min_samples_split
        mf,                                  // 12: min_samples_leaf
        mb,                                  // 13: max_bins
        mg,                                  // 14: min_gain
        py.None().into_bound(py).into_any(), // 15: monotonic_constraints
    ])
}

/// Helper: builds a build_tree args tuple with one position replaced by a bad value.
fn make_build_tree_args_with_bad_at(
    py: Python<'_>,
    bad_index: usize,
) -> Result<Bound<'_, PyTuple>, ClearGbmError> {
    let mut args_vec = match make_valid_build_tree_arg_vec(py) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let bad = match str_to_py(py, "bad") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    args_vec[bad_index] = bad;
    match PyTuple::new(py, args_vec) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

#[test]
fn test_build_tree_from_args_wrong_n_reg_bins_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_bad_at(py, 5_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_max_leaves_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_bad_at(py, 8_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_reg_alpha_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_bad_at(py, 9_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_reg_lambda_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_bad_at(py, 10_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_min_split_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_bad_at(py, 11_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_min_leaf_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_bad_at(py, 12_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_max_bins_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_bad_at(py, 13_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_min_gain_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_bad_at(py, 14_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_wrong_constraints_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_bad_at(py, 15_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_negative_max_depth_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let mut args_vec = match make_valid_build_tree_arg_vec(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let neg = match i64_to_py(py, -1_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        args_vec[7_usize] = neg;
        let args = match PyTuple::new(py, args_vec) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_empty_tuple_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// PyTree _from_args: empty-tuple tests (covers get_item(0) Err paths)
// =============================================================================

#[test]
fn test_py_tree_max_depth_from_args_empty_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(py_tree_max_depth_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_n_leaves_from_args_empty_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(py_tree_n_leaves_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_n_nodes_from_args_empty_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(py_tree_n_nodes_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_to_json_from_args_empty_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(py_tree_to_json_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_from_json_from_args_empty_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(py_tree_from_json_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_repr_from_args_empty_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(py_tree_repr_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_py_tree_from_json_from_args_wrong_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let bad = match i64_to_py(py, 42_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let args = match PyTuple::new(py, [bad]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(py_tree_from_json_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// Short-tuple tests for build_tree_from_args (cover get_item(N) Err N=1..7)
// =============================================================================

/// Helper: builds a build_tree args tuple with only the first N elements.
fn make_build_tree_args_short(
    py: Python<'_>,
    count: usize,
) -> Result<Bound<'_, PyTuple>, ClearGbmError> {
    let args_vec = match make_valid_build_tree_arg_vec(py) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let short = args_vec.into_iter().take(count).collect::<Vec<_>>();
    match PyTuple::new(py, short) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

#[test]
fn test_build_tree_from_args_one_arg_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 1_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_two_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 2_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_three_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 3_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_four_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 4_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_five_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 5_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_six_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 6_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_seven_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 7_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_eight_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 8_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_nine_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 9_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_ten_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 10_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_eleven_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 11_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_twelve_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 12_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_thirteen_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 13_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_fourteen_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_short(py, 14_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// Negative i64_to_usize tests (cover lines 472..492 in build_tree_from_args)
// =============================================================================

/// Helper: builds a build_tree args tuple with one position replaced by a negative i64.
fn make_build_tree_args_with_neg_at(
    py: Python<'_>,
    neg_index: usize,
) -> Result<Bound<'_, PyTuple>, ClearGbmError> {
    let mut args_vec = match make_valid_build_tree_arg_vec(py) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let neg = match i64_to_py(py, -1_i64) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    args_vec[neg_index] = neg;
    match PyTuple::new(py, args_vec) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

#[test]
fn test_build_tree_from_args_negative_n_regular_bins_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_neg_at(py, 5_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_negative_max_leaves_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_neg_at(py, 8_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_negative_min_samples_split_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_neg_at(py, 11_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_negative_min_samples_leaf_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_neg_at(py, 12_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_negative_max_bins_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_with_neg_at(py, 13_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// Non-contiguous array tests for build_tree_from_args as_slice Err paths
// =============================================================================

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

/// Helper: creates a non-contiguous 1D i64 numpy array by slicing with stride 2.
fn make_non_contiguous_i64<'py>(py: Python<'py>) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    let numpy = match py.import("numpy") {
        Ok(m) => m,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let dtype = match numpy.getattr("int64") {
        Ok(d) => d,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let kwargs = pyo3::types::PyDict::new(py);
    match kwargs.set_item("dtype", dtype) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let arr = match numpy.call_method(
        "array",
        (vec![0_i64, 1_i64, 2_i64, 3_i64, 4_i64, 5_i64],),
        Some(&kwargs),
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

/// Helper: builds build_tree args with one array position replaced by a non-contiguous array.
fn make_build_tree_args_non_contiguous_at(
    py: Python<'_>,
    nc_index: usize,
    is_i64: bool,
) -> Result<Bound<'_, PyTuple>, ClearGbmError> {
    let mut args_vec = match make_valid_build_tree_arg_vec(py) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let nc = if is_i64 {
        match make_non_contiguous_i64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        }
    } else {
        match make_non_contiguous_f64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        }
    };
    args_vec[nc_index] = nc;
    match PyTuple::new(py, args_vec) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

#[test]
fn test_build_tree_from_args_non_contiguous_sample_indices_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_non_contiguous_at(py, 0_usize, true) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_non_contiguous_gradients_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_non_contiguous_at(py, 1_usize, false) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_non_contiguous_hessians_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_non_contiguous_at(py, 2_usize, false) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_non_contiguous_bins_flat_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_non_contiguous_at(py, 3_usize, true) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_tree_from_args_non_contiguous_bin_thresholds_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_build_tree_args_non_contiguous_at(py, 6_usize, false) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// Additional coverage tests for tree_fns.rs
// =============================================================================

/// Line 462: get_item(15) returns Err → None (optional monotonic_constraints absent)
#[test]
fn test_build_tree_from_args_fifteen_args_no_constraints() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Build a valid 15-arg tuple (omit monotonic_constraints at position 15)
        let args = match make_build_tree_args_short(py, 15_usize) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        // Should succeed — constraints are optional and default to None
        let result = build_tree_from_args(&args);
        assert!(result.is_ok());
        Ok(())
    })
}

/// Line 507: i64_slice_to_usize_vec fails when sample_indices contain negative values.
#[test]
fn test_build_tree_from_args_negative_sample_index_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let mut args_vec = match make_valid_build_tree_arg_vec(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        // Replace sample_indices with one containing a negative value
        let bad_indices = PyArray1::from_vec(py, vec![-1_i64, 0_i64, 1_i64, 2_i64]);
        args_vec[0_usize] = bad_indices.into_any();
        let args = match PyTuple::new(py, args_vec) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

/// Line 572: SplitConfig::new fails with max_bins=0
#[test]
fn test_build_tree_from_args_zero_max_bins_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let mut args_vec = match make_valid_build_tree_arg_vec(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        // max_bins is position 13 — set to 0 to trigger SplitConfig validation error
        let zero = match i64_to_py(py, 0_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        args_vec[13_usize] = zero;
        let args = match PyTuple::new(py, args_vec) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

/// Line 583: TreeBuildConfig::new fails with negative reg_alpha
#[test]
fn test_build_tree_from_args_neg_reg_alpha_config_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let mut args_vec = match make_valid_build_tree_arg_vec(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        // reg_alpha is position 9 — set to -1.0 to trigger TreeBuildConfig validation error
        let neg = match f64_to_py(py, -1.0_f64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        args_vec[9_usize] = neg;
        let args = match PyTuple::new(py, args_vec) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

/// Lines 246-247: convert_monotonic_constraints as_slice Err (non-contiguous constraints)
#[test]
fn test_build_tree_from_args_non_contiguous_constraints_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let mut args_vec = match make_valid_build_tree_arg_vec(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        // Replace monotonic_constraints (position 15) with a non-contiguous i64 array
        let nc = match make_non_contiguous_i64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        args_vec[15_usize] = nc;
        let args = match PyTuple::new(py, args_vec) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

/// Line 600: build_tree core Err (mismatched sample_indices vs gradients length)
#[test]
fn test_build_tree_from_args_core_build_tree_error_propagates() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let mut args_vec = match make_valid_build_tree_arg_vec(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        // Replace sample_indices with 10 indices, but gradients only has 4 elements
        // build_tree will fail with ShapeMismatch: gradients.len() < n_samples
        let big_indices = PyArray1::from_vec(
            py,
            vec![
                0_i64, 1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64, 9_i64,
            ],
        );
        args_vec[0_usize] = big_indices.into_any();
        let args = match PyTuple::new(py, args_vec) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_tree_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// Error helper: ser_err
// =============================================================================

#[test]
fn test_ser_err_produces_pyerr() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let py_err = ser_err("test serialization failure".to_string());
        let err_str = py_err.to_string();
        assert!(
            err_str.contains("test serialization failure"),
            "expected error message, got: {err_str}"
        );
        let _ = py;
        Ok(())
    })
}
