//! Tests for PyO3 histogram binding functions.
//!
//! Tests [`super::super::histogram_fns::build_histogram_rs`],
//! [`super::super::histogram_fns::subtract_histogram_rs`], and their
//! `_from_args` wrappers through the PyO3 runtime.

use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::ClearGbmError;
use crate::pyo3_module::histogram_fns::{
    build_histogram_from_args, build_histogram_rs, subtract_histogram_from_args,
    subtract_histogram_rs,
};

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
// build_histogram_rs
// =============================================================================

#[test]
fn test_build_histogram_rs_valid_input() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64, 1_i64, 2_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64, 0.2_f64, 0.3_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64, 1_i64, 0_i64]);

        let result = build_histogram_rs(
            py,
            &indices.readonly(),
            &grads.readonly(),
            &hess.readonly(),
            &bins.readonly(),
            3_i64,
        );

        let (grad_sums, hess_sums, counts) = match result {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Bin 0: samples 0,2 -> grad=0.1+0.3=0.4, hess=2.0, count=2
        // Bin 1: sample 1 -> grad=0.2, hess=1.0, count=1
        assert_eq!(grad_sums.len(), 3_usize);
        assert_eq!(hess_sums.len(), 3_usize);
        assert_eq!(counts.len(), 3_usize);

        let g = grad_sums.readonly();
        let g_slice = match g.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::EmptyInput {
                    context: format!("grad_sums: {e}"),
                })
            }
        };
        assert!((g_slice[0_usize] - 0.4_f64).abs() < 1e-10_f64);
        assert!((g_slice[1_usize] - 0.2_f64).abs() < 1e-10_f64);
        assert!((g_slice[2_usize] - 0.0_f64).abs() < 1e-10_f64);

        let c = counts.readonly();
        let c_slice = match c.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::EmptyInput {
                    context: format!("counts: {e}"),
                })
            }
        };
        assert_eq!(c_slice[0_usize], 2_u64);
        assert_eq!(c_slice[1_usize], 1_u64);
        assert_eq!(c_slice[2_usize], 0_u64);

        Ok(())
    })
}

#[test]
fn test_build_histogram_rs_single_sample() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let grads = PyArray1::from_vec(py, vec![0.5_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64]);

        let result = build_histogram_rs(
            py,
            &indices.readonly(),
            &grads.readonly(),
            &hess.readonly(),
            &bins.readonly(),
            2_i64,
        );

        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_build_histogram_rs_negative_index_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64, -1_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64, 0.2_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64, 0_i64]);

        let result = build_histogram_rs(
            py,
            &indices.readonly(),
            &grads.readonly(),
            &hess.readonly(),
            &bins.readonly(),
            2_i64,
        );

        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_histogram_rs_negative_n_bins_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64]);

        let result = build_histogram_rs(
            py,
            &indices.readonly(),
            &grads.readonly(),
            &hess.readonly(),
            &bins.readonly(),
            -1_i64,
        );

        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_histogram_rs_negative_bin_assignment_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![-1_i64]);

        let result = build_histogram_rs(
            py,
            &indices.readonly(),
            &grads.readonly(),
            &hess.readonly(),
            &bins.readonly(),
            2_i64,
        );

        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// build_histogram_from_args
// =============================================================================

#[test]
fn test_build_histogram_from_args_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64, 1_i64, 2_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64, 0.2_f64, 0.3_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64, 1_i64, 0_i64]);
        let n_bins = match i64_to_py(py, 3_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(
            py,
            [
                indices.into_any(),
                grads.into_any(),
                hess.into_any(),
                bins.into_any(),
                n_bins,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = build_histogram_from_args(&args);
        assert!(result.is_ok());

        Ok(())
    })
}

#[test]
fn test_build_histogram_from_args_wrong_indices_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let grads = PyArray1::from_vec(py, vec![0.1_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64]);
        let n_bins = match i64_to_py(py, 2_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(
            py,
            [
                not_array,
                grads.into_any(),
                hess.into_any(),
                bins.into_any(),
                n_bins,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = build_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_build_histogram_from_args_wrong_grads_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64]);
        let n_bins = match i64_to_py(py, 2_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(
            py,
            [
                indices.into_any(),
                not_array,
                hess.into_any(),
                bins.into_any(),
                n_bins,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = build_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_build_histogram_from_args_wrong_hess_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64]);
        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let bins = PyArray1::from_vec(py, vec![0_i64]);
        let n_bins = match i64_to_py(py, 2_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(
            py,
            [
                indices.into_any(),
                grads.into_any(),
                not_array,
                bins.into_any(),
                n_bins,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = build_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_build_histogram_from_args_wrong_bins_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let n_bins = match i64_to_py(py, 2_i64) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(
            py,
            [
                indices.into_any(),
                grads.into_any(),
                hess.into_any(),
                not_array,
                n_bins,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = build_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_build_histogram_from_args_wrong_n_bins_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64]);
        let not_int = match str_to_py(py, "not_an_int") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(
            py,
            [
                indices.into_any(),
                grads.into_any(),
                hess.into_any(),
                bins.into_any(),
                not_int,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = build_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

// =============================================================================
// subtract_histogram_rs
// =============================================================================

#[test]
fn test_subtract_histogram_rs_valid_input() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Parent: grad=[1.0, 2.0], hess=[3.0, 4.0], counts=[5, 6]
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64, 4.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64, 6_u64]);

        // Child: grad=[0.3, 0.7], hess=[1.0, 1.5], counts=[2, 3]
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64, 0.7_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.5_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64, 3_u64]);

        let result = subtract_histogram_rs(
            py,
            &p_grads.readonly(),
            &p_hess.readonly(),
            &p_counts.readonly(),
            &c_grads.readonly(),
            &c_hess.readonly(),
            &c_counts.readonly(),
        );

        let (grad_sums, hess_sums, counts) = match result {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Sibling: grad=[0.7, 1.3], hess=[2.0, 2.5], counts=[3, 3]
        let g = grad_sums.readonly();
        let g_slice = match g.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::EmptyInput {
                    context: format!("grad_sums: {e}"),
                })
            }
        };
        assert!((g_slice[0_usize] - 0.7_f64).abs() < 1e-10_f64);
        assert!((g_slice[1_usize] - 1.3_f64).abs() < 1e-10_f64);

        let h = hess_sums.readonly();
        let h_slice = match h.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::EmptyInput {
                    context: format!("hess_sums: {e}"),
                })
            }
        };
        assert!((h_slice[0_usize] - 2.0_f64).abs() < 1e-10_f64);
        assert!((h_slice[1_usize] - 2.5_f64).abs() < 1e-10_f64);

        let c = counts.readonly();
        let c_slice = match c.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::EmptyInput {
                    context: format!("counts: {e}"),
                })
            }
        };
        assert_eq!(c_slice[0_usize], 3_u64);
        assert_eq!(c_slice[1_usize], 3_u64);

        Ok(())
    })
}

#[test]
fn test_subtract_histogram_rs_mismatched_lengths_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64, 4.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64, 6_u64]);

        // Child has different length (3 bins vs 2)
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64, 0.7_f64, 0.1_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.5_f64, 0.5_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64, 3_u64, 1_u64]);

        let result = subtract_histogram_rs(
            py,
            &p_grads.readonly(),
            &p_hess.readonly(),
            &p_counts.readonly(),
            &c_grads.readonly(),
            &c_hess.readonly(),
            &c_counts.readonly(),
        );

        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_rs_hess_length_mismatch_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Parent with mismatched hess length
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64]); // Wrong length
        let p_counts = PyArray1::from_vec(py, vec![5_u64, 6_u64]);

        let c_grads = PyArray1::from_vec(py, vec![0.3_f64, 0.7_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.5_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64, 3_u64]);

        let result = subtract_histogram_rs(
            py,
            &p_grads.readonly(),
            &p_hess.readonly(),
            &p_counts.readonly(),
            &c_grads.readonly(),
            &c_hess.readonly(),
            &c_counts.readonly(),
        );

        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_rs_count_length_mismatch_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Parent with mismatched count length
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64, 4.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64]); // Wrong length

        let c_grads = PyArray1::from_vec(py, vec![0.3_f64, 0.7_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.5_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64, 3_u64]);

        let result = subtract_histogram_rs(
            py,
            &p_grads.readonly(),
            &p_hess.readonly(),
            &p_counts.readonly(),
            &c_grads.readonly(),
            &c_hess.readonly(),
            &c_counts.readonly(),
        );

        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// subtract_histogram_from_args
// =============================================================================

#[test]
fn test_subtract_histogram_from_args_valid() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64, 4.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64, 6_u64]);
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64, 0.7_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.5_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64, 3_u64]);

        let args = match PyTuple::new(
            py,
            [
                p_grads.into_any(),
                p_hess.into_any(),
                p_counts.into_any(),
                c_grads.into_any(),
                c_hess.into_any(),
                c_counts.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = subtract_histogram_from_args(&args);
        assert!(result.is_ok());

        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_wrong_parent_grads_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64]);
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64]);

        let args = match PyTuple::new(
            py,
            [
                not_array,
                p_hess.into_any(),
                p_counts.into_any(),
                c_grads.into_any(),
                c_hess.into_any(),
                c_counts.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = subtract_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_wrong_child_grads_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64]);
        let not_array = match str_to_py(py, "not_an_array") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64]);

        let args = match PyTuple::new(
            py,
            [
                p_grads.into_any(),
                p_hess.into_any(),
                p_counts.into_any(),
                not_array,
                c_hess.into_any(),
                c_counts.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = subtract_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_wrong_parent_hess_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let not_array = match str_to_py(py, "bad") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let p_counts = PyArray1::from_vec(py, vec![5_u64]);
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64]);

        let args = match PyTuple::new(
            py,
            [
                p_grads.into_any(),
                not_array,
                p_counts.into_any(),
                c_grads.into_any(),
                c_hess.into_any(),
                c_counts.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = subtract_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_wrong_parent_counts_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64]);
        let not_array = match str_to_py(py, "bad") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64]);

        let args = match PyTuple::new(
            py,
            [
                p_grads.into_any(),
                p_hess.into_any(),
                not_array,
                c_grads.into_any(),
                c_hess.into_any(),
                c_counts.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = subtract_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_wrong_child_hess_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64]);
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64]);
        let not_array = match str_to_py(py, "bad") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let c_counts = PyArray1::from_vec(py, vec![2_u64]);

        let args = match PyTuple::new(
            py,
            [
                p_grads.into_any(),
                p_hess.into_any(),
                p_counts.into_any(),
                c_grads.into_any(),
                not_array,
                c_counts.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = subtract_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_wrong_child_counts_type_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64]);
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let not_array = match str_to_py(py, "bad") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

        let args = match PyTuple::new(
            py,
            [
                p_grads.into_any(),
                p_hess.into_any(),
                p_counts.into_any(),
                c_grads.into_any(),
                c_hess.into_any(),
                not_array,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = subtract_histogram_from_args(&args);
        assert!(result.is_err());

        Ok(())
    })
}

// =============================================================================
// Additional error path tests
// =============================================================================

#[test]
fn test_build_histogram_rs_out_of_bounds_index_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Index 100 is out-of-bounds for arrays of length 3
        let indices = PyArray1::from_vec(py, vec![0_i64, 100_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64, 0.2_f64, 0.3_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64, 1_i64, 0_i64]);

        let result = build_histogram_rs(
            py,
            &indices.readonly(),
            &grads.readonly(),
            &hess.readonly(),
            &bins.readonly(),
            3_i64,
        );

        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_rs_child_hess_mismatch_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Valid parent
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64, 4.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64, 6_u64]);

        // Child with internal hess length mismatch
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64, 0.7_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64]); // Wrong length
        let c_counts = PyArray1::from_vec(py, vec![2_u64, 3_u64]);

        let result = subtract_histogram_rs(
            py,
            &p_grads.readonly(),
            &p_hess.readonly(),
            &p_counts.readonly(),
            &c_grads.readonly(),
            &c_hess.readonly(),
            &c_counts.readonly(),
        );

        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_rs_child_count_mismatch_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Valid parent
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![3.0_f64, 4.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![5_u64, 6_u64]);

        // Child with internal count length mismatch
        let c_grads = PyArray1::from_vec(py, vec![0.3_f64, 0.7_f64]);
        let c_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.5_f64]);
        let c_counts = PyArray1::from_vec(py, vec![2_u64]); // Wrong length

        let result = subtract_histogram_rs(
            py,
            &p_grads.readonly(),
            &p_hess.readonly(),
            &p_counts.readonly(),
            &c_grads.readonly(),
            &c_hess.readonly(),
            &c_counts.readonly(),
        );

        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// Empty tuple tests (cover get_item(0) Err paths in _from_args)
// =============================================================================

#[test]
fn test_build_histogram_from_args_empty_tuple_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(build_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_empty_tuple_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = PyTuple::empty(py);
        assert!(subtract_histogram_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// Non-contiguous array tests (cover as_slice() Err paths in _rs functions)
// =============================================================================

/// Creates a non-contiguous 1D f64 numpy array by slicing with stride 2.
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

/// Creates a non-contiguous 1D i64 numpy array by slicing with stride 2.
fn make_non_contiguous_i64<'py>(py: Python<'py>) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    let numpy = match py.import("numpy") {
        Ok(m) => m,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    // Explicitly use dtype=numpy.int64 so extraction as PyReadonlyArray1<i64> succeeds.
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

#[test]
fn test_build_histogram_rs_non_contiguous_indices_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let nc_indices = match make_non_contiguous_i64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let indices: numpy::PyReadonlyArray1<'_, i64> = match nc_indices.extract() {
            Ok(v) => v,
            Err(e) => {
                let py_err: PyErr = e.into();
                return Err(wrap_py_err(&py_err));
            }
        };
        let grads = PyArray1::from_vec(py, vec![0.1_f64, 0.2_f64, 0.3_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64, 1_i64, 0_i64]);

        let result = build_histogram_rs(
            py,
            &indices,
            &grads.readonly(),
            &hess.readonly(),
            &bins.readonly(),
            3_i64,
        );
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_histogram_rs_non_contiguous_grads_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64, 1_i64, 2_i64]);
        let nc_grads = match make_non_contiguous_f64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let grads: numpy::PyReadonlyArray1<'_, f64> = match nc_grads.extract() {
            Ok(v) => v,
            Err(e) => {
                let py_err: PyErr = e.into();
                return Err(wrap_py_err(&py_err));
            }
        };
        let hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64, 1_i64, 0_i64]);

        let result = build_histogram_rs(
            py,
            &indices.readonly(),
            &grads,
            &hess.readonly(),
            &bins.readonly(),
            3_i64,
        );
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_histogram_rs_non_contiguous_hess_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64, 1_i64, 2_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64, 0.2_f64, 0.3_f64]);
        let nc_hess = match make_non_contiguous_f64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let hess: numpy::PyReadonlyArray1<'_, f64> = match nc_hess.extract() {
            Ok(v) => v,
            Err(e) => {
                let py_err: PyErr = e.into();
                return Err(wrap_py_err(&py_err));
            }
        };
        let bins = PyArray1::from_vec(py, vec![0_i64, 1_i64, 0_i64]);

        let result = build_histogram_rs(
            py,
            &indices.readonly(),
            &grads.readonly(),
            &hess,
            &bins.readonly(),
            3_i64,
        );
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_build_histogram_rs_non_contiguous_bins_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64, 1_i64, 2_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64, 0.2_f64, 0.3_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let nc_bins = match make_non_contiguous_i64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let bins: numpy::PyReadonlyArray1<'_, i64> = match nc_bins.extract() {
            Ok(v) => v,
            Err(e) => {
                let py_err: PyErr = e.into();
                return Err(wrap_py_err(&py_err));
            }
        };

        let result = build_histogram_rs(
            py,
            &indices.readonly(),
            &grads.readonly(),
            &hess.readonly(),
            &bins,
            3_i64,
        );
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// Short tuple tests (cover get_item(1) Err paths in _from_args)
// =============================================================================

#[test]
fn test_build_histogram_from_args_one_arg_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let args = match PyTuple::new(py, [indices.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_histogram_from_args_two_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64]);
        let args = match PyTuple::new(py, [indices.into_any(), grads.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_histogram_from_args_three_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let args = match PyTuple::new(py, [indices.into_any(), grads.into_any(), hess.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_build_histogram_from_args_four_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![0_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64]);
        let args = match PyTuple::new(
            py,
            [
                indices.into_any(),
                grads.into_any(),
                hess.into_any(),
                bins.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_one_arg_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let args = match PyTuple::new(py, [p_grads.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(subtract_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_two_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let args = match PyTuple::new(py, [p_grads.into_any(), p_hess.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(subtract_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_three_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![1_u64]);
        let args = match PyTuple::new(
            py,
            [p_grads.into_any(), p_hess.into_any(), p_counts.into_any()],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(subtract_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_four_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![1_u64]);
        let c_grads = PyArray1::from_vec(py, vec![0.5_f64]);
        let args = match PyTuple::new(
            py,
            [
                p_grads.into_any(),
                p_hess.into_any(),
                p_counts.into_any(),
                c_grads.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(subtract_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_five_args_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![1.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![1_u64]);
        let c_grads = PyArray1::from_vec(py, vec![0.5_f64]);
        let c_hess = PyArray1::from_vec(py, vec![0.5_f64]);
        let args = match PyTuple::new(
            py,
            [
                p_grads.into_any(),
                p_hess.into_any(),
                p_counts.into_any(),
                c_grads.into_any(),
                c_hess.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(subtract_histogram_from_args(&args).is_err());
        Ok(())
    })
}

// =============================================================================
// Non-contiguous arrays in subtract_histogram_rs (cover build_histogram_buffer_from_arrays)
// =============================================================================

/// Creates a non-contiguous 1D u64 numpy array by slicing with stride 2.
fn make_non_contiguous_u64<'py>(py: Python<'py>) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    let numpy = match py.import("numpy") {
        Ok(m) => m,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let dtype = match numpy.getattr("uint64") {
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
        (vec![1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64],),
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

#[test]
fn test_subtract_histogram_rs_non_contiguous_parent_grads_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let nc = match make_non_contiguous_f64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let parent_grads: numpy::PyReadonlyArray1<'_, f64> = match nc.extract() {
            Ok(v) => v,
            Err(e) => {
                let py_err: PyErr = e.into();
                return Err(wrap_py_err(&py_err));
            }
        };
        let parent_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let parent_counts = PyArray1::from_vec(py, vec![1_u64, 1_u64, 1_u64]);
        let child_grads = PyArray1::from_vec(py, vec![0.5_f64, 0.5_f64, 0.5_f64]);
        let child_hess = PyArray1::from_vec(py, vec![0.5_f64, 0.5_f64, 0.5_f64]);
        let child_counts = PyArray1::from_vec(py, vec![1_u64, 1_u64, 1_u64]);

        let result = subtract_histogram_rs(
            py,
            &parent_grads,
            &parent_hess.readonly(),
            &parent_counts.readonly(),
            &child_grads.readonly(),
            &child_hess.readonly(),
            &child_counts.readonly(),
        );
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_rs_non_contiguous_parent_hess_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let nc = match make_non_contiguous_f64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let parent_hess: numpy::PyReadonlyArray1<'_, f64> = match nc.extract() {
            Ok(v) => v,
            Err(e) => {
                let py_err: PyErr = e.into();
                return Err(wrap_py_err(&py_err));
            }
        };
        let parent_grads = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let parent_counts = PyArray1::from_vec(py, vec![1_u64, 1_u64, 1_u64]);
        let child_grads = PyArray1::from_vec(py, vec![0.5_f64, 0.5_f64, 0.5_f64]);
        let child_hess = PyArray1::from_vec(py, vec![0.5_f64, 0.5_f64, 0.5_f64]);
        let child_counts = PyArray1::from_vec(py, vec![1_u64, 1_u64, 1_u64]);

        let result = subtract_histogram_rs(
            py,
            &parent_grads.readonly(),
            &parent_hess,
            &parent_counts.readonly(),
            &child_grads.readonly(),
            &child_hess.readonly(),
            &child_counts.readonly(),
        );
        assert!(result.is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_rs_non_contiguous_parent_counts_fails() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let nc = match make_non_contiguous_u64(py) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let parent_counts: numpy::PyReadonlyArray1<'_, u64> = match nc.extract() {
            Ok(v) => v,
            Err(e) => {
                let py_err: PyErr = e.into();
                return Err(wrap_py_err(&py_err));
            }
        };
        let parent_grads = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let parent_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let child_grads = PyArray1::from_vec(py, vec![0.5_f64, 0.5_f64, 0.5_f64]);
        let child_hess = PyArray1::from_vec(py, vec![0.5_f64, 0.5_f64, 0.5_f64]);
        let child_counts = PyArray1::from_vec(py, vec![1_u64, 1_u64, 1_u64]);

        let result = subtract_histogram_rs(
            py,
            &parent_grads.readonly(),
            &parent_hess.readonly(),
            &parent_counts,
            &child_grads.readonly(),
            &child_hess.readonly(),
            &child_counts.readonly(),
        );
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// _from_args core error propagation
// =============================================================================

#[test]
fn test_build_histogram_from_args_core_error_propagates() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let indices = PyArray1::from_vec(py, vec![100_i64]);
        let grads = PyArray1::from_vec(py, vec![0.1_f64, 0.2_f64, 0.3_f64]);
        let hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let bins = PyArray1::from_vec(py, vec![0_i64, 1_i64, 0_i64]);
        let n_bins_val = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("i64 to py failed: {e}"),
                })
            }
        };
        let args = match PyTuple::new(
            py,
            [
                indices.into_any(),
                grads.into_any(),
                hess.into_any(),
                bins.into_any(),
                n_bins_val,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(build_histogram_from_args(&args).is_err());
        Ok(())
    })
}

#[test]
fn test_subtract_histogram_from_args_core_error_propagates() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let p_grads = PyArray1::from_vec(py, vec![1.0_f64, 2.0_f64, 3.0_f64]);
        let p_hess = PyArray1::from_vec(py, vec![1.0_f64, 1.0_f64, 1.0_f64]);
        let p_counts = PyArray1::from_vec(py, vec![1_u64, 1_u64, 1_u64]);
        let c_grads = PyArray1::from_vec(py, vec![0.5_f64, 0.5_f64]);
        let c_hess = PyArray1::from_vec(py, vec![0.5_f64, 0.5_f64]);
        let c_counts = PyArray1::from_vec(py, vec![1_u64, 1_u64]);
        let args = match PyTuple::new(
            py,
            [
                p_grads.into_any(),
                p_hess.into_any(),
                p_counts.into_any(),
                c_grads.into_any(),
                c_hess.into_any(),
                c_counts.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(subtract_histogram_from_args(&args).is_err());
        Ok(())
    })
}
