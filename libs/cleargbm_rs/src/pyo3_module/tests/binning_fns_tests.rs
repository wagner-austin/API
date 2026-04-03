//! Tests for PyO3 binning function bindings.
//!
//! Tests [`super::super::binning_fns`] functions through the PyO3 runtime.

use numpy::{PyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::error::ClearGbmError;
use crate::pyo3_module::binning_fns::{
    bin_samples_from_args, build_bins_array, compute_bin_edges_from_args,
    precompute_feature_bins_from_args,
};

/// Helper: wraps a PyErr into ClearGbmError for test return types.
fn wrap_py_err(e: &PyErr) -> ClearGbmError {
    ClearGbmError::TreeConstructionFailed {
        reason: format!("PyErr: {e}"),
    }
}

// =============================================================================
// precompute_feature_bins_from_args
// =============================================================================

#[test]
fn test_precompute_feature_bins_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![
            vec![1.0_f64, 10.0_f64],
            vec![2.0_f64, 20.0_f64],
            vec![3.0_f64, 30.0_f64],
            vec![4.0_f64, 40.0_f64],
        ];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let max_bins = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = match precompute_feature_bins_from_args(&tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Result is a tuple of (bin_thresholds, sample_bins, n_regular_bins)
        let result_tuple: &Bound<'_, PyTuple> = match result.bind(py).cast::<PyTuple>() {
            Ok(t) => t,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("downcast failed: {e}"),
                })
            }
        };
        assert_eq!(result_tuple.len(), 3_usize);
        Ok(())
    })
}

#[test]
fn test_precompute_feature_bins_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match precompute_feature_bins_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for empty args".to_string(),
            }),
        }
    })
}

#[test]
fn test_precompute_feature_bins_wrong_features_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Pass i64 array instead of f64
        let data = vec![vec![1_i64, 2_i64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let max_bins = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match precompute_feature_bins_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong features type".to_string(),
            }),
        }
    })
}

#[test]
fn test_precompute_feature_bins_wrong_max_bins_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        // Pass string instead of int
        let max_bins = match "three".into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match precompute_feature_bins_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong max_bins type".to_string(),
            }),
        }
    })
}

// =============================================================================
// compute_bin_edges_from_args
// =============================================================================

#[test]
fn test_compute_bin_edges_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![
            vec![1.0_f64, 10.0_f64],
            vec![2.0_f64, 20.0_f64],
            vec![3.0_f64, 30.0_f64],
        ];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let max_bins = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = match compute_bin_edges_from_args(&tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Result is a list of lists
        let result_list: &Bound<'_, PyList> = match result.bind(py).cast::<PyList>() {
            Ok(l) => l,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("downcast failed: {e}"),
                })
            }
        };
        // Should have one entry per feature (2 features)
        assert_eq!(result_list.len(), 2_usize);
        Ok(())
    })
}

#[test]
fn test_compute_bin_edges_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match compute_bin_edges_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for empty args".to_string(),
            }),
        }
    })
}

#[test]
fn test_compute_bin_edges_wrong_features_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1_i64, 2_i64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let max_bins = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match compute_bin_edges_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong features type".to_string(),
            }),
        }
    })
}

// =============================================================================
// bin_samples_from_args
// =============================================================================

#[test]
fn test_bin_samples_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![
            vec![1.0_f64, 10.0_f64],
            vec![2.0_f64, 20.0_f64],
            vec![3.0_f64, 30.0_f64],
        ];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };

        // First compute edges
        let max_bins = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let edges_tuple = match PyTuple::new(py, [features.as_any().clone(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let edges_result = match compute_bin_edges_from_args(&edges_tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Now bin samples
        let n_regular_bins = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let bin_tuple = match PyTuple::new(
            py,
            [
                features.into_any(),
                edges_result.into_bound(py).into_any(),
                n_regular_bins,
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = match bin_samples_from_args(&bin_tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Result should be a 2D i64 array
        let arr: &Bound<'_, PyArray2<i64>> = match result.bind(py).cast::<PyArray2<i64>>() {
            Ok(a) => a,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("downcast failed: {e}"),
                })
            }
        };
        let shape = arr.shape();
        assert_eq!(shape[0_usize], 3_usize);
        assert_eq!(shape[1_usize], 2_usize);
        Ok(())
    })
}

#[test]
fn test_bin_samples_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match bin_samples_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for empty args".to_string(),
            }),
        }
    })
}

#[test]
fn test_bin_samples_wrong_features_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1_i64, 2_i64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let edges = PyList::empty(py);
        let n_regular_bins = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), edges.into_any(), n_regular_bins])
        {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match bin_samples_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong features type".to_string(),
            }),
        }
    })
}

#[test]
fn test_bin_samples_missing_n_regular_bins() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let edges = PyList::empty(py);
        // Only pass 2 args (missing n_regular_bins)
        let tuple = match PyTuple::new(py, [features.into_any(), edges.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match bin_samples_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for missing n_regular_bins".to_string(),
            }),
        }
    })
}

// =============================================================================
// NaN handling
// =============================================================================

#[test]
fn test_precompute_feature_bins_with_nan() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![
            vec![1.0_f64, f64::NAN],
            vec![2.0_f64, 20.0_f64],
            vec![f64::NAN, 30.0_f64],
        ];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let max_bins = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // NaN values should be handled by the binning module
        let result = precompute_feature_bins_from_args(&tuple);
        assert!(result.is_ok());
        Ok(())
    })
}

// =============================================================================
// Additional coverage tests
// =============================================================================

/// `precompute_feature_bins_from_args` with max_bins=1 (below minimum) triggers binning error.
#[test]
fn test_precompute_feature_bins_max_bins_one() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64], vec![3.0_f64, 4.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let max_bins = match 1_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match precompute_feature_bins_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for max_bins=1".to_string(),
            }),
        }
    })
}

/// `precompute_feature_bins_from_args` with 1-item tuple (missing max_bins).
#[test]
fn test_precompute_feature_bins_one_arg_only() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let tuple = match PyTuple::new(py, [features.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match precompute_feature_bins_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for 1-arg tuple".to_string(),
            }),
        }
    })
}

/// `compute_bin_edges_from_args` with max_bins=1 triggers binning error.
#[test]
fn test_compute_bin_edges_max_bins_one() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64], vec![3.0_f64, 4.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let max_bins = match 1_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match compute_bin_edges_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for max_bins=1".to_string(),
            }),
        }
    })
}

/// `compute_bin_edges_from_args` with only 1 arg (missing max_bins).
#[test]
fn test_compute_bin_edges_one_arg_only() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let tuple = match PyTuple::new(py, [features.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match compute_bin_edges_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for 1-arg tuple".to_string(),
            }),
        }
    })
}

/// `compute_bin_edges_from_args` with wrong type for max_bins (string).
#[test]
fn test_compute_bin_edges_wrong_max_bins_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let max_bins = match "three".into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match compute_bin_edges_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong max_bins type".to_string(),
            }),
        }
    })
}

/// `compute_bin_edges_from_args` with negative max_bins.
#[test]
fn test_compute_bin_edges_negative_max_bins() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let max_bins = match (-1_i64).into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match compute_bin_edges_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for negative max_bins".to_string(),
            }),
        }
    })
}

/// `bin_samples_from_args` with only 1 arg (missing bin_edges and n_regular_bins).
#[test]
fn test_bin_samples_one_arg_only() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let tuple = match PyTuple::new(py, [features.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match bin_samples_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for 1-arg tuple".to_string(),
            }),
        }
    })
}

/// `bin_samples_from_args` with wrong type for bin_edges (int instead of list).
#[test]
fn test_bin_samples_wrong_edges_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let bad_edges = match 42_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let nrb = match 3_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), bad_edges, nrb]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match bin_samples_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong edges type".to_string(),
            }),
        }
    })
}

/// `bin_samples_from_args` with wrong type for n_regular_bins (string).
#[test]
fn test_bin_samples_wrong_nrb_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let edges = PyList::empty(py);
        let nrb = match "three".into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), edges.into_any(), nrb]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match bin_samples_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong nrb type".to_string(),
            }),
        }
    })
}

/// `bin_samples_from_args` with negative n_regular_bins.
#[test]
fn test_bin_samples_negative_nrb() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let edges = PyList::empty(py);
        let nrb = match (-1_i64).into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), edges.into_any(), nrb]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match bin_samples_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for negative nrb".to_string(),
            }),
        }
    })
}

/// Direct `build_bins_array` with empty input.
#[test]
fn test_build_bins_array_empty() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let empty: Vec<Vec<usize>> = vec![];
        // build_bins_array with empty input should return an empty array
        let result = build_bins_array(py, &empty);
        assert!(result.is_ok());
        Ok(())
    })
}

/// Direct `build_bins_array` with non-uniform rows.
#[test]
fn test_build_bins_array_non_uniform_rows() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let jagged = vec![vec![0_usize, 1_usize], vec![2_usize]];
        let result = build_bins_array(py, &jagged);
        assert!(result.is_err());
        Ok(())
    })
}

// =============================================================================
// Negative max_bins
// =============================================================================

#[test]
fn test_precompute_feature_bins_negative_max_bins() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let data = vec![vec![1.0_f64, 2.0_f64]];
        let features = match PyArray2::from_vec2(py, &data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        // Negative max_bins → i64_to_usize error
        let max_bins = match (-1_i64).into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let tuple = match PyTuple::new(py, [features.into_any(), max_bins]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match precompute_feature_bins_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for negative max_bins".to_string(),
            }),
        }
    })
}
