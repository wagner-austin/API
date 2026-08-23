//! Tests for the prediction bindings (`predict_proba_model_rs`,
//! `predict_raw_model_rs`) through their `*_from_args` wrappers.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::helpers::{train_model, wrap_py_err};
use crate::error::ClearGbmError;
use crate::pyo3_module::training_fns::{
    predict_proba_model_from_args, predict_raw_model_from_args,
};

// =============================================================================
// predict_proba_model_from_args
// =============================================================================

#[test]
fn test_predict_proba_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };

        let test_data = vec![vec![0.2_f64, 0.3_f64], vec![0.8_f64, 0.9_f64]];
        let x_test = match PyArray2::from_vec2(py, &test_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let tuple = match PyTuple::new(py, [model.into_bound(py).into_any(), x_test.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = match predict_proba_model_from_args(&tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Result should be a 2D f64 array of shape (2, 2)
        let arr: &Bound<'_, PyArray2<f64>> = match result.bind(py).cast::<PyArray2<f64>>() {
            Ok(a) => a,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("downcast failed: {e}"),
                })
            }
        };
        let shape = arr.shape();
        assert_eq!(shape[0_usize], 2_usize);
        assert_eq!(shape[1_usize], 2_usize);

        // Probabilities should sum to ~1.0 per row
        let readonly = arr.readonly();
        let flat = match readonly.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("as_slice failed: {e}"),
                })
            }
        };
        let row0_sum = flat[0_usize] + flat[1_usize];
        assert!((row0_sum - 1.0_f64).abs() < 1e-10_f64);
        let row1_sum = flat[2_usize] + flat[3_usize];
        assert!((row1_sum - 1.0_f64).abs() < 1e-10_f64);
        Ok(())
    })
}

#[test]
fn test_predict_proba_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match predict_proba_model_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for empty args".to_string(),
            }),
        }
    })
}

#[test]
fn test_predict_proba_wrong_model_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Pass a string instead of PyGbmModel
        let fake_model = match "not_a_model".into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let test_data = vec![vec![0.5_f64, 0.5_f64]];
        let x_test = match PyArray2::from_vec2(py, &test_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let tuple = match PyTuple::new(py, [fake_model, x_test.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match predict_proba_model_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong model type".to_string(),
            }),
        }
    })
}

// =============================================================================
// predict_raw_model_from_args
// =============================================================================

#[test]
fn test_predict_raw_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };

        let test_data = vec![vec![0.2_f64, 0.3_f64], vec![0.8_f64, 0.9_f64]];
        let x_test = match PyArray2::from_vec2(py, &test_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let tuple = match PyTuple::new(py, [model.into_bound(py).into_any(), x_test.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = match predict_raw_model_from_args(&tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Result should be a 1D f64 array of length 2
        let arr: &Bound<'_, PyArray1<f64>> = match result.bind(py).cast::<PyArray1<f64>>() {
            Ok(a) => a,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("downcast failed: {e}"),
                })
            }
        };
        assert_eq!(arr.len(), 2_usize);
        Ok(())
    })
}

#[test]
fn test_predict_raw_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match predict_raw_model_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for empty args".to_string(),
            }),
        }
    })
}

#[test]
fn test_predict_raw_wrong_model_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let fake_model = match "not_a_model".into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let test_data = vec![vec![0.5_f64, 0.5_f64]];
        let x_test = match PyArray2::from_vec2(py, &test_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let tuple = match PyTuple::new(py, [fake_model, x_test.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match predict_raw_model_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong model type".to_string(),
            }),
        }
    })
}
/// Predict proba with only model (missing features).
#[test]
fn test_predict_proba_one_arg_only() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let tuple = match PyTuple::new(py, [model.into_bound(py).into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match predict_proba_model_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for 1-arg tuple".to_string(),
            }),
        }
    })
}

/// Predict proba with wrong features type (i64 instead of f64).
#[test]
fn test_predict_proba_wrong_features_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let bad_data = vec![vec![1_i64, 2_i64]];
        let bad_features = match PyArray2::from_vec2(py, &bad_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let tuple = match PyTuple::new(
            py,
            [model.into_bound(py).into_any(), bad_features.into_any()],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match predict_proba_model_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong features type".to_string(),
            }),
        }
    })
}

/// Predict raw with only model (missing features).
#[test]
fn test_predict_raw_one_arg_only() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let tuple = match PyTuple::new(py, [model.into_bound(py).into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match predict_raw_model_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for 1-arg tuple".to_string(),
            }),
        }
    })
}

/// Predict raw with wrong features type (i64 instead of f64).
#[test]
fn test_predict_raw_wrong_features_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let bad_data = vec![vec![1_i64, 2_i64]];
        let bad_features = match PyArray2::from_vec2(py, &bad_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let tuple = match PyTuple::new(
            py,
            [model.into_bound(py).into_any(), bad_features.into_any()],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match predict_raw_model_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong features type".to_string(),
            }),
        }
    })
}
#[test]
fn test_predict_proba_rejects_empty_feature_matrix() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };

        // A zero-row matrix has no samples to score. Returning an empty result
        // would let a caller silently believe every row was predicted, so the
        // extraction boundary rejects it instead.
        let empty: Vec<Vec<f64>> = Vec::new();
        let x_empty = match PyArray2::from_vec2(py, &empty) {
            Ok(a) => a,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let tuple = match PyTuple::new(py, [model.into_bound(py).into_any(), x_empty.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match predict_proba_model_from_args(&tuple) {
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "empty feature matrix must be rejected".to_string(),
            }),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("zero rows"),
                    "error should name the empty matrix, got: {text}"
                );
                Ok(())
            }
        }
    })
}
