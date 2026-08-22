//! Tests for PyO3 training function bindings.
//!
//! Tests [`super::super::training_fns`] functions through the PyO3 runtime.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::helpers::{
    fail, make_config_dict, make_training_args, set_config_i64, train_model, wrap_py_err,
};
use crate::error::ClearGbmError;
use crate::pyo3_module::training_fns::{
    predict_proba_model_from_args, predict_raw_model_from_args, train_gradient_boosting_from_args,
};

// =============================================================================
// train_gradient_boosting_from_args
// =============================================================================

#[test]
fn test_train_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        // Model should be a PyGbmModel
        assert!(!model.is_none(py));
        Ok(())
    })
}

#[test]
fn test_train_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match train_gradient_boosting_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for empty args".to_string(),
            }),
        }
    })
}

#[test]
fn test_train_wrong_x_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Pass i64 array instead of f64 for x_train
        let x_data = vec![vec![1_i64, 2_i64]];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 1_i64]);
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match train_gradient_boosting_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong x_train type".to_string(),
            }),
        }
    })
}

#[test]
fn test_train_wrong_y_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Pass f64 array instead of i64 for y_train
        let x_data = vec![vec![1.0_f64, 2.0_f64]];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0.0_f64, 1.0_f64]);
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match train_gradient_boosting_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong y_train type".to_string(),
            }),
        }
    })
}

#[test]
fn test_train_missing_config_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);

        // Empty config dict → missing "n_estimators"
        let config = PyDict::new(py);
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match train_gradient_boosting_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for missing config key".to_string(),
            }),
        }
    })
}

#[test]
fn test_train_invalid_config_value() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);

        // learning_rate = 0.0 → invalid
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let lr_val = match 0.0_f64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        match config.set_item("learning_rate", lr_val) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match train_gradient_boosting_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for learning_rate=0.0".to_string(),
            }),
        }
    })
}

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

// =============================================================================
// Additional coverage tests for arg extraction
// =============================================================================

/// Training with only 1 arg — triggers get_item(1) error.
#[test]
fn test_train_one_arg_only() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![vec![1.0_f64, 2.0_f64]];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let tuple = match PyTuple::new(py, [x_train.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            }),
        }
    })
}

/// Training with validation data.
#[test]
fn test_train_with_validation_data() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);

        let x_val_data = vec![vec![0.2_f64, 0.3_f64], vec![0.8_f64, 0.9_f64]];
        let x_val = match PyArray2::from_vec2(py, &x_val_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let y_val = PyArray1::from_vec(py, vec![0_i64, 1_i64]);

        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                x_val.into_any(),
                y_val.into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = train_gradient_boosting_from_args(&tuple);
        assert!(result.is_ok());
        Ok(())
    })
}

/// Training with early stopping rounds.
#[test]
fn test_train_with_early_stopping() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);

        let x_val_data = vec![vec![0.2_f64, 0.3_f64], vec![0.8_f64, 0.9_f64]];
        let x_val = match PyArray2::from_vec2(py, &x_val_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let y_val = PyArray1::from_vec(py, vec![0_i64, 1_i64]);

        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        // Add early_stopping_rounds = 1
        match set_config_i64(&config, "early_stopping_rounds", 1_i64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        };
        // Set more estimators so early stopping can trigger
        match set_config_i64(&config, "n_estimators", 10_i64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        };

        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                x_val.into_any(),
                y_val.into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = train_gradient_boosting_from_args(&tuple);
        assert!(result.is_ok());
        Ok(())
    })
}

/// Training with decreasing monotonic constraints (-1).
#[test]
fn test_train_with_decreasing_constraints() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        // Decreasing on feature 0, none on feature 1
        let constraints = match PyList::new(py, [-1_i64, 0_i64]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match config.set_item("monotonic_constraints", constraints) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = train_gradient_boosting_from_args(&tuple);
        assert!(result.is_ok());
        Ok(())
    })
}

/// Training with explicit None for monotonic_constraints.
#[test]
fn test_train_with_none_constraints() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        // Explicitly set constraints to None
        match config.set_item("monotonic_constraints", py.None()) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = train_gradient_boosting_from_args(&tuple);
        assert!(result.is_ok());
        Ok(())
    })
}

/// Training with explicit None for early_stopping_rounds.
#[test]
fn test_train_with_none_early_stopping() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        match config.set_item("early_stopping_rounds", py.None()) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result = train_gradient_boosting_from_args(&tuple);
        assert!(result.is_ok());
        Ok(())
    })
}

/// Training with invalid label values (-1 can't convert to u8).
#[test]
fn test_train_invalid_label_value() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![vec![0.1_f64, 0.2_f64], vec![0.3_f64, 0.4_f64]];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        // Label -1 can't convert to u8
        let y_train = PyArray1::from_vec(py, vec![0_i64, -1_i64]);
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for invalid label".to_string(),
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

/// Missing an f64 config key (learning_rate).
#[test]
fn test_train_missing_f64_config_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);

        // Config with i64 fields but missing learning_rate (f64)
        let config = PyDict::new(py);
        match set_config_i64(&config, "n_estimators", 2_i64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        };
        match set_config_i64(&config, "max_depth", 2_i64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        };
        // learning_rate intentionally missing
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for missing f64 config key".to_string(),
            }),
        }
    })
}

// =============================================================================
// Monotonic constraints and early stopping
// =============================================================================

#[test]
fn test_train_with_monotonic_constraints() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);

        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        // Add monotonic constraints [1, 0] (increasing on feature 0, none on feature 1)
        let constraints = match PyList::new(py, [1_i64, 0_i64]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match config.set_item("monotonic_constraints", constraints) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = train_gradient_boosting_from_args(&tuple);
        assert!(result.is_ok());
        Ok(())
    })
}

#[test]
fn test_train_invalid_monotonic_constraint_value() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_data = vec![
            vec![0.1_f64, 0.2_f64],
            vec![0.3_f64, 0.4_f64],
            vec![0.5_f64, 0.6_f64],
            vec![0.7_f64, 0.8_f64],
            vec![0.9_f64, 1.0_f64],
            vec![1.1_f64, 1.2_f64],
        ];
        let x_train = match PyArray2::from_vec2(py, &x_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("PyArray2 creation failed: {e}"),
                })
            }
        };
        let y_train = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);

        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        // Invalid constraint value: 2 (only -1, 0, 1 allowed)
        let constraints = match PyList::new(py, [2_i64, 0_i64]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match config.set_item("monotonic_constraints", constraints) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let tuple = match PyTuple::new(
            py,
            [
                x_train.into_any(),
                y_train.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match train_gradient_boosting_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for invalid monotonic constraint value".to_string(),
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

// =============================================================================
// growth_strategy extraction
// =============================================================================

/// Builds the standard training args, rewrites `growth_strategy` in the config
/// dict, and returns the resulting rejection text.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `value` - `Some(spelling)` to overwrite the key, `None` to delete it.
///
/// # Returns
///
/// The `PyErr` text produced by the training call.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if the args cannot be
/// built or if training unexpectedly succeeds.
fn train_error_with_growth_strategy(
    py: Python<'_>,
    value: Option<&str>,
) -> Result<String, ClearGbmError> {
    let args = match make_training_args(py) {
        Ok(a) => a,
        Err(e) => return Err(e),
    };
    let item = match args.get_item(4_usize) {
        Ok(v) => v,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let config: Bound<'_, PyDict> = match item.extract() {
        Ok(d) => d,
        Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
    };
    match value {
        Some(spelling) => match config.set_item("growth_strategy", spelling) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        },
        None => match config.del_item("growth_strategy") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        },
    };
    match train_gradient_boosting_from_args(&args) {
        Ok(_) => Err(fail(
            "a growth_strategy defect must be rejected".to_string(),
        )),
        Err(e) => Ok(e.to_string()),
    }
}

/// A missing `growth_strategy` key is an error, not a silent depth-wise run.
#[test]
fn test_train_rejects_missing_growth_strategy() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let text = match train_error_with_growth_strategy(py, None) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(
            text.contains("missing required key 'growth_strategy'"),
            "error should name the missing key, got: {text}"
        );
        Ok(())
    })
}

/// An unrecognised spelling names itself in the rejection.
#[test]
fn test_train_rejects_unknown_growth_strategy() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let text = match train_error_with_growth_strategy(py, Some("lossguide")) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(
            text.contains("lossguide"),
            "error should quote the offending value, got: {text}"
        );
        Ok(())
    })
}

/// `leaf_wise` without a leaf budget is rejected at the pairing check.
#[test]
fn test_train_rejects_leaf_wise_without_a_budget() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let text = match train_error_with_growth_strategy(py, Some("leaf_wise")) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(
            text.contains("num_leaves"),
            "error should name the missing budget, got: {text}"
        );
        Ok(())
    })
}

/// `leaf_wise` with a budget trains through the real binding.
#[test]
fn test_train_accepts_leaf_wise_with_a_budget() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_training_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(4_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.set_item("growth_strategy", "leaf_wise") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match config.set_item("num_leaves", 3_i64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Ok(()),
            Err(e) => Err(wrap_py_err(&e)),
        }
    })
}

/// A missing `num_leaves` key is an error even under depth-wise growth.
#[test]
fn test_train_rejects_missing_num_leaves_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_training_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(4_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.del_item("num_leaves") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Err(fail(
                "a missing num_leaves key must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("missing required key 'num_leaves'"),
                    "error should name the missing key, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// A non-integer `num_leaves` fails at extraction.
#[test]
fn test_train_rejects_non_integer_num_leaves() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_training_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(4_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.set_item("num_leaves", "thirty-one") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Err(fail(
                "a non-integer num_leaves must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("TypeError") || text.contains("int"),
                    "error should report the type mismatch, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// A non-string `growth_strategy` fails at extraction.
#[test]
fn test_train_rejects_non_string_growth_strategy() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_training_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(4_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.set_item("growth_strategy", 1_i64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Err(fail(
                "a non-string growth_strategy must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("TypeError") || text.contains("str"),
                    "error should report the type mismatch, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// The accepted spelling trains, so the axis is not merely a rejection gate.
#[test]
fn test_train_accepts_depth_wise_growth_strategy() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_training_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(4_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.set_item("growth_strategy", "depth_wise") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Ok(()),
            Err(e) => Err(wrap_py_err(&e)),
        }
    })
}
