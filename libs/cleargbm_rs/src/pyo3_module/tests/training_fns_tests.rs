//! Tests for the binary training entry binding.
//!
//! Prediction bindings are covered in `predict_fns_tests`; optional-feature
//! paths (validation data, early stopping, monotonic constraints) in
//! `training_options_tests`; required-config-key contracts in
//! `training_config_key_tests`; the regression entry in
//! `training_regression_entry_tests`.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::helpers::{make_config_dict, set_config_i64, train_model, wrap_py_err};
use crate::error::ClearGbmError;
use crate::pyo3_module::entry_args::train_gradient_boosting_from_args;

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
