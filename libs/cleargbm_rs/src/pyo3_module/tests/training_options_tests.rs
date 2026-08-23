//! Tests for optional training features through the binary entry binding:
//! validation data, early stopping, and monotonic constraints.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::helpers::{make_config_dict, set_config_i64, wrap_py_err};
use crate::error::ClearGbmError;
use crate::pyo3_module::training_fns::train_gradient_boosting_from_args;

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

// =============================================================================
// Validation both-or-neither pairing
// =============================================================================

/// Builds a six-arg training tuple with the given third and fourth items.
fn args_with_val<'py>(
    py: Python<'py>,
    x_val: Bound<'py, PyAny>,
    y_val: Bound<'py, PyAny>,
) -> Result<Bound<'py, PyTuple>, ClearGbmError> {
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
    let names = match PyList::new(py, ["f0", "f1"]) {
        Ok(l) => l,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match PyTuple::new(
        py,
        [
            x_train.into_any(),
            y_train.into_any(),
            x_val,
            y_val,
            config.into_any(),
            names.into_any(),
        ],
    ) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// `x_val` without `y_val` is rejected at the binding boundary.
#[test]
fn test_train_rejects_x_val_without_y_val() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_val_data = vec![vec![0.2_f64, 0.3_f64]];
        let x_val = match PyArray2::from_vec2(py, &x_val_data) {
            Ok(f) => f,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("{e}"),
                })
            }
        };
        let tuple = match args_with_val(py, x_val.into_any(), py.None().into_bound(py).into_any()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        match train_gradient_boosting_from_args(&tuple) {
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "x_val without y_val must be rejected".to_string(),
            }),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("y_val must be provided"),
                    "rejection should name the missing pair, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// `y_val` without `x_val` is rejected at the binding boundary.
#[test]
fn test_train_rejects_y_val_without_x_val() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_val = PyArray1::from_vec(py, vec![0_i64]);
        let tuple = match args_with_val(py, py.None().into_bound(py).into_any(), y_val.into_any()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        match train_gradient_boosting_from_args(&tuple) {
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "y_val without x_val must be rejected".to_string(),
            }),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("x_val must be provided"),
                    "rejection should name the missing pair, got: {text}"
                );
                Ok(())
            }
        }
    })
}
