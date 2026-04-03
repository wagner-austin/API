//! Tests for PyO3 loss function bindings.
//!
//! Tests [`super::super::loss_fns`] functions through the PyO3 runtime.

use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::ClearGbmError;
use crate::pyo3_module::loss_fns::{
    binary_log_loss_from_args, binary_log_loss_gradients_from_args,
    binary_log_loss_hessians_from_args, binary_log_loss_initial_prediction_from_args,
    sigmoid_array_from_args,
};

/// Helper: wraps a PyErr into ClearGbmError for test return types.
fn wrap_py_err(e: &PyErr) -> ClearGbmError {
    ClearGbmError::TreeConstructionFailed {
        reason: format!("PyErr: {e}"),
    }
}

// --- binary_log_loss_from_args ---

#[test]
fn test_binary_log_loss_from_args_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64, 1_i64, 0_i64]);
        let y_pred = PyArray1::from_slice(py, &[0.9_f64, 0.1_f64, 0.8_f64, 0.2_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let loss = match binary_log_loss_from_args(&tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        assert!(loss > 0.0_f64);
        assert!(loss < 1.0_f64);
        Ok(())
    })
}

#[test]
fn test_binary_log_loss_from_args_invalid_label() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[0_i64, 5_i64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64, 0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for invalid label, got {v}"),
            }),
        }
    })
}

// --- binary_log_loss_gradients_from_args ---

#[test]
fn test_gradients_from_args_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let y_pred = PyArray1::from_slice(py, &[0.7_f64, 0.3_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result_obj = match binary_log_loss_gradients_from_args(&tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result: &Bound<'_, PyArray1<f64>> = match result_obj.bind(py).cast::<PyArray1<f64>>() {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("cast failed: {e}"),
                })
            }
        };
        assert_eq!(result.len(), 2_usize);

        let readonly = result.readonly();
        let grads = match readonly.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("as_slice failed: {e}"),
                })
            }
        };
        // y=1, p=0.7 → gradient = -0.3
        assert!((grads[0_usize] - (-0.3_f64)).abs() < 1e-10_f64);
        // y=0, p=0.3 → gradient = 0.3
        assert!((grads[1_usize] - 0.3_f64).abs() < 1e-10_f64);
        Ok(())
    })
}

// --- binary_log_loss_hessians_from_args ---

#[test]
fn test_hessians_from_args_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64, 0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result_obj = match binary_log_loss_hessians_from_args(&tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result: &Bound<'_, PyArray1<f64>> = match result_obj.bind(py).cast::<PyArray1<f64>>() {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("cast failed: {e}"),
                })
            }
        };
        assert_eq!(result.len(), 2_usize);

        let readonly = result.readonly();
        let hess = match readonly.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("as_slice failed: {e}"),
                })
            }
        };
        // p=0.5: 0.5 * 0.5 = 0.25
        assert!((hess[0_usize] - 0.25_f64).abs() < 1e-10_f64);
        assert!((hess[1_usize] - 0.25_f64).abs() < 1e-10_f64);
        Ok(())
    })
}

// --- binary_log_loss_initial_prediction_from_args ---

#[test]
fn test_initial_prediction_from_args_balanced() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[0_i64, 1_i64]);
        let tuple = match PyTuple::new(py, [y_true.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result = match binary_log_loss_initial_prediction_from_args(&tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        // 50/50 → log-odds ≈ 0
        assert!(result.abs() < 1e-10_f64);
        Ok(())
    })
}

#[test]
fn test_initial_prediction_from_args_all_same_labels() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 1_i64, 1_i64]);
        let tuple = match PyTuple::new(py, [y_true.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_initial_prediction_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for all-same labels, got {v}"),
            }),
        }
    })
}

// --- sigmoid_array_from_args ---

#[test]
fn test_sigmoid_array_from_args_basic() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x = PyArray1::from_slice(py, &[0.0_f64, 1.0_f64, -1.0_f64]);
        let tuple = match PyTuple::new(py, [x.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let result_obj = match sigmoid_array_from_args(&tuple) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let result: &Bound<'_, PyArray1<f64>> = match result_obj.bind(py).cast::<PyArray1<f64>>() {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("cast failed: {e}"),
                })
            }
        };
        assert_eq!(result.len(), 3_usize);

        let readonly = result.readonly();
        let values = match readonly.as_slice() {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("as_slice failed: {e}"),
                })
            }
        };
        // sigmoid(0) = 0.5
        assert!((values[0_usize] - 0.5_f64).abs() < 1e-10_f64);
        // sigmoid(1) > 0.5
        assert!(values[1_usize] > 0.5_f64);
        // sigmoid(-1) < 0.5
        assert!(values[2_usize] < 0.5_f64);
        Ok(())
    })
}

// --- Error path tests: wrong argument types ---

#[test]
fn test_binary_log_loss_from_args_wrong_y_true_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Pass f64 instead of i64 for y_true
        let y_true = PyArray1::from_slice(py, &[1.0_f64, 0.0_f64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64, 0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for wrong y_true type, got {v}"),
            }),
        }
    })
}

#[test]
fn test_binary_log_loss_from_args_wrong_y_pred_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // Pass i64 instead of f64 for y_pred
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let y_pred = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for wrong y_pred type, got {v}"),
            }),
        }
    })
}

#[test]
fn test_binary_log_loss_from_args_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match binary_log_loss_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for empty args, got {v}"),
            }),
        }
    })
}

#[test]
fn test_binary_log_loss_from_args_missing_y_pred() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let tuple = match PyTuple::new(py, [y_true.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for missing y_pred, got {v}"),
            }),
        }
    })
}

#[test]
fn test_gradients_from_args_wrong_y_true_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1.0_f64, 0.0_f64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64, 0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_gradients_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong y_true type".to_string(),
            }),
        }
    })
}

#[test]
fn test_gradients_from_args_wrong_y_pred_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let y_pred = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_gradients_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong y_pred type".to_string(),
            }),
        }
    })
}

#[test]
fn test_gradients_from_args_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match binary_log_loss_gradients_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for empty args".to_string(),
            }),
        }
    })
}

#[test]
fn test_hessians_from_args_wrong_y_true_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1.0_f64, 0.0_f64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64, 0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_hessians_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong y_true type".to_string(),
            }),
        }
    })
}

#[test]
fn test_hessians_from_args_wrong_y_pred_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let y_pred = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_hessians_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong y_pred type".to_string(),
            }),
        }
    })
}

#[test]
fn test_hessians_from_args_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match binary_log_loss_hessians_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for empty args".to_string(),
            }),
        }
    })
}

#[test]
fn test_initial_prediction_from_args_wrong_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1.0_f64, 0.0_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_initial_prediction_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for wrong type, got {v}"),
            }),
        }
    })
}

#[test]
fn test_initial_prediction_from_args_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match binary_log_loss_initial_prediction_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for empty args, got {v}"),
            }),
        }
    })
}

#[test]
fn test_sigmoid_array_from_args_wrong_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x = PyArray1::from_slice(py, &[0_i64, 1_i64]);
        let tuple = match PyTuple::new(py, [x.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match sigmoid_array_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for wrong type".to_string(),
            }),
        }
    })
}

#[test]
fn test_sigmoid_array_from_args_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);

        match sigmoid_array_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for empty args".to_string(),
            }),
        }
    })
}

// --- Error path tests: negative labels in gradients/hessians/initial_prediction ---

#[test]
fn test_gradients_from_args_negative_label() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[0_i64, -1_i64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64, 0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_gradients_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for negative label in gradients".to_string(),
            }),
        }
    })
}

#[test]
fn test_hessians_from_args_negative_label() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[0_i64, -1_i64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64, 0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_hessians_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for negative label in hessians".to_string(),
            }),
        }
    })
}

#[test]
fn test_initial_prediction_from_args_negative_label() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[0_i64, -1_i64]);
        let tuple = match PyTuple::new(py, [y_true.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_initial_prediction_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for negative label, got {v}"),
            }),
        }
    })
}

// --- Error path tests: missing y_pred argument ---

#[test]
fn test_gradients_from_args_missing_y_pred() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let tuple = match PyTuple::new(py, [y_true.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_gradients_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for missing y_pred in gradients".to_string(),
            }),
        }
    })
}

#[test]
fn test_hessians_from_args_missing_y_pred() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let tuple = match PyTuple::new(py, [y_true.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_hessians_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for missing y_pred in hessians".to_string(),
            }),
        }
    })
}

// --- Error path tests: shape mismatch (triggers _rs wrapper error paths) ---

#[test]
fn test_gradients_from_args_shape_mismatch() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_gradients_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for shape mismatch in gradients".to_string(),
            }),
        }
    })
}

#[test]
fn test_hessians_from_args_shape_mismatch() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[1_i64, 0_i64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_hessians_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for shape mismatch in hessians".to_string(),
            }),
        }
    })
}

// --- Error path tests: non-contiguous arrays ---

/// Helper: creates a non-contiguous numpy array via Python slicing.
///
/// `arr[::2]` produces a view with stride 2, which is non-contiguous.
fn make_non_contiguous_i64<'py>(
    py: Python<'py>,
    data: &[i64],
) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    let arr = PyArray1::from_slice(py, data);
    let slice = pyo3::types::PySlice::new(py, 0_isize, 4_isize, 2_isize);
    match arr.call_method1("__getitem__", (slice,)) {
        Ok(v) => Ok(v),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Helper: creates a non-contiguous f64 numpy array via Python slicing.
fn make_non_contiguous_f64<'py>(
    py: Python<'py>,
    data: &[f64],
) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    let arr = PyArray1::from_slice(py, data);
    let slice = pyo3::types::PySlice::new(py, 0_isize, 4_isize, 2_isize);
    match arr.call_method1("__getitem__", (slice,)) {
        Ok(v) => Ok(v),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

#[test]
fn test_binary_log_loss_from_args_non_contiguous_y_true() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let nc_y_true = match make_non_contiguous_i64(py, &[0_i64, 1_i64, 0_i64, 1_i64]) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let y_pred = PyArray1::from_slice(py, &[0.5_f64, 0.5_f64]);
        let tuple = match PyTuple::new(py, [nc_y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Non-contiguous i64 → extract_labels_u8 as_slice error
        match binary_log_loss_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for non-contiguous y_true, got {v}"),
            }),
        }
    })
}

#[test]
fn test_binary_log_loss_from_args_non_contiguous_y_pred() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[0_i64, 1_i64]);
        let nc_y_pred = match make_non_contiguous_f64(py, &[0.5_f64, 0.5_f64, 0.5_f64, 0.5_f64]) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let tuple = match PyTuple::new(py, [y_true.as_any(), nc_y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Non-contiguous f64 → extract_f64_slice as_slice error
        match binary_log_loss_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for non-contiguous y_pred, got {v}"),
            }),
        }
    })
}

#[test]
fn test_gradients_from_args_non_contiguous_y_pred() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[0_i64, 1_i64]);
        let nc_y_pred = match make_non_contiguous_f64(py, &[0.5_f64, 0.5_f64, 0.5_f64, 0.5_f64]) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let tuple = match PyTuple::new(py, [y_true.as_any(), nc_y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_gradients_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for non-contiguous y_pred in gradients".to_string(),
            }),
        }
    })
}

#[test]
fn test_hessians_from_args_non_contiguous_y_pred() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[0_i64, 1_i64]);
        let nc_y_pred = match make_non_contiguous_f64(py, &[0.5_f64, 0.5_f64, 0.5_f64, 0.5_f64]) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let tuple = match PyTuple::new(py, [y_true.as_any(), nc_y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_hessians_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for non-contiguous y_pred in hessians".to_string(),
            }),
        }
    })
}

#[test]
fn test_sigmoid_array_from_args_non_contiguous() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let nc_x = match make_non_contiguous_f64(py, &[0.0_f64, 1.0_f64, -1.0_f64, 2.0_f64]) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let tuple = match PyTuple::new(py, [nc_x.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match sigmoid_array_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for non-contiguous array in sigmoid_array".to_string(),
            }),
        }
    })
}

#[test]
fn test_negative_label_rejected_by_i64_to_u8() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let y_true = PyArray1::from_slice(py, &[0_i64, -1_i64]);
        let y_pred = PyArray1::from_slice(py, &[0.5_f64, 0.5_f64]);
        let tuple = match PyTuple::new(py, [y_true.as_any(), y_pred.as_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        match binary_log_loss_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(v) => Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected error for negative label, got {v}"),
            }),
        }
    })
}
