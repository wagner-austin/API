//! Tests for sample weights through the PyO3 training entries.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::helpers::{
    fail, make_config_dict, make_regression_config_dict, regression_targets, training_labels,
    training_rows, wrap_py_err,
};
use crate::error::ClearGbmError;
use crate::pyo3_module::entry_args::{
    train_gradient_boosting_from_args, train_gradient_boosting_regression_from_args,
};

/// Builds an 8-element args tuple with the given weight elements.
fn args_with_weights<'py>(
    py: Python<'py>,
    binary: bool,
    sample_weight: Bound<'py, PyAny>,
    val_sample_weight: Bound<'py, PyAny>,
) -> Result<Bound<'py, PyTuple>, ClearGbmError> {
    let x_train = match PyArray2::from_vec2(py, &training_rows()) {
        Ok(f) => f,
        Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
    };
    let y_train: Bound<'py, PyAny> = if binary {
        PyArray1::from_vec(py, training_labels()).into_any()
    } else {
        PyArray1::from_vec(py, regression_targets()).into_any()
    };
    let config = if binary {
        match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        }
    } else {
        match make_regression_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        }
    };
    let names = match PyList::new(py, ["f0", "f1"]) {
        Ok(l) => l,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match PyTuple::new(
        py,
        [
            x_train.into_any(),
            y_train,
            sample_weight,
            py.None().into_bound(py).into_any(),
            py.None().into_bound(py).into_any(),
            val_sample_weight,
            config.into_any(),
            names.into_any(),
        ],
    ) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// A weighted binary train through the binding produces a model that differs
/// from the unweighted one.
#[test]
fn test_binary_entry_accepts_and_honors_sample_weights() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let weights = PyArray1::from_vec(
            py,
            vec![5.0_f64, 0.5_f64, 5.0_f64, 0.5_f64, 5.0_f64, 0.5_f64],
        );
        let weighted_args = match args_with_weights(
            py,
            true,
            weights.into_any(),
            py.None().into_bound(py).into_any(),
        ) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let weighted = match train_gradient_boosting_from_args(&weighted_args) {
            Ok(m) => m,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let unweighted_args = match args_with_weights(
            py,
            true,
            py.None().into_bound(py).into_any(),
            py.None().into_bound(py).into_any(),
        ) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let unweighted = match train_gradient_boosting_from_args(&unweighted_args) {
            Ok(m) => m,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        // Serialize both and compare the documents: the weighted model must
        // differ (knob sensitivity through the full binding).
        let ser = |m: &Py<PyAny>| -> Result<String, ClearGbmError> {
            let tuple = match PyTuple::new(py, [m.clone_ref(py)]) {
                Ok(t) => t,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            match crate::pyo3_module::model_fns::py_gbm_model_to_json_from_args(&tuple) {
                Ok(s) => Ok(s),
                Err(e) => Err(wrap_py_err(&e)),
            }
        };
        let weighted_json = match ser(&weighted) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        let unweighted_json = match ser(&unweighted) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        assert_ne!(
            weighted_json, unweighted_json,
            "weights through the binding produced an identical model"
        );
        Ok(())
    })
}

/// A weighted regression train through the binding succeeds with validation
/// weights alongside a validation split.
#[test]
fn test_regression_entry_accepts_weights_and_val_weights() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x_train = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_train = PyArray1::from_vec(py, regression_targets());
        let weights = PyArray1::from_vec(
            py,
            vec![1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 2.0_f64, 3.0_f64],
        );
        let x_val = match PyArray2::from_vec2(py, &[vec![0.4_f64, 0.5_f64]]) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_val = PyArray1::from_vec(py, vec![0.9_f64]);
        let val_weights = PyArray1::from_vec(py, vec![2.0_f64]);
        let config = match make_regression_config_dict(py) {
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
                weights.into_any(),
                x_val.into_any(),
                y_val.into_any(),
                val_weights.into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_regression_from_args(&tuple) {
            Ok(_) => Ok(()),
            Err(e) => Err(wrap_py_err(&e)),
        }
    })
}

/// A zero weight is rejected through the binding, naming its index.
#[test]
fn test_binary_entry_rejects_zero_weight() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let weights = PyArray1::from_vec(
            py,
            vec![1.0_f64, 1.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64],
        );
        let args = match args_with_weights(
            py,
            true,
            weights.into_any(),
            py.None().into_bound(py).into_any(),
        ) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Err(fail("a zero weight must be rejected".to_string())),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("index 2"),
                    "rejection should name the offending index, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// A wrong-dtype weight array is rejected at extraction.
#[test]
fn test_regression_entry_rejects_integer_weights() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let weights = PyArray1::from_vec(py, vec![1_i64, 2_i64, 3_i64, 1_i64, 2_i64, 3_i64]);
        let args = match args_with_weights(
            py,
            false,
            weights.into_any(),
            py.None().into_bound(py).into_any(),
        ) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        match train_gradient_boosting_regression_from_args(&args) {
            Ok(_) => Err(fail("an i64 weight array must be rejected".to_string())),
            Err(_) => Ok(()),
        }
    })
}

/// A validation weight without a validation split is rejected, not ignored.
#[test]
fn test_binary_entry_rejects_val_weight_without_val_split() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let val_weights = PyArray1::from_vec(py, vec![1.0_f64]);
        let args = match args_with_weights(
            py,
            true,
            py.None().into_bound(py).into_any(),
            val_weights.into_any(),
        ) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Err(fail(
                "a val weight without a val split must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("y_val must be provided"),
                    "rejection should name the missing split, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// The regression entry likewise rejects a dangling validation weight.
#[test]
fn test_regression_entry_rejects_val_weight_without_val_split() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let val_weights = PyArray1::from_vec(py, vec![1.0_f64]);
        let args = match args_with_weights(
            py,
            false,
            py.None().into_bound(py).into_any(),
            val_weights.into_any(),
        ) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        match train_gradient_boosting_regression_from_args(&args) {
            Ok(_) => Err(fail(
                "a val weight without a val split must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("y_val must be provided"),
                    "rejection should name the missing split, got: {text}"
                );
                Ok(())
            }
        }
    })
}
