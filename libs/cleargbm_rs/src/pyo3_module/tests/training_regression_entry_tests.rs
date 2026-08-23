//! Tests for the regression training entry binding
//! (`train_gradient_boosting_regression_rs`).

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::helpers::{
    fail, init_module, make_config_dict, make_regression_config_dict,
    make_regression_training_args, module_fn, regression_targets, train_regression_model,
    training_rows, wrap_py_err,
};
use crate::error::ClearGbmError;
use crate::pyo3_module::training_fns::{
    predict_proba_model_from_args, train_gradient_boosting_regression_from_args,
};

/// The regression entry trains and its predictions track the target.
#[test]
fn test_regression_train_and_predict_raw() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let model = match train_regression_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let x = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let predict = match module_fn(&module, "predict_raw_model_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        let raw = match predict.call1((model, x)) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let preds = match raw.extract::<Vec<f64>>() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let targets = regression_targets();
        assert_eq!(preds.len(), targets.len());
        // The fit must discriminate: the largest target draws a larger
        // prediction than the smallest.
        assert!(
            preds[5_usize] > preds[0_usize],
            "regression predictions do not track the target: {preds:?}"
        );
        Ok(())
    })
}

/// The regression entry is registered on the module and trains through the
/// registration closure.
#[test]
fn test_regression_entry_registered_on_module() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let train_fn = match module_fn(&module, "train_gradient_boosting_regression_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        let x = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y = PyArray1::from_vec(py, regression_targets());
        let config = match make_regression_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let names = match PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let model = match train_fn.call1((x, y, py.None(), py.None(), config, names)) {
            Ok(m) => m,
            Err(e) => return Err(fail(format!("registered regression trainer failed: {e}"))),
        };
        let class = match module.getattr("PyGbmModel") {
            Ok(c) => c,
            Err(e) => return Err(fail(format!("getattr PyGbmModel failed: {e}"))),
        };
        let is_model = match model.is_instance(&class) {
            Ok(v) => v,
            Err(e) => return Err(fail(format!("is_instance failed: {e}"))),
        };
        assert!(
            is_model,
            "regression trainer returned a non-PyGbmModel object"
        );
        Ok(())
    })
}

/// A regression model refuses probabilities through the binding.
#[test]
fn test_regression_model_rejects_predict_proba() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let model = match train_regression_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let x = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let tuple = match PyTuple::new(py, [model.into_bound(py).into_any(), x.into_any()]) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match predict_proba_model_from_args(&tuple) {
            Ok(_) => Err(fail(
                "predict_proba must be rejected for a squared_error model".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("predict_raw"),
                    "rejection should point at predict_raw, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// The binary entry rejects a squared-error config: entry and objective must
/// agree.
#[test]
fn test_binary_entry_rejects_squared_error_config() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);
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
                x.into_any(),
                y.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match crate::pyo3_module::training_fns::train_gradient_boosting_from_args(&tuple) {
            Ok(_) => Err(fail(
                "the binary entry must reject a squared_error config".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("squared_error"),
                    "rejection should name the objective mismatch, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// The regression entry rejects a binary config the same way.
#[test]
fn test_regression_entry_rejects_binary_config() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y = PyArray1::from_vec(py, regression_targets());
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
                x.into_any(),
                y.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_regression_from_args(&tuple) {
            Ok(_) => Err(fail(
                "the regression entry must reject a binary_log_loss config".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("binary_log_loss"),
                    "rejection should name the objective mismatch, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// The regression entry rejects an i64 target array — targets are f64.
#[test]
fn test_regression_entry_rejects_integer_targets() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y = PyArray1::from_vec(py, vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]);
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
                x.into_any(),
                y.into_any(),
                py.None().into_bound(py).into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_regression_from_args(&tuple) {
            Ok(_) => Err(fail(
                "an i64 target array must be rejected by the f64 entry".to_string(),
            )),
            Err(_) => Ok(()),
        }
    })
}

/// The regression entry accepts validation data and NaN targets are rejected.
#[test]
fn test_regression_entry_with_validation_and_nan_rejection() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // With validation data: trains.
        let x = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y = PyArray1::from_vec(py, regression_targets());
        let x_val = match PyArray2::from_vec2(py, &[vec![0.4_f64, 0.5_f64]]) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_val = PyArray1::from_vec(py, vec![0.9_f64]);
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
                x.into_any(),
                y.into_any(),
                x_val.into_any(),
                y_val.into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_regression_from_args(&tuple) {
            Ok(_) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // With a NaN target: rejected, naming the index.
        let args = match make_regression_training_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let y_item = match args.get_item(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let y_arr: Bound<'_, PyArray1<f64>> = match y_item.extract() {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("y extract failed: {e}"))),
        };
        {
            let mut rw = y_arr.readwrite();
            let slice = match rw.as_slice_mut() {
                Ok(s) => s,
                Err(e) => return Err(fail(format!("as_slice_mut failed: {e}"))),
            };
            slice[2_usize] = f64::NAN;
        }
        match train_gradient_boosting_regression_from_args(&args) {
            Ok(_) => Err(fail("a NaN target must be rejected".to_string())),
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

/// The regression entry rejects a malformed args tuple.
#[test]
fn test_regression_entry_too_few_args() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let tuple = PyTuple::empty(py);
        match train_gradient_boosting_regression_from_args(&tuple) {
            Err(_) => Ok(()),
            Ok(_) => Err(fail("expected error for empty args".to_string())),
        }
    })
}

/// The regression entry surfaces the both-or-neither validation pairing.
#[test]
fn test_regression_entry_rejects_x_val_without_y_val() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y = PyArray1::from_vec(py, regression_targets());
        let x_val = match PyArray2::from_vec2(py, &[vec![0.4_f64, 0.5_f64]]) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
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
                x.into_any(),
                y.into_any(),
                x_val.into_any(),
                py.None().into_bound(py).into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_regression_from_args(&tuple) {
            Ok(_) => Err(fail("x_val without y_val must be rejected".to_string())),
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

/// `y_val` without `x_val` is rejected at the regression binding boundary.
#[test]
fn test_regression_entry_rejects_y_val_without_x_val() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let x = match PyArray2::from_vec2(py, &training_rows()) {
            Ok(a) => a,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y = PyArray1::from_vec(py, regression_targets());
        let y_val = PyArray1::from_vec(py, vec![0.9_f64]);
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
                x.into_any(),
                y.into_any(),
                py.None().into_bound(py).into_any(),
                y_val.into_any(),
                config.into_any(),
                names.into_any(),
            ],
        ) {
            Ok(t) => t,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_regression_from_args(&tuple) {
            Ok(_) => Err(fail("y_val without x_val must be rejected".to_string())),
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
