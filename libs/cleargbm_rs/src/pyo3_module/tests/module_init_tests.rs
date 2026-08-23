//! Tests for PyO3 module initialization and registered function invocation.
//!
//! Tests that [`crate::pyo3_module::cleargbm_rs`] registers all functions and
//! classes, and that each registered function is callable through the module.
//!
//! The model persistence and introspection entry points are covered separately
//! in [`super::model_persistence_tests`].

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};

use super::helpers::{
    fail, init_module, make_config_dict, make_regression_config_dict, set_config_str,
    training_labels, training_rows, wrap_py_err,
};
use crate::error::ClearGbmError;

// =============================================================================
// Module initialization
// =============================================================================

#[test]
fn test_module_init_succeeds() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let _module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        Ok(())
    })
}

#[test]
fn test_module_registers_every_public_entry_point() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        // The full surface `cleargbm._rust` imports. A registration dropped
        // from the `and_then` chain in `pyo3_module::cleargbm_rs` would break
        // the Python package at import time, so assert on all of it.
        for name in [
            "PyGbmModel",
            "train_gradient_boosting_rs",
            "train_gradient_boosting_regression_rs",
            "predict_proba_model_rs",
            "predict_raw_model_rs",
            "py_gbm_model_to_json_rs",
            "py_gbm_model_from_json_rs",
            "py_gbm_model_feature_importances_rs",
            "py_gbm_model_n_trees_rs",
        ] {
            let present = match module.hasattr(name) {
                Ok(v) => v,
                Err(e) => return Err(fail(format!("hasattr({name}) failed: {e}"))),
            };
            assert!(present, "module does not register '{name}'");
        }
        Ok(())
    })
}

// =============================================================================
// Call registered functions through module (covers mod.rs closure bodies)
// =============================================================================

/// The positional training arguments for the registered trainer.
struct TrainingCallArgs<'py> {
    /// Feature matrix.
    x_train: Bound<'py, PyArray2<f64>>,
    /// Label vector.
    y_train: Bound<'py, PyArray1<i64>>,
    /// Hyperparameter dict.
    config: Bound<'py, pyo3::types::PyDict>,
    /// Feature-name list.
    names: Bound<'py, PyList>,
}

/// Builds the positional training arguments for the registered trainer.
fn training_call_args(py: Python<'_>) -> Result<TrainingCallArgs<'_>, ClearGbmError> {
    let x_train = match PyArray2::from_vec2(py, &training_rows()) {
        Ok(f) => f,
        Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
    };
    let y_train = PyArray1::from_vec(py, training_labels());
    let config = match make_config_dict(py) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let names = match PyList::new(py, ["f0", "f1"]) {
        Ok(l) => l,
        Err(e) => return Err(fail(format!("PyList creation failed: {e}"))),
    };
    Ok(TrainingCallArgs {
        x_train,
        y_train,
        config,
        names,
    })
}

#[test]
fn test_module_call_train_gradient_boosting() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let func = match module.getattr("train_gradient_boosting_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr failed: {e}"))),
        };
        let parts = match training_call_args(py) {
            Ok(parts) => parts,
            Err(e) => return Err(e),
        };

        let model = match func.call1((
            parts.x_train,
            parts.y_train,
            py.None(),
            py.None(),
            py.None(),
            py.None(),
            parts.config,
            parts.names,
        )) {
            Ok(m) => m,
            Err(e) => return Err(fail(format!("training call failed: {e}"))),
        };

        // The registered trainer must hand back a real PyGbmModel: a wrong
        // return type would still satisfy a bare `is_ok()` check.
        let class = match module.getattr("PyGbmModel") {
            Ok(c) => c,
            Err(e) => return Err(fail(format!("getattr PyGbmModel failed: {e}"))),
        };
        let is_model = match model.is_instance(&class) {
            Ok(v) => v,
            Err(e) => return Err(fail(format!("is_instance failed: {e}"))),
        };
        assert!(is_model, "trainer returned a non-PyGbmModel object");
        Ok(())
    })
}

#[test]
fn test_module_call_predict_proba_and_raw() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let train_func = match module.getattr("train_gradient_boosting_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr failed: {e}"))),
        };
        let parts = match training_call_args(py) {
            Ok(parts) => parts,
            Err(e) => return Err(e),
        };
        let model = match train_func.call1((
            parts.x_train,
            parts.y_train,
            py.None(),
            py.None(),
            py.None(),
            py.None(),
            parts.config,
            parts.names,
        )) {
            Ok(m) => m,
            Err(e) => return Err(fail(format!("training call failed: {e}"))),
        };

        let test_data = vec![vec![0.2_f64, 0.3_f64], vec![0.8_f64, 0.9_f64]];

        let predict_func = match module.getattr("predict_proba_model_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr failed: {e}"))),
        };
        let x_test = match PyArray2::from_vec2(py, &test_data) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let proba = match predict_func.call1((model.clone(), x_test)) {
            Ok(v) => v,
            Err(e) => return Err(fail(format!("predict_proba call failed: {e}"))),
        };
        let proba_rows = match proba.extract::<Vec<Vec<f64>>>() {
            Ok(v) => v,
            Err(e) => return Err(fail(format!("proba extract failed: {e}"))),
        };
        assert_eq!(proba_rows.len(), test_data.len());
        for row in &proba_rows {
            assert_eq!(row.len(), 2_usize);
            let total = row[0_usize] + row[1_usize];
            assert!(
                (total - 1.0_f64).abs() < 1e-10_f64,
                "probabilities sum to {total}, expected 1.0"
            );
        }

        let raw_func = match module.getattr("predict_raw_model_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr failed: {e}"))),
        };
        let x_test2 = match PyArray2::from_vec2(py, &test_data) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let raw = match raw_func.call1((model, x_test2)) {
            Ok(v) => v,
            Err(e) => return Err(fail(format!("predict_raw call failed: {e}"))),
        };
        let raw_scores = match raw.extract::<Vec<f64>>() {
            Ok(v) => v,
            Err(e) => return Err(fail(format!("raw extract failed: {e}"))),
        };
        assert_eq!(raw_scores.len(), test_data.len());

        // The raw score and the class-1 probability must agree through the
        // logistic link, which ties the two registered entry points to the
        // same underlying ensemble rather than checking each in isolation.
        for i in 0_usize..raw_scores.len() {
            let expected = 1.0_f64 / (1.0_f64 + (-raw_scores[i]).exp());
            assert!(
                (proba_rows[i][1_usize] - expected).abs() < 1e-10_f64,
                "sample {i}: proba {} does not match sigmoid(raw) {expected}",
                proba_rows[i][1_usize]
            );
        }
        Ok(())
    })
}

#[test]
fn test_module_registers_the_multiclass_surface() -> Result<(), ClearGbmError> {
    // The four multiclass registrations are callable through the module
    // object itself: train, then drive each predictor.
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let x_rows = vec![
            vec![0.0_f64, 0.0_f64],
            vec![1.0_f64, 0.0_f64],
            vec![10.0_f64, 0.0_f64],
            vec![11.0_f64, 0.0_f64],
            vec![20.0_f64, 0.0_f64],
            vec![21.0_f64, 0.0_f64],
        ];
        let x_train = match numpy::PyArray2::from_vec2(py, &x_rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_train = numpy::PyArray1::from_vec(py, vec![0_i64, 0, 1, 1, 2, 2]);
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        match set_config_str(&config, "objective", "multiclass_softmax") {
            Ok(()) => {}
            Err(e) => return Err(e),
        };
        match config.set_item("scale_pos_weight", py.None()) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match config.set_item("n_classes", 3_i64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match config.set_item("max_bins", 16_i64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let names = match pyo3::types::PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let train_func = match module.getattr("train_gradient_boosting_multiclass_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr failed: {e}"))),
        };
        let features_again = match numpy::PyArray2::from_vec2(py, &x_rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let model = match train_func.call1((
            x_train,
            y_train,
            py.None(),
            py.None(),
            py.None(),
            py.None(),
            config,
            names,
        )) {
            Ok(m) => m,
            Err(e) => return Err(fail(format!("training call failed: {e}"))),
        };

        for func_name in [
            "predict_raw_multiclass_model_rs",
            "predict_proba_multiclass_model_rs",
            "predict_class_model_rs",
        ] {
            let func = match module.getattr(func_name) {
                Ok(f) => f,
                Err(e) => return Err(fail(format!("getattr {func_name} failed: {e}"))),
            };
            match func.call1((model.clone(), features_again.clone())) {
                Ok(_) => {}
                Err(e) => return Err(fail(format!("{func_name} call failed: {e}"))),
            }
        }
        Ok(())
    })
}
#[test]
fn test_module_registers_the_ranking_surface() -> Result<(), ClearGbmError> {
    // The ranking registration is callable through the module object
    // itself: train on a two-query corpus, then score with predict_raw.
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let x_rows = vec![
            vec![0.0_f64, 0.0_f64],
            vec![1.0_f64, 0.0_f64],
            vec![2.0_f64, 0.0_f64],
            vec![0.1_f64, 0.0_f64],
            vec![1.1_f64, 0.0_f64],
            vec![2.1_f64, 0.0_f64],
        ];
        let x_train = match numpy::PyArray2::from_vec2(py, &x_rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_train = numpy::PyArray1::from_vec(py, vec![0_i64, 1, 2, 0, 1, 2]);
        let group = numpy::PyArray1::from_vec(py, vec![3_i64, 3_i64]);
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        match set_config_str(&config, "objective", "lambdarank") {
            Ok(()) => {}
            Err(e) => return Err(e),
        };
        match config.set_item("scale_pos_weight", py.None()) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match config.set_item("lambdarank_truncation_level", 3_i64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match config.set_item("max_bins", 16_i64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let names = match pyo3::types::PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let train_func = match module.getattr("train_gradient_boosting_ranking_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr failed: {e}"))),
        };
        let features_again = match numpy::PyArray2::from_vec2(py, &x_rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let model = match train_func.call1((
            x_train,
            y_train,
            group,
            py.None(),
            py.None(),
            py.None(),
            py.None(),
            config,
            names,
        )) {
            Ok(m) => m,
            Err(e) => return Err(fail(format!("training call failed: {e}"))),
        };

        let score_func = match module.getattr("predict_raw_model_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr predict_raw failed: {e}"))),
        };
        match score_func.call1((model, features_again)) {
            Ok(_) => Ok(()),
            Err(e) => Err(fail(format!("predict_raw call failed: {e}"))),
        }
    })
}
#[test]
fn test_module_registers_the_continuation_surface() -> Result<(), ClearGbmError> {
    // Both continuation registrations are callable through the module
    // object itself: train, then continue, for binary and regression.
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let x_rows = vec![
            vec![0.0_f64, 0.1_f64],
            vec![0.2_f64, 0.2_f64],
            vec![0.8_f64, 0.9_f64],
            vec![1.0_f64, 1.0_f64],
        ];
        let x_train = match numpy::PyArray2::from_vec2(py, &x_rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_train = numpy::PyArray1::from_vec(py, vec![0_i64, 0, 1, 1]);
        let config = match make_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let names = match pyo3::types::PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let train_func = match module.getattr("train_gradient_boosting_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr failed: {e}"))),
        };
        let model = match train_func.call1((
            x_train,
            y_train,
            py.None(),
            py.None(),
            py.None(),
            py.None(),
            config,
            names,
        )) {
            Ok(m) => m,
            Err(e) => return Err(fail(format!("training call failed: {e}"))),
        };

        let continue_func = match module.getattr("continue_gradient_boosting_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr continue failed: {e}"))),
        };
        let x_again = match numpy::PyArray2::from_vec2(py, &x_rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let y_again = numpy::PyArray1::from_vec(py, vec![0_i64, 0, 1, 1]);
        match continue_func.call1((
            model,
            x_again,
            y_again,
            py.None(),
            py.None(),
            py.None(),
            py.None(),
            1_i64,
            1_i64,
        )) {
            Ok(_) => {}
            Err(e) => return Err(fail(format!("binary continuation call failed: {e}"))),
        }

        // Regression: train then continue through the module surface.
        let reg_config = match make_regression_config_dict(py) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let reg_names = match pyo3::types::PyList::new(py, ["f0", "f1"]) {
            Ok(l) => l,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let reg_x = match numpy::PyArray2::from_vec2(py, &x_rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let reg_y = numpy::PyArray1::from_vec(py, vec![0.0_f64, 0.2_f64, 0.8_f64, 1.0_f64]);
        let reg_train_func = match module.getattr("train_gradient_boosting_regression_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr failed: {e}"))),
        };
        let reg_model = match reg_train_func.call1((
            reg_x,
            reg_y,
            py.None(),
            py.None(),
            py.None(),
            py.None(),
            reg_config,
            reg_names,
        )) {
            Ok(m) => m,
            Err(e) => return Err(fail(format!("regression training call failed: {e}"))),
        };
        let reg_continue_func = match module.getattr("continue_gradient_boosting_regression_rs") {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("getattr continue failed: {e}"))),
        };
        let reg_x2 = match numpy::PyArray2::from_vec2(py, &x_rows) {
            Ok(f) => f,
            Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
        };
        let reg_y2 = numpy::PyArray1::from_vec(py, vec![0.0_f64, 0.2_f64, 0.8_f64, 1.0_f64]);
        match reg_continue_func.call1((
            reg_model,
            reg_x2,
            reg_y2,
            py.None(),
            py.None(),
            py.None(),
            py.None(),
            1_i64,
            1_i64,
        )) {
            Ok(_) => Ok(()),
            Err(e) => Err(fail(format!("regression continuation call failed: {e}"))),
        }
    })
}
