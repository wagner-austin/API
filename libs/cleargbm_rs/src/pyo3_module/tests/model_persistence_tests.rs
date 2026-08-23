//! Tests for the model persistence and introspection bindings.
//!
//! Covers `py_gbm_model_to_json_rs`, `py_gbm_model_from_json_rs`,
//! `py_gbm_model_feature_importances_rs` and `py_gbm_model_n_trees_rs` — the
//! surface `cleargbm.ensemble` uses to save, reload and inspect a trained
//! model.
//!
//! Each function is exercised twice: once through the registered module
//! attribute (which runs the registration closure in
//! [`crate::pyo3_module`] and the `*_from_args` extraction wrapper), and once
//! through the wrapper directly for the argument-error paths that cannot be
//! produced from Python without constructing a malformed call.

use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyTuple};

use super::helpers::{fail, init_module, module_fn, train_model, wrap_py_err};
use crate::error::ClearGbmError;
use crate::pyo3_module::model_fns::{
    de_err, py_gbm_model_feature_importances_from_args, py_gbm_model_from_json_from_args,
    py_gbm_model_n_trees_from_args, py_gbm_model_to_json_from_args, ser_err,
};

/// Builds a one-element positional args tuple.
fn args1<'py>(
    py: Python<'py>,
    item: Bound<'py, PyAny>,
) -> Result<Bound<'py, PyTuple>, ClearGbmError> {
    match PyTuple::new(py, [item]) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Builds an empty positional args tuple, to drive the `get_item(0)` failure.
fn args0(py: Python<'_>) -> Bound<'_, PyTuple> {
    PyTuple::empty(py)
}

/// Serializes a freshly trained model through the registered module function.
fn model_json(py: Python<'_>) -> Result<String, ClearGbmError> {
    let module = match init_module(py) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let model = match train_model(py) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let to_json = match module_fn(&module, "py_gbm_model_to_json_rs") {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    let out = match to_json.call1((model,)) {
        Ok(v) => v,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match out.extract::<String>() {
        Ok(s) => Ok(s),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

// =============================================================================
// Round trip
// =============================================================================

/// A model reloaded from its own JSON must predict identically to the original.
///
/// This is the property that makes persistence usable: a saved model that
/// scores differently after a reload is silently wrong, and a shape-only
/// assertion would not catch it.
#[test]
fn test_model_json_roundtrip_predicts_identically() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let original = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };

        let to_json = match module_fn(&module, "py_gbm_model_to_json_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        let json = match to_json.call1((original.clone_ref(py),)) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        let from_json = match module_fn(&module, "py_gbm_model_from_json_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        let restored = match from_json.call1((json,)) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        // Score both models on points that straddle the training range, so a
        // dropped tree or a rescaled leaf value shows up as a difference.
        let probe = vec![
            vec![0.15_f64, 0.25_f64],
            vec![0.55_f64, 0.65_f64],
            vec![1.05_f64, 1.15_f64],
        ];
        let predict = match module_fn(&module, "predict_raw_model_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };

        let mut outputs: Vec<Vec<f64>> = Vec::with_capacity(2_usize);
        for model in [original, restored.unbind()] {
            let x = match PyArray2::from_vec2(py, &probe) {
                Ok(a) => a,
                Err(e) => return Err(fail(format!("PyArray2 creation failed: {e}"))),
            };
            let raw = match predict.call1((model, x)) {
                Ok(v) => v,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            let arr = match raw.extract::<Vec<f64>>() {
                Ok(v) => v,
                Err(e) => return Err(wrap_py_err(&e)),
            };
            outputs.push(arr);
        }

        let before = &outputs[0_usize];
        let after = &outputs[1_usize];
        assert_eq!(before.len(), probe.len());
        assert_eq!(after.len(), probe.len());
        for i in 0_usize..probe.len() {
            assert!(
                (before[i] - after[i]).abs() < 1e-12_f64,
                "sample {i}: {} != {} after JSON round trip",
                before[i],
                after[i]
            );
        }
        // A model that scored every probe identically would pass the loop
        // above even if prediction were stubbed out, so require that the
        // ensemble actually discriminates between the extremes.
        assert!(
            (before[0_usize] - before[2_usize]).abs() > 1e-9_f64,
            "ensemble produced a constant score; round-trip check is vacuous"
        );
        Ok(())
    })
}

/// The serialized document names the fields the deserializer requires.
#[test]
fn test_model_json_contains_required_fields() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let json = match model_json(py) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        for field in [
            "trees",
            "base_prediction",
            "learning_rate",
            "feature_names",
            "config",
        ] {
            assert!(json.contains(field), "serialized model lacks '{field}'");
        }
        // The objective travels inside the embedded config — a saved model
        // must state the loss it was trained under.
        assert!(json.contains(r#""objective":"binary_log_loss""#));
        Ok(())
    })
}

// =============================================================================
// Introspection accessors
// =============================================================================

/// Tree count reflects the configured `n_estimators`.
#[test]
fn test_n_trees_matches_configured_estimators() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let n_trees_fn = match module_fn(&module, "py_gbm_model_n_trees_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        let out = match n_trees_fn.call1((model,)) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let n_trees = match out.extract::<usize>() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        // The shared config sets n_estimators = 2 with no early stopping.
        assert_eq!(n_trees, 2_usize);
        Ok(())
    })
}

/// Feature importances are named, one per feature, and normalized to sum to 1.
#[test]
fn test_feature_importances_are_named_and_normalized() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let model = match train_model(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let importances_fn = match module_fn(&module, "py_gbm_model_feature_importances_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        let out = match importances_fn.call1((model,)) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let pairs = match out.extract::<Vec<(String, f64)>>() {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };

        assert_eq!(pairs.len(), 2_usize);
        assert_eq!(pairs[0_usize].0, "f0");
        assert_eq!(pairs[1_usize].0, "f1");

        let total: f64 = pairs.iter().map(|&(_, v)| v).sum();
        // The trained ensemble has at least one internal node, so the
        // normalization branch (rather than the all-zero branch) applies.
        assert!(
            (total - 1.0_f64).abs() < 1e-12_f64,
            "importances sum to {total}, expected 1.0"
        );
        for &(ref name, value) in &pairs {
            assert!(value >= 0.0_f64, "feature {name} has negative importance");
        }
        Ok(())
    })
}

// =============================================================================
// Deserialization failures
// =============================================================================

/// Malformed JSON is rejected rather than panicking.
#[test]
fn test_from_json_rejects_malformed_document() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let from_json = match module_fn(&module, "py_gbm_model_from_json_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        match from_json.call1(("{not json at all",)) {
            Ok(_) => Err(fail("malformed JSON must not deserialize".to_string())),
            Err(_) => Ok(()),
        }
    })
}

/// A structurally valid document missing a required field is rejected.
#[test]
fn test_from_json_rejects_missing_required_field() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let from_json = match module_fn(&module, "py_gbm_model_from_json_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        // Valid JSON, valid field names, but `config` is absent.
        let json = r#"{"trees":[],"base_prediction":0.0,"learning_rate":0.1,"feature_names":[]}"#;
        match from_json.call1((json,)) {
            Ok(_) => Err(fail(
                "document missing 'config' must not deserialize".to_string(),
            )),
            Err(_) => Ok(()),
        }
    })
}

/// A model whose payload violates config validation is rejected on load.
///
/// Guards the documented contract that deserialization routes through the
/// validating constructor rather than reconstructing the struct field-wise.
#[test]
fn test_from_json_rejects_invalid_config_payload() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let json = match model_json(py) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        // Drive learning_rate out of range; the config validator must reject it.
        let tampered = json.replace("\"learning_rate\":0.1", "\"learning_rate\":-1.0");
        assert_ne!(
            tampered, json,
            "tamper step did not match the serialized learning_rate"
        );

        let module = match init_module(py) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let from_json = match module_fn(&module, "py_gbm_model_from_json_rs") {
            Ok(f) => f,
            Err(e) => return Err(e),
        };
        match from_json.call1((tampered,)) {
            Ok(_) => Err(fail(
                "negative learning_rate must fail config validation".to_string(),
            )),
            Err(_) => Ok(()),
        }
    })
}

// =============================================================================
// Argument extraction failures
// =============================================================================

/// Every model accessor rejects a call with no arguments.
#[test]
fn test_accessors_reject_missing_argument() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let empty = args0(py);
        assert!(py_gbm_model_to_json_from_args(&empty).is_err());
        assert!(py_gbm_model_from_json_from_args(&empty).is_err());
        assert!(py_gbm_model_feature_importances_from_args(&empty).is_err());
        assert!(py_gbm_model_n_trees_from_args(&empty).is_err());
        Ok(())
    })
}

/// Every model accessor rejects an argument that is not a `PyGbmModel`.
#[test]
fn test_accessors_reject_wrong_argument_type() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // An empty list is unambiguously not a PyGbmModel.
        let not_a_model = pyo3::types::PyList::empty(py).into_any();
        let tuple = match args1(py, not_a_model) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(py_gbm_model_to_json_from_args(&tuple).is_err());
        assert!(py_gbm_model_feature_importances_from_args(&tuple).is_err());
        assert!(py_gbm_model_n_trees_from_args(&tuple).is_err());
        Ok(())
    })
}

/// `from_json` rejects an argument that is not a string.
#[test]
fn test_from_json_rejects_non_string_argument() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // An empty list is unambiguously not a JSON string.
        let not_a_str = pyo3::types::PyList::empty(py).into_any();
        let tuple = match args1(py, not_a_str) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(py_gbm_model_from_json_from_args(&tuple).is_err());
        Ok(())
    })
}

// =============================================================================
// Error constructors
// =============================================================================

/// Serialization failures surface to Python as `RuntimeError` with the reason.
///
/// Called directly: `serde_json::to_string` cannot fail for the model type, so
/// this arm is unreachable through `py_gbm_model_to_json_rs`.
#[test]
fn test_ser_err_maps_to_runtime_error() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let err = ser_err("disk exploded".to_string());
        assert!(err.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
        assert!(err.to_string().contains("disk exploded"));
        Ok(())
    })
}

/// Deserialization failures surface to Python as `RuntimeError` with the reason.
#[test]
fn test_de_err_maps_to_runtime_error() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let err = de_err("trailing garbage".to_string());
        assert!(err.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
        assert!(err.to_string().contains("trailing garbage"));
        Ok(())
    })
}
