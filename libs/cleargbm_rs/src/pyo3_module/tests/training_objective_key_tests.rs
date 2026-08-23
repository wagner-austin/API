//! Tests for the required `objective` and `scale_pos_weight` config keys.
//!
//! Both keys must be present so a caller can never fall into a silent
//! binary run or a silent weight of 1, and the objective/weight pairing is
//! enforced at the boundary. Each contract is driven through the real
//! binding.

use pyo3::prelude::*;

use super::helpers::{train_error_with_key, wrap_py_err};
use crate::error::ClearGbmError;

/// A missing `objective` key is an error, not a silent binary run.
#[test]
fn test_train_rejects_missing_objective_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let text = match train_error_with_key(py, "objective", None) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(
            text.contains("missing required key 'objective'"),
            "error should name the missing key, got: {text}"
        );
        Ok(())
    })
}

/// An unrecognised objective spelling names itself in the rejection.
#[test]
fn test_train_rejects_unknown_objective_spelling() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let value = match "reg:squarederror".into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let text = match train_error_with_key(py, "objective", Some(value)) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(
            text.contains("reg:squarederror"),
            "error should quote the offending value, got: {text}"
        );
        Ok(())
    })
}

/// A non-string `objective` fails at extraction.
#[test]
fn test_train_rejects_non_string_objective() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let value = match 7_i64.into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let text = match train_error_with_key(py, "objective", Some(value)) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(
            text.contains("TypeError") || text.contains("str"),
            "error should report the type mismatch, got: {text}"
        );
        Ok(())
    })
}

/// A missing `scale_pos_weight` key is an error, not a silent weight of 1.
#[test]
fn test_train_rejects_missing_scale_pos_weight_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let text = match train_error_with_key(py, "scale_pos_weight", None) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(
            text.contains("missing required key 'scale_pos_weight'"),
            "error should name the missing key, got: {text}"
        );
        Ok(())
    })
}

/// A null `scale_pos_weight` under the binary objective is a pairing error.
#[test]
fn test_train_rejects_null_weight_under_binary_objective() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let text =
            match train_error_with_key(py, "scale_pos_weight", Some(py.None().into_bound(py))) {
                Ok(t) => t,
                Err(e) => return Err(e),
            };
        assert!(
            text.contains("must be set"),
            "error should say the weight is required under binary_log_loss, got: {text}"
        );
        Ok(())
    })
}

/// A non-float `scale_pos_weight` fails at extraction.
#[test]
fn test_train_rejects_non_float_scale_pos_weight() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let value = match "heavy".into_pyobject(py) {
            Ok(v) => v.into_any(),
            Err(e) => return Err(wrap_py_err(&e.into())),
        };
        let text = match train_error_with_key(py, "scale_pos_weight", Some(value)) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(
            text.contains("TypeError") || text.contains("float"),
            "error should report the type mismatch, got: {text}"
        );
        Ok(())
    })
}
