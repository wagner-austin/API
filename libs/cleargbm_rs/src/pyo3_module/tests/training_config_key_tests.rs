//! Tests for the required-config-key contracts at the training boundary.
//!
//! Every axis a config can state must be present by key — `growth_strategy`,
//! `num_leaves`, `objective`, `scale_pos_weight`, `max_features` — so a
//! caller can never fall into a silent default. Each contract is driven
//! through the real binding.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::helpers::{fail, make_training_args, wrap_py_err};
use crate::error::ClearGbmError;
use crate::pyo3_module::entry_args::train_gradient_boosting_from_args;

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
    let item = match args.get_item(6_usize) {
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
        let item = match args.get_item(6_usize) {
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
        let item = match args.get_item(6_usize) {
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
        let item = match args.get_item(6_usize) {
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
        let item = match args.get_item(6_usize) {
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
        let item = match args.get_item(6_usize) {
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

/// `max_features` with an in-range count trains through the real binding.
#[test]
fn test_train_accepts_max_features_count() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_training_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(6_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.set_item("max_features", 1_i64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Ok(()),
            Err(e) => Err(wrap_py_err(&e)),
        }
    })
}

/// A missing `max_features` key is an error, not "all features".
#[test]
fn test_train_rejects_missing_max_features_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_training_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(6_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.del_item("max_features") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Err(fail(
                "a missing max_features key must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("missing required key 'max_features'"),
                    "error should name the missing key, got: {text}"
                );
                Ok(())
            }
        }
    })
}

/// A non-integer `max_features` fails at extraction.
#[test]
fn test_train_rejects_non_integer_max_features() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let args = match make_training_args(py) {
            Ok(a) => a,
            Err(e) => return Err(e),
        };
        let item = match args.get_item(6_usize) {
            Ok(v) => v,
            Err(e) => return Err(wrap_py_err(&e)),
        };
        let config: Bound<'_, PyDict> = match item.extract() {
            Ok(d) => d,
            Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
        };
        match config.set_item("max_features", "half") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Err(fail(
                "a non-integer max_features must be rejected".to_string(),
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

// =============================================================================
// objective and scale_pos_weight extraction
// =============================================================================

/// Builds the standard training args and rewrites one config key.
///
/// # Args
///
/// * `py` - Python GIL token.
/// * `key` - The config key to rewrite.
/// * `value` - `Some(object)` to overwrite the key, `None` to delete it.
///
/// # Returns
///
/// The `PyErr` text produced by the training call.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if the args cannot be
/// built or if training unexpectedly succeeds.
fn train_error_with_key(
    py: Python<'_>,
    key: &str,
    value: Option<Bound<'_, PyAny>>,
) -> Result<String, ClearGbmError> {
    let args = match make_training_args(py) {
        Ok(a) => a,
        Err(e) => return Err(e),
    };
    let item = match args.get_item(6_usize) {
        Ok(v) => v,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    let config: Bound<'_, PyDict> = match item.extract() {
        Ok(d) => d,
        Err(e) => return Err(fail(format!("config arg is not a dict: {e}"))),
    };
    match value {
        Some(obj) => match config.set_item(key, obj) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        },
        None => match config.del_item(key) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        },
    };
    match train_gradient_boosting_from_args(&args) {
        Ok(_) => Err(fail(format!("a defect in '{key}' must be rejected"))),
        Err(e) => Ok(e.to_string()),
    }
}

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
