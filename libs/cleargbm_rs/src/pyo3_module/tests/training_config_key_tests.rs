//! Tests for the required `growth_strategy` and `num_leaves` config keys.
//!
//! Both growth-policy keys must be present so a caller can never fall into
//! a silent depth-wise run or an inferred leaf budget. The other required
//! keys live in [`super::training_objective_key_tests`] and
//! [`super::training_feature_budget_key_tests`]. Each contract is driven
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
#[test]
fn test_train_accepts_goss_rates_and_rejects_a_missing_goss_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // A run with both GOSS rates set trains (rates as floats cross
        // the boundary and validate).
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
        match config.set_item("goss_top_rate", 0.3_f64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match config.set_item("goss_other_rate", 0.2_f64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        }

        // A missing goss_top_rate key is an error, not a silent "off".
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
        match config.del_item("goss_top_rate") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Err(fail(
                "a missing goss_top_rate key must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("missing required key 'goss_top_rate'"),
                    "got: {text}"
                );
                Ok(())
            }
        }
    })
}

#[test]
fn test_train_accepts_quantized_bins_and_rejects_a_missing_key() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        // A run with quantized_gradient_bins set trains (the int crosses
        // the boundary and validates).
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
        match config.set_item("quantized_gradient_bins", 4_i64) {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        }

        // A missing quantized_gradient_bins key is an error, not a
        // silent "off".
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
        match config.del_item("quantized_gradient_bins") {
            Ok(()) => {}
            Err(e) => return Err(wrap_py_err(&e)),
        };
        match train_gradient_boosting_from_args(&args) {
            Ok(_) => Err(fail(
                "a missing quantized_gradient_bins key must be rejected".to_string(),
            )),
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("missing required key 'quantized_gradient_bins'"),
                    "got: {text}"
                );
                Ok(())
            }
        }
    })
}
