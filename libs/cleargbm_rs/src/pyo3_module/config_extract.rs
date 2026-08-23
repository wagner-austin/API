//! Config-dict extraction for the PyO3 training entries.
//!
//! Translates the Python-side hyperparameter dict into a validated
//! [`GradientBoostingConfig`]. Every key is required by presence — the
//! optional-valued axes (`num_leaves`, `max_features`, `scale_pos_weight`)
//! are required-with-null, so an absent key is an error rather than a
//! silent default. Cross-field pairings (growth/budget, objective/weight)
//! are enforced once, by `GradientBoostingConfig::new`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::ClearGbmError;
use crate::pyo3_module::array_helpers::{i64_to_usize, try_convert_int};
use crate::split::MonotonicConstraint;
use crate::training::{
    GradientBoostingConfig, GradientBoostingConfigParams, GrowthStrategy, Objective,
};

/// Extracts a required i64 value from a Python dict.
///
/// # Errors
///
/// Returns `PyErr` if the key is missing or the value is not i64.
pub(super) fn dict_get_i64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    let opt = propagate!(dict.get_item(key));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: key.to_string(),
                reason: format!("missing required key '{key}'"),
            }
            .into())
        }
    };
    item.extract::<i64>()
}

/// Extracts a required f64 value from a Python dict.
///
/// # Errors
///
/// Returns `PyErr` if the key is missing or the value is not f64.
pub(super) fn dict_get_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    let opt = propagate!(dict.get_item(key));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: key.to_string(),
                reason: format!("missing required key '{key}'"),
            }
            .into())
        }
    };
    item.extract::<f64>()
}

/// Extracts a `GradientBoostingConfig` from a Python dict.
///
/// # Errors
///
/// Returns `PyErr` if any required key is missing, has wrong type, or validation fails.
pub(super) fn extract_config(dict: &Bound<'_, PyDict>) -> PyResult<GradientBoostingConfig> {
    let n_estimators = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "n_estimators")),
        "n_estimators"
    ));
    let max_depth = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "max_depth")),
        "max_depth"
    ));
    let learning_rate = propagate!(dict_get_f64(dict, "learning_rate"));
    let min_samples_split = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "min_samples_split")),
        "min_samples_split"
    ));
    let min_samples_leaf = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "min_samples_leaf")),
        "min_samples_leaf"
    ));
    let max_bins = propagate_into!(i64_to_usize(
        propagate!(dict_get_i64(dict, "max_bins")),
        "max_bins"
    ));
    let subsample = propagate!(dict_get_f64(dict, "subsample"));
    let random_state: u64 = propagate_into!(try_convert_int(
        propagate!(dict_get_i64(dict, "random_state")),
        "random_state"
    ));
    let reg_alpha = propagate!(dict_get_f64(dict, "reg_alpha"));
    let reg_lambda = propagate!(dict_get_f64(dict, "reg_lambda"));
    let monotonic_constraints = propagate!(extract_monotonic_constraints(dict));
    let early_stopping_rounds = propagate!(extract_early_stopping_rounds(dict));
    let growth_strategy = propagate!(extract_growth_strategy(dict));
    let num_leaves = propagate!(extract_num_leaves(dict));
    let objective = propagate!(extract_objective(dict));
    let scale_pos_weight = propagate!(extract_scale_pos_weight(dict));
    let max_features = propagate!(extract_max_features(dict));
    let colsample_bytree = propagate!(extract_colsample_bytree(dict));

    let params = GradientBoostingConfigParams {
        n_estimators,
        max_depth,
        learning_rate,
        min_samples_split,
        min_samples_leaf,
        max_bins,
        subsample,
        random_state,
        monotonic_constraints,
        reg_alpha,
        reg_lambda,
        early_stopping_rounds,
        growth_strategy,
        num_leaves,
        objective,
        scale_pos_weight,
        max_features,
        colsample_bytree,
    };

    Ok(propagate_into!(GradientBoostingConfig::new(params)))
}

/// Extracts the optional leaf budget from a config dict.
///
/// The key `"num_leaves"` is required to be present; its value may be `None`.
/// Presence is mandatory for the same reason `growth_strategy` is: an absent
/// key would read as "no budget" and quietly turn a bounded leaf-wise arm into
/// an unbounded one. Whether the value pairs correctly with the growth policy
/// is decided by `GradientBoostingConfig::new`, not here.
///
/// # Errors
///
/// Returns `PyErr` if the key is missing or the value is neither `None` nor a
/// non-negative integer.
fn extract_num_leaves(dict: &Bound<'_, PyDict>) -> PyResult<Option<usize>> {
    let opt = propagate!(dict.get_item("num_leaves"));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: "num_leaves".to_string(),
                reason: "missing required key 'num_leaves'".to_string(),
            }
            .into())
        }
    };

    if item.is_none() {
        return Ok(None);
    }

    let val: i64 = propagate!(item.extract());
    Ok(Some(propagate_into!(i64_to_usize(val, "num_leaves"))))
}

/// Extracts the optional per-split feature budget from a config dict.
///
/// The key `"max_features"` is required to be present; its value may be
/// `None` (all features). The same presence contract as `num_leaves`: an
/// absent key would silently read as "all features".
///
/// # Errors
///
/// Returns `PyErr` if the key is missing or the value is neither `None`
/// nor a non-negative integer.
fn extract_max_features(dict: &Bound<'_, PyDict>) -> PyResult<Option<usize>> {
    let opt = propagate!(dict.get_item("max_features"));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: "max_features".to_string(),
                reason: "missing required key 'max_features'".to_string(),
            }
            .into())
        }
    };

    if item.is_none() {
        return Ok(None);
    }

    let val: i64 = propagate!(item.extract());
    Ok(Some(propagate_into!(i64_to_usize(val, "max_features"))))
}

/// Extracts the optional per-tree feature fraction from a config dict.
///
/// The key `"colsample_bytree"` is required to be present; its value may be
/// `None` (all features). Same presence contract as `max_features`: an
/// absent key would silently read as "all features".
///
/// # Errors
///
/// Returns `PyErr` if the key is missing or the value is neither `None`
/// nor a float.
fn extract_colsample_bytree(dict: &Bound<'_, PyDict>) -> PyResult<Option<f64>> {
    let opt = propagate!(dict.get_item("colsample_bytree"));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: "colsample_bytree".to_string(),
                reason: "missing required key 'colsample_bytree'".to_string(),
            }
            .into())
        }
    };

    if item.is_none() {
        return Ok(None);
    }

    let val: f64 = propagate!(item.extract());
    Ok(Some(val))
}

/// Extracts the optional positive-class weight from a config dict.
///
/// The key `"scale_pos_weight"` is required to be present; its value may be
/// `None` (regression) or a float (binary classification). Whether the value
/// pairs correctly with the objective is decided by
/// `GradientBoostingConfig::new`, not here.
///
/// # Errors
///
/// Returns `PyErr` if the key is missing or the value is neither `None` nor
/// a float.
fn extract_scale_pos_weight(dict: &Bound<'_, PyDict>) -> PyResult<Option<f64>> {
    let opt = propagate!(dict.get_item("scale_pos_weight"));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: "scale_pos_weight".to_string(),
                reason: "missing required key 'scale_pos_weight'".to_string(),
            }
            .into())
        }
    };

    if item.is_none() {
        return Ok(None);
    }

    let val: f64 = propagate!(item.extract());
    Ok(Some(val))
}

/// Extracts the training objective from a config dict.
///
/// The key `"objective"` is required and must be the string
/// `"binary_log_loss"` or `"squared_error"`. A missing key is an error
/// rather than a default, per the same rule as `growth_strategy`: a run
/// must name the loss it descends.
///
/// # Errors
///
/// Returns `PyErr` if the key is missing, is not a string, or is not one of
/// the two spellings.
fn extract_objective(dict: &Bound<'_, PyDict>) -> PyResult<Objective> {
    let opt = propagate!(dict.get_item("objective"));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: "objective".to_string(),
                reason: "missing required key 'objective'".to_string(),
            }
            .into())
        }
    };
    let value: String = propagate!(item.extract());
    Ok(propagate_into!(Objective::from_wire(&value)))
}

/// Extracts the tree growth policy from a config dict.
///
/// The key `"growth_strategy"` is required and must be the string
/// `"depth_wise"` or `"leaf_wise"`. Unlike `monotonic_constraints`, a missing
/// key is an error rather than a default: a benchmark arm that meant to name a
/// policy and silently got another one is exactly the failure this axis exists
/// to prevent.
///
/// # Errors
///
/// Returns `PyErr` if the key is missing, is not a string, or is not one of
/// the two spellings.
fn extract_growth_strategy(dict: &Bound<'_, PyDict>) -> PyResult<GrowthStrategy> {
    let opt = propagate!(dict.get_item("growth_strategy"));
    let item = match opt {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: "growth_strategy".to_string(),
                reason: "missing required key 'growth_strategy'".to_string(),
            }
            .into())
        }
    };
    let value: String = propagate!(item.extract());
    Ok(propagate_into!(GrowthStrategy::from_wire(&value)))
}

/// Extracts optional monotonic constraints from a config dict.
///
/// The key `"monotonic_constraints"` should be `None` or a list of ints
/// where -1 = decreasing, 0 = none, 1 = increasing.
///
/// # Errors
///
/// Returns `PyErr` if the value is present but not a valid list of ints.
fn extract_monotonic_constraints(
    dict: &Bound<'_, PyDict>,
) -> PyResult<Option<Vec<MonotonicConstraint>>> {
    let opt = propagate!(dict.get_item("monotonic_constraints"));
    let item = match opt {
        Some(v) => v,
        None => return Ok(None),
    };

    if item.is_none() {
        return Ok(None);
    }

    let py_list: Bound<'_, PyList> = propagate_into!(item.extract());

    let mut constraints = Vec::with_capacity(py_list.len());
    for i in 0_usize..py_list.len() {
        let val = propagate!(py_list.get_item(i));
        let int_val: i64 = propagate!(val.extract());
        let constraint = match int_val {
            -1_i64 => MonotonicConstraint::Decreasing,
            0_i64 => MonotonicConstraint::None,
            1_i64 => MonotonicConstraint::Increasing,
            other => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "monotonic_constraints".to_string(),
                    reason: format!("invalid value {other}, expected -1, 0, or 1"),
                }
                .into())
            }
        };
        constraints.push(constraint);
    }

    Ok(Some(constraints))
}

/// Extracts optional early stopping rounds from a config dict.
///
/// # Errors
///
/// Returns `PyErr` if the value is present but not a valid int.
fn extract_early_stopping_rounds(dict: &Bound<'_, PyDict>) -> PyResult<Option<usize>> {
    let opt = propagate!(dict.get_item("early_stopping_rounds"));
    let item = match opt {
        Some(v) => v,
        None => return Ok(None),
    };

    if item.is_none() {
        return Ok(None);
    }

    let val: i64 = propagate!(item.extract());
    Ok(Some(propagate_into!(i64_to_usize(
        val,
        "early_stopping_rounds"
    ))))
}
