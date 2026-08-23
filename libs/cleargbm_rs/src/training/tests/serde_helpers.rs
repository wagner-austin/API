//! Shared helpers for the training serde tests.

use crate::error::ClearGbmError;
use crate::training::{GradientBoostingConfig, GradientBoostingModel};

use super::train_helpers::{
    make_config, make_regression_config, make_simple_dataset, train_binary, train_regression,
};

/// Trains a small binary model for serde round-trip tests.
pub(super) fn make_test_model() -> Result<GradientBoostingModel, ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    train_binary(&x_train, &y_train, None, &config, &feature_names)
}

/// Trains a small squared-error model for serde round-trip tests.
pub(super) fn make_test_regression_model() -> Result<GradientBoostingModel, ClearGbmError> {
    let (rows, _, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_train: Vec<f64> = rows
        .iter()
        .map(|r| 2.0_f64 * r[0_usize] - r[1_usize])
        .collect();
    let config = match make_regression_config(5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    train_regression(&x_train, &y_train, None, &config, &feature_names)
}

/// Test helper: serialize via `serde_json`, mapping errors into
/// `ClearGbmError::SerializationFailed` so a caller can `propagate!` on the
/// result. Tests that need to observe a serialize error use `serde_json`
/// directly and match on `Result`.
pub(super) fn to_json<T: serde::Serialize>(value: &T) -> Result<String, ClearGbmError> {
    match serde_json::to_string(value) {
        Ok(s) => Ok(s),
        Err(e) => Err(ClearGbmError::SerializationFailed {
            reason: e.to_string(),
        }),
    }
}

/// Test helper: deserialize via `serde_json`, mapping errors into
/// `ClearGbmError::DeserializationFailed`. Tests that need to observe a
/// deserialize error use `serde_json` directly and match on `Result`.
pub(super) fn from_json<'a, T: serde::Deserialize<'a>>(json: &'a str) -> Result<T, ClearGbmError> {
    match serde_json::from_str::<T>(json) {
        Ok(v) => Ok(v),
        Err(e) => Err(ClearGbmError::DeserializationFailed {
            reason: e.to_string(),
        }),
    }
}

/// Serializes `value` and reparses it as a mutable JSON object.
///
/// Building the reference payload from the real `Serialize` impl (rather than
/// a hand-written literal) keeps these tests honest if a field is renamed: the
/// per-field assertions fail loudly instead of quietly testing nothing.
///
/// # Args
///
/// * `value` - The value to serialize.
///
/// # Returns
///
/// The serialized form as a JSON object map.
///
/// # Errors
///
/// Returns [`ClearGbmError::SerializationFailed`] if serialization fails, or
/// [`ClearGbmError::DeserializationFailed`] if the result is not a JSON object.
pub(super) fn as_json_object<T: serde::Serialize>(
    value: &T,
) -> Result<serde_json::Map<String, serde_json::Value>, ClearGbmError> {
    let text = match to_json(value) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };
    let parsed: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    match parsed {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(ClearGbmError::DeserializationFailed {
            reason: "expected a JSON object".to_string(),
        }),
    }
}

/// Builds the reference binary config used by the error-path tests.
pub(super) fn reference_config() -> Result<GradientBoostingConfig, ClearGbmError> {
    GradientBoostingConfig::new(super::train_helpers::default_params())
}
