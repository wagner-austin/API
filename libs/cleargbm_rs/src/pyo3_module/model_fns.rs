//! PyO3 bindings for trained-model persistence and introspection.
//!
//! Owns the [`PyGbmModel`] opaque class plus the module-level functions that
//! operate on one: JSON round-tripping, feature importances, and tree count.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::ClearGbmError;
use crate::training::{feature_importances, GradientBoostingModel};

/// Opaque Python wrapper around a trained [`GradientBoostingModel`].
///
/// Created by the training entry points in
/// [`super::training_fns`] and consumed by the prediction and persistence
/// functions.
///
/// JSON persistence and feature-importance extraction are exposed as
/// module-level functions (`py_gbm_model_to_json_rs`,
/// `py_gbm_model_from_json_rs`, `py_gbm_model_feature_importances_rs`,
/// `py_gbm_model_n_trees_rs`) rather than `#[pymethods]` so the crate's
/// `question_mark_used` / `useless_conversion` forbids stay clean
/// (`#[pymethods]` expansion is incompatible).
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct PyGbmModel {
    /// The underlying trained model.
    pub(crate) inner: GradientBoostingModel,
}

/// Converts a serialization failure description into a [`PyErr`].
///
/// # Args
///
/// * `reason` - Human-readable description of the serialization failure.
///
/// # Returns
///
/// A Python `RuntimeError` wrapping the serialization error.
///
/// `pub(super)` rather than private so [`crate::pyo3_module::tests`] can assert
/// the mapping directly. `serde_json::to_string` cannot fail for
/// [`GradientBoostingModel`] — the type contains no non-string map keys, and
/// writing to a `String` is infallible — so this arm is unreachable through
/// [`py_gbm_model_to_json_rs`] and can only be covered by calling it.
pub(super) fn ser_err(reason: String) -> PyErr {
    ClearGbmError::SerializationFailed { reason }.into()
}

/// Converts a deserialization failure description into a [`PyErr`].
///
/// # Args
///
/// * `reason` - Human-readable description of the deserialization failure.
///
/// # Returns
///
/// A Python `RuntimeError` wrapping the deserialization error.
pub(super) fn de_err(reason: String) -> PyErr {
    ClearGbmError::DeserializationFailed { reason }.into()
}

/// Serializes a [`PyGbmModel`] to a JSON string.
///
/// # Args
///
/// * `model` - The `PyGbmModel` reference.
///
/// # Returns
///
/// JSON string representation of the model. Round-trips through
/// [`py_gbm_model_from_json_rs`] without loss beyond one ULP on float text
/// representation; see the Rust unit test
/// `test_model_roundtrip_predictions_identical` for the per-sample prediction
/// preservation guarantee at 1e-15.
///
/// # Errors
///
/// Returns `RuntimeError` if serialization fails.
pub(crate) fn py_gbm_model_to_json_rs(model: &PyGbmModel) -> PyResult<String> {
    serde_json::to_string(&model.inner).map_err(|e| ser_err(e.to_string()))
}

/// Deserializes a [`PyGbmModel`] from a JSON string.
///
/// # Args
///
/// * `json_str` - JSON string previously produced by [`py_gbm_model_to_json_rs`].
///
/// # Returns
///
/// A new `PyGbmModel` instance.
///
/// # Errors
///
/// Returns `RuntimeError` on parse failures or on validation errors from the
/// model's config validator (e.g. an invalid `learning_rate` value in the
/// payload).
pub(crate) fn py_gbm_model_from_json_rs(json_str: &str) -> PyResult<PyGbmModel> {
    let inner: GradientBoostingModel = match serde_json::from_str(json_str) {
        Ok(m) => m,
        Err(e) => return Err(de_err(e.to_string())),
    };
    Ok(PyGbmModel { inner })
}

/// Returns per-feature split-count importance from a [`PyGbmModel`], normalized
/// to sum to 1.0.
///
/// A feature that never appears at an internal (split) node has importance 0.0.
/// If the ensemble has zero internal nodes (every tree is a single leaf), every
/// feature has importance 0.0.
///
/// # Args
///
/// * `model` - The `PyGbmModel` reference.
///
/// # Returns
///
/// A list of `(feature_name, importance)` pairs in feature-index order.
pub(crate) fn py_gbm_model_feature_importances_rs(model: &PyGbmModel) -> Vec<(String, f64)> {
    feature_importances(&model.inner)
}

/// Returns the number of trees in a [`PyGbmModel`] ensemble.
///
/// # Args
///
/// * `model` - The `PyGbmModel` reference.
///
/// # Returns
///
/// Tree count (equal to `n_estimators` unless early stopping trimmed the
/// ensemble).
pub(crate) fn py_gbm_model_n_trees_rs(model: &PyGbmModel) -> usize {
    model.inner.n_trees()
}

/// Extracts a [`PyGbmModel`] from arg 0 and serializes it to JSON.
///
/// # Args (positional)
///
/// 0. `model` (`PyGbmModel`)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or serialization fails.
pub(crate) fn py_gbm_model_to_json_from_args(args: &Bound<'_, PyTuple>) -> PyResult<String> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let model: PyRef<'_, PyGbmModel> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    py_gbm_model_to_json_rs(&model)
}

/// Extracts a JSON string from arg 0 and deserializes it into a [`PyGbmModel`].
///
/// # Args (positional)
///
/// 0. `json_str` (str)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction or deserialization fails.
pub(crate) fn py_gbm_model_from_json_from_args(args: &Bound<'_, PyTuple>) -> PyResult<PyGbmModel> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let json_str: String = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    py_gbm_model_from_json_rs(&json_str)
}

/// Extracts a [`PyGbmModel`] from arg 0 and returns its feature-importance
/// list.
///
/// # Args (positional)
///
/// 0. `model` (`PyGbmModel`)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction fails.
pub(crate) fn py_gbm_model_feature_importances_from_args(
    args: &Bound<'_, PyTuple>,
) -> PyResult<Vec<(String, f64)>> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let model: PyRef<'_, PyGbmModel> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    Ok(py_gbm_model_feature_importances_rs(&model))
}

/// Extracts a [`PyGbmModel`] from arg 0 and returns its tree count.
///
/// # Args (positional)
///
/// 0. `model` (`PyGbmModel`)
///
/// # Errors
///
/// Returns `PyErr` if argument extraction fails.
pub(crate) fn py_gbm_model_n_trees_from_args(args: &Bound<'_, PyTuple>) -> PyResult<usize> {
    let arg0 = match args.get_item(0_usize) {
        Ok(obj) => obj,
        Err(e) => return Err(e),
    };
    let model: PyRef<'_, PyGbmModel> = match arg0.extract() {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };
    Ok(py_gbm_model_n_trees_rs(&model))
}
