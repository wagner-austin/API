//! Shared fixtures for the PyO3 binding tests.
//!
//! Holds the dataset, config-dict and module-construction helpers used by more
//! than one test module in [`super`]. Everything here builds *real* Python
//! objects and calls the real registered bindings — there are no stand-ins for
//! the code under test.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::error::ClearGbmError;
use crate::pyo3_module::training_fns::train_gradient_boosting_from_args;

/// Wraps a [`PyErr`] into a [`ClearGbmError`] so tests can return `Result`.
///
/// # Args
///
/// * `e` - The Python error to describe.
///
/// # Returns
///
/// A [`ClearGbmError::TreeConstructionFailed`] carrying the Python error text.
pub(super) fn wrap_py_err(e: &PyErr) -> ClearGbmError {
    ClearGbmError::TreeConstructionFailed {
        reason: format!("PyErr: {e}"),
    }
}

/// Wraps an arbitrary failure description into a [`ClearGbmError`].
///
/// # Args
///
/// * `reason` - Human-readable description of what failed.
///
/// # Returns
///
/// A [`ClearGbmError::TreeConstructionFailed`] carrying `reason`.
pub(super) fn fail(reason: String) -> ClearGbmError {
    ClearGbmError::TreeConstructionFailed { reason }
}

/// Sets an `i64` value in a config dict.
///
/// # Args
///
/// * `dict` - The config dict to mutate.
/// * `key` - The config key to set.
/// * `val` - The value to store.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if the insert fails.
pub(super) fn set_config_i64(
    dict: &Bound<'_, PyDict>,
    key: &str,
    val: i64,
) -> Result<(), ClearGbmError> {
    match dict.set_item(key, val) {
        Ok(()) => Ok(()),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Sets an `f64` value in a config dict.
///
/// # Args
///
/// * `dict` - The config dict to mutate.
/// * `key` - The config key to set.
/// * `val` - The value to store.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if the insert fails.
pub(super) fn set_config_f64(
    dict: &Bound<'_, PyDict>,
    key: &str,
    val: f64,
) -> Result<(), ClearGbmError> {
    match dict.set_item(key, val) {
        Ok(()) => Ok(()),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Sets a string value in a config dict.
///
/// # Args
///
/// * `dict` - The config dict to mutate.
/// * `key` - The config key to set.
/// * `val` - The value to store.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if the insert fails.
pub(super) fn set_config_str(
    dict: &Bound<'_, PyDict>,
    key: &str,
    val: &str,
) -> Result<(), ClearGbmError> {
    match dict.set_item(key, val) {
        Ok(()) => Ok(()),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Builds a config dict with valid training hyperparameters.
///
/// `n_jobs` is pinned to `1` so tests are deterministic and do not spawn a
/// worker pool per training call.
///
/// # Args
///
/// * `py` - Python GIL token.
///
/// # Returns
///
/// A dict accepted by `train_gradient_boosting_rs`.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if any insert fails.
pub(super) fn make_config_dict<'py>(py: Python<'py>) -> Result<Bound<'py, PyDict>, ClearGbmError> {
    let config = PyDict::new(py);
    match set_config_i64(&config, "n_estimators", 2_i64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_i64(&config, "max_depth", 2_i64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_f64(&config, "learning_rate", 0.1_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_i64(&config, "min_samples_split", 2_i64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_i64(&config, "min_samples_leaf", 1_i64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_i64(&config, "max_bins", 4_i64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_f64(&config, "subsample", 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_i64(&config, "random_state", 42_i64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_f64(&config, "reg_alpha", 0.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_f64(&config, "reg_lambda", 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_i64(&config, "n_jobs", 1_i64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    match set_config_str(&config, "growth_strategy", "depth_wise") {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    // Present and null: depth-wise growth carries no leaf budget, and the
    // extractor requires the key rather than inferring absence as "no budget".
    match config.set_item("num_leaves", py.None()) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match set_config_f64(&config, "scale_pos_weight", 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    };
    // Present and null, like num_leaves: absence must be an error, not a
    // silent "all features".
    match config.set_item("max_features", py.None()) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    Ok(config)
}

/// The 6-sample, 2-feature training matrix shared by the binding tests.
///
/// Linearly separable on either feature, so a depth-2 tree splits rather than
/// collapsing to a single leaf — which keeps feature-importance and tree-count
/// assertions meaningful.
///
/// # Returns
///
/// Six rows of two `f64` features.
pub(super) fn training_rows() -> Vec<Vec<f64>> {
    vec![
        vec![0.1_f64, 0.2_f64],
        vec![0.3_f64, 0.4_f64],
        vec![0.5_f64, 0.6_f64],
        vec![0.7_f64, 0.8_f64],
        vec![0.9_f64, 1.0_f64],
        vec![1.1_f64, 1.2_f64],
    ]
}

/// The labels paired with [`training_rows`].
///
/// # Returns
///
/// Six binary labels, three of each class.
pub(super) fn training_labels() -> Vec<i64> {
    vec![0_i64, 0_i64, 0_i64, 1_i64, 1_i64, 1_i64]
}

/// Builds the positional args tuple for `train_gradient_boosting_from_args`.
///
/// # Args
///
/// * `py` - Python GIL token.
///
/// # Returns
///
/// A six-element tuple `(x_train, y_train, None, None, config, feature_names)`.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if any Python object
/// construction fails.
pub(super) fn make_training_args<'py>(
    py: Python<'py>,
) -> Result<Bound<'py, PyTuple>, ClearGbmError> {
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
        Err(e) => return Err(wrap_py_err(&e)),
    };

    match PyTuple::new(
        py,
        [
            x_train.into_any(),
            y_train.into_any(),
            py.None().into_bound(py).into_any(),
            py.None().into_bound(py).into_any(),
            config.into_any(),
            names.into_any(),
        ],
    ) {
        Ok(t) => Ok(t),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Trains a model through the real binding and returns the Python handle.
///
/// # Args
///
/// * `py` - Python GIL token.
///
/// # Returns
///
/// The trained `PyGbmModel` as an opaque Python object.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if training fails.
pub(super) fn train_model(py: Python<'_>) -> Result<Py<PyAny>, ClearGbmError> {
    let args = match make_training_args(py) {
        Ok(a) => a,
        Err(e) => return Err(e),
    };
    match train_gradient_boosting_from_args(&args) {
        Ok(m) => Ok(m),
        Err(e) => Err(wrap_py_err(&e)),
    }
}

/// Creates and initializes a real `cleargbm_rs` module.
///
/// Every function the extension exposes is registered by this call, so tests
/// that go through the returned module exercise the registration closures in
/// [`crate::pyo3_module`] as well as the functions themselves.
///
/// # Args
///
/// * `py` - Python GIL token.
///
/// # Returns
///
/// The initialized module.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if module creation or
/// registration fails.
pub(super) fn init_module<'py>(py: Python<'py>) -> Result<Bound<'py, PyModule>, ClearGbmError> {
    let module = match pyo3::types::PyModule::new(py, "cleargbm_rs") {
        Ok(m) => m,
        Err(e) => return Err(wrap_py_err(&e)),
    };
    match crate::pyo3_module::cleargbm_rs(&module) {
        Ok(()) => {}
        Err(e) => return Err(wrap_py_err(&e)),
    };
    Ok(module)
}

/// Looks up a registered function on an initialized module.
///
/// # Args
///
/// * `module` - A module produced by [`init_module`].
/// * `name` - The registered function name.
///
/// # Returns
///
/// The bound callable.
///
/// # Errors
///
/// Returns [`ClearGbmError::TreeConstructionFailed`] if `name` is not
/// registered.
pub(super) fn module_fn<'py>(
    module: &Bound<'py, PyModule>,
    name: &str,
) -> Result<Bound<'py, PyAny>, ClearGbmError> {
    match module.getattr(name) {
        Ok(f) => Ok(f),
        Err(e) => Err(wrap_py_err(&e)),
    }
}
