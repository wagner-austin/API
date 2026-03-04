//! Conversion from [`ClearGbmError`] to Python exceptions.
//!
//! Maps each error variant to the most appropriate Python exception type:
//! - Index errors → `IndexError`
//! - Validation/input errors → `ValueError`
//! - Internal/construction errors → `RuntimeError`

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

use crate::error::ClearGbmError;

/// Converts a [`ClearGbmError`] into the corresponding Python exception.
///
/// # Mapping
///
/// | Rust Variant | Python Exception |
/// |-------------|-----------------|
/// | `FeatureIndexOutOfBounds` | `IndexError` |
/// | `SampleIndexOutOfBounds` | `IndexError` |
/// | `BinIndexOutOfBounds` | `IndexError` |
/// | `NodeNotFound` | `IndexError` |
/// | `ShapeMismatch` | `ValueError` |
/// | `EmptyInput` | `ValueError` |
/// | `InvalidParameter` | `ValueError` |
/// | `IntegerConversion` | `ValueError` |
/// | `TreeConstructionFailed` | `RuntimeError` |
/// | `SerializationFailed` | `RuntimeError` |
/// | `DeserializationFailed` | `RuntimeError` |
impl From<ClearGbmError> for PyErr {
    fn from(err: ClearGbmError) -> Self {
        let message = err.to_string();
        match err {
            // Index-related errors → IndexError
            ClearGbmError::FeatureIndexOutOfBounds { .. }
            | ClearGbmError::SampleIndexOutOfBounds { .. }
            | ClearGbmError::BinIndexOutOfBounds { .. }
            | ClearGbmError::NodeNotFound { .. } => PyIndexError::new_err(message),

            // Validation/input errors → ValueError
            ClearGbmError::ShapeMismatch { .. }
            | ClearGbmError::EmptyInput { .. }
            | ClearGbmError::InvalidParameter { .. }
            | ClearGbmError::IntegerConversion { .. } => PyValueError::new_err(message),

            // Internal/construction errors → RuntimeError
            ClearGbmError::TreeConstructionFailed { .. }
            | ClearGbmError::SerializationFailed { .. }
            | ClearGbmError::DeserializationFailed { .. } => PyRuntimeError::new_err(message),
        }
    }
}
