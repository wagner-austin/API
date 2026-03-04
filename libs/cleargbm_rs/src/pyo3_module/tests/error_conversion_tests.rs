//! Tests for ClearGbmError → PyErr conversion.
//!
//! Verifies that each error variant maps to the correct Python exception type.

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::types::PyTypeMethods;
use pyo3::PyErr;

use crate::error::ClearGbmError;

/// Helper: converts a ClearGbmError to PyErr and checks its Python type name.
fn assert_pyerr_type(err: ClearGbmError, expected_type: &str) -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let py_err: PyErr = err.into();
        let err_type = py_err.get_type(py);
        let type_name_bound = match err_type.name() {
            Ok(n) => n,
            Err(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "failed to get error type name".to_string(),
                })
            }
        };
        let type_name = type_name_bound.to_string();
        assert_eq!(type_name, expected_type);
        Ok(())
    })
}

#[test]
fn test_feature_index_out_of_bounds_maps_to_index_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::FeatureIndexOutOfBounds {
            index: 10_usize,
            n_features: 5_usize,
        },
        "IndexError",
    )
}

#[test]
fn test_sample_index_out_of_bounds_maps_to_index_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::SampleIndexOutOfBounds {
            index: 100_usize,
            n_samples: 50_usize,
        },
        "IndexError",
    )
}

#[test]
fn test_bin_index_out_of_bounds_maps_to_index_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::BinIndexOutOfBounds {
            bin: 64_usize,
            n_bins: 64_usize,
        },
        "IndexError",
    )
}

#[test]
fn test_node_not_found_maps_to_index_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::NodeNotFound { node_id: 42_usize },
        "IndexError",
    )
}

#[test]
fn test_shape_mismatch_maps_to_value_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::ShapeMismatch {
            expected: "100".to_string(),
            got: "50".to_string(),
        },
        "ValueError",
    )
}

#[test]
fn test_empty_input_maps_to_value_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::EmptyInput {
            context: "test".to_string(),
        },
        "ValueError",
    )
}

#[test]
fn test_invalid_parameter_maps_to_value_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::InvalidParameter {
            name: "max_depth".to_string(),
            reason: "must be positive".to_string(),
        },
        "ValueError",
    )
}

#[test]
fn test_integer_conversion_maps_to_value_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::IntegerConversion {
            context: "negative i64 to usize".to_string(),
        },
        "ValueError",
    )
}

#[test]
fn test_tree_construction_failed_maps_to_runtime_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::TreeConstructionFailed {
            reason: "cycle detected".to_string(),
        },
        "RuntimeError",
    )
}

#[test]
fn test_serialization_failed_maps_to_runtime_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::SerializationFailed {
            reason: "io error".to_string(),
        },
        "RuntimeError",
    )
}

#[test]
fn test_deserialization_failed_maps_to_runtime_error() -> Result<(), ClearGbmError> {
    assert_pyerr_type(
        ClearGbmError::DeserializationFailed {
            reason: "unexpected token".to_string(),
        },
        "RuntimeError",
    )
}

#[test]
fn test_pyerr_message_contains_original_error() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let err = ClearGbmError::FeatureIndexOutOfBounds {
            index: 10_usize,
            n_features: 5_usize,
        };
        let expected_msg = err.to_string();
        let py_err: PyErr = err.into();
        let msg = py_err.value(py).to_string();
        assert!(
            msg.contains(&expected_msg),
            "expected message to contain '{expected_msg}', got '{msg}'"
        );
        Ok(())
    })
}

#[test]
fn test_index_error_is_instance() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let err = ClearGbmError::FeatureIndexOutOfBounds {
            index: 0_usize,
            n_features: 0_usize,
        };
        let py_err: PyErr = err.into();
        assert!(py_err.is_instance_of::<PyIndexError>(py));
        Ok(())
    })
}

#[test]
fn test_value_error_is_instance() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let err = ClearGbmError::EmptyInput {
            context: "test".to_string(),
        };
        let py_err: PyErr = err.into();
        assert!(py_err.is_instance_of::<PyValueError>(py));
        Ok(())
    })
}

#[test]
fn test_runtime_error_is_instance() -> Result<(), ClearGbmError> {
    pyo3::Python::initialize();
    pyo3::Python::attach(|py| {
        let err = ClearGbmError::TreeConstructionFailed {
            reason: "test".to_string(),
        };
        let py_err: PyErr = err.into();
        assert!(py_err.is_instance_of::<PyRuntimeError>(py));
        Ok(())
    })
}
