//! Error types for `ClearGBM` Rust core.
//!
//! All errors are explicit and propagate via `Result<T, ClearGbmError>`.
//! No panics, no unwrap, no expect in production code.

use thiserror::Error;

/// Errors that can occur during `ClearGBM` operations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ClearGbmError {
    /// Feature index is out of bounds.
    #[error("feature index {index} out of bounds (n_features={n_features})")]
    FeatureIndexOutOfBounds {
        /// The invalid index that was provided.
        index: usize,
        /// The total number of features.
        n_features: usize,
    },

    /// Sample index is out of bounds.
    #[error("sample index {index} out of bounds (n_samples={n_samples})")]
    SampleIndexOutOfBounds {
        /// The invalid index that was provided.
        index: usize,
        /// The total number of samples.
        n_samples: usize,
    },

    /// Bin index is out of bounds.
    #[error("bin index {bin} out of bounds (n_bins={n_bins})")]
    BinIndexOutOfBounds {
        /// The invalid bin index.
        bin: usize,
        /// The total number of bins.
        n_bins: usize,
    },

    /// Array shape mismatch.
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        got: String,
    },

    /// Empty input where non-empty required.
    #[error("empty input: {context}")]
    EmptyInput {
        /// Context describing what was empty.
        context: String,
    },

    /// Invalid parameter value.
    #[error("invalid parameter {name}: {reason}")]
    InvalidParameter {
        /// Parameter name.
        name: String,
        /// Reason it's invalid.
        reason: String,
    },

    /// Tree construction failed.
    #[error("tree construction failed: {reason}")]
    TreeConstructionFailed {
        /// Reason for failure.
        reason: String,
    },

    /// Node not found in tree.
    #[error("node {node_id} not found in tree")]
    NodeNotFound {
        /// The missing node ID.
        node_id: usize,
    },

    /// Integer conversion failed.
    #[error("integer conversion failed: {context}")]
    IntegerConversion {
        /// Context describing the conversion.
        context: String,
    },

    /// Serialization failed.
    #[error("serialization failed: {reason}")]
    SerializationFailed {
        /// Reason for the serialization failure.
        reason: String,
    },

    /// Deserialization failed.
    #[error("deserialization failed: {reason}")]
    DeserializationFailed {
        /// Reason for the deserialization failure.
        reason: String,
    },

    /// Invalid label value (expected 0 or 1).
    #[error("invalid label {value} at index {index} (expected 0 or 1)")]
    InvalidLabel {
        /// The invalid label value that was provided.
        value: u8,
        /// The index of the invalid label.
        index: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_feature_index_out_of_bounds() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::FeatureIndexOutOfBounds {
            index: 10,
            n_features: 5,
        };
        assert_eq!(
            err.to_string(),
            "feature index 10 out of bounds (n_features=5)"
        );
        Ok(())
    }

    #[test]
    fn test_error_display_sample_index_out_of_bounds() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::SampleIndexOutOfBounds {
            index: 100,
            n_samples: 50,
        };
        assert_eq!(
            err.to_string(),
            "sample index 100 out of bounds (n_samples=50)"
        );
        Ok(())
    }

    #[test]
    fn test_error_display_bin_index_out_of_bounds() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::BinIndexOutOfBounds {
            bin: 64,
            n_bins: 64,
        };
        assert_eq!(err.to_string(), "bin index 64 out of bounds (n_bins=64)");
        Ok(())
    }

    #[test]
    fn test_error_display_shape_mismatch() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::ShapeMismatch {
            expected: "length 100".to_string(),
            got: "length 50".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "shape mismatch: expected length 100, got length 50"
        );
        Ok(())
    }

    #[test]
    fn test_error_display_empty_input() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::EmptyInput {
            context: "sample_indices cannot be empty".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "empty input: sample_indices cannot be empty"
        );
        Ok(())
    }

    #[test]
    fn test_error_display_invalid_parameter() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::InvalidParameter {
            name: "max_depth".to_string(),
            reason: "must be positive".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "invalid parameter max_depth: must be positive"
        );
        Ok(())
    }

    #[test]
    fn test_error_display_tree_construction_failed() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::TreeConstructionFailed {
            reason: "no valid split found".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "tree construction failed: no valid split found"
        );
        Ok(())
    }

    #[test]
    fn test_error_display_node_not_found() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::NodeNotFound { node_id: 42 };
        assert_eq!(err.to_string(), "node 42 not found in tree");
        Ok(())
    }

    #[test]
    fn test_error_display_integer_conversion() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::IntegerConversion {
            context: "i64 to usize".to_string(),
        };
        assert_eq!(err.to_string(), "integer conversion failed: i64 to usize");
        Ok(())
    }

    #[test]
    fn test_error_display_invalid_label() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::InvalidLabel { value: 5, index: 3 };
        assert_eq!(
            err.to_string(),
            "invalid label 5 at index 3 (expected 0 or 1)"
        );
        Ok(())
    }

    #[test]
    fn test_error_clone() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::BinIndexOutOfBounds { bin: 5, n_bins: 4 };
        let cloned = err.clone();
        assert_eq!(err, cloned);
        Ok(())
    }

    #[test]
    fn test_error_debug() -> Result<(), ClearGbmError> {
        let err = ClearGbmError::BinIndexOutOfBounds { bin: 5, n_bins: 4 };
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("BinIndexOutOfBounds"));
        assert!(debug_str.contains("bin: 5"));
        assert!(debug_str.contains("n_bins: 4"));
        Ok(())
    }
}
