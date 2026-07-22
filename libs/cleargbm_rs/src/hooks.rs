//! Dependency injection hooks for testing.
//!
//! This module provides the `Hooks` struct for injecting behavior into the tree
//! building process. Production code uses `Hooks::default()` which calls the real
//! implementations. Tests can create custom `Hooks` to exercise error handling paths.
//!
//! # Pattern
//!
//! Production code sets hooks to real implementations at startup. Tests set them to
//! controlled implementations that can trigger specific error conditions. There are
//! no conditionals - the hook is always called directly.
//!
//! # Example
//!
//! ```rust,no_run
//! use cleargbm_rs::{Hooks, ClearGbmError, build_tree, BuildTreeInput};
//! use cleargbm_rs::types::HistogramBuffer;
//!
//! // Production: use default hooks (real implementations)
//! let hooks = Hooks::default();
//! // let result = build_tree(&input, &hooks);
//!
//! // Testing: inject custom behavior
//! fn error_histogram(
//!     _: &[usize], _: &[f64], _: &[f64], _: &[usize], _: usize,
//! ) -> Result<HistogramBuffer, ClearGbmError> {
//!     Err(ClearGbmError::EmptyInput { context: "injected".to_string() })
//! }
//! let test_hooks = Hooks::with_histogram_builder(error_histogram);
//! // let result = build_tree(&input, &test_hooks);
//! // assert!(result.is_err());
//! ```

use crate::error::ClearGbmError;
use crate::types::HistogramBuffer;

/// Function signature for building a single histogram.
///
/// Takes sample indices, gradients, hessians, bin assignments (`u8` per
/// sample), and number of bins. Returns a histogram buffer or an error.
pub type BuildHistogramFn = fn(
    sample_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[u8],
    n_bins: usize,
) -> Result<HistogramBuffer, ClearGbmError>;

/// Dependency injection hooks for tree building.
///
/// Contains function pointers for operations that might need to be injected
/// for testing purposes. Production code uses `Hooks::default()` which provides
/// the real implementations.
///
/// # Design
///
/// This struct enables testing of error propagation paths that would otherwise
/// be unreachable. By injecting a histogram builder that returns errors, tests
/// can exercise the `?` error propagation in `build_tree` and related functions.
#[derive(Clone)]
pub struct Hooks {
    /// Hook for building histograms from sample data.
    ///
    /// Default: `crate::histogram::build_histogram`
    pub build_histogram: BuildHistogramFn,

    /// Optional error to inject in finalize_nodes.
    ///
    /// If Some, finalize_nodes returns this error immediately.
    /// Used for testing error propagation in build_tree.
    pub finalize_nodes_error: Option<ClearGbmError>,
}

impl Default for Hooks {
    /// Creates hooks with the default (real) implementations.
    ///
    /// This is what production code should use.
    fn default() -> Self {
        Self {
            build_histogram: crate::histogram::build_histogram,
            finalize_nodes_error: None,
        }
    }
}

impl Hooks {
    /// Creates hooks with a custom histogram builder.
    ///
    /// # Args
    ///
    /// * `build_histogram` - Custom function for building histograms.
    ///
    /// # Returns
    ///
    /// A new `Hooks` instance with the custom histogram builder.
    pub const fn with_histogram_builder(build_histogram: BuildHistogramFn) -> Self {
        Self {
            build_histogram,
            finalize_nodes_error: None,
        }
    }

    /// Creates hooks with a finalize_nodes error injection.
    ///
    /// When finalize_nodes is called with these hooks, it will return the
    /// provided error immediately. Used for testing error propagation.
    ///
    /// # Args
    ///
    /// * `error` - The error to return from finalize_nodes.
    ///
    /// # Returns
    ///
    /// A new `Hooks` instance that will cause finalize_nodes to fail.
    pub fn with_finalize_nodes_error(error: ClearGbmError) -> Self {
        Self {
            build_histogram: crate::histogram::build_histogram,
            finalize_nodes_error: Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hooks_default() -> Result<(), ClearGbmError> {
        let hooks = Hooks::default();
        // Verify default hooks work with valid input
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];
        let result =
            (hooks.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_hooks_with_custom_histogram_builder() -> Result<(), ClearGbmError> {
        fn error_histogram(
            _: &[usize],
            _: &[f64],
            _: &[f64],
            _: &[u8],
            _: usize,
        ) -> Result<HistogramBuffer, ClearGbmError> {
            Err(ClearGbmError::EmptyInput {
                context: "test injected error".to_string(),
            })
        }

        let hooks = Hooks::with_histogram_builder(error_histogram);
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];
        let result =
            (hooks.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_hooks_clone() -> Result<(), ClearGbmError> {
        let hooks1 = Hooks::default();
        let hooks2 = hooks1.clone();
        // Both should work independently - verify by calling both
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];

        let result1 =
            (hooks1.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        let result2 =
            (hooks2.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        Ok(())
    }

    #[test]
    fn test_hooks_with_finalize_nodes_error() -> Result<(), ClearGbmError> {
        let hooks = Hooks::with_finalize_nodes_error(ClearGbmError::TreeConstructionFailed {
            reason: "test error".to_string(),
        });

        // finalize_nodes_error should be set
        assert!(hooks.finalize_nodes_error.is_some());

        // build_histogram should still be the default
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];
        let result =
            (hooks.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_hooks_default_has_no_finalize_error() -> Result<(), ClearGbmError> {
        let hooks = Hooks::default();
        assert!(hooks.finalize_nodes_error.is_none());
        Ok(())
    }

    #[test]
    fn test_hooks_with_histogram_builder_has_no_finalize_error() -> Result<(), ClearGbmError> {
        fn custom_histogram(
            _: &[usize],
            _: &[f64],
            _: &[f64],
            _: &[u8],
            _: usize,
        ) -> Result<HistogramBuffer, ClearGbmError> {
            Ok(HistogramBuffer::new(3_usize))
        }

        let hooks = Hooks::with_histogram_builder(custom_histogram);
        assert!(hooks.finalize_nodes_error.is_none());

        // Actually call the custom histogram to cover it
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];
        let result =
            (hooks.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        assert!(result.is_ok());
        Ok(())
    }
}
