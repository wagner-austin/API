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
/// Takes sample indices, gradients, hessians, bin assignments, and number of bins.
/// Returns a histogram buffer or an error.
pub type BuildHistogramFn = fn(
    sample_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[usize],
    n_bins: usize,
) -> std::result::Result<HistogramBuffer, ClearGbmError>;

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
#[derive(Clone, Copy)]
pub struct Hooks {
    /// Hook for building histograms from sample data.
    ///
    /// Default: `crate::histogram::build_histogram`
    pub build_histogram: BuildHistogramFn,
}

impl Default for Hooks {
    /// Creates hooks with the default (real) implementations.
    ///
    /// This is what production code should use.
    fn default() -> Self {
        Self {
            build_histogram: crate::histogram::build_histogram,
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
        Self { build_histogram }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hooks_default() -> std::result::Result<(), ClearGbmError> {
        let hooks = Hooks::default();
        // Verify default hooks work with valid input
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_usize, 1_usize];
        let result =
            (hooks.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_hooks_with_custom_histogram_builder() -> std::result::Result<(), ClearGbmError> {
        fn error_histogram(
            _: &[usize],
            _: &[f64],
            _: &[f64],
            _: &[usize],
            _: usize,
        ) -> std::result::Result<HistogramBuffer, ClearGbmError> {
            Err(ClearGbmError::EmptyInput {
                context: "test injected error".to_string(),
            })
        }

        let hooks = Hooks::with_histogram_builder(error_histogram);
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_usize, 1_usize];
        let result =
            (hooks.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_hooks_copy() -> std::result::Result<(), ClearGbmError> {
        let hooks1 = Hooks::default();
        let hooks2 = hooks1;
        // Both should work independently - verify by calling both
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_usize, 1_usize];

        let result1 =
            (hooks1.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        let result2 =
            (hooks2.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        Ok(())
    }
}
