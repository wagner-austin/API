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
//!     _: &[u32], _: &[f32], _: &[f32], _: &[u8], _: usize,
//! ) -> Result<HistogramBuffer, ClearGbmError> {
//!     Err(ClearGbmError::EmptyInput { context: "injected".to_string() })
//! }
//! let test_hooks = Hooks::with_histogram_builder(error_histogram);
//! // let result = build_tree(&input, &test_hooks);
//! // assert!(result.is_err());
//! ```

use crate::error::ClearGbmError;
use crate::types::HistogramBuffer;

/// Function signature for the per-feature histogram builder called by the
/// tree builder.
///
/// Args:
/// - `sample_indices` — sample IDs at this node (used for the per-feature
///   bin gather; NOT indexed into the gradient/hessian arrays).
/// - `ordered_gradients` — pre-permuted gradient stream: `ordered_gradients[i]`
///   equals `gradients[sample_indices[i]]`. Length equals `sample_indices.len()`.
/// - `ordered_hessians` — pre-permuted hessian stream, same shape.
/// - `bins` — per-sample bin assignments (`u8`) for one feature.
/// - `n_bins` — number of bins (including NaN bin).
///
/// The tree builder does one node-scoped reorder pass (via
/// [`crate::histogram::reorder_grad_hess_into`]) before dispatching this
/// hook per feature, so every per-feature call gets sequential-access-shaped
/// gradient + hessian streams.
///
/// Returns a `Result` so error-injection tests can force failure through the
/// same hook the production path uses — no separate injection surface.
pub type BuildHistogramFn = fn(
    sample_indices: &[u32],
    ordered_gradients: &[f32],
    ordered_hessians: &[f32],
    bins: &[u8],
    n_bins: usize,
) -> Result<HistogramBuffer, ClearGbmError>;

/// Dependency injection hooks for tree building.
///
/// Contains function pointers for operations that might need to be injected
/// for testing purposes. Production code uses `Hooks::default()` which provides
/// the real implementations. Tests inject error-returning implementations to
/// exercise error-propagation paths that would otherwise be unreachable.
#[derive(Clone)]
pub struct Hooks {
    /// Hook for building one feature's histogram from pre-reordered inputs.
    ///
    /// Default: [`default_trusted_build_histogram`], wrapping
    /// [`crate::histogram::build_histogram_ordered_trusted`] in the fallible
    /// signature.
    pub build_histogram: BuildHistogramFn,

    /// Optional error to inject in finalize_nodes.
    ///
    /// If Some, finalize_nodes returns this error immediately.
    /// Used for testing error propagation in build_tree.
    pub finalize_nodes_error: Option<ClearGbmError>,
}

/// Default histogram builder used by [`Hooks::default`].
///
/// Wraps [`crate::histogram::build_histogram_ordered_trusted`] in the
/// fallible signature that [`BuildHistogramFn`] demands. The wrapped
/// function reads `ordered_gradients` / `ordered_hessians` sequentially by
/// loop position; only the bin lookup is a gather. The tree builder
/// establishes the caller-side invariants (index bounds, ordered-array
/// length) by construction — see the docs on
/// [`crate::histogram::build_histogram_ordered_trusted`].
#[inline]
fn default_trusted_build_histogram(
    sample_indices: &[u32],
    ordered_gradients: &[f32],
    ordered_hessians: &[f32],
    bins: &[u8],
    n_bins: usize,
) -> Result<HistogramBuffer, ClearGbmError> {
    Ok(crate::histogram::build_histogram_ordered_trusted(
        sample_indices,
        ordered_gradients,
        ordered_hessians,
        bins,
        n_bins,
    ))
}

impl Default for Hooks {
    /// Creates hooks with the default (real) implementations.
    ///
    /// This is what production code should use.
    fn default() -> Self {
        Self {
            build_histogram: default_trusted_build_histogram,
            finalize_nodes_error: None,
        }
    }
}

impl Hooks {
    /// Creates hooks with a custom histogram builder.
    ///
    /// Used by error-injection tests to force `?`-propagation paths in the
    /// tree builder — the injected function receives the same
    /// (`sample_indices`, `ordered_gradients`, `ordered_hessians`, `bins`,
    /// `n_bins`) args the default does, and returns `Err(...)` to trigger
    /// the failure path.
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
    pub fn with_finalize_nodes_error(error: ClearGbmError) -> Self {
        Self {
            build_histogram: default_trusted_build_histogram,
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
        let sample_indices = vec![0_u32, 1_u32];
        let gradients = vec![1.0_f32, 2.0_f32];
        let hessians = vec![1.0_f32, 1.0_f32];
        let bins = vec![0_u8, 1_u8];
        let result =
            (hooks.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_hooks_with_custom_histogram_builder() -> Result<(), ClearGbmError> {
        fn error_histogram(
            _: &[u32],
            _: &[f32],
            _: &[f32],
            _: &[u8],
            _: usize,
        ) -> Result<HistogramBuffer, ClearGbmError> {
            Err(ClearGbmError::EmptyInput {
                context: "test injected error".to_string(),
            })
        }

        let hooks = Hooks::with_histogram_builder(error_histogram);
        let sample_indices = vec![0_u32, 1_u32];
        let gradients = vec![1.0_f32, 2.0_f32];
        let hessians = vec![1.0_f32, 1.0_f32];
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
        let sample_indices = vec![0_u32, 1_u32];
        let gradients = vec![1.0_f32, 2.0_f32];
        let hessians = vec![1.0_f32, 1.0_f32];
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
        let sample_indices = vec![0_u32, 1_u32];
        let gradients = vec![1.0_f32, 2.0_f32];
        let hessians = vec![1.0_f32, 1.0_f32];
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
            _: &[u32],
            _: &[f32],
            _: &[f32],
            _: &[u8],
            _: usize,
        ) -> Result<HistogramBuffer, ClearGbmError> {
            Ok(HistogramBuffer::new(3_usize))
        }

        let hooks = Hooks::with_histogram_builder(custom_histogram);
        assert!(hooks.finalize_nodes_error.is_none());

        // Actually call the custom histogram to cover it
        let sample_indices = vec![0_u32, 1_u32];
        let gradients = vec![1.0_f32, 2.0_f32];
        let hessians = vec![1.0_f32, 1.0_f32];
        let bins = vec![0_u8, 1_u8];
        let result =
            (hooks.build_histogram)(&sample_indices, &gradients, &hessians, &bins, 3_usize);
        assert!(result.is_ok());
        Ok(())
    }
}
