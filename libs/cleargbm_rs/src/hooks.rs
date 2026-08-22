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
//! use cleargbm_rs::histogram::NodeHistogramRequest;
//!
//! // Production: use default hooks (real implementations)
//! let hooks = Hooks::default();
//! // let result = build_tree(&input, &hooks);
//!
//! // Testing: inject custom behavior
//! fn error_histogram(
//!     _: NodeHistogramRequest<'_>,
//! ) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
//!     Err(ClearGbmError::EmptyInput { context: "injected".to_string() })
//! }
//! let test_hooks = Hooks::with_histogram_builder(error_histogram);
//! // let result = build_tree(&input, &test_hooks);
//! // assert!(result.is_err());
//! ```

use crate::error::ClearGbmError;
use crate::histogram::NodeHistogramRequest;
use crate::types::HistogramBuffer;

/// Dependency injection hooks for tree building.
///
/// Contains function pointers for operations that might need to be injected
/// for testing purposes. Production code uses `Hooks::default()` which provides
/// the real implementations. Tests inject error-returning implementations to
/// exercise error-propagation paths that would otherwise be unreachable.
///
/// The node histogram builder receives pre-reordered inputs — the tree
/// builder does one node-scoped reorder pass (via
/// [`crate::histogram::reorder_grad_hess_into`]) before dispatching this hook
/// once per node, which builds every feature's histogram in a single sample
/// walk. Returns a `Result` so error-injection tests can force failure
/// through the same hook the production path uses — no separate injection
/// surface.
#[derive(Clone)]
pub struct Hooks {
    /// Hook for building all of one node's feature histograms in one pass.
    ///
    /// Default: [`crate::histogram::build_node_histograms_single_pass`]
    /// wrapped in `Ok(...)`.
    pub build_node_histograms:
        fn(NodeHistogramRequest<'_>) -> Result<Vec<HistogramBuffer>, ClearGbmError>,

    /// Optional error to inject in finalize_nodes.
    ///
    /// If Some, finalize_nodes returns this error immediately.
    /// Used for testing error propagation in build_tree.
    pub finalize_nodes_error: Option<ClearGbmError>,

    /// Hook for building the run-scoped rayon worker pool.
    ///
    /// Default: `rayon::ThreadPoolBuilder::new().num_threads(n).build()`.
    ///
    /// Typed with rayon's own error rather than [`ClearGbmError`] so the
    /// default is a single expression with no error arm of its own; the
    /// translation into a `ClearGbmError` happens at the one call site, in
    /// [`crate::training::train_gradient_boosting`]. Pool construction is
    /// injectable because it is the only fallible step in training that no
    /// input can provoke — a caller cannot ask for a thread count the OS
    /// refuses — so an error-injection test is the only way to reach the
    /// failure path.
    pub build_pool:
        fn(core::num::NonZeroUsize) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError>,
}

impl Default for Hooks {
    /// Creates hooks with the default (real) implementations.
    ///
    /// This is what production code should use.
    fn default() -> Self {
        Self {
            build_node_histograms: |request| {
                Ok(crate::histogram::build_node_histograms_single_pass(request))
            },
            finalize_nodes_error: None,
            build_pool: |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads.get())
                    .build()
            },
        }
    }
}

impl Hooks {
    /// Creates hooks with a custom histogram builder.
    ///
    /// Used by error-injection tests to force `?`-propagation paths in the
    /// tree builder — the injected function receives the same
    /// [`NodeHistogramRequest`] the default does, and returns `Err(...)` to
    /// trigger the failure path.
    pub fn with_histogram_builder(
        build_node_histograms: fn(
            NodeHistogramRequest<'_>,
        ) -> Result<Vec<HistogramBuffer>, ClearGbmError>,
    ) -> Self {
        Self {
            build_node_histograms,
            ..Self::default()
        }
    }

    /// Creates hooks with a custom worker-pool builder.
    ///
    /// Used by error-injection tests to reach the pool-construction failure
    /// path in [`crate::training::train_gradient_boosting`], which no caller
    /// input can trigger.
    ///
    /// # Args
    ///
    /// * `build_pool` - Replacement pool constructor.
    ///
    /// # Returns
    ///
    /// Hooks with the default histogram builder and the supplied pool builder.
    #[must_use]
    pub fn with_pool_builder(
        build_pool: fn(
            core::num::NonZeroUsize,
        ) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError>,
    ) -> Self {
        Self {
            build_pool,
            ..Self::default()
        }
    }

    /// Creates hooks with a finalize_nodes error injection.
    ///
    /// When finalize_nodes is called with these hooks, it will return the
    /// provided error immediately. Used for testing error propagation.
    pub fn with_finalize_nodes_error(error: ClearGbmError) -> Self {
        Self {
            finalize_nodes_error: Some(error),
            ..Self::default()
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
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];
        let result = (hooks.build_node_histograms)(NodeHistogramRequest {
            sample_indices: &sample_indices,
            ordered_gradients: &gradients,
            ordered_hessians: &hessians,
            bins_rows: &bins,
            n_features: 1_usize,
            n_bins: 3_usize,
        });
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_hooks_with_custom_histogram_builder() -> Result<(), ClearGbmError> {
        fn error_histogram(
            _: NodeHistogramRequest<'_>,
        ) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
            Err(ClearGbmError::EmptyInput {
                context: "test injected error".to_string(),
            })
        }

        let hooks = Hooks::with_histogram_builder(error_histogram);
        let sample_indices = vec![0_u32, 1_u32];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];
        let result = (hooks.build_node_histograms)(NodeHistogramRequest {
            sample_indices: &sample_indices,
            ordered_gradients: &gradients,
            ordered_hessians: &hessians,
            bins_rows: &bins,
            n_features: 1_usize,
            n_bins: 3_usize,
        });
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_hooks_clone() -> Result<(), ClearGbmError> {
        let hooks1 = Hooks::default();
        let hooks2 = hooks1.clone();
        // Both should work independently - verify by calling both
        let sample_indices = vec![0_u32, 1_u32];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];

        let result1 = (hooks1.build_node_histograms)(NodeHistogramRequest {
            sample_indices: &sample_indices,
            ordered_gradients: &gradients,
            ordered_hessians: &hessians,
            bins_rows: &bins,
            n_features: 1_usize,
            n_bins: 3_usize,
        });
        let result2 = (hooks2.build_node_histograms)(NodeHistogramRequest {
            sample_indices: &sample_indices,
            ordered_gradients: &gradients,
            ordered_hessians: &hessians,
            bins_rows: &bins,
            n_features: 1_usize,
            n_bins: 3_usize,
        });

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
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];
        let result = (hooks.build_node_histograms)(NodeHistogramRequest {
            sample_indices: &sample_indices,
            ordered_gradients: &gradients,
            ordered_hessians: &hessians,
            bins_rows: &bins,
            n_features: 1_usize,
            n_bins: 3_usize,
        });
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
            _: NodeHistogramRequest<'_>,
        ) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
            Ok(vec![HistogramBuffer::new(3_usize)])
        }

        let hooks = Hooks::with_histogram_builder(custom_histogram);
        assert!(hooks.finalize_nodes_error.is_none());

        // Actually call the custom histogram to cover it
        let sample_indices = vec![0_u32, 1_u32];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![0_u8, 1_u8];
        let result = (hooks.build_node_histograms)(NodeHistogramRequest {
            sample_indices: &sample_indices,
            ordered_gradients: &gradients,
            ordered_hessians: &hessians,
            bins_rows: &bins,
            n_features: 1_usize,
            n_bins: 3_usize,
        });
        assert!(result.is_ok());
        Ok(())
    }
}
