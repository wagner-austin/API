//! Error-path tests for leaf-wise growth: input validation, the
//! assignment-discarding wrapper, and propagation of injected hook
//! failures from every stage of the grow loop.

use super::leafwise_helpers::{
    fixture_bins, fixture_gradients, fixture_hessians, fixture_indices, fixture_thresholds,
    make_config, ROOT_SAMPLES,
};

use crate::error::ClearGbmError;
use crate::histogram::NodeHistogramRequest;
use crate::hooks::Hooks;
use crate::tree::{
    build_tree_leaf_wise, build_tree_leaf_wise_with_leaf_assignment, BuildTreeInput,
    TreeBuildConfig,
};
use crate::types::HistogramBuffer;
// =========================================================================
// Shared validation + the discarding wrapper
// =========================================================================

#[test]
fn test_leaf_wise_rejects_empty_sample_indices() -> Result<(), ClearGbmError> {
    let config = match make_config(3_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let sample_indices: Vec<u32> = Vec::new();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 8_usize, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 8_usize,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &config,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    match build_tree_leaf_wise(&input, &Hooks::default()) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "empty sample_indices must be rejected".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert_eq!(context, "sample_indices");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

// =========================================================================
// Error propagation
// =========================================================================

/// Runs the leaf-wise builder over the fixture with injected hooks.
fn grow_with_hooks(
    config: &TreeBuildConfig,
    hooks: &Hooks,
) -> Result<(crate::tree::Tree, Vec<f64>), ClearGbmError> {
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, ROOT_SAMPLES, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: ROOT_SAMPLES,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    build_tree_leaf_wise_with_leaf_assignment(&input, hooks)
}

/// Histogram builder that always fails.
fn error_histogram(_: NodeHistogramRequest<'_>) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    Err(ClearGbmError::EmptyInput {
        context: "injected error from hook".to_string(),
    })
}

/// Histogram builder that fails only below the root.
///
/// Keyed on the request's sample count rather than a call counter: the tree
/// suite runs its tests concurrently, so a shared counter would make which
/// call fails depend on scheduling. The root is the only node holding all
/// eight fixture samples, so this reliably lets root construction succeed and
/// fails the first child.
fn error_histogram_below_root(
    request: NodeHistogramRequest<'_>,
) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    if request.sample_indices.len() == ROOT_SAMPLES {
        return Ok(crate::histogram::build_node_histograms_single_pass(request));
    }
    Err(ClearGbmError::EmptyInput {
        context: "injected child histogram failure".to_string(),
    })
}

/// Histogram builder that always returns a too-small buffer.
fn undersized_histogram(
    _: NodeHistogramRequest<'_>,
) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    Ok(vec![HistogramBuffer::new(2_usize)])
}

#[test]
fn test_leaf_wise_propagates_a_root_histogram_failure() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let hooks = Hooks::with_histogram_builder(error_histogram);
    match grow_with_hooks(&config, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a root histogram failure must propagate".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert_eq!(context, "injected error from hook");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_leaf_wise_propagates_a_root_split_search_failure() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let hooks = Hooks::with_histogram_builder(undersized_histogram);
    match grow_with_hooks(&config, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a root split-search failure must propagate".to_string(),
        }),
        Err(_) => Ok(()),
    }
}

#[test]
fn test_leaf_wise_propagates_a_child_histogram_failure() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let hooks = Hooks::with_histogram_builder(error_histogram_below_root);
    match grow_with_hooks(&config, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a child histogram failure must propagate".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert_eq!(context, "injected child histogram failure");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_leaf_wise_propagates_a_finalize_failure() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let hooks = Hooks {
        finalize_nodes_error: Some(ClearGbmError::TreeConstructionFailed {
            reason: "injected finalize failure".to_string(),
        }),
        ..Hooks::default()
    };
    match grow_with_hooks(&config, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a finalize failure must propagate".to_string(),
        }),
        Err(ClearGbmError::TreeConstructionFailed { reason }) => {
            assert_eq!(reason, "injected finalize failure");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_leaf_wise_handles_an_unsplittable_root() -> Result<(), ClearGbmError> {
    // Every sample in one bin: no split has positive gain, so the root never
    // reaches the frontier and growth ends before consuming any budget.
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins: Vec<u8> = vec![0_u8; 16_usize];
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, ROOT_SAMPLES, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: ROOT_SAMPLES,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &config,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    let (tree, _assignment) =
        match build_tree_leaf_wise_with_leaf_assignment(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
    assert_eq!(tree.n_leaves(), 1_usize);
    assert_eq!(tree.n_nodes(), 1_usize);
    assert_eq!(tree.max_depth(), 0_usize);
    Ok(())
}

#[test]
fn test_build_tree_leaf_wise_discards_the_assignment() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 8_usize, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 8_usize,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &config,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    let tree = match build_tree_leaf_wise(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    assert_eq!(tree.n_leaves(), 3_usize);
    Ok(())
}
