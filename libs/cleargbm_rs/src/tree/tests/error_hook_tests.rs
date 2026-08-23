//! Hook-based error injection tests for `build_tree`: histogram
//! building, child-histogram subtraction, split finding and node
//! finalization each fail on cue and the error must surface intact.

use crate::error::ClearGbmError;
use crate::histogram::NodeHistogramRequest;
use crate::hooks::Hooks;
use crate::tree::histograms::{compute_child_histograms, ChildHistogramConfig, OrderedScratch};
use crate::tree::{build_tree, BuildTreeInput, TreeBuildConfig};
use crate::types::{HistogramBuffer, SplitConfig};

// =========================================================================
// Hook-based error injection tests
// =========================================================================

/// Histogram builder that always returns an error (for testing error propagation)
fn error_histogram(_: NodeHistogramRequest<'_>) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    Err(ClearGbmError::EmptyInput {
        context: "injected error from hook".to_string(),
    })
}

#[test]
fn test_build_tree_hooks_error_in_histogram_building() -> Result<(), ClearGbmError> {
    // Use hooks to inject error during histogram building
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32, 3_u32];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 3_u8];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 4_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
    };

    // Inject error via hook
    let error_hooks = Hooks::with_histogram_builder(error_histogram);
    let result = build_tree(&input, &error_hooks);

    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(ClearGbmError::EmptyInput { context }) if context.contains("injected")
    ));
    Ok(())
}

#[test]
fn test_compute_child_histograms_hooks_error() -> Result<(), ClearGbmError> {
    // Test error propagation from hooks in compute_child_histograms
    let left_indices = vec![0_u32, 1_u32];
    let right_indices = vec![2_u32, 3_u32];
    let gradients = vec![1.0_f64, 2.0_f64, -1.0_f64, -2.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins: Vec<u8> = vec![0_u8, 0_u8, 1_u8, 1_u8];

    let mut parent_hist = HistogramBuffer::new(3_usize);
    match parent_hist.accumulate(0_usize, 3.0_f64, 2.0_f64) {
        Ok(_) => {}
        Err(e) => return Err(e),
    }
    match parent_hist.accumulate(1_usize, -3.0_f64, 2.0_f64) {
        Ok(_) => {}
        Err(e) => return Err(e),
    }
    let parent_histograms = vec![parent_hist];

    // Inject error via hook
    let error_hooks = Hooks::with_histogram_builder(error_histogram);
    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 4_usize, 1_usize);
    let config = ChildHistogramConfig {
        left_indices: &left_indices,
        right_indices: &right_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_features: 1_usize,
        n_bins: 3_usize,
        parent_histograms: &parent_histograms,
        hooks: &error_hooks,
    };

    let result = compute_child_histograms(
        &config,
        &mut OrderedScratch::new(config.left_indices.len() + config.right_indices.len()),
    );
    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(ClearGbmError::EmptyInput { context }) if context.contains("injected")
    ));
    Ok(())
}

/// Histogram builder that returns undersized histogram (for testing error propagation)
fn undersized_histogram(
    _: NodeHistogramRequest<'_>,
) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    // Return a histogram with only 2 bins, regardless of requested size
    // This will cause find_best_split_from_histogram to fail when n_regular_bins > 2
    Ok(vec![HistogramBuffer::new(2_usize)])
}

#[test]
fn test_build_tree_hooks_error_in_split_finding() -> Result<(), ClearGbmError> {
    // Use hooks to inject undersized histogram, causing split finding to fail
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32, 3_u32];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 3_u8];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 4_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize, // 4 regular bins, but hook returns 2-bin histogram
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
    };

    // Inject undersized histogram via hook - this causes split finding error
    let undersized_hooks = Hooks::with_histogram_builder(undersized_histogram);
    let result = build_tree(&input, &undersized_hooks);

    // Should fail because n_regular_bins (4) > histogram.n_bins() (2)
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_build_tree_finalize_nodes_error_via_hook() -> Result<(), ClearGbmError> {
    // Test error propagation from finalize_nodes via hook injection.
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32, 3_u32];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 3_u8];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 4_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
    };

    // Inject error via finalize_nodes hook
    let error_hooks = Hooks::with_finalize_nodes_error(ClearGbmError::TreeConstructionFailed {
        reason: "injected finalize_nodes error".to_string(),
    });
    let result = build_tree(&input, &error_hooks);

    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(ClearGbmError::TreeConstructionFailed { reason }) if reason.contains("injected")
    ));
    Ok(())
}
