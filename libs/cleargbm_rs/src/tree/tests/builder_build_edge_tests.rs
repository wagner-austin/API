//! Tests for `build_tree` on malformed inputs and hook-injected
//! failures: shape mismatches, degenerate feature sets, and error
//! propagation from histogram building and split finding.

use crate::error::ClearGbmError;
use crate::histogram::NodeHistogramRequest;
use crate::hooks::Hooks;
use crate::split::MonotonicConstraint;
use crate::tree::{build_tree, BuildTreeInput, TreeBuildConfig};
use crate::types::SplitConfig;

#[test]
fn test_build_tree_no_features() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    // n_features = 0 -> flat slice empty
    let bins: Vec<u8> = vec![];
    let bin_thresholds: Vec<Vec<f64>> = vec![];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 3_usize, 0_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 3_usize,
        n_features: 0_usize,
        n_regular_bins: 3_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
        quantized: None,
    };

    let result = build_tree(&input, &Hooks::default());
    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(ClearGbmError::EmptyInput { .. })
    ));
    Ok(())
}

#[test]
fn test_build_tree_bins_shape_mismatch() -> Result<(), ClearGbmError> {
    // n_features = 1, n_samples = 3, but bins slice has wrong length.
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    // Declared n_samples=3 * n_features=1 = 3, but supplied only 2 bytes.
    // Passed directly (not through a transpose) because the wrong length IS
    // the subject: build_tree's shape check must reject it.
    let bins_rows: Vec<u8> = vec![0_u8, 0_u8];
    let bin_thresholds: Vec<Vec<f64>> = vec![vec![0.5_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 3_usize,
        n_features: 1_usize,
        n_regular_bins: 3_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
        quantized: None,
    };

    let result = build_tree(&input, &Hooks::default());
    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(ClearGbmError::ShapeMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_build_tree_with_monotonic_constraints() -> Result<(), ClearGbmError> {
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
    let constraints = vec![MonotonicConstraint::Increasing];

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
        monotonic_constraints: Some(&constraints),
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
        quantized: None,
    };

    // Should succeed (constraint may or may not affect the split)
    let _ = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    Ok(())
}

#[test]
fn test_build_tree_with_l1_regularization() -> Result<(), ClearGbmError> {
    // Use L1 regularization
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.5_f64, 0.0_f64, sc) {
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
        quantized: None,
    };

    let _ = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    Ok(())
}

#[test]
fn test_build_tree_left_larger_than_right() -> Result<(), ClearGbmError> {
    // Test where left child has more samples than right
    // This exercises the else branch in compute_child_histograms
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    // 6 samples: 4 in low bins (left), 2 in high bins (right).
    // Column-major flat: bins for feature 0 = [0, 0, 1, 1, 2, 3].
    let sample_indices = vec![0_u32, 1_u32, 2_u32, 3_u32, 4_u32, 5_u32];
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, -2.0_f64, -2.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins: Vec<u8> = vec![0_u8, 0_u8, 1_u8, 1_u8, 2_u8, 3_u8];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 6_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 6_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
        quantized: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Should have split into left (4 samples) and right (2 samples)
    assert!(tree.n_nodes() >= 3_usize);
    Ok(())
}

#[test]
fn test_build_tree_deep_tree() -> Result<(), ClearGbmError> {
    // Test building a deeper tree to exercise more code paths
    // Allow deep tree
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(10_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    // 8 samples with varying gradients
    let sample_indices = vec![0_u32, 1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32];
    let gradients = vec![
        4.0_f64, 3.0_f64, 2.0_f64, 1.0_f64, -1.0_f64, -2.0_f64, -3.0_f64, -4.0_f64,
    ];
    let hessians = vec![
        1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64,
    ];
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8];
    let bin_thresholds = vec![vec![
        0.125_f64, 0.25_f64, 0.375_f64, 0.5_f64, 0.625_f64, 0.75_f64, 0.875_f64, 1.0_f64,
    ]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 8_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 8_usize,
        n_features: 1_usize,
        n_regular_bins: 8_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
        quantized: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Should have multiple nodes
    assert!(tree.n_nodes() > 1_usize);
    // Should have at least 2 leaves
    assert!(tree.n_leaves() >= 2_usize);
    // Test nodes accessor
    let nodes = tree.nodes();
    assert_eq!(nodes.len(), tree.n_nodes());
    Ok(())
}

// =============================================================================
// Error path tests using hooks
// =============================================================================

/// Track call count for histogram building
use std::cell::Cell;
thread_local! {
    static HISTOGRAM_CALL_COUNT: Cell<usize> = const { Cell::new(0) };
}

fn counting_error_histogram(
    request: NodeHistogramRequest<'_>,
) -> Result<Vec<crate::types::HistogramBuffer>, ClearGbmError> {
    HISTOGRAM_CALL_COUNT.with(|count| {
        let current = count.get();
        count.set(current + 1_usize);
        // First call (root histogram) succeeds with real data, subsequent calls fail
        if current < 1_usize {
            Ok(crate::histogram::build_node_histograms_single_pass(request))
        } else {
            Err(ClearGbmError::EmptyInput {
                context: "injected histogram build error".to_string(),
            })
        }
    })
}

#[test]
fn test_build_tree_child_histogram_error() -> Result<(), ClearGbmError> {
    // Reset call count
    HISTOGRAM_CALL_COUNT.with(|count| count.set(0_usize));

    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    // Create data that will produce a split
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
        quantized: None,
    };

    // Inject hook that fails on child histogram building
    let hooks = Hooks::with_histogram_builder(counting_error_histogram);
    let result = build_tree(&input, &hooks);

    // Should fail when trying to build child histograms
    assert!(result.is_err());
    Ok(())
}

thread_local! {
    static WRONG_SIZE_CALL_COUNT: Cell<usize> = const { Cell::new(0) };
}

fn wrong_size_histogram(
    request: NodeHistogramRequest<'_>,
) -> Result<Vec<crate::types::HistogramBuffer>, ClearGbmError> {
    WRONG_SIZE_CALL_COUNT.with(|count| {
        let current = count.get();
        count.set(current + 1_usize);
        // First call (root histogram) succeeds with correct size
        // Second call (child histogram) returns wrong size to trigger subtract_histogram error
        if current < 1_usize {
            Ok(crate::histogram::build_node_histograms_single_pass(request))
        } else {
            // Return histogram with wrong size to break sibling subtraction
            Ok(vec![crate::types::HistogramBuffer::new(
                request.n_bins + 10_usize,
            )])
        }
    })
}

#[test]
fn test_build_tree_subtract_histogram_error() -> Result<(), ClearGbmError> {
    // Reset call count
    WRONG_SIZE_CALL_COUNT.with(|count| count.set(0_usize));

    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    // Create data that will produce a split
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
        quantized: None,
    };

    // Inject hook that returns wrong-sized histogram
    let hooks = Hooks::with_histogram_builder(wrong_size_histogram);
    let result = build_tree(&input, &hooks);

    // Should fail when trying to subtract histograms with different sizes
    assert!(result.is_err());
    Ok(())
}
