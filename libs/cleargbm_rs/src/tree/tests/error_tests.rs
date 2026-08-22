//! Error path tests for tree building internals.

use crate::error::ClearGbmError;
use crate::histogram::HistogramRequest;
use crate::hooks::Hooks;
use crate::split::MonotonicConstraint;
use crate::tree::histograms::{
    build_feature_histograms, compute_child_histograms, find_best_split_across_features_internal,
    BuildHistogramConfig, ChildHistogramConfig, OrderedScratch,
};
use crate::tree::nodes::{finalize_nodes, BuildNode};
use crate::tree::{build_tree, BuildTreeInput, TreeBuildConfig};
use crate::types::{HistogramBuffer, SplitConfig};

// =========================================================================
// finalize_nodes error path tests
// =========================================================================

#[test]
fn test_finalize_nodes_internal_node_missing_feature_index() -> Result<(), ClearGbmError> {
    // Create an internal node (is_leaf=false) without feature_index
    let build_nodes = vec![BuildNode {
        node_id: 0_usize,
        is_leaf: false, // internal node
        value: 0.0_f64,
        n_samples: 10_usize,
        feature_index: None, // missing!
        split_bin: Some(1_usize),
        nan_goes_left: true,
    }];
    let child_pointers = vec![(Some(1_usize), Some(2_usize))];
    let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

    let result = finalize_nodes(
        &build_nodes,
        &child_pointers,
        &bin_thresholds,
        &Hooks::default(),
    );
    assert!(matches!(
        result,
        Err(ClearGbmError::TreeConstructionFailed { .. })
    ));
    Ok(())
}

#[test]
fn test_finalize_nodes_internal_node_missing_split_bin() -> Result<(), ClearGbmError> {
    // Create an internal node without split_bin
    let build_nodes = vec![BuildNode {
        node_id: 0_usize,
        is_leaf: false,
        value: 0.0_f64,
        n_samples: 10_usize,
        feature_index: Some(0_usize),
        split_bin: None, // missing!
        nan_goes_left: true,
    }];
    let child_pointers = vec![(Some(1_usize), Some(2_usize))];
    let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

    let result = finalize_nodes(
        &build_nodes,
        &child_pointers,
        &bin_thresholds,
        &Hooks::default(),
    );
    assert!(matches!(
        result,
        Err(ClearGbmError::TreeConstructionFailed { .. })
    ));
    Ok(())
}

#[test]
fn test_finalize_nodes_internal_node_missing_left_child() -> Result<(), ClearGbmError> {
    // Create an internal node with missing left child in child_pointers
    let build_nodes = vec![BuildNode {
        node_id: 0_usize,
        is_leaf: false,
        value: 0.0_f64,
        n_samples: 10_usize,
        feature_index: Some(0_usize),
        split_bin: Some(0_usize),
        nan_goes_left: true,
    }];
    let child_pointers = vec![(None, Some(2_usize))]; // left is None!
    let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

    let result = finalize_nodes(
        &build_nodes,
        &child_pointers,
        &bin_thresholds,
        &Hooks::default(),
    );
    assert!(matches!(
        result,
        Err(ClearGbmError::TreeConstructionFailed { .. })
    ));
    Ok(())
}

#[test]
fn test_finalize_nodes_internal_node_missing_right_child() -> Result<(), ClearGbmError> {
    // Create an internal node with missing right child in child_pointers
    let build_nodes = vec![BuildNode {
        node_id: 0_usize,
        is_leaf: false,
        value: 0.0_f64,
        n_samples: 10_usize,
        feature_index: Some(0_usize),
        split_bin: Some(0_usize),
        nan_goes_left: true,
    }];
    let child_pointers = vec![(Some(1_usize), None)]; // right is None!
    let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

    let result = finalize_nodes(
        &build_nodes,
        &child_pointers,
        &bin_thresholds,
        &Hooks::default(),
    );
    assert!(matches!(
        result,
        Err(ClearGbmError::TreeConstructionFailed { .. })
    ));
    Ok(())
}

#[test]
fn test_finalize_nodes_leaf_node_success() -> Result<(), ClearGbmError> {
    // Leaf nodes should finalize without needing feature_index, split_bin, or children
    let build_nodes = vec![BuildNode {
        node_id: 0_usize,
        is_leaf: true,
        value: 1.5_f64,
        n_samples: 10_usize,
        feature_index: None,
        split_bin: None,
        nan_goes_left: false,
    }];
    let child_pointers = vec![(None, None)];
    let bin_thresholds: Vec<Vec<f64>> = vec![];

    let nodes = match finalize_nodes(
        &build_nodes,
        &child_pointers,
        &bin_thresholds,
        &Hooks::default(),
    ) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };
    assert_eq!(nodes.len(), 1_usize);
    assert!(nodes[0_usize].is_leaf());
    assert!((nodes[0_usize].value() - 1.5_f64).abs() < 1e-10_f64);
    Ok(())
}

// =========================================================================
// Internal function error path tests
// =========================================================================

#[test]
fn test_build_feature_histograms_empty_features() -> Result<(), ClearGbmError> {
    // Test with n_features = 0
    let sample_indices = vec![0_u32, 1_u32];
    let gradients = vec![1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64];
    // n_features = 0 → empty flat bin slice
    let bins: Vec<u8> = vec![];

    let hooks = Hooks::default();
    let config = BuildHistogramConfig {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 2_usize,
        n_features: 0_usize,
        n_bins: 3_usize,
        hooks: &hooks,
    };
    let result = build_feature_histograms(
        &config,
        &mut OrderedScratch::new(config.sample_indices.len()),
    );

    // Should return Ok with empty vec (no error, but no histograms)
    assert!(matches!(result, Ok(ref h) if h.is_empty()));
    Ok(())
}

#[test]
fn test_find_best_split_across_features_internal_error() -> Result<(), ClearGbmError> {
    // Create a histogram and config that will trigger an error
    // n_regular_bins > n_bins should cause an error
    let histogram = HistogramBuffer::new(3_usize);
    let histograms = vec![histogram];
    let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let result = find_best_split_across_features_internal(
        &histograms,
        &config,
        10_usize, // n_regular_bins > n_bins (3)
        None,
    );

    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_find_best_split_across_features_internal_multiple_features() -> Result<(), ClearGbmError> {
    // Test with 2 features that both have valid splits to cover the comparison closure
    let mut hist0 = HistogramBuffer::new(4_usize);
    for _ in 0_usize..10_usize {
        match hist0.accumulate(0_usize, 0.5_f64, 1.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }
    for _ in 0_usize..10_usize {
        match hist0.accumulate(1_usize, -0.5_f64, 1.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }

    let mut hist1 = HistogramBuffer::new(4_usize);
    for _ in 0_usize..10_usize {
        match hist1.accumulate(0_usize, 0.3_f64, 1.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }
    for _ in 0_usize..10_usize {
        match hist1.accumulate(1_usize, -0.3_f64, 1.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
    }

    let histograms = vec![hist0, hist1];
    let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let result = match find_best_split_across_features_internal(&histograms, &config, 3_usize, None)
    {
        Ok(r) => r,
        Err(e) => return Err(e),
    };

    // Should find the best split (feature 0 has higher gain due to larger gradient magnitude)
    assert!(matches!(result, Some(ref s) if s.feature_index() == 0_usize));
    Ok(())
}

#[test]
fn test_compute_child_histograms_parent_histograms_too_short() -> Result<(), ClearGbmError> {
    // Test error when parent_histograms has fewer entries than n_features
    let left_indices = vec![0_u32, 1_u32];
    let right_indices = vec![2_u32, 3_u32];
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    // 4 samples, 2 features column-major:
    // feat 0: [0, 1, 0, 1]; feat 1: [0, 1, 0, 1]
    let bins: Vec<u8> = vec![0_u8, 1_u8, 0_u8, 1_u8, 0_u8, 1_u8, 0_u8, 1_u8];

    // Only 1 parent histogram, but n_features = 2
    let parent_histograms = vec![HistogramBuffer::new(3_usize)];

    let hooks = Hooks::default();
    let config = ChildHistogramConfig {
        left_indices: &left_indices,
        right_indices: &right_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 4_usize,
        n_features: 2_usize, // 2 features, but only 1 parent histogram
        n_bins: 3_usize,
        parent_histograms: &parent_histograms,
        hooks: &hooks,
    };

    let result = compute_child_histograms(
        &config,
        &mut OrderedScratch::new(config.left_indices.len() + config.right_indices.len()),
    );
    assert!(matches!(
        result,
        Err(ClearGbmError::FeatureIndexOutOfBounds { .. })
    ));
    Ok(())
}

#[test]
fn test_compute_child_histograms_success() -> Result<(), ClearGbmError> {
    // Test successful child histogram computation
    let left_indices = vec![0_u32, 1_u32];
    let right_indices = vec![2_u32, 3_u32];
    let gradients = vec![1.0_f64, 2.0_f64, -1.0_f64, -2.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    // 4 samples, 1 feature. Column-major flat: [0, 0, 1, 1]
    let bins: Vec<u8> = vec![0_u8, 0_u8, 1_u8, 1_u8];

    // Create parent histogram with proper values
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

    let hooks = Hooks::default();
    let config = ChildHistogramConfig {
        left_indices: &left_indices,
        right_indices: &right_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_bins: 3_usize,
        parent_histograms: &parent_histograms,
        hooks: &hooks,
    };

    let (left_hists, right_hists) = match compute_child_histograms(
        &config,
        &mut OrderedScratch::new(config.left_indices.len() + config.right_indices.len()),
    ) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };
    assert_eq!(left_hists.len(), 1_usize);
    assert_eq!(right_hists.len(), 1_usize);
    Ok(())
}

#[test]
fn test_build_tree_with_large_n_regular_bins() -> Result<(), ClearGbmError> {
    // Test with n_regular_bins much larger than actual bins used
    // This succeeds because histogram is built with n_bins = n_regular_bins + 1
    let sc = match SplitConfig::new(2_usize, 1_usize, 4_usize, 0.0_f64, 0.0_f64) {
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

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 100_usize, // Large but histogram will have n_bins = 101
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
    };

    // This succeeds because histogram.n_bins() = n_regular_bins + 1 = 101 > 100
    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    assert!(tree.n_nodes() >= 1_usize);
    Ok(())
}

// (removed: `test_build_tree_bins_out_of_bounds_via_validated_hook` — this
// test targeted the deleted `build_histogram` validated shim. Under the new
// architecture there is no per-histogram validation surface; bin bounds are
// established by construction upstream in `FeatureBins`, which guarantees
// `bin <= n_regular_bins <= 255` and is validated at the pyo3 boundary by
// `train_gradient_boosting_rs` — not at every internal histogram build.)

#[test]
fn test_is_increasing_is_decreasing_methods() -> Result<(), ClearGbmError> {
    // Test the is_increasing and is_decreasing methods
    let inc = MonotonicConstraint::Increasing;
    let dec = MonotonicConstraint::Decreasing;
    let none = MonotonicConstraint::None;

    assert!(inc.is_increasing());
    assert!(!inc.is_decreasing());
    assert!(!inc.is_none());

    assert!(!dec.is_increasing());
    assert!(dec.is_decreasing());
    assert!(!dec.is_none());

    assert!(!none.is_increasing());
    assert!(!none.is_decreasing());
    assert!(none.is_none());

    Ok(())
}

// =========================================================================
// Hook-based error injection tests
// =========================================================================

/// Histogram builder that always returns an error (for testing error propagation)
fn error_histogram(_: HistogramRequest<'_>) -> Result<HistogramBuffer, ClearGbmError> {
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

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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
    let config = ChildHistogramConfig {
        left_indices: &left_indices,
        right_indices: &right_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 4_usize,
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
fn undersized_histogram(_: HistogramRequest<'_>) -> Result<HistogramBuffer, ClearGbmError> {
    // Return a histogram with only 2 bins, regardless of requested size
    // This will cause find_best_split_from_histogram to fail when n_regular_bins > 2
    Ok(HistogramBuffer::new(2_usize))
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

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize, // 4 regular bins, but hook returns 2-bin histogram
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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
