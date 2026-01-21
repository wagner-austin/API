//! Tests for tree building functions.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::split::MonotonicConstraint;
use crate::tree::builder::{compute_leaf_value, compute_sums, should_stop, split_samples, EPSILON};
use crate::tree::{build_tree, BuildTreeInput, TreeBuildConfig};
use crate::types::SplitConfig;

// =========================================================================
// compute_leaf_value tests
// =========================================================================

#[test]
fn test_compute_leaf_value_basic() -> Result<(), ClearGbmError> {
    // Simple case: -G/H = -2.0/10.0 = -0.2
    let value = compute_leaf_value(2.0_f64, 10.0_f64, 0.0_f64, 0.0_f64);
    assert!((value - (-0.2_f64)).abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_leaf_value_with_l2() -> Result<(), ClearGbmError> {
    // With L2: -G/(H + lambda) = -2.0/(10.0 + 1.0) = -2.0/11.0
    let value = compute_leaf_value(2.0_f64, 10.0_f64, 0.0_f64, 1.0_f64);
    let expected = -2.0_f64 / 11.0_f64;
    assert!((value - expected).abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_leaf_value_with_l1() -> Result<(), ClearGbmError> {
    // With L1: soft threshold
    // G = 2.0, alpha = 0.5
    // sign(G) = 1, |G| = 2.0 > alpha
    // value = -1 * (2.0 - 0.5) / (10.0 + 0.0) = -1.5 / 10.0 = -0.15
    let value = compute_leaf_value(2.0_f64, 10.0_f64, 0.5_f64, 0.0_f64);
    let expected = -1.5_f64 / 10.0_f64;
    assert!((value - expected).abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_leaf_value_l1_below_threshold() -> Result<(), ClearGbmError> {
    // With L1: |G| <= alpha, value = 0
    let value = compute_leaf_value(0.3_f64, 10.0_f64, 0.5_f64, 0.0_f64);
    assert!(value.abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_leaf_value_zero_hessian() -> Result<(), ClearGbmError> {
    // Zero hessian should return 0
    let value = compute_leaf_value(2.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    assert!(value.abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_leaf_value_negative_gradient() -> Result<(), ClearGbmError> {
    // Negative gradient: -(-2.0)/10.0 = 0.2
    let value = compute_leaf_value(-2.0_f64, 10.0_f64, 0.0_f64, 0.0_f64);
    assert!((value - 0.2_f64).abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_leaf_value_negative_gradient_with_l1() -> Result<(), ClearGbmError> {
    // Negative gradient with L1: soft threshold
    // G = -2.0, alpha = 0.5
    // sign(G) = -1, |G| = 2.0 > alpha
    // value = -(-1) * (2.0 - 0.5) / (10.0 + 0.0) = 1.5 / 10.0 = 0.15
    let value = compute_leaf_value(-2.0_f64, 10.0_f64, 0.5_f64, 0.0_f64);
    let expected = 1.5_f64 / 10.0_f64;
    assert!((value - expected).abs() < EPSILON);
    Ok(())
}

// =========================================================================
// should_stop tests
// =========================================================================

#[test]
fn test_should_stop_max_depth() -> Result<(), ClearGbmError> {
    assert!(should_stop(
        5_usize, 100_usize, 0_usize, 5_usize, 0_usize, 2_usize, 1_usize
    ));
    assert!(!should_stop(
        4_usize, 100_usize, 0_usize, 5_usize, 0_usize, 2_usize, 1_usize
    ));
    Ok(())
}

#[test]
fn test_should_stop_unlimited_depth() -> Result<(), ClearGbmError> {
    // max_depth = 0 means unlimited
    assert!(!should_stop(
        100_usize, 100_usize, 0_usize, 0_usize, 0_usize, 2_usize, 1_usize
    ));
    Ok(())
}

#[test]
fn test_should_stop_max_leaves() -> Result<(), ClearGbmError> {
    // max_leaves = 10, n_leaves = 9, would add 1 more -> stop
    assert!(should_stop(
        2_usize, 100_usize, 9_usize, 0_usize, 10_usize, 2_usize, 1_usize
    ));
    assert!(!should_stop(
        2_usize, 100_usize, 8_usize, 0_usize, 10_usize, 2_usize, 1_usize
    ));
    Ok(())
}

#[test]
fn test_should_stop_min_samples_split() -> Result<(), ClearGbmError> {
    assert!(should_stop(
        2_usize, 5_usize, 0_usize, 0_usize, 0_usize, 10_usize, 1_usize
    ));
    assert!(!should_stop(
        2_usize, 15_usize, 0_usize, 0_usize, 0_usize, 10_usize, 1_usize
    ));
    Ok(())
}

#[test]
fn test_should_stop_min_samples_leaf() -> Result<(), ClearGbmError> {
    // n_samples = 5, min_samples_leaf = 3, need 6 samples minimum
    assert!(should_stop(
        2_usize, 5_usize, 0_usize, 0_usize, 0_usize, 2_usize, 3_usize
    ));
    assert!(!should_stop(
        2_usize, 10_usize, 0_usize, 0_usize, 0_usize, 2_usize, 3_usize
    ));
    Ok(())
}

// =========================================================================
// split_samples tests
// =========================================================================

#[test]
fn test_split_samples_basic() -> Result<(), ClearGbmError> {
    // 5 samples, bins for feature 0
    let bins = vec![
        vec![0_usize], // sample 0, feature 0 -> bin 0
        vec![1_usize], // sample 1, feature 0 -> bin 1
        vec![2_usize], // sample 2, feature 0 -> bin 2
        vec![0_usize], // sample 3, feature 0 -> bin 0
        vec![1_usize], // sample 4, feature 0 -> bin 1
    ];
    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize, 4_usize];

    // Split at bin 0 (samples in bin <= 0 go left)
    let (left, right) = split_samples(&sample_indices, &bins, 0_usize, 0_usize, true, 3_usize);

    // Left: bins 0 (samples 0, 3)
    assert_eq!(left.len(), 2_usize);
    assert!(left.contains(&0_usize));
    assert!(left.contains(&3_usize));

    // Right: bins 1, 2 (samples 1, 2, 4)
    assert_eq!(right.len(), 3_usize);
    assert!(right.contains(&1_usize));
    assert!(right.contains(&2_usize));
    assert!(right.contains(&4_usize));
    Ok(())
}

#[test]
fn test_split_samples_nan_handling() -> Result<(), ClearGbmError> {
    // Sample with NaN bin (= n_regular_bins)
    let bins = vec![
        vec![0_usize], // sample 0 -> bin 0
        vec![3_usize], // sample 1 -> NaN bin (n_regular_bins = 3)
    ];
    let sample_indices = vec![0_usize, 1_usize];

    // NaN goes left
    let (left, right) = split_samples(&sample_indices, &bins, 0_usize, 0_usize, true, 3_usize);
    assert!(left.contains(&0_usize)); // bin 0
    assert!(left.contains(&1_usize)); // NaN goes left
    assert!(right.is_empty());

    // NaN goes right
    let (left2, right2) = split_samples(&sample_indices, &bins, 0_usize, 0_usize, false, 3_usize);
    assert!(left2.contains(&0_usize)); // bin 0
    assert!(right2.contains(&1_usize)); // NaN goes right
    Ok(())
}

// =========================================================================
// build_tree tests
// =========================================================================

#[test]
fn test_build_tree_single_leaf() -> Result<(), ClearGbmError> {
    // Create simple data that results in a single leaf
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(1_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_usize, 1_usize, 2_usize];
    let gradients = vec![0.1_f64, 0.1_f64, 0.1_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![vec![0_usize], vec![0_usize], vec![0_usize]];
    let bin_thresholds = vec![vec![0.5_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 1_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Should be a single leaf (max_depth = 1, samples in same bin)
    assert_eq!(tree.n_leaves(), 1_usize);
    let _ = match tree.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    Ok(())
}

#[test]
fn test_build_tree_with_split() -> Result<(), ClearGbmError> {
    // Create data with clear split
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
    // Left samples (bins 0,1) have positive gradients
    // Right samples (bins 2,3) have negative gradients
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Should have split
    assert!(tree.n_nodes() >= 3_usize);
    assert!(tree.n_leaves() >= 2_usize);

    // Root should not be a leaf
    let root = match tree.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    assert!(!root.is_leaf());
    Ok(())
}

#[test]
fn test_build_tree_empty_input() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices: Vec<usize> = vec![];
    let gradients: Vec<f64> = vec![];
    let hessians: Vec<f64> = vec![];
    let bins: Vec<Vec<usize>> = vec![];
    let bin_thresholds: Vec<Vec<f64>> = vec![];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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
fn test_build_tree_max_depth_constraint() -> Result<(), ClearGbmError> {
    // max_depth = 1 should create root + 2 leaves max
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(1_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Max depth = 1, so max 3 nodes (root + 2 leaves)
    assert!(tree.n_nodes() <= 3_usize);
    assert!(tree.max_depth() <= 1_usize);
    Ok(())
}

#[test]
fn test_build_tree_max_leaves_constraint() -> Result<(), ClearGbmError> {
    // max_leaves = 2
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(10_usize, 2_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    assert!(tree.n_leaves() <= 2_usize);
    Ok(())
}

#[test]
fn test_build_tree_gradients_too_short() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_usize, 1_usize, 2_usize];
    let gradients = vec![1.0_f64]; // Too short!
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize]];
    let bin_thresholds = vec![vec![0.5_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 3_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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
fn test_build_tree_hessians_too_short() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_usize, 1_usize, 2_usize];
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64]; // Too short!
    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize]];
    let bin_thresholds = vec![vec![0.5_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 3_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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
fn test_build_tree_no_features() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_usize, 1_usize, 2_usize];
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    // No features - empty inner vecs
    let bins: Vec<Vec<usize>> = vec![vec![], vec![], vec![]];
    let bin_thresholds: Vec<Vec<f64>> = vec![];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 3_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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
fn test_build_tree_empty_bins_vec() -> Result<(), ClearGbmError> {
    // Test where bins vec itself is empty (different from empty inner vecs)
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_usize, 1_usize, 2_usize];
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    // Completely empty bins vec
    let bins: Vec<Vec<usize>> = vec![];
    let bin_thresholds: Vec<Vec<f64>> = vec![];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 3_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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
fn test_build_tree_with_monotonic_constraints() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];
    let constraints = vec![MonotonicConstraint::Increasing];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: Some(&constraints),
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

    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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

    // 6 samples: 4 in low bins (left), 2 in high bins (right)
    // This makes left child larger than right child
    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize, 4_usize, 5_usize];
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, -2.0_f64, -2.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    // First 4 samples in bins 0-1 (left), last 2 in bins 2-3 (right)
    let bins = vec![
        vec![0_usize],
        vec![0_usize],
        vec![1_usize],
        vec![1_usize],
        vec![2_usize],
        vec![3_usize],
    ];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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
    let sample_indices = vec![
        0_usize, 1_usize, 2_usize, 3_usize, 4_usize, 5_usize, 6_usize, 7_usize,
    ];
    let gradients = vec![
        4.0_f64, 3.0_f64, 2.0_f64, 1.0_f64, -1.0_f64, -2.0_f64, -3.0_f64, -4.0_f64,
    ];
    let hessians = vec![
        1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64,
    ];
    let bins = vec![
        vec![0_usize],
        vec![1_usize],
        vec![2_usize],
        vec![3_usize],
        vec![4_usize],
        vec![5_usize],
        vec![6_usize],
        vec![7_usize],
    ];
    let bin_thresholds = vec![vec![
        0.125_f64, 0.25_f64, 0.375_f64, 0.5_f64, 0.625_f64, 0.75_f64, 0.875_f64, 1.0_f64,
    ]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 8_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
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
// compute_sums edge case tests
// =============================================================================

/// Test compute_sums with all indices in bounds.
#[test]
fn test_compute_sums_basic() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_usize, 1_usize, 2_usize];
    let gradients = vec![1.0_f64, 2.0_f64, 3.0_f64];
    let hessians = vec![0.5_f64, 0.5_f64, 0.5_f64];

    let (g_sum, h_sum) = compute_sums(&sample_indices, &gradients, &hessians);
    assert!((g_sum - 6.0_f64).abs() < EPSILON);
    assert!((h_sum - 1.5_f64).abs() < EPSILON);
    Ok(())
}

/// Test compute_sums with sample indices that exceed gradient array bounds.
/// This covers the `idx >= gradients.len()` branch at line 429.
#[test]
fn test_compute_sums_gradient_out_of_bounds() -> Result<(), ClearGbmError> {
    // Sample indices include index 5, but gradients only has indices 0-2
    let sample_indices = vec![0_usize, 1_usize, 5_usize];
    let gradients = vec![1.0_f64, 2.0_f64, 3.0_f64]; // length 3
    let hessians = vec![0.5_f64, 0.5_f64, 0.5_f64, 0.5_f64, 0.5_f64, 0.5_f64]; // length 6

    let (g_sum, h_sum) = compute_sums(&sample_indices, &gradients, &hessians);
    // g_sum = 1.0 + 2.0 + 0 (index 5 skipped) = 3.0
    // h_sum = 0.5 + 0.5 + 0.5 (index 5 is valid for hessians) = 1.5
    assert!((g_sum - 3.0_f64).abs() < EPSILON);
    assert!((h_sum - 1.5_f64).abs() < EPSILON);
    Ok(())
}

/// Test compute_sums with sample indices that exceed hessian array bounds.
/// This covers the `idx >= hessians.len()` branch at line 432.
#[test]
fn test_compute_sums_hessian_out_of_bounds() -> Result<(), ClearGbmError> {
    // Sample indices include index 5, but hessians only has indices 0-2
    let sample_indices = vec![0_usize, 1_usize, 5_usize];
    let gradients = vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64]; // length 6
    let hessians = vec![0.5_f64, 0.5_f64, 0.5_f64]; // length 3

    let (g_sum, h_sum) = compute_sums(&sample_indices, &gradients, &hessians);
    // g_sum = 1.0 + 2.0 + 6.0 (index 5 is valid for gradients) = 9.0
    // h_sum = 0.5 + 0.5 + 0 (index 5 skipped) = 1.0
    assert!((g_sum - 9.0_f64).abs() < EPSILON);
    assert!((h_sum - 1.0_f64).abs() < EPSILON);
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
    sample_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[usize],
    n_bins: usize,
) -> Result<crate::types::HistogramBuffer, ClearGbmError> {
    HISTOGRAM_CALL_COUNT.with(|count| {
        let current = count.get();
        count.set(current + 1_usize);
        // First call (root histogram) succeeds with real data, subsequent calls fail
        if current < 1_usize {
            crate::histogram::build_histogram(sample_indices, gradients, hessians, bins, n_bins)
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
    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
    };

    // Inject hook that fails on child histogram building (line 360)
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
    sample_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[usize],
    n_bins: usize,
) -> Result<crate::types::HistogramBuffer, ClearGbmError> {
    WRONG_SIZE_CALL_COUNT.with(|count| {
        let current = count.get();
        count.set(current + 1_usize);
        // First call (root histogram) succeeds with correct size
        // Second call (child histogram) returns wrong size to trigger subtract_histogram error
        if current < 1_usize {
            crate::histogram::build_histogram(sample_indices, gradients, hessians, bins, n_bins)
        } else {
            // Return histogram with wrong size (line 671)
            Ok(crate::types::HistogramBuffer::new(n_bins + 10_usize))
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
    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
    };

    // Inject hook that returns wrong-sized histogram (line 671)
    let hooks = Hooks::with_histogram_builder(wrong_size_histogram);
    let result = build_tree(&input, &hooks);

    // Should fail when trying to subtract histograms with different sizes
    assert!(result.is_err());
    Ok(())
}
