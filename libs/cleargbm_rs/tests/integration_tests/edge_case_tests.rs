//! Edge case integration tests.
//!
//! Tests behavior under degenerate conditions: uniform gradients (no useful
//! split), accessor coverage, and sibling histogram subtraction correctness.

use cleargbm_rs::{
    build_tree, subtract_histogram, BuildTreeInput, ClearGbmError, HistogramBuffer, Hooks,
    SplitConfig, TreeBuildConfig,
};

use super::EPSILON;

/// Test handling of all-same gradients (no useful split)
#[test]
fn test_no_useful_split() -> std::result::Result<(), ClearGbmError> {
    let sample_indices: Vec<usize> = (0_usize..4_usize).collect();
    // All same gradients - no information gain from splitting
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];

    // 4 samples, 1 feature, column-major flat.
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 3_u8];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let split_config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let tree_config = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // With all same gradients, there's no gain from splitting
    // Tree should be a single leaf
    assert_eq!(
        tree.n_leaves(),
        1_usize,
        "Tree should be single leaf when all gradients are same"
    );

    let root = match tree.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    assert!(root.is_leaf(), "Root should be a leaf");

    // Leaf value should be -sum(G)/sum(H) = -4/4 = -1
    let expected_value = -4.0_f64 / 4.0_f64;
    assert!(
        (root.value() - expected_value).abs() < EPSILON,
        "Leaf value should be {expected_value}, got {}",
        root.value()
    );

    Ok(())
}

/// Test all TreeNode accessor methods for coverage
#[test]
fn test_tree_node_accessors() -> std::result::Result<(), ClearGbmError> {
    // Build a simple tree to get TreeNode instances
    let sample_indices: Vec<usize> = (0_usize..6_usize).collect();
    let gradients = vec![-1.0_f64, -1.0_f64, -1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];

    // 6 samples, 1 feature, column-major flat.
    let bins: Vec<u8> = vec![0_u8, 0_u8, 0_u8, 1_u8, 1_u8, 1_u8];
    let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

    let split_config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let tree_config = match TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_samples: 6_usize,
        n_features: 1_usize,
        n_regular_bins: 2_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Exercise all TreeNode accessors on root
    let root = match tree.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };

    // All accessor methods
    let _ = root.node_id();
    let _ = root.is_leaf();
    let _ = root.feature_index();
    let _ = root.threshold();
    let _ = root.value();
    let _ = root.n_samples();
    let _ = root.left_child();
    let _ = root.right_child();
    let _ = root.nan_goes_left();

    // If root has children, exercise accessors on them too
    if let Some(left_id) = root.left_child() {
        let left = match tree.node(left_id) {
            Ok(n) => n,
            Err(e) => return Err(e),
        };
        let _ = left.node_id();
        let _ = left.is_leaf();
        let _ = left.feature_index();
        let _ = left.threshold();
        let _ = left.value();
        let _ = left.n_samples();
        let _ = left.left_child();
        let _ = left.right_child();
        let _ = left.nan_goes_left();
    }

    Ok(())
}

/// Test that sibling histogram subtraction is mathematically correct
#[test]
fn test_sibling_subtraction_correctness() -> std::result::Result<(), ClearGbmError> {
    // Parent histogram
    let mut parent = HistogramBuffer::new(3_usize);
    match parent.accumulate(0_usize, 1.0_f64, 2.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match parent.accumulate(1_usize, 3.0_f64, 4.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match parent.accumulate(2_usize, 5.0_f64, 6.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    // Child histogram (subset of parent)
    let mut child = HistogramBuffer::new(3_usize);
    match child.accumulate(0_usize, 0.3_f64, 0.5_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match child.accumulate(1_usize, 1.0_f64, 1.5_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match child.accumulate(2_usize, 2.0_f64, 2.5_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    // Sibling = Parent - Child
    let sibling = match subtract_histogram(&parent, &child) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    // Verify each bin
    let g0 = match sibling.gradient_sum(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        (g0 - 0.7_f64).abs() < EPSILON,
        "Bin 0 gradient: expected 0.7, got {g0}"
    );

    let g1 = match sibling.gradient_sum(1_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        (g1 - 2.0_f64).abs() < EPSILON,
        "Bin 1 gradient: expected 2.0, got {g1}"
    );

    let g2 = match sibling.gradient_sum(2_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        (g2 - 3.0_f64).abs() < EPSILON,
        "Bin 2 gradient: expected 3.0, got {g2}"
    );

    // Verify hessians too
    let h0 = match sibling.hessian_sum(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        (h0 - 1.5_f64).abs() < EPSILON,
        "Bin 0 hessian: expected 1.5, got {h0}"
    );

    Ok(())
}
