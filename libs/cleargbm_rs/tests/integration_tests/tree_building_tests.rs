//! End-to-end tree building integration tests.
//!
//! Verifies tree construction on binary classification data, squared error
//! minimization, monotonic constraint enforcement, and regularization effects.

use cleargbm_rs::{
    build_tree, BuildTreeInput, ClearGbmError, Hooks, MonotonicConstraint, SplitConfig,
    TreeBuildConfig,
};

use super::EPSILON;

/// Test building a tree on simple binary classification data
/// and verify the predictions are correct
#[test]
fn test_tree_binary_classification_correctness() -> std::result::Result<(), ClearGbmError> {
    // Simple dataset: 8 samples, 1 feature
    // Feature values: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    // Labels: [1, 1, 1, 1, 0, 0, 0, 0]
    //
    // For gradient boosting with log loss:
    // Initial prediction = 0.5 (balanced)
    // Gradient = pred - label
    // Hessian = pred * (1 - pred) = 0.25

    let sample_indices: Vec<usize> = (0_usize..8_usize).collect();

    // Gradients: pred(0.5) - label
    // First 4 samples (label=1): 0.5 - 1 = -0.5
    // Last 4 samples (label=0): 0.5 - 0 = 0.5
    let gradients = vec![
        -0.5_f64, -0.5_f64, -0.5_f64, -0.5_f64, 0.5_f64, 0.5_f64, 0.5_f64, 0.5_f64,
    ];
    let hessians = vec![
        0.25_f64, 0.25_f64, 0.25_f64, 0.25_f64, 0.25_f64, 0.25_f64, 0.25_f64, 0.25_f64,
    ];

    // Bins: 8 bins (one per sample for simplicity)
    let bins: Vec<Vec<usize>> = (0_usize..8_usize).map(|i| vec![i]).collect();
    let bin_thresholds = vec![vec![
        0.15_f64, 0.25_f64, 0.35_f64, 0.45_f64, 0.55_f64, 0.65_f64, 0.75_f64, 0.85_f64,
    ]];

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
        n_regular_bins: 8_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Tree should have found a split
    assert!(
        tree.n_nodes() >= 3_usize,
        "Tree should have at least 3 nodes"
    );
    assert!(
        tree.n_leaves() >= 2_usize,
        "Tree should have at least 2 leaves"
    );

    // Root should not be a leaf (we have a clear split point)
    let root = match tree.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    assert!(!root.is_leaf(), "Root should be an internal node");

    // The split should be around the middle (between samples 3 and 4)
    // where the label changes from 1 to 0
    let threshold = root.threshold().ok_or_else(|| ClearGbmError::EmptyInput {
        context: "missing threshold".to_string(),
    });
    let threshold = match threshold {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    assert!(
        threshold > 0.3_f64 && threshold < 0.6_f64,
        "Split threshold should be between 0.3 and 0.6, got {threshold}"
    );

    Ok(())
}

/// Test that tree predictions minimize the squared error
#[test]
fn test_tree_minimizes_squared_error() -> std::result::Result<(), ClearGbmError> {
    // For squared error loss:
    // Gradient = prediction - target = -residual (with initial pred=0)
    // Hessian = 1 (constant)
    //
    // Targets: [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    // With initial prediction = 0:
    // Gradients = [0-1, 0-1, 0-1, 0-(-1), 0-(-1), 0-(-1)] = [-1, -1, -1, 1, 1, 1]

    let sample_indices: Vec<usize> = (0_usize..6_usize).collect();
    let gradients = vec![-1.0_f64, -1.0_f64, -1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];

    // Two bins: first 3 samples in bin 0, last 3 in bin 1
    let bins = vec![
        vec![0_usize],
        vec![0_usize],
        vec![0_usize],
        vec![1_usize],
        vec![1_usize],
        vec![1_usize],
    ];
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
        n_regular_bins: 2_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Should split into two leaves
    assert_eq!(tree.n_leaves(), 2_usize, "Should have exactly 2 leaves");

    // Get leaf values by traversing the tree
    // Left leaf (bin 0): G_sum = -3, H_sum = 3, value = -(-3)/3 = 1.0
    // Right leaf (bin 1): G_sum = 3, H_sum = 3, value = -3/3 = -1.0

    // The root should point to two children
    let root = match tree.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    assert!(!root.is_leaf(), "Root should be internal node");

    let left_id = root.left_child().ok_or_else(|| ClearGbmError::EmptyInput {
        context: "missing left child".to_string(),
    });
    let left_id = match left_id {
        Ok(id) => id,
        Err(e) => return Err(e),
    };
    let right_id = root.right_child().ok_or_else(|| ClearGbmError::EmptyInput {
        context: "missing right child".to_string(),
    });
    let right_id = match right_id {
        Ok(id) => id,
        Err(e) => return Err(e),
    };
    let left_child = match tree.node(left_id) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };
    let right_child = match tree.node(right_id) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };

    // Verify leaf values are correct
    // Left leaf should predict +1 (to correct negative gradients from targets=1)
    assert!(left_child.is_leaf(), "Left child should be leaf");
    let left_value = left_child.value();
    assert!(
        (left_value - 1.0_f64).abs() < EPSILON,
        "Left leaf should predict 1.0, got {left_value}"
    );

    // Right leaf should predict -1 (to correct positive gradients from targets=-1)
    assert!(right_child.is_leaf(), "Right child should be leaf");
    let right_value = right_child.value();
    assert!(
        (right_value - (-1.0_f64)).abs() < EPSILON,
        "Right leaf should predict -1.0, got {right_value}"
    );

    Ok(())
}

/// Test that monotonic constraints are enforced correctly
#[test]
fn test_monotonic_constraint_enforcement() -> std::result::Result<(), ClearGbmError> {
    let sample_indices: Vec<usize> = (0_usize..6_usize).collect();
    // Gradients that would give decreasing leaf values as feature increases
    let gradients = vec![-1.0_f64, -1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];

    let bins = vec![
        vec![0_usize],
        vec![0_usize],
        vec![1_usize],
        vec![1_usize],
        vec![2_usize],
        vec![2_usize],
    ];
    let bin_thresholds = vec![vec![0.33_f64, 0.66_f64, 1.0_f64]];

    let split_config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let tree_config = match TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let constraints = vec![MonotonicConstraint::Increasing];

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 3_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: Some(&constraints),
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // If tree has split, verify monotonicity is maintained
    if tree.n_leaves() >= 2_usize {
        let root = match tree.root() {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        if !root.is_leaf() {
            if let (Some(left_id), Some(right_id)) = (root.left_child(), root.right_child()) {
                let left = match tree.node(left_id) {
                    Ok(n) => n,
                    Err(e) => return Err(e),
                };
                let right = match tree.node(right_id) {
                    Ok(n) => n,
                    Err(e) => return Err(e),
                };

                if left.is_leaf() && right.is_leaf() {
                    // For increasing constraint: left_value <= right_value
                    assert!(
                        left.value() <= right.value() + EPSILON,
                        "Monotonic increasing violated: left={} > right={}",
                        left.value(),
                        right.value()
                    );
                }
            }
        }
    }

    Ok(())
}

/// Test regularization effects on tree structure
#[test]
fn test_regularization_effects() -> std::result::Result<(), ClearGbmError> {
    let sample_indices: Vec<usize> = (0_usize..8_usize).collect();
    let gradients = vec![
        1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64, -1.0_f64, -1.0_f64,
    ];
    let hessians = vec![
        1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64,
    ];

    let bins: Vec<Vec<usize>> = (0_usize..8_usize).map(|i| vec![i]).collect();
    let bin_thresholds = vec![vec![
        0.125_f64, 0.25_f64, 0.375_f64, 0.5_f64, 0.625_f64, 0.75_f64, 0.875_f64, 1.0_f64,
    ]];

    // Build tree without regularization
    let split_config_no_reg = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let tree_config_no_reg =
        match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, split_config_no_reg) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

    let input_no_reg = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 8_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config_no_reg,
        monotonic_constraints: None,
    };

    let tree_no_reg = match build_tree(&input_no_reg, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Build tree with strong L2 regularization
    let split_config_l2 = match SplitConfig::new(2_usize, 1_usize, 64_usize, 10.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let tree_config_l2 =
        match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 10.0_f64, split_config_l2) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

    let input_l2 = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 8_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config_l2,
        monotonic_constraints: None,
    };

    let tree_l2 = match build_tree(&input_l2, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // With regularization, leaf values should be shrunk toward zero
    let root_no_reg = match tree_no_reg.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    let root_l2 = match tree_l2.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };

    // Both should have similar structure but different leaf values
    if !root_no_reg.is_leaf() && !root_l2.is_leaf() {
        if let (Some(left_id_no_reg), Some(left_id_l2)) =
            (root_no_reg.left_child(), root_l2.left_child())
        {
            let left_no_reg = match tree_no_reg.node(left_id_no_reg) {
                Ok(n) => n,
                Err(e) => return Err(e),
            };
            let left_l2 = match tree_l2.node(left_id_l2) {
                Ok(n) => n,
                Err(e) => return Err(e),
            };

            if left_no_reg.is_leaf() && left_l2.is_leaf() {
                // Regularized values should be smaller in absolute value
                assert!(
                    left_l2.value().abs() <= left_no_reg.value().abs() + EPSILON,
                    "Regularization should shrink leaf values"
                );
            }
        }
    }

    Ok(())
}
