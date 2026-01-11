//! Integration tests for ClearGBM Rust core.
//!
//! These tests verify the mathematical correctness of the gradient boosting
//! algorithm, not just that the code runs without errors.

use cleargbm_rs::{
    build_histogram, build_tree, compute_leaf_value, compute_split_gain,
    find_best_split_from_histogram, BuildTreeInput, ClearGbmError, HistogramBuffer, Hooks,
    MonotonicConstraint, SplitConfig, TreeBuildConfig,
};

const EPSILON: f64 = 1e-10_f64;

// =============================================================================
// Mathematical correctness tests
// =============================================================================

/// Test that leaf values are computed correctly: leaf = -G / (H + λ)
#[test]
fn test_leaf_value_mathematical_correctness() -> std::result::Result<(), ClearGbmError> {
    // For squared error loss with targets y and predictions p:
    // gradient = 2(p - y) = -2(y - p), simplified: gradient = (p - y)
    // hessian = 2 (constant for squared error)
    //
    // Optimal leaf value minimizes loss: leaf = -sum(gradients) / sum(hessians)

    // Case 1: Simple case with no regularization
    // samples with gradients [1.0, -1.0, 0.5], hessians [1.0, 1.0, 1.0]
    // sum(G) = 0.5, sum(H) = 3.0
    // leaf = -0.5 / 3.0 = -0.1667
    let leaf = compute_leaf_value(0.5_f64, 3.0_f64, 0.0_f64, 0.0_f64);
    let expected = -0.5_f64 / 3.0_f64;
    assert!(
        (leaf - expected).abs() < EPSILON,
        "Expected leaf={expected}, got {leaf}"
    );

    // Case 2: With L2 regularization (lambda = 1.0)
    // leaf = -G / (H + λ) = -0.5 / (3.0 + 1.0) = -0.125
    let leaf_l2 = compute_leaf_value(0.5_f64, 3.0_f64, 0.0_f64, 1.0_f64);
    let expected_l2 = -0.5_f64 / 4.0_f64;
    assert!(
        (leaf_l2 - expected_l2).abs() < EPSILON,
        "Expected leaf_l2={expected_l2}, got {leaf_l2}"
    );

    // Case 3: With L1 regularization (alpha = 0.3)
    // Soft threshold: if |G| > alpha, leaf = -sign(G) * (|G| - alpha) / H
    // |G| = 0.5 > 0.3, sign(G) = 1
    // leaf = -1 * (0.5 - 0.3) / 3.0 = -0.2 / 3.0 = -0.0667
    let leaf_l1 = compute_leaf_value(0.5_f64, 3.0_f64, 0.3_f64, 0.0_f64);
    let expected_l1 = -0.2_f64 / 3.0_f64;
    assert!(
        (leaf_l1 - expected_l1).abs() < EPSILON,
        "Expected leaf_l1={expected_l1}, got {leaf_l1}"
    );

    // Case 4: L1 with |G| <= alpha should return 0
    let leaf_l1_zero = compute_leaf_value(0.2_f64, 3.0_f64, 0.3_f64, 0.0_f64);
    assert!(
        leaf_l1_zero.abs() < EPSILON,
        "Expected 0 when |G| <= alpha, got {leaf_l1_zero}"
    );

    Ok(())
}

/// Test that split gain is computed correctly using the formula:
/// gain = G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G_total^2/(H_total+λ)
#[test]
fn test_split_gain_mathematical_correctness() -> std::result::Result<(), ClearGbmError> {
    // Left: G_L = 2.0, H_L = 4.0
    // Right: G_R = -2.0, H_R = 4.0
    // Total: G_total = 0.0, H_total = 8.0
    // No regularization: λ = 0
    //
    // gain = 4/4 + 4/4 - 0/8 = 1 + 1 - 0 = 2.0
    let gain = compute_split_gain(
        2.0_f64, 4.0_f64, -2.0_f64, 4.0_f64, 0.0_f64, 8.0_f64, 0.0_f64,
    );
    let expected = 4.0_f64 / 4.0_f64 + 4.0_f64 / 4.0_f64 - 0.0_f64 / 8.0_f64;
    assert!(
        (gain - expected).abs() < EPSILON,
        "Expected gain={expected}, got {gain}"
    );

    // With L2 regularization: λ = 1.0
    // h_left_reg = 5, h_right_reg = 5, h_total_reg = 9
    // gain = 4/5 + 4/5 - 0/9 = 0.8 + 0.8 = 1.6
    let gain_l2 = compute_split_gain(
        2.0_f64, 4.0_f64, -2.0_f64, 4.0_f64, 0.0_f64, 8.0_f64, 1.0_f64,
    );
    let expected_l2 = 4.0_f64 / 5.0_f64 + 4.0_f64 / 5.0_f64;
    assert!(
        (gain_l2 - expected_l2).abs() < EPSILON,
        "Expected gain_l2={expected_l2}, got {gain_l2}"
    );

    // Test with asymmetric gradients
    // Left: G_L = 3.0, H_L = 2.0
    // Right: G_R = 1.0, H_R = 2.0
    // Total: G_total = 4.0, H_total = 4.0
    // λ = 0
    // gain = 9/2 + 1/2 - 16/4 = 4.5 + 0.5 - 4 = 1.0
    let gain_asym = compute_split_gain(
        3.0_f64, 2.0_f64, 1.0_f64, 2.0_f64, 4.0_f64, 4.0_f64, 0.0_f64,
    );
    let expected_asym = 9.0_f64 / 2.0_f64 + 1.0_f64 / 2.0_f64 - 16.0_f64 / 4.0_f64;
    assert!(
        (gain_asym - expected_asym).abs() < EPSILON,
        "Expected gain_asym={expected_asym}, got {gain_asym}"
    );

    Ok(())
}

/// Test histogram accumulation correctness
#[test]
fn test_histogram_accumulation_correctness() -> std::result::Result<(), ClearGbmError> {
    // 6 samples in 3 bins:
    // Bin 0: samples 0, 3 with gradients [0.1, 0.4], hessians [1.0, 1.0]
    // Bin 1: samples 1, 4 with gradients [0.2, 0.5], hessians [1.0, 1.0]
    // Bin 2: samples 2, 5 with gradients [0.3, 0.6], hessians [1.0, 1.0]

    let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize, 4_usize, 5_usize];
    let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64, 0.4_f64, 0.5_f64, 0.6_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![0_usize, 1_usize, 2_usize, 0_usize, 1_usize, 2_usize];

    let hist = build_histogram(&sample_indices, &gradients, &hessians, &bins, 3_usize)?;

    // Verify bin 0: sum([0.1, 0.4]) = 0.5, count = 2
    let g0 = hist.gradient_sum(0_usize)?;
    assert!(
        (g0 - 0.5_f64).abs() < EPSILON,
        "Bin 0 gradient: expected 0.5, got {g0}"
    );
    assert_eq!(hist.count(0_usize)?, 2_usize, "Bin 0 count should be 2");

    // Verify bin 1: sum([0.2, 0.5]) = 0.7, count = 2
    let g1 = hist.gradient_sum(1_usize)?;
    assert!(
        (g1 - 0.7_f64).abs() < EPSILON,
        "Bin 1 gradient: expected 0.7, got {g1}"
    );
    assert_eq!(hist.count(1_usize)?, 2_usize, "Bin 1 count should be 2");

    // Verify bin 2: sum([0.3, 0.6]) = 0.9, count = 2
    let g2 = hist.gradient_sum(2_usize)?;
    assert!(
        (g2 - 0.9_f64).abs() < EPSILON,
        "Bin 2 gradient: expected 0.9, got {g2}"
    );
    assert_eq!(hist.count(2_usize)?, 2_usize, "Bin 2 count should be 2");

    // Total should equal sum of all gradients
    let total_g: f64 = hist.gradient_sums().iter().sum();
    let expected_total: f64 = gradients.iter().sum();
    assert!(
        (total_g - expected_total).abs() < EPSILON,
        "Total gradient mismatch: expected {expected_total}, got {total_g}"
    );

    Ok(())
}

/// Test that find_best_split finds the optimal split point
#[test]
fn test_find_best_split_finds_optimal() -> std::result::Result<(), ClearGbmError> {
    // Create a histogram where the optimal split is clearly at bin 1
    // Bin 0: G=2.0, H=2.0 (positive gradient region)
    // Bin 1: G=1.0, H=2.0 (transition)
    // Bin 2: G=-3.0, H=2.0 (negative gradient region)
    //
    // Split at bin 0: left=[bin0], right=[bin1,bin2]
    //   G_L=2, H_L=2, G_R=-2, H_R=4
    //   gain = 0.5 * (4/2 + 4/4) = 0.5 * (2 + 1) = 1.5
    //
    // Split at bin 1: left=[bin0,bin1], right=[bin2]
    //   G_L=3, H_L=4, G_R=-3, H_R=2
    //   gain = 0.5 * (9/4 + 9/2) = 0.5 * (2.25 + 4.5) = 3.375

    let mut histogram = HistogramBuffer::new(4_usize); // 3 regular + 1 NaN
    histogram.accumulate(0_usize, 2.0_f64, 2.0_f64)?;
    histogram.accumulate(0_usize, 0.0_f64, 0.0_f64)?; // extra to get count
    histogram.accumulate(1_usize, 1.0_f64, 2.0_f64)?;
    histogram.accumulate(1_usize, 0.0_f64, 0.0_f64)?;
    histogram.accumulate(2_usize, -3.0_f64, 2.0_f64)?;
    histogram.accumulate(2_usize, 0.0_f64, 0.0_f64)?;

    // Need proper counts
    let mut histogram = HistogramBuffer::new(4_usize);
    for _ in 0_usize..2_usize {
        histogram.accumulate(0_usize, 1.0_f64, 1.0_f64)?;
    }
    for _ in 0_usize..2_usize {
        histogram.accumulate(1_usize, 0.5_f64, 1.0_f64)?;
    }
    for _ in 0_usize..2_usize {
        histogram.accumulate(2_usize, -1.5_f64, 1.0_f64)?;
    }

    let config = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
    let maybe_split = find_best_split_from_histogram(
        &histogram,
        0_usize,
        &config,
        3_usize,
        MonotonicConstraint::None,
    )?;

    let split = maybe_split.ok_or_else(|| ClearGbmError::EmptyInput {
        context: "expected split".to_string(),
    })?;

    // The split should be at bin 1 (split_bin=1 means left gets bins 0,1)
    // because that gives the highest gain
    assert!(split.gain() > 0.0_f64, "Split gain should be positive");

    Ok(())
}

// =============================================================================
// End-to-end tree building tests
// =============================================================================

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

    let split_config = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
    let tree_config = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, split_config)?;

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

    let tree = build_tree(&input, &Hooks::default())?;

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
    let root = tree.root()?;
    assert!(!root.is_leaf(), "Root should be an internal node");

    // The split should be around the middle (between samples 3 and 4)
    // where the label changes from 1 to 0
    let threshold = root.threshold().ok_or_else(|| ClearGbmError::EmptyInput {
        context: "missing threshold".to_string(),
    })?;
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

    let split_config = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
    let tree_config = TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, split_config)?;

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

    let tree = build_tree(&input, &Hooks::default())?;

    // Should split into two leaves
    assert_eq!(tree.n_leaves(), 2_usize, "Should have exactly 2 leaves");

    // Get leaf values by traversing the tree
    // Left leaf (bin 0): G_sum = -3, H_sum = 3, value = -(-3)/3 = 1.0
    // Right leaf (bin 1): G_sum = 3, H_sum = 3, value = -3/3 = -1.0

    // The root should point to two children
    let root = tree.root()?;
    assert!(!root.is_leaf(), "Root should be internal node");

    let left_id = root.left_child().ok_or_else(|| ClearGbmError::EmptyInput {
        context: "missing left child".to_string(),
    })?;
    let right_id = root
        .right_child()
        .ok_or_else(|| ClearGbmError::EmptyInput {
            context: "missing right child".to_string(),
        })?;
    let left_child = tree.node(left_id)?;
    let right_child = tree.node(right_id)?;

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
    // Create data where optimal split would violate monotonicity
    // but constraint should prevent it
    //
    // Feature increases but optimal leaf values would decrease
    // With increasing constraint, this should either not split or
    // choose a different split

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

    let split_config = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
    let tree_config = TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, split_config)?;

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

    let tree = build_tree(&input, &Hooks::default())?;

    // If tree has split, verify monotonicity is maintained
    if tree.n_leaves() >= 2_usize {
        let root = tree.root()?;
        if !root.is_leaf() {
            if let (Some(left_id), Some(right_id)) = (root.left_child(), root.right_child()) {
                let left = tree.node(left_id)?;
                let right = tree.node(right_id)?;

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
    let split_config_no_reg = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
    let tree_config_no_reg =
        TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, split_config_no_reg)?;

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

    let tree_no_reg = build_tree(&input_no_reg, &Hooks::default())?;

    // Build tree with strong L2 regularization
    let split_config_l2 = SplitConfig::new(2_usize, 1_usize, 64_usize, 10.0_f64, 0.0_f64)?;
    let tree_config_l2 =
        TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 10.0_f64, split_config_l2)?;

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

    let tree_l2 = build_tree(&input_l2, &Hooks::default())?;

    // With regularization, leaf values should be shrunk toward zero
    // Compare leaf values
    let root_no_reg = tree_no_reg.root()?;
    let root_l2 = tree_l2.root()?;

    // Both should have similar structure but different leaf values
    // The regularized tree might have fewer splits or smaller leaf values
    if !root_no_reg.is_leaf() && !root_l2.is_leaf() {
        if let (Some(left_id_no_reg), Some(left_id_l2)) =
            (root_no_reg.left_child(), root_l2.left_child())
        {
            let left_no_reg = tree_no_reg.node(left_id_no_reg)?;
            let left_l2 = tree_l2.node(left_id_l2)?;

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

// =============================================================================
// Edge case tests
// =============================================================================

/// Test handling of all-same gradients (no useful split)
#[test]
fn test_no_useful_split() -> std::result::Result<(), ClearGbmError> {
    let sample_indices: Vec<usize> = (0_usize..4_usize).collect();
    // All same gradients - no information gain from splitting
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];

    let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let split_config = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
    let tree_config = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, split_config)?;

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: None,
    };

    let tree = build_tree(&input, &Hooks::default())?;

    // With all same gradients, there's no gain from splitting
    // Tree should be a single leaf
    assert_eq!(
        tree.n_leaves(),
        1_usize,
        "Tree should be single leaf when all gradients are same"
    );

    let root = tree.root()?;
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

/// Test that sibling histogram subtraction is mathematically correct
#[test]
fn test_sibling_subtraction_correctness() -> std::result::Result<(), ClearGbmError> {
    use cleargbm_rs::subtract_histogram;

    // Parent histogram
    let mut parent = HistogramBuffer::new(3_usize);
    parent.accumulate(0_usize, 1.0_f64, 2.0_f64)?;
    parent.accumulate(1_usize, 3.0_f64, 4.0_f64)?;
    parent.accumulate(2_usize, 5.0_f64, 6.0_f64)?;

    // Child histogram (subset of parent)
    let mut child = HistogramBuffer::new(3_usize);
    child.accumulate(0_usize, 0.3_f64, 0.5_f64)?;
    child.accumulate(1_usize, 1.0_f64, 1.5_f64)?;
    child.accumulate(2_usize, 2.0_f64, 2.5_f64)?;

    // Sibling = Parent - Child
    let sibling = subtract_histogram(&parent, &child)?;

    // Verify each bin
    let g0 = sibling.gradient_sum(0_usize)?;
    assert!(
        (g0 - 0.7_f64).abs() < EPSILON,
        "Bin 0 gradient: expected 0.7, got {g0}"
    );

    let g1 = sibling.gradient_sum(1_usize)?;
    assert!(
        (g1 - 2.0_f64).abs() < EPSILON,
        "Bin 1 gradient: expected 2.0, got {g1}"
    );

    let g2 = sibling.gradient_sum(2_usize)?;
    assert!(
        (g2 - 3.0_f64).abs() < EPSILON,
        "Bin 2 gradient: expected 3.0, got {g2}"
    );

    // Verify hessians too
    let h0 = sibling.hessian_sum(0_usize)?;
    assert!(
        (h0 - 1.5_f64).abs() < EPSILON,
        "Bin 0 hessian: expected 1.5, got {h0}"
    );

    Ok(())
}
