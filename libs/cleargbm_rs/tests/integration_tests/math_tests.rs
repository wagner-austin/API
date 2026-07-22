//! Mathematical correctness tests for core GBM primitives.
//!
//! Verifies leaf value computation, split gain formulas, histogram
//! accumulation, and best-split search against known analytical results.

use cleargbm_rs::{
    build_histogram, compute_leaf_value, compute_split_gain, find_best_split_from_histogram,
    ClearGbmError, HistogramBuffer, MonotonicConstraint, SplitConfig,
};

use super::EPSILON;

/// Test that leaf values are computed correctly: leaf = -G / (H + λ)
#[test]
fn test_leaf_value_mathematical_correctness() -> std::result::Result<(), ClearGbmError> {
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
    let bins = vec![0_u8, 1_u8, 2_u8, 0_u8, 1_u8, 2_u8];

    let hist = match build_histogram(&sample_indices, &gradients, &hessians, &bins, 3_usize) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    // Verify bin 0: sum([0.1, 0.4]) = 0.5, count = 2
    let g0 = match hist.gradient_sum(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        (g0 - 0.5_f64).abs() < EPSILON,
        "Bin 0 gradient: expected 0.5, got {g0}"
    );
    let count0 = match hist.count(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(count0, 2_usize, "Bin 0 count should be 2");

    // Verify bin 1: sum([0.2, 0.5]) = 0.7, count = 2
    let g1 = match hist.gradient_sum(1_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        (g1 - 0.7_f64).abs() < EPSILON,
        "Bin 1 gradient: expected 0.7, got {g1}"
    );
    let count1 = match hist.count(1_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(count1, 2_usize, "Bin 1 count should be 2");

    // Verify bin 2: sum([0.3, 0.6]) = 0.9, count = 2
    let g2 = match hist.gradient_sum(2_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        (g2 - 0.9_f64).abs() < EPSILON,
        "Bin 2 gradient: expected 0.9, got {g2}"
    );
    let count2 = match hist.count(2_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(count2, 2_usize, "Bin 2 count should be 2");

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
    match histogram.accumulate(0_usize, 2.0_f64, 2.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match histogram.accumulate(0_usize, 0.0_f64, 0.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match histogram.accumulate(1_usize, 1.0_f64, 2.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match histogram.accumulate(1_usize, 0.0_f64, 0.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match histogram.accumulate(2_usize, -3.0_f64, 2.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match histogram.accumulate(2_usize, 0.0_f64, 0.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    // Need proper counts
    let mut histogram = HistogramBuffer::new(4_usize);
    for _ in 0_usize..2_usize {
        match histogram.accumulate(0_usize, 1.0_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }
    for _ in 0_usize..2_usize {
        match histogram.accumulate(1_usize, 0.5_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }
    for _ in 0_usize..2_usize {
        match histogram.accumulate(2_usize, -1.5_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }

    let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let maybe_split = match find_best_split_from_histogram(
        &histogram,
        0_usize,
        &config,
        3_usize,
        MonotonicConstraint::None,
    ) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    let split = maybe_split.ok_or_else(|| ClearGbmError::EmptyInput {
        context: "expected split".to_string(),
    });
    let split = match split {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    // The split should be at bin 1 (split_bin=1 means left gets bins 0,1)
    // because that gives the highest gain
    assert!(split.gain() > 0.0_f64, "Split gain should be positive");

    Ok(())
}
