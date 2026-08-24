//! Tests for the node-level tree building helpers: `compute_leaf_value`,
//! `should_stop`, `split_samples`, `compute_sums` and leaf-value
//! recording. The `build_tree` tests live in
//! [`super::builder_build_tests`] and [`super::builder_build_edge_tests`].

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::split::SplitDecision;
use crate::tree::nodes::EPSILON;
use crate::tree::nodes::{compute_leaf_value, compute_sums, should_stop, split_samples};
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
    // 5 samples, 1 feature, column-major flat storage.
    // Row-major layout (pre-refactor): [[0], [1], [2], [0], [1]]
    // Column-major flat: [0, 1, 2, 0, 1]
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 0_u8, 1_u8];
    let sample_indices = vec![0_u32, 1_u32, 2_u32, 3_u32, 4_u32];

    // Split at bin 0 (samples in bin <= 0 go left)
    let (left, right) = split_samples(
        &sample_indices,
        &bins,
        1_usize,
        0_usize,
        SplitDecision::Threshold { split_bin: 0_usize },
        true,
        3_usize,
    );

    // Left: bins 0 (samples 0, 3)
    assert_eq!(left.len(), 2_usize);
    assert!(left.contains(&0_u32));
    assert!(left.contains(&3_u32));

    // Right: bins 1, 2 (samples 1, 2, 4)
    assert_eq!(right.len(), 3_usize);
    assert!(right.contains(&1_u32));
    assert!(right.contains(&2_u32));
    assert!(right.contains(&4_u32));
    Ok(())
}

#[test]
fn test_split_samples_nan_handling() -> Result<(), ClearGbmError> {
    // Sample with NaN bin (= n_regular_bins). 2 samples, 1 feature.
    // Row-major layout (pre-refactor): [[0], [3]]
    // Column-major flat: [0, 3] where 3 is the NaN bin (n_regular_bins = 3).
    let bins: Vec<u8> = vec![0_u8, 3_u8];
    let sample_indices = vec![0_u32, 1_u32];

    // NaN goes left
    let (left, right) = split_samples(
        &sample_indices,
        &bins,
        1_usize,
        0_usize,
        SplitDecision::Threshold { split_bin: 0_usize },
        true,
        3_usize,
    );
    assert!(left.contains(&0_u32)); // bin 0
    assert!(left.contains(&1_u32)); // NaN goes left
    assert!(right.is_empty());

    // NaN goes right
    let (left2, right2) = split_samples(
        &sample_indices,
        &bins,
        1_usize,
        0_usize,
        SplitDecision::Threshold { split_bin: 0_usize },
        false,
        3_usize,
    );
    assert!(left2.contains(&0_u32)); // bin 0
    assert!(right2.contains(&1_u32)); // NaN goes right
    Ok(())
}

#[test]
fn test_split_samples_index_out_of_range_treated_as_nan() -> Result<(), ClearGbmError> {
    // A sample index that exceeds n_samples should route via the NaN branch.
    // Guards the missing-row-Vec behavior of the pre-refactor code.
    let bins: Vec<u8> = vec![0_u8, 0_u8];
    let sample_indices = vec![0_u32, 5_u32]; // 5 is out of range for n_samples = 2

    let (left, right) = split_samples(
        &sample_indices,
        &bins,
        1_usize,
        0_usize,
        SplitDecision::Threshold { split_bin: 0_usize },
        true,
        3_usize,
    );
    // sample 0 has bin 0 -> left. sample 5 is out of range -> NaN -> left.
    assert_eq!(left.len(), 2_usize);
    assert!(right.is_empty());

    let (left2, right2) = split_samples(
        &sample_indices,
        &bins,
        1_usize,
        0_usize,
        SplitDecision::Threshold { split_bin: 0_usize },
        false,
        3_usize,
    );
    // sample 0 -> left. sample 5 -> NaN -> right.
    assert!(left2.contains(&0_u32));
    assert!(right2.contains(&5_u32));
    Ok(())
}

// =============================================================================
// compute_sums edge case tests
// =============================================================================

/// Test compute_sums with all indices in bounds.
#[test]
fn test_compute_sums_basic() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![1.0_f64, 2.0_f64, 3.0_f64];
    let hessians = vec![0.5_f64, 0.5_f64, 0.5_f64];

    let (g_sum, h_sum) = compute_sums(&sample_indices, &gradients, &hessians);
    assert!((g_sum - 6.0_f64).abs() < EPSILON);
    assert!((h_sum - 1.5_f64).abs() < EPSILON);
    Ok(())
}

/// Test compute_sums with sample indices that exceed gradient array bounds.
/// This covers the `idx >= gradients.len()` branch.
#[test]
fn test_compute_sums_gradient_out_of_bounds() -> Result<(), ClearGbmError> {
    // Sample indices include index 5, but gradients only has indices 0-2
    let sample_indices = vec![0_u32, 1_u32, 5_u32];
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
/// This covers the `idx >= hessians.len()` branch.
#[test]
fn test_compute_sums_hessian_out_of_bounds() -> Result<(), ClearGbmError> {
    // Sample indices include index 5, but hessians only has indices 0-2
    let sample_indices = vec![0_u32, 1_u32, 5_u32];
    let gradients = vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64]; // length 6
    let hessians = vec![0.5_f64, 0.5_f64, 0.5_f64]; // length 3

    let (g_sum, h_sum) = compute_sums(&sample_indices, &gradients, &hessians);
    // g_sum = 1.0 + 2.0 + 6.0 (index 5 is valid for gradients) = 9.0
    // h_sum = 0.5 + 0.5 + 0 (index 5 skipped) = 1.0
    assert!((g_sum - 9.0_f64).abs() < EPSILON);
    assert!((h_sum - 1.0_f64).abs() < EPSILON);
    Ok(())
}

// =========================================================================
// record_leaf_values tests
// =========================================================================

#[test]
fn test_record_leaf_values_writes_every_index() -> Result<(), ClearGbmError> {
    use crate::tree::builder::record_leaf_values;

    let mut out = vec![f64::NAN; 5_usize];
    record_leaf_values(&[0_u32, 2_u32, 4_u32], 0.75_f64, &mut out);
    assert!((out[0_usize] - 0.75_f64).abs() < 1e-15_f64);
    assert!((out[2_usize] - 0.75_f64).abs() < 1e-15_f64);
    assert!((out[4_usize] - 0.75_f64).abs() < 1e-15_f64);
    // Untouched samples keep the NaN sentinel the caller reads as
    // "subsampled out this round".
    assert!(out[1_usize].is_nan());
    assert!(out[3_usize].is_nan());
    Ok(())
}

#[test]
fn test_build_tree_rejects_out_of_range_sample_index() -> Result<(), ClearGbmError> {
    // The per-sample leaf-value buffer is sized `input.n_samples`, so an index
    // past it has no slot. Rejecting up front is what lets the leaf-recording
    // loop index directly; silently skipping would instead leave that sample
    // at the NaN sentinel, which the caller reads as "subsampled out this
    // round" and quietly rescores through a full tree walk.
    let n_samples = 4_usize;
    let n_features = 1_usize;
    let split_config = match SplitConfig::new(2_usize, 1_usize, 4_usize, 1.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = match TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 1.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 3_u8];
    let gradients = vec![0.1_f64, -0.2_f64, 0.3_f64, -0.4_f64, 0.5_f64, -0.6_f64];
    let hessians = vec![1.0_f64; 6_usize];
    let thresholds: Vec<Vec<f64>> = vec![vec![0.5_f64, 1.5_f64, 2.5_f64]];

    // Index 9 addresses no slot in a 4-sample buffer.
    let sample_indices = vec![0_u32, 1_u32, 9_u32];
    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, n_samples, n_features);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples,
        n_features,
        n_regular_bins: 4_usize,
        bin_thresholds: &thresholds,
        config: &config,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
        categorical: None,
        quantized: None,
    };

    match build_tree(&input, &Hooks::default()) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "sample index 9 into a 4-sample buffer must be rejected".to_string(),
        }),
        Err(ClearGbmError::SampleIndexOutOfBounds {
            index,
            n_samples: n,
        }) => {
            assert_eq!(index, 9_usize);
            assert_eq!(n, 4_usize);
            Ok(())
        }
        Err(other) => Err(other),
    }
}
