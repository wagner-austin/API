//! Example-based unit tests for the histogram module.
//!
//! Tests target the trusted hot loop [`build_node_histograms_single_pass`]
//! directly — the one histogram-build entry point in the crate. Inputs are
//! pre-permuted in the test (mimicking what the tree builder does via
//! [`reorder_grad_hess_into`]). There is no validated public shim, so no
//! error-path tests: the hot loop is trusted, and the tree builder
//! establishes its invariants by construction.
//!
//! The single-feature cases carry the exact expected values the per-feature
//! predecessor pinned, so they double as the equivalence record across the
//! 2026-08-21 single-pass refactor: at `n_features = 1` the row-major and
//! column-major layouts coincide, and every pinned sum is unchanged.

use crate::error::ClearGbmError;
use crate::histogram::{
    build_node_histograms_single_pass, reorder_grad_hess_into, subtract_histogram,
    NodeHistogramRequest,
};
use crate::types::HistogramBuffer;

// ============================================================================
// build_node_histograms_single_pass — the one trusted hot loop
// ============================================================================

#[test]
fn test_build_histogram_ordered_simple() -> Result<(), ClearGbmError> {
    // sample_indices identity => ordered_g == gradients (same for hessians).
    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![0_u8, 1_u8, 0_u8];
    let n_bins = 3_usize;

    let hists = build_node_histograms_single_pass(NodeHistogramRequest {
        sample_indices: &sample_indices,
        ordered_gradients: &gradients,
        ordered_hessians: &hessians,
        bins_rows: &bins,
        n_features: 1_usize,
        n_bins,
    });
    let hist = &hists[0_usize];

    // Bin 0: samples 0 and 2 (gradients 0.1 + 0.3 = 0.4).
    assert!((hist.gradient_sums()[0_usize] - 0.4_f64).abs() < 1e-6_f64);
    assert_eq!(hist.counts()[0_usize], 2_usize);
    // Bin 1: sample 1 only (gradient 0.2).
    assert!((hist.gradient_sums()[1_usize] - 0.2_f64).abs() < 1e-6_f64);
    assert_eq!(hist.counts()[1_usize], 1_usize);
    // Bin 2: empty.
    assert!(hist.gradient_sums()[2_usize].abs() < 1e-9_f64);
    assert_eq!(hist.counts()[2_usize], 0_usize);
    Ok(())
}

#[test]
fn test_build_histogram_ordered_subset_via_reorder() -> Result<(), ClearGbmError> {
    // Non-identity permutation — the tree builder emits these at every non-root
    // node. Reorder into position-space before calling the trusted hot loop.
    let sample_indices = vec![0_u32, 2_u32]; // skip sample 1
    let full_gradients = vec![0.1_f64, 0.2_f64, 0.3_f64];
    let full_hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let bins = vec![0_u8, 1_u8, 0_u8];
    let n_bins = 2_usize;

    let n = sample_indices.len();
    let mut ordered_g: Vec<f64> = vec![0.0_f64; n];
    let mut ordered_h: Vec<f64> = vec![0.0_f64; n];
    reorder_grad_hess_into(
        &sample_indices,
        &full_gradients,
        &full_hessians,
        &mut ordered_g,
        &mut ordered_h,
    );

    let hists = build_node_histograms_single_pass(NodeHistogramRequest {
        sample_indices: &sample_indices,
        ordered_gradients: &ordered_g,
        ordered_hessians: &ordered_h,
        bins_rows: &bins,
        n_features: 1_usize,
        n_bins,
    });
    let hist = &hists[0_usize];

    // Only samples 0 and 2 count (both in bin 0).
    assert!((hist.gradient_sums()[0_usize] - 0.4_f64).abs() < 1e-6_f64);
    assert_eq!(hist.counts()[0_usize], 2_usize);
    // Bin 1 empty (sample 1 skipped).
    assert_eq!(hist.counts()[1_usize], 0_usize);
    Ok(())
}

#[test]
fn test_build_histogram_ordered_large_unrolled_chunks() -> Result<(), ClearGbmError> {
    // Exercise the 8-wide unroll — 10000 samples = 1250 full chunks + 0 tail.
    let n = 10000_usize;
    let sample_indices: Vec<u32> = (0_u32..).take(n).collect();
    let ordered_g: Vec<f64> = (0_u32..)
        .take(n)
        .map(|i| f64::from(u8::try_from(i % 251).unwrap_or(0)) * 0.001_f64)
        .collect();
    let ordered_h: Vec<f64> = vec![1.0_f64; n];
    let bins: Vec<u8> = (0_usize..n)
        .map(|i| u8::try_from(i % 64).unwrap_or(0_u8))
        .collect();
    let n_bins = 64_usize;

    let hists = build_node_histograms_single_pass(NodeHistogramRequest {
        sample_indices: &sample_indices,
        ordered_gradients: &ordered_g,
        ordered_hessians: &ordered_h,
        bins_rows: &bins,
        n_features: 1_usize,
        n_bins,
    });
    let hist = &hists[0_usize];

    // Total count across all bins == n.
    let total_count: usize = hist.counts().iter().sum();
    assert_eq!(total_count, n);
    // Total hessian sum == n (each hessian is 1.0).
    let total_hess: f64 = hist.hessian_sums().iter().sum();
    let n_f64 = f64::from(u32::try_from(n).unwrap_or_default());
    assert!((total_hess - n_f64).abs() < 1e-6_f64);
    Ok(())
}

#[test]
fn test_build_histogram_ordered_tail_remainder() -> Result<(), ClearGbmError> {
    // Force the scalar tail — 11 samples = 1 chunk of 8 + 3 remainder.
    let n = 11_usize;
    let sample_indices: Vec<u32> = (0_u32..).take(n).collect();
    let ordered_g: Vec<f64> = (0_u32..)
        .take(n)
        .map(|i| f64::from(u8::try_from(i).unwrap_or(0)))
        .collect();
    let ordered_h: Vec<f64> = (10_u32..)
        .take(n)
        .map(|i| f64::from(u8::try_from(i).unwrap_or(0)))
        .collect();
    let bins: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2];
    let n_bins = 4_usize;

    let hists = build_node_histograms_single_pass(NodeHistogramRequest {
        sample_indices: &sample_indices,
        ordered_gradients: &ordered_g,
        ordered_hessians: &ordered_h,
        bins_rows: &bins,
        n_features: 1_usize,
        n_bins,
    });
    let hist = &hists[0_usize];

    // Bin 0: samples 0, 4, 8 (gradients 0 + 4 + 8 = 12).
    assert!((hist.gradient_sums()[0_usize] - 12.0_f64).abs() < 1e-6_f64);
    assert_eq!(hist.counts()[0_usize], 3_usize);
    // Bin 3: samples 3, 7 (gradients 3 + 7 = 10).
    assert!((hist.gradient_sums()[3_usize] - 10.0_f64).abs() < 1e-6_f64);
    assert_eq!(hist.counts()[3_usize], 2_usize);
    Ok(())
}

#[test]
fn test_build_histogram_ordered_permuted_non_monotonic() -> Result<(), ClearGbmError> {
    // Non-monotonic sample_indices — what the tree builder produces after
    // splits partition the sample space. Reorder + trusted call.
    let sample_indices = vec![5_u32, 2_u32, 8_u32, 1_u32, 9_u32, 0_u32, 4_u32];
    let full_gradients: Vec<f64> = (0_u32..10_u32)
        .map(|i| f64::from(u8::try_from(i).unwrap_or(0)) * 0.5_f64)
        .collect();
    let full_hessians: Vec<f64> = vec![1.0_f64; 10_usize];
    let bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0];
    let n_bins = 3_usize;

    let n = sample_indices.len();
    let mut ordered_g: Vec<f64> = vec![0.0_f64; n];
    let mut ordered_h: Vec<f64> = vec![0.0_f64; n];
    reorder_grad_hess_into(
        &sample_indices,
        &full_gradients,
        &full_hessians,
        &mut ordered_g,
        &mut ordered_h,
    );

    let hists = build_node_histograms_single_pass(NodeHistogramRequest {
        sample_indices: &sample_indices,
        ordered_gradients: &ordered_g,
        ordered_hessians: &ordered_h,
        bins_rows: &bins,
        n_features: 1_usize,
        n_bins,
    });
    let hist = &hists[0_usize];

    // Sum of gradients equals the sum of full_gradients at the selected sample_indices.
    let expected_total: f64 = sample_indices
        .iter()
        .map(|&i| full_gradients[crate::narrow::index_widen(i)])
        .sum();
    let actual_total: f64 = hist.gradient_sums().iter().sum();
    assert!((actual_total - expected_total).abs() < 1e-6_f64);
    // Total count == number of sample_indices.
    let total_count: usize = hist.counts().iter().sum();
    assert_eq!(total_count, sample_indices.len());
    Ok(())
}

#[test]
fn test_build_node_histograms_multi_feature_exact_values() -> Result<(), ClearGbmError> {
    // Three samples, two features, row-major rows: sample i holds
    // [bin_feat0, bin_feat1]. One walk must fill both features' histograms
    // with exactly the sums the per-feature passes produced.
    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![0.1_f64, 0.2_f64, 0.4_f64];
    let hessians = vec![1.0_f64, 2.0_f64, 4.0_f64];
    // feature 0 bins: [0, 1, 0]; feature 1 bins: [1, 1, 0]
    let bins_rows: Vec<u8> = vec![0_u8, 1_u8, 1_u8, 1_u8, 0_u8, 0_u8];
    let n_bins = 2_usize;

    let hists = build_node_histograms_single_pass(NodeHistogramRequest {
        sample_indices: &sample_indices,
        ordered_gradients: &gradients,
        ordered_hessians: &hessians,
        bins_rows: &bins_rows,
        n_features: 2_usize,
        n_bins,
    });

    assert_eq!(hists.len(), 2_usize);
    // Feature 0, bin 0: samples 0 and 2 (0.1 + 0.4 gradient, 1 + 4 hessian).
    assert!((hists[0_usize].gradient_sums()[0_usize] - 0.5_f64).abs() < 1e-12_f64);
    assert!((hists[0_usize].hessian_sums()[0_usize] - 5.0_f64).abs() < 1e-12_f64);
    assert_eq!(hists[0_usize].counts()[0_usize], 2_usize);
    // Feature 0, bin 1: sample 1 only.
    assert!((hists[0_usize].gradient_sums()[1_usize] - 0.2_f64).abs() < 1e-12_f64);
    assert_eq!(hists[0_usize].counts()[1_usize], 1_usize);
    // Feature 1, bin 0: sample 2 only.
    assert!((hists[1_usize].gradient_sums()[0_usize] - 0.4_f64).abs() < 1e-12_f64);
    assert_eq!(hists[1_usize].counts()[0_usize], 1_usize);
    // Feature 1, bin 1: samples 0 and 1.
    assert!((hists[1_usize].gradient_sums()[1_usize] - 0.3_f64).abs() < 1e-12_f64);
    assert_eq!(hists[1_usize].counts()[1_usize], 2_usize);
    Ok(())
}

#[test]
fn test_build_node_histograms_matches_per_feature_composition() -> Result<(), ClearGbmError> {
    // Self-consistency: one multi-feature walk must equal n_features
    // independent single-feature walks over that feature's own column.
    let sample_indices = vec![4_u32, 1_u32, 3_u32, 0_u32];
    let ordered_g = vec![0.7_f64, -0.3_f64, 0.9_f64, 0.05_f64];
    let ordered_h = vec![1.5_f64, 0.5_f64, 2.0_f64, 1.0_f64];
    let n_features = 3_usize;
    let n_bins = 4_usize;
    // 5 samples x 3 features, row-major.
    let bins_rows: Vec<u8> = vec![
        0_u8, 3_u8, 2_u8, // sample 0
        1_u8, 1_u8, 0_u8, // sample 1
        2_u8, 0_u8, 3_u8, // sample 2
        3_u8, 2_u8, 1_u8, // sample 3
        0_u8, 2_u8, 2_u8, // sample 4
    ];

    let combined = build_node_histograms_single_pass(NodeHistogramRequest {
        sample_indices: &sample_indices,
        ordered_gradients: &ordered_g,
        ordered_hessians: &ordered_h,
        bins_rows: &bins_rows,
        n_features,
        n_bins,
    });

    for feat_idx in 0_usize..n_features {
        let column: Vec<u8> = (0_usize..5_usize)
            .map(|s| bins_rows[s * n_features + feat_idx])
            .collect();
        let single = build_node_histograms_single_pass(NodeHistogramRequest {
            sample_indices: &sample_indices,
            ordered_gradients: &ordered_g,
            ordered_hessians: &ordered_h,
            bins_rows: &column,
            n_features: 1_usize,
            n_bins,
        });
        assert_eq!(combined[feat_idx], single[0_usize]);
    }
    Ok(())
}

// ============================================================================
// reorder_grad_hess_into — the amortization step
// ============================================================================

#[test]
fn test_reorder_grad_hess_into_populates_correctly() -> Result<(), ClearGbmError> {
    let sample_indices = vec![3_u32, 0_u32, 5_u32, 2_u32];
    let gradients = vec![10.0_f64, 11.0_f64, 12.0_f64, 13.0_f64, 14.0_f64, 15.0_f64];
    let hessians = vec![20.0_f64, 21.0_f64, 22.0_f64, 23.0_f64, 24.0_f64, 25.0_f64];
    let mut ordered_g = vec![0.0_f64; 4_usize];
    let mut ordered_h = vec![0.0_f64; 4_usize];

    reorder_grad_hess_into(
        &sample_indices,
        &gradients,
        &hessians,
        &mut ordered_g,
        &mut ordered_h,
    );

    // ordered_[i] == full_[sample_indices[i]].
    assert_eq!(ordered_g, vec![13.0_f64, 10.0_f64, 15.0_f64, 12.0_f64]);
    assert_eq!(ordered_h, vec![23.0_f64, 20.0_f64, 25.0_f64, 22.0_f64]);
    Ok(())
}

/// The output-length assertion fires when the caller sizes the ordered
/// buffers wrongly.
///
/// Catches the unwind explicitly rather than using `#[should_panic]`, which
/// Rust requires to return `()` and so could not follow the crate's
/// tests-return-`Result` convention. Asserting on the caught result is also
/// stronger: it pins the panic to this one call instead of accepting a panic
/// from anywhere in the test body.
#[test]
fn test_reorder_grad_hess_into_panics_on_output_length_mismatch() -> Result<(), ClearGbmError> {
    let sample_indices = vec![0_u32, 1_u32];
    let gradients = vec![1.0_f64, 2.0_f64];
    let hessians = vec![1.0_f64, 2.0_f64];

    // Silence the default panic hook so the expected unwind does not print a
    // backtrace into the test output; restored immediately afterwards.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(|| {
        // Wrong size — trips the length assertion in reorder_grad_hess_into.
        let mut ordered_g = vec![0.0_f64; 3_usize];
        let mut ordered_h = vec![0.0_f64; 3_usize];
        reorder_grad_hess_into(
            &sample_indices,
            &gradients,
            &hessians,
            &mut ordered_g,
            &mut ordered_h,
        );
    });
    std::panic::set_hook(previous_hook);

    assert!(outcome.is_err());
    Ok(())
}

// ============================================================================
// subtract_histogram — sibling derivation via H_L + H_R = H_P
// ============================================================================

#[test]
fn test_subtract_histogram_simple() -> Result<(), ClearGbmError> {
    let mut parent = HistogramBuffer::new(3_usize);
    match parent.accumulate(0_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match parent.accumulate(0_usize, 0.3_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match parent.accumulate(1_usize, 0.2_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut child = HistogramBuffer::new(3_usize);
    match child.accumulate(0_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let sibling = match subtract_histogram(&parent, &child) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    // Bin 0: parent has 2 samples (0.5+0.3=0.8), child has 1 (0.5)
    // Sibling should have 1 sample with gradient 0.3.
    assert_eq!(sibling.counts()[0_usize], 1_usize);
    assert!((sibling.gradient_sums()[0_usize] - 0.3_f64).abs() < 1e-10_f64);
    // Bin 1: parent has 1 sample, child has 0.
    assert_eq!(sibling.counts()[1_usize], 1_usize);
    Ok(())
}

#[test]
fn test_subtract_histogram_shape_mismatch() -> Result<(), ClearGbmError> {
    let parent = HistogramBuffer::new(3_usize);
    let child = HistogramBuffer::new(5_usize);

    let result = subtract_histogram(&parent, &child);

    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_subtract_histogram_empty() -> Result<(), ClearGbmError> {
    let parent = HistogramBuffer::new(3_usize);
    let child = HistogramBuffer::new(3_usize);

    let sibling = match subtract_histogram(&parent, &child) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    for i in 0_usize..3_usize {
        assert_eq!(sibling.counts()[i], 0_usize);
        assert!(sibling.gradient_sums()[i].abs() < f64::EPSILON);
    }
    Ok(())
}
