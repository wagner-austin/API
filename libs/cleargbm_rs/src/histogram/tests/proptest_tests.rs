//! Property-based tests for histogram module.

use super::helpers::{
    subtract_for_test, validate_all_zeros, validate_histogram_sums,
    validate_subtraction_correctness,
};
use crate::error::ClearGbmError;
use crate::histogram::{build_histogram_ordered_trusted, reorder_grad_hess_into, HistogramRequest};
use crate::types::HistogramBuffer;

// =========================================================================
// Inner functions for proptest
// =========================================================================

/// Inner function for prop_histogram_sums - uses validate_histogram_sums.
///
/// Reorders `gradients` / `hessians` into position-space (per
/// [`reorder_grad_hess_into`]), then dispatches [`build_histogram_ordered_trusted`]
/// — the same shape the tree builder uses in production. The tree-builder path
/// is trusted (invariants established by construction), so this inner function
/// is infallible.
fn prop_histogram_sums_inner(
    sample_indices: &[u32],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[u8],
    n_bins: usize,
) -> Result<(), proptest::test_runner::TestCaseError> {
    let n = sample_indices.len();
    let mut ordered_g: Vec<f64> = vec![0.0_f64; n];
    let mut ordered_h: Vec<f64> = vec![0.0_f64; n];
    reorder_grad_hess_into(
        sample_indices,
        gradients,
        hessians,
        &mut ordered_g,
        &mut ordered_h,
    );
    let hist = build_histogram_ordered_trusted(HistogramRequest {
        sample_indices,
        ordered_gradients: &ordered_g,
        ordered_hessians: &ordered_h,
        bins,
        n_bins,
    });

    // Gradients, the accumulator and the sums are all f64, so the expected
    // total is an exact sum with no conversion anywhere in the path.
    let expected_grad: f64 = gradients.iter().sum();
    validate_histogram_sums(&hist, expected_grad, sample_indices.len())
}

/// Inner function for prop_subtract_histogram_identity - uses validate_all_zeros.
fn prop_subtract_identity_inner(
    parent: &HistogramBuffer,
    other: &HistogramBuffer,
) -> Result<(), proptest::test_runner::TestCaseError> {
    let sibling = match subtract_for_test(parent, other) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    validate_all_zeros(&sibling)
}

/// Inner function for prop_subtract_histogram_correctness - uses validate_subtraction_correctness.
fn prop_subtract_correctness_inner(
    parent: &HistogramBuffer,
    child: &HistogramBuffer,
) -> Result<(), proptest::test_runner::TestCaseError> {
    let sibling = match subtract_for_test(parent, child) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    validate_subtraction_correctness(&sibling, parent, child)
}

// =========================================================================
// Tests for inner function error branches
// =========================================================================
//
// `prop_histogram_sums_inner` no longer has an error branch — the trusted
// `build_histogram_ordered_trusted` is infallible (invariants established by
// construction). Subtract-based inners still have shape-mismatch errors that
// are worth covering.

#[test]
fn test_prop_subtract_identity_inner_error() -> Result<(), ClearGbmError> {
    // Cover error path by using mismatched histogram sizes
    let parent = HistogramBuffer::new(3_usize);
    let other = HistogramBuffer::new(5_usize);
    let result = prop_subtract_identity_inner(&parent, &other);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_prop_subtract_correctness_inner_error() -> Result<(), ClearGbmError> {
    // Cover error path by using mismatched histogram sizes
    let parent = HistogramBuffer::new(3_usize);
    let child = HistogramBuffer::new(5_usize);
    let result = prop_subtract_correctness_inner(&parent, &child);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Property-based tests
// =========================================================================

#[test]
fn prop_histogram_sums_equal_input_sums() -> Result<(), ClearGbmError> {
    let config = proptest::test_runner::Config::with_cases(100);
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(
            &(1_usize..50_usize, 2_usize..10_usize),
            |(n_samples, n_bins)| {
                let gradients: Vec<f64> = {
                    let mut v = Vec::with_capacity(n_samples);
                    let mut acc = 0.0_f64;
                    for _ in 0_usize..n_samples {
                        v.push(acc);
                        acc += 0.1_f64;
                    }
                    v
                };
                let hessians: Vec<f64> = (0_usize..n_samples).map(|_| 1.0_f64).collect();
                let sample_indices: Vec<u32> = (0_u32..).take(n_samples).collect();
                let bins: Vec<u8> = (0_usize..n_samples)
                    .map(|i| u8::try_from(i % n_bins).unwrap_or_default())
                    .collect();
                prop_histogram_sums_inner(&sample_indices, &gradients, &hessians, &bins, n_bins)
            },
        )
        .map_err(|e| ClearGbmError::InvalidParameter {
            name: "proptest".to_string(),
            reason: format!("{}", e),
        })
}

#[test]
fn prop_subtract_histogram_identity() -> Result<(), ClearGbmError> {
    let config = proptest::test_runner::Config::with_cases(100);
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(&(2_usize..10_usize), |n_bins| {
            let mut parent = HistogramBuffer::new(n_bins);
            for bin in 0_usize..n_bins {
                let _ = parent.accumulate(bin, 1.0_f64, 1.0_f64);
            }
            prop_subtract_identity_inner(&parent, &parent)
        })
        .map_err(|e| ClearGbmError::InvalidParameter {
            name: "proptest".to_string(),
            reason: format!("{}", e),
        })
}

#[test]
fn prop_subtract_histogram_correctness() -> Result<(), ClearGbmError> {
    let config = proptest::test_runner::Config::with_cases(100);
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(
            &(2_usize..8_usize, 0.1_f64..0.9_f64),
            |(n_bins, child_fraction)| {
                let mut parent = HistogramBuffer::new(n_bins);
                for bin in 0_usize..n_bins {
                    let _ = parent.accumulate(bin, 10.0_f64, 5.0_f64);
                }

                let mut child = HistogramBuffer::new(n_bins);
                for bin in 0_usize..n_bins {
                    let _ =
                        child.accumulate(bin, 10.0_f64 * child_fraction, 5.0_f64 * child_fraction);
                }

                prop_subtract_correctness_inner(&parent, &child)
            },
        )
        .map_err(|e| ClearGbmError::InvalidParameter {
            name: "proptest".to_string(),
            reason: format!("{}", e),
        })
}
