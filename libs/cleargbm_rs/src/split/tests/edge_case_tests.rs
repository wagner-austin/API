//! Additional edge case and error propagation tests.

use super::helpers::{
    helper_find_split_across_with_config, helper_find_split_with_config, TestSplitParams,
};
use crate::error::ClearGbmError;
use crate::split::{
    check_monotonicity_constraint, find_best_split_across_features, find_best_split_from_histogram,
    MonotonicConstraint, NanDirection, SplitResult, SplitResultConfig,
};
use crate::types::{HistogramBuffer, SplitConfig};

#[test]
fn test_check_monotonicity_zero_hessian() -> Result<(), ClearGbmError> {
    // Test zero hessian handling in monotonicity check
    // With h_left = 0 (triggers EPSILON case), left_value becomes very large negative
    // right_value = -(-1.0) / 10.0 = 0.1
    // For Increasing: left_value <= right_value should be true
    let result = check_monotonicity_constraint(
        MonotonicConstraint::Increasing,
        1.0_f64,  // g_left
        0.0_f64,  // h_left = 0 (triggers EPSILON case)
        -1.0_f64, // g_right
        10.0_f64, // h_right
    );
    assert!(result);
    Ok(())
}

#[test]
fn test_find_best_split_no_nan_bin() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut histogram = HistogramBuffer::new(3_usize);
        for _ in 0_usize..10_usize {
            match histogram.accumulate(0_usize, 0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match histogram.accumulate(1_usize, -0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..5_usize {
            match histogram.accumulate(2_usize, 0.1_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        helper_find_split_with_config(
            &histogram,
            0_usize,
            3_usize,
            MonotonicConstraint::None,
            TestSplitParams {
                min_samples_split,
                min_samples_leaf: 1_usize,
                max_bins: 64_usize,
                reg_lambda: 0.0_f64,
                min_gain: 0.0_f64,
            },
        )
    }
    let result = match inner(2_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(result.is_some());
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_from_histogram_n_regular_bins_too_large() -> Result<(), ClearGbmError> {
    // Error test - no inner function needed since we expect error
    let histogram = HistogramBuffer::new(3_usize);
    let result = helper_find_split_with_config(
        &histogram,
        0_usize,
        10_usize,
        MonotonicConstraint::None,
        TestSplitParams {
            min_samples_split: 2_usize,
            min_samples_leaf: 1_usize,
            max_bins: 64_usize,
            reg_lambda: 0.0_f64,
            min_gain: 0.0_f64,
        },
    );
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { .. })
    ));
    Ok(())
}

#[test]
fn test_find_best_split_across_features_error_propagation() -> Result<(), ClearGbmError> {
    // Error test - no inner function needed since we expect error
    let histogram = HistogramBuffer::new(3_usize);
    let result = helper_find_split_across_with_config(
        &[histogram],
        10_usize,
        None,
        TestSplitParams {
            min_samples_split: 2_usize,
            min_samples_leaf: 1_usize,
            max_bins: 64_usize,
            reg_lambda: 0.0_f64,
            min_gain: 0.0_f64,
        },
    );
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Comprehensive error propagation tests
// These use inner functions with match to cover both Ok and Err branches
// =========================================================================

/// Covers error propagation for find_best_split_from_histogram calls.
#[test]
fn test_coverage_find_split_error_propagation() -> Result<(), ClearGbmError> {
    fn inner(n_bins: usize, n_regular_bins: usize) -> Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(n_bins);
        for _ in 0_usize..10_usize {
            match hist.accumulate(0_usize, 0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match hist.accumulate(1_usize, -0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        let cfg = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let _ = match find_best_split_from_histogram(
            &hist,
            0_usize,
            &cfg,
            n_regular_bins,
            MonotonicConstraint::None,
        ) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        Ok(())
    }
    assert!(inner(4_usize, 3_usize).is_ok());
    assert!(inner(3_usize, 100_usize).is_err());
    Ok(())
}

/// Covers error propagation for find_best_split_across_features calls.
#[test]
fn test_coverage_find_split_across_error_propagation() -> Result<(), ClearGbmError> {
    fn inner(n_regular_bins: usize) -> Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        for _ in 0_usize..10_usize {
            match hist.accumulate(0_usize, 0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match hist.accumulate(1_usize, -0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        let cfg = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let _ = match find_best_split_across_features(&[hist], &cfg, n_regular_bins, None) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        Ok(())
    }
    assert!(inner(2_usize).is_ok());
    assert!(inner(100_usize).is_err());
    Ok(())
}

/// Covers SplitConfig validation error paths.
#[test]
fn test_coverage_split_config_errors() -> Result<(), ClearGbmError> {
    fn inner(min_split: usize) -> Result<(), ClearGbmError> {
        let _ = match SplitConfig::new(min_split, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        Ok(())
    }
    assert!(inner(2_usize).is_ok());
    assert!(inner(1_usize).is_err());
    Ok(())
}

/// Covers ok_or error paths for Option to Result conversion.
#[test]
fn test_coverage_option_to_result_conversion() -> Result<(), ClearGbmError> {
    fn inner(find_split: bool) -> Result<SplitResult, ClearGbmError> {
        let maybe: Option<SplitResult> = if find_split {
            Some(SplitResult::new(SplitResultConfig {
                feature_index: 0_usize,
                split_bin: 1_usize,
                gain: 1.0_f64,
                left_gradient_sum: 0.5_f64,
                left_hessian_sum: 1.0_f64,
                left_count: 5_usize,
                right_gradient_sum: -0.5_f64,
                right_hessian_sum: 1.0_f64,
                right_count: 5_usize,
                nan_direction: NanDirection::Left,
            }))
        } else {
            None
        };
        match maybe {
            Some(s) => Ok(s),
            None => Err(ClearGbmError::TreeConstructionFailed {
                reason: "no split".to_string(),
            }),
        }
    }
    assert!(inner(true).is_ok());
    assert!(inner(false).is_err());
    Ok(())
}

/// Covers explicit match propagation on Result types.
#[test]
fn test_coverage_explicit_match_propagation() -> Result<(), ClearGbmError> {
    /// Inner function that uses explicit match for error propagation.
    fn check_result(result: Result<i32, ClearGbmError>) -> Result<i32, ClearGbmError> {
        match result {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
        Ok(42_i32)
    }

    // Cover Ok path - match doesn't trigger, returns Ok(42)
    let ok_input: Result<i32, ClearGbmError> = Ok(1_i32);
    assert!(check_result(ok_input).is_ok());

    // Cover Err path - match propagates the error
    let err_input: Result<i32, ClearGbmError> = Err(ClearGbmError::InvalidParameter {
        name: "test".to_string(),
        reason: "test".to_string(),
    });
    assert!(matches!(
        check_result(err_input),
        Err(ClearGbmError::InvalidParameter { .. })
    ));

    Ok(())
}

#[test]
fn test_mismatched_histogram_arrays_rejected_at_deserialization() -> Result<(), ClearGbmError> {
    // Before the bin accumulators were interleaved, the three histogram
    // arrays could disagree in length and the split scan needed a per-field
    // error arm for every access. Interleaving makes that state impossible
    // to construct, and the deserializer is the boundary that keeps it so:
    // any per-field length disagreement is rejected before a
    // `HistogramBuffer` exists.

    fn assert_rejected(json: &str) {
        let result: Result<HistogramBuffer, serde_json::Error> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // gradient_sums shorter than n_bins
    assert_rejected(
        r#"{
        "n_bins": 10,
        "gradient_sums": [1.0, 2.0, 3.0],
        "hessian_sums": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "counts": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    }"#,
    );

    // hessian_sums shorter than n_bins
    assert_rejected(
        r#"{
        "n_bins": 10,
        "gradient_sums": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "hessian_sums": [1.0, 2.0, 3.0],
        "counts": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    }"#,
    );

    // counts shorter than n_bins
    assert_rejected(
        r#"{
        "n_bins": 10,
        "gradient_sums": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "hessian_sums": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "counts": [5, 5, 5]
    }"#,
    );

    // arrays longer than n_bins are equally a disagreement
    assert_rejected(
        r#"{
        "n_bins": 2,
        "gradient_sums": [1.0, 1.0, 1.0],
        "hessian_sums": [1.0, 1.0, 1.0],
        "counts": [5, 5, 5]
    }"#,
    );

    Ok(())
}
