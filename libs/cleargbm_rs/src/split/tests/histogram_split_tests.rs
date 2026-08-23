//! Tests for find_best_split_from_histogram function.

use super::helpers::{helper_find_split_with_config, TestSplitParams};
use crate::error::ClearGbmError;
use crate::split::{MonotonicConstraint, SplitDecision, SplitResult};
use crate::types::HistogramBuffer;

#[test]
fn test_find_best_split_simple() -> Result<(), ClearGbmError> {
    // Inner function for full branch coverage
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut histogram = HistogramBuffer::new(4_usize);
        for _ in 0_usize..10_usize {
            match histogram.accumulate(0_usize, 0.05_f64, 0.1_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match histogram.accumulate(1_usize, 0.03_f64, 0.1_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match histogram.accumulate(2_usize, -0.08_f64, 0.1_f64) {
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
    // Cover Ok path
    let maybe_split = match inner(2_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let split = match maybe_split {
        Some(s) => s,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected split".to_string(),
            })
        }
    };
    assert_eq!(split.feature_index(), 0_usize);
    assert!(split.gain() > 0.0_f64);
    assert_eq!(
        split.decision(),
        SplitDecision::Threshold { split_bin: 1_usize }
    );
    // Cover Err path (invalid min_samples_split)
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_with_nan_bin() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut histogram = HistogramBuffer::new(4_usize);
        for _ in 0_usize..10_usize {
            match histogram.accumulate(0_usize, 0.1_f64, 0.1_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match histogram.accumulate(1_usize, 0.1_f64, 0.1_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match histogram.accumulate(2_usize, -0.2_f64, 0.1_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..5_usize {
            match histogram.accumulate(3_usize, 0.05_f64, 0.1_f64) {
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
    let maybe_split = match inner(2_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let split = match maybe_split {
        Some(s) => s,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected split".to_string(),
            })
        }
    };
    assert_eq!(split.left_count() + split.right_count(), 35_usize);
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_min_samples_leaf() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut histogram = HistogramBuffer::new(3_usize);
        match histogram.accumulate(0_usize, 0.1_f64, 0.1_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match histogram.accumulate(0_usize, 0.1_f64, 0.1_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match histogram.accumulate(1_usize, -0.1_f64, 0.1_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match histogram.accumulate(1_usize, -0.1_f64, 0.1_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        helper_find_split_with_config(
            &histogram,
            0_usize,
            2_usize,
            MonotonicConstraint::None,
            TestSplitParams {
                min_samples_split,
                min_samples_leaf: 3_usize,
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
    assert!(result.is_none());
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_min_gain_threshold() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut histogram = HistogramBuffer::new(3_usize);
        for _ in 0_usize..10_usize {
            match histogram.accumulate(0_usize, 0.01_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match histogram.accumulate(1_usize, 0.01_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        helper_find_split_with_config(
            &histogram,
            0_usize,
            2_usize,
            MonotonicConstraint::None,
            TestSplitParams {
                min_samples_split,
                min_samples_leaf: 1_usize,
                max_bins: 64_usize,
                reg_lambda: 0.0_f64,
                min_gain: 1.0_f64,
            },
        )
    }
    let result = match inner(2_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(result.is_none());
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_monotonicity_constraint() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut histogram = HistogramBuffer::new(3_usize);
        for _ in 0_usize..10_usize {
            match histogram.accumulate(0_usize, -0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match histogram.accumulate(1_usize, 0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        helper_find_split_with_config(
            &histogram,
            0_usize,
            2_usize,
            MonotonicConstraint::Increasing,
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
    assert!(result.is_none());
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_empty_histogram() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let histogram = HistogramBuffer::new(3_usize);
        helper_find_split_with_config(
            &histogram,
            0_usize,
            2_usize,
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
    assert!(result.is_none());
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_n_regular_bins_exceeds_n_bins() -> Result<(), ClearGbmError> {
    let histogram = HistogramBuffer::new(3_usize);
    let result = helper_find_split_with_config(
        &histogram,
        0_usize,
        5_usize,
        MonotonicConstraint::None,
        TestSplitParams {
            min_samples_split: 2_usize,
            min_samples_leaf: 1_usize,
            max_bins: 64_usize,
            reg_lambda: 0.0_f64,
            min_gain: 0.0_f64,
        },
    );
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "n_regular_bins"
    ));
    Ok(())
}

#[test]
fn test_find_best_split_zero_regular_bins() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let histogram = HistogramBuffer::new(3_usize);
        helper_find_split_with_config(
            &histogram,
            0_usize,
            0_usize,
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
    assert!(result.is_none());
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_no_nan_bin() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut histogram = HistogramBuffer::new(3_usize);
        for _ in 0_usize..10_usize {
            match histogram.accumulate(0_usize, 0.1_f64, 0.1_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match histogram.accumulate(1_usize, 0.1_f64, 0.1_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match histogram.accumulate(2_usize, -0.2_f64, 0.1_f64) {
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
    let maybe_split = match inner(2_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let split = match maybe_split {
        Some(s) => s,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected split".to_string(),
            })
        }
    };
    assert_eq!(split.left_count() + split.right_count(), 30_usize);
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_from_histogram_n_regular_bins_too_large() -> Result<(), ClearGbmError> {
    let histogram = HistogramBuffer::new(10_usize);
    let result = helper_find_split_with_config(
        &histogram,
        0_usize,
        15_usize, // More than histogram bins
        MonotonicConstraint::None,
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

/// Test that covers the `dominated = true` branch in find_best_split_for_histogram.
/// When scanning bins, if a later bin has gain <= current best, it should be skipped.
/// This requires an early bin to have higher gain than later bins.
#[test]
fn test_find_best_split_early_bin_wins() -> Result<(), ClearGbmError> {
    // Create a histogram where:
    // - Split after bin 0 has HIGH gain (large gradient difference between left and right)
    // - Split after bin 1 has LOWER gain (smaller gradient difference)
    // This ensures `dominated = true` when evaluating the split after bin 1.
    let mut histogram = HistogramBuffer::new(4_usize);

    // Bin 0: Strong positive gradient (left side of optimal split)
    for _ in 0_usize..10_usize {
        match histogram.accumulate(0_usize, 0.9_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }

    // Bin 1: Weak negative gradient (makes split after bin 0 the best)
    for _ in 0_usize..10_usize {
        match histogram.accumulate(1_usize, -0.1_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }

    // Bin 2: Strong negative gradient (makes split after bin 1 have lower gain)
    for _ in 0_usize..10_usize {
        match histogram.accumulate(2_usize, -0.8_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }

    // The best split should be after bin 0, where:
    // - Left: bin 0 with gradient 0.9 * 10 = 9.0
    // - Right: bins 1+2 with gradient -0.1*10 + -0.8*10 = -9.0
    // This has the highest gain.
    //
    // Split after bin 1 would have:
    // - Left: bins 0+1 with gradient 9.0 + (-1.0) = 8.0
    // - Right: bin 2 with gradient -8.0
    // This has lower gain, so dominated = true.

    let maybe_split = match helper_find_split_with_config(
        &histogram,
        0_usize,
        3_usize,
        MonotonicConstraint::None,
        TestSplitParams {
            min_samples_split: 2_usize,
            min_samples_leaf: 1_usize,
            max_bins: 64_usize,
            reg_lambda: 0.0_f64,
            min_gain: 0.0_f64,
        },
    ) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let split = match maybe_split {
        Some(s) => s,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected split".to_string(),
            })
        }
    };

    // Best split should be after bin 0
    assert_eq!(
        split.decision(),
        SplitDecision::Threshold { split_bin: 0_usize }
    );
    Ok(())
}
