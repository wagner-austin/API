//! Tests for find_best_split_across_features function.

use super::helpers::{helper_find_split_across_with_config, TestSplitParams};
use crate::error::ClearGbmError;
use crate::split::{MonotonicConstraint, SplitResult};
use crate::types::HistogramBuffer;

#[test]
fn test_find_best_split_across_features_single() -> Result<(), ClearGbmError> {
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
        helper_find_split_across_with_config(
            &[histogram],
            2_usize,
            None,
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
    assert_eq!(split.feature_index(), 0_usize);
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_across_features_multiple() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut hist0 = HistogramBuffer::new(3_usize);
        for _ in 0_usize..10_usize {
            match hist0.accumulate(0_usize, 0.1_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match hist0.accumulate(1_usize, -0.1_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        let mut hist1 = HistogramBuffer::new(3_usize);
        for _ in 0_usize..10_usize {
            match hist1.accumulate(0_usize, 0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match hist1.accumulate(1_usize, -0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        helper_find_split_across_with_config(
            &[hist0, hist1],
            2_usize,
            None,
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
    assert_eq!(split.feature_index(), 1_usize);
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_across_features_with_constraints() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut hist0 = HistogramBuffer::new(3_usize);
        for _ in 0_usize..10_usize {
            match hist0.accumulate(0_usize, -0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match hist0.accumulate(1_usize, 0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        let mut hist1 = HistogramBuffer::new(3_usize);
        for _ in 0_usize..10_usize {
            match hist1.accumulate(0_usize, 0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match hist1.accumulate(1_usize, -0.5_f64, 1.0_f64) {
                Ok(()) => {}
                Err(e) => return Err(e),
            }
        }
        let constraints = vec![
            MonotonicConstraint::Increasing,
            MonotonicConstraint::Increasing,
        ];
        helper_find_split_across_with_config(
            &[hist0, hist1],
            2_usize,
            Some(&constraints),
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
    assert_eq!(split.feature_index(), 1_usize);
    assert!(inner(0_usize).is_err());
    Ok(())
}

#[test]
fn test_find_best_split_across_features_no_valid_split() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let mut hist0 = HistogramBuffer::new(3_usize);
        match hist0.accumulate(0_usize, 0.1_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist0.accumulate(1_usize, -0.1_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        let mut hist1 = HistogramBuffer::new(3_usize);
        match hist1.accumulate(0_usize, 0.1_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist1.accumulate(1_usize, -0.1_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        helper_find_split_across_with_config(
            &[hist0, hist1],
            2_usize,
            None,
            TestSplitParams {
                min_samples_split,
                min_samples_leaf: 5_usize,
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
fn test_find_best_split_across_features_empty() -> Result<(), ClearGbmError> {
    fn inner(min_samples_split: usize) -> Result<Option<SplitResult>, ClearGbmError> {
        let histograms: Vec<HistogramBuffer> = vec![];
        helper_find_split_across_with_config(
            &histograms,
            2_usize,
            None,
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

/// Test that covers the `is_better = false` branch in find_best_split_across_features.
/// When feature 0 has a higher gain than feature 1, the second feature should NOT
/// replace the best split (is_better evaluates to false).
#[test]
fn test_find_best_split_first_feature_wins() -> Result<(), ClearGbmError> {
    // Feature 0: HIGH gain (large gradient difference)
    let mut hist0 = HistogramBuffer::new(3_usize);
    for _ in 0_usize..10_usize {
        match hist0.accumulate(0_usize, 0.9_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }
    for _ in 0_usize..10_usize {
        match hist0.accumulate(1_usize, -0.9_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }

    // Feature 1: LOW gain (small gradient difference)
    let mut hist1 = HistogramBuffer::new(3_usize);
    for _ in 0_usize..10_usize {
        match hist1.accumulate(0_usize, 0.1_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }
    for _ in 0_usize..10_usize {
        match hist1.accumulate(1_usize, -0.1_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }

    // Feature 0 should win because it has higher gain.
    // When feature 1 is evaluated, is_better = false (split1.gain < split0.gain).
    let maybe_split = match helper_find_split_across_with_config(
        &[hist0, hist1],
        2_usize,
        None,
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

    // Feature 0 should be selected (not feature 1)
    assert_eq!(split.feature_index(), 0_usize);
    Ok(())
}
