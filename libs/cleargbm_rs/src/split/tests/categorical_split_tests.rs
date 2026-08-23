//! Tests for the many-vs-many categorical split search and its bin set.
//!
//! The load-bearing test is the non-contiguous subset: a category layout
//! whose optimal partition is NOT any prefix of bin order proves the
//! search runs over the gradient-sorted order (Fisher's sufficient
//! ordering), not a disguised threshold scan.

use crate::error::ClearGbmError;
use crate::split::{
    find_best_categorical_split_from_histogram, CategoryBinSet, NanDirection, SplitDecision,
};
use crate::types::{HistogramBuffer, SplitConfig};

/// Builds a split config with permissive constraints for these tests.
fn make_config(min_samples_leaf: usize) -> Result<SplitConfig, ClearGbmError> {
    SplitConfig::new(2_usize, min_samples_leaf, 8_usize, 0.0_f64, 0.0_f64)
}

/// Accumulates `n` identical samples into one bin.
fn fill_bin(
    hist: &mut HistogramBuffer,
    bin: usize,
    n: usize,
    gradient: f64,
    hessian: f64,
) -> Result<(), ClearGbmError> {
    for _ in 0_usize..n {
        match hist.accumulate(bin, gradient, hessian) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

// =============================================================================
// CategoryBinSet
// =============================================================================

#[test]
fn test_bin_set_insert_contains_and_order() {
    let mut set = CategoryBinSet::new();
    assert!(set.is_empty());
    set.insert(200_usize);
    set.insert(3_usize);
    set.insert(64_usize);
    assert_eq!(set.len(), 3_usize);
    assert!(set.contains(3_usize) && set.contains(64_usize) && set.contains(200_usize));
    assert!(!set.contains(4_usize));
    assert_eq!(set.bins(), vec![3_usize, 64_usize, 200_usize]);
}

#[test]
fn test_bin_set_ignores_out_of_range_bins() {
    // Bins at or above 256 are unrepresentable by the u8 invariant; insert
    // ignores them and contains reports false rather than indexing outside
    // the mask.
    let mut set = CategoryBinSet::new();
    set.insert(256_usize);
    set.insert(1_000_usize);
    assert!(set.is_empty());
    assert!(!set.contains(256_usize));
}

// =============================================================================
// find_best_categorical_split_from_histogram
// =============================================================================

#[test]
fn test_finds_a_non_contiguous_subset() -> Result<(), ClearGbmError> {
    // Categories 0 and 2 push positive (negative gradients), category 1
    // pushes negative: the optimal partition {0, 2} vs {1} is not a prefix
    // of bin order, so no threshold over bins can express it. 4 bins total
    // (3 categories + NaN slot), all NaN-free.
    let mut hist = HistogramBuffer::new(4_usize);
    propagate!(fill_bin(&mut hist, 0_usize, 4_usize, -1.0_f64, 1.0_f64));
    propagate!(fill_bin(&mut hist, 1_usize, 4_usize, 1.0_f64, 1.0_f64));
    propagate!(fill_bin(&mut hist, 2_usize, 4_usize, -1.0_f64, 1.0_f64));

    let config = propagate!(make_config(1_usize));
    let split = match propagate!(find_best_categorical_split_from_histogram(
        &hist, 7_usize, &config, 3_usize, 3_usize,
    )) {
        Some(s) => s,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected a categorical split".to_string(),
            })
        }
    };

    assert_eq!(split.feature_index(), 7_usize);
    let left_bins = match split.decision() {
        SplitDecision::CategorySubset { left_bins } => left_bins,
        SplitDecision::Threshold { .. } => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "categorical search returned a threshold decision".to_string(),
            })
        }
    };
    // One side is exactly {0, 2}; which side depends on the sort direction,
    // and the complement {1} is equivalent. Assert the partition itself.
    let bins = left_bins.bins();
    let is_pair = bins == vec![0_usize, 2_usize];
    let is_singleton = bins == vec![1_usize];
    assert!(
        is_pair || is_singleton,
        "expected {{0,2}} vs {{1}} partition, got left = {bins:?}"
    );
    assert_eq!(split.left_count() + split.right_count(), 12_usize);
    assert!(split.gain() > 0.0_f64);
    Ok(())
}

#[test]
fn test_nan_partition_joins_its_gradient_side() -> Result<(), ClearGbmError> {
    // NaN samples share the sign of category 1: the winning split must put
    // the NaN partition on category 1's side, which the search finds by
    // trying both directions.
    let mut hist = HistogramBuffer::new(3_usize);
    propagate!(fill_bin(&mut hist, 0_usize, 4_usize, -1.0_f64, 1.0_f64));
    propagate!(fill_bin(&mut hist, 1_usize, 4_usize, 1.0_f64, 1.0_f64));
    // NaN bin sits at n_regular_bins = 2.
    propagate!(fill_bin(&mut hist, 2_usize, 4_usize, 1.0_f64, 1.0_f64));

    let config = propagate!(make_config(1_usize));
    let split = match propagate!(find_best_categorical_split_from_histogram(
        &hist, 0_usize, &config, 2_usize, 2_usize,
    )) {
        Some(s) => s,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected a categorical split".to_string(),
            })
        }
    };

    let left_bins = match split.decision() {
        SplitDecision::CategorySubset { left_bins } => left_bins,
        SplitDecision::Threshold { .. } => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "categorical search returned a threshold decision".to_string(),
            })
        }
    };
    // Whichever side bin 1 landed on, the NaN direction must point the
    // same way — 8 samples of matching gradient against 4.
    let bin_one_left = left_bins.contains(1_usize);
    assert_eq!(
        split.nan_direction() == NanDirection::Left,
        bin_one_left,
        "NaN must join category 1's side"
    );
    Ok(())
}

#[test]
fn test_single_category_yields_no_split() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    propagate!(fill_bin(&mut hist, 0_usize, 8_usize, -1.0_f64, 1.0_f64));
    let config = propagate!(make_config(1_usize));
    let result = propagate!(find_best_categorical_split_from_histogram(
        &hist, 0_usize, &config, 2_usize, 2_usize,
    ));
    assert!(result.is_none(), "one populated category cannot split");
    Ok(())
}

#[test]
fn test_min_samples_leaf_blocks_thin_subsets() -> Result<(), ClearGbmError> {
    // The only useful partition leaves 2 samples on one side; a floor of 3
    // must reject it.
    let mut hist = HistogramBuffer::new(3_usize);
    propagate!(fill_bin(&mut hist, 0_usize, 2_usize, -1.0_f64, 1.0_f64));
    propagate!(fill_bin(&mut hist, 1_usize, 6_usize, 1.0_f64, 1.0_f64));
    let config = propagate!(make_config(3_usize));
    let result = propagate!(find_best_categorical_split_from_histogram(
        &hist, 0_usize, &config, 2_usize, 2_usize,
    ));
    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_rejects_categories_beyond_regular_bins() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let config = propagate!(make_config(1_usize));
    match find_best_categorical_split_from_histogram(&hist, 0_usize, &config, 3_usize, 2_usize) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "n_categories > n_regular_bins must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_categories");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_rejects_regular_bins_beyond_histogram() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let config = propagate!(make_config(1_usize));
    match find_best_categorical_split_from_histogram(&hist, 0_usize, &config, 2_usize, 5_usize) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "n_regular_bins > n_bins must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_regular_bins");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_split_without_a_nan_bin() -> Result<(), ClearGbmError> {
    // n_bins == n_regular_bins: no NaN partition exists, the zero-NaN arm
    // runs, and the split still lands.
    let mut hist = HistogramBuffer::new(2_usize);
    propagate!(fill_bin(&mut hist, 0_usize, 4_usize, -1.0_f64, 1.0_f64));
    propagate!(fill_bin(&mut hist, 1_usize, 4_usize, 1.0_f64, 1.0_f64));
    let config = propagate!(make_config(1_usize));
    let split = match propagate!(find_best_categorical_split_from_histogram(
        &hist, 0_usize, &config, 2_usize, 2_usize,
    )) {
        Some(s) => s,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected a categorical split".to_string(),
            })
        }
    };
    assert!(split.gain() > 0.0_f64);
    assert_eq!(split.left_count() + split.right_count(), 8_usize);
    Ok(())
}
