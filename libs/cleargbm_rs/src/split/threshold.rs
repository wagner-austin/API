//! Threshold (numeric) split search: the O(K) prefix-sum scan over
//! ordered histogram bins, with both NaN directions tried per candidate
//! and monotonicity enforced per feature.

use crate::error::ClearGbmError;
use crate::types::{HistogramBuffer, SplitConfig};

use super::{
    check_monotonicity_constraint, compute_split_gain, MonotonicConstraint, NanDirection,
    SplitDecision, SplitResult, SplitResultConfig,
};

/// Internal struct to track the best split candidate during search.
#[derive(Debug, Clone, Copy)]
struct SplitCandidate {
    /// Bin index for the split.
    bin_index: usize,
    /// Gain from this split.
    gain: f64,
    /// Sum of gradients going left (including NaN if applicable).
    g_left: f64,
    /// Sum of hessians going left (including NaN if applicable).
    h_left: f64,
    /// Count of samples going left (including NaN if applicable).
    n_left: usize,
    /// Direction for NaN values.
    nan_direction: NanDirection,
}

/// Finds the best split for a single feature from its histogram.
///
/// Scans all possible split points (bin boundaries) and returns the best one.
/// This is O(K) where K is the number of bins.
///
/// The algorithm:
/// 1. Compute total gradient/hessian sums across all bins
/// 2. Extract NaN bin statistics (last bin if present)
/// 3. Scan through regular bins, maintaining prefix sums
/// 4. For each split point, evaluate both NaN-goes-left and NaN-goes-right
/// 5. Check min_samples_leaf and monotonicity constraints
/// 6. Track the split with maximum gain
///
/// # Args
///
/// * `histogram` - Histogram for this feature (includes NaN bin as last bin).
/// * `feature_index` - Index of this feature.
/// * `config` - Split configuration (min_samples_leaf, reg_lambda, min_gain).
/// * `n_regular_bins` - Number of regular bins (excluding NaN bin).
/// * `monotonic_constraint` - Monotonicity constraint for this feature.
///
/// # Returns
///
/// * `Ok(Some(SplitResult))` - Best split if one exists meeting all constraints.
/// * `Ok(None)` - No valid split found.
///
/// # Errors
///
/// Returns error if histogram access fails.
pub fn find_best_split_from_histogram(
    histogram: &HistogramBuffer,
    feature_index: usize,
    config: &SplitConfig,
    n_regular_bins: usize,
    monotonic_constraint: MonotonicConstraint,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let n_bins = histogram.n_bins();

    // Validate n_regular_bins
    if n_regular_bins == 0_usize {
        return Ok(None);
    }
    if n_regular_bins > n_bins {
        return Err(ClearGbmError::InvalidParameter {
            name: "n_regular_bins".to_string(),
            reason: format!(
                "n_regular_bins ({n_regular_bins}) cannot exceed histogram n_bins ({n_bins})"
            ),
        });
    }

    // Check if histogram has a NaN bin (last bin beyond regular bins)
    let has_nan_bin = n_bins > n_regular_bins;
    let nan_bin_idx = n_regular_bins;

    // Extract NaN bin statistics.
    //
    // Direct indexing, not the fallible accessors: `bins.len() == n_bins` is
    // a `HistogramBuffer` construction invariant (`new`, serde, subtraction
    // and copy all enforce it), and `nan_bin_idx < n_bins` follows from the
    // `n_regular_bins > n_bins` guard above. Before the accumulators were
    // interleaved the three arrays could disagree in length and each access
    // needed its own error arm; a per-field length disagreement can no
    // longer be constructed, so those arms would be dead code.
    let (g_nan, h_nan, n_nan) = if has_nan_bin {
        let acc = histogram.bins[nan_bin_idx];
        (acc.gradient_sum, acc.hessian_sum, acc.count)
    } else {
        (0.0_f64, 0.0_f64, 0_usize)
    };

    // Compute totals for regular bins and cache bin data for the split search
    let mut g_regular = 0.0_f64;
    let mut h_regular = 0.0_f64;
    let mut n_regular = 0_usize;

    // Cache bin data to avoid redundant histogram access in the split search loop
    let mut bin_data: Vec<(f64, f64, usize)> = Vec::with_capacity(n_regular_bins);

    for acc in histogram.bins.iter().take(n_regular_bins) {
        g_regular += acc.gradient_sum;
        h_regular += acc.hessian_sum;
        n_regular += acc.count;
        bin_data.push((acc.gradient_sum, acc.hessian_sum, acc.count));
    }

    // Total including NaN
    let g_total = g_regular + g_nan;
    let h_total = h_regular + h_nan;
    let n_total = n_regular + n_nan;

    // Early exit if not enough samples to split
    let min_samples_leaf = config.min_samples_leaf();
    if n_total < 2_usize * min_samples_leaf {
        return Ok(None);
    }

    let reg_lambda = config.reg_lambda();
    let min_gain = config.min_gain();

    // Track best split
    let mut best: Option<SplitCandidate> = None;

    // Prefix sums for left side (regular bins only, before adding NaN)
    let mut g_left_base = 0.0_f64;
    let mut h_left_base = 0.0_f64;
    let mut n_left_base = 0_usize;

    // Scan regular bins (split after each bin except the last)
    // After bin i, samples in bins 0..=i go left, bins i+1..n_regular_bins go right
    for (bin_idx, &(g, h, n)) in bin_data
        .iter()
        .enumerate()
        .take(n_regular_bins.saturating_sub(1_usize))
    {
        g_left_base += g;
        h_left_base += h;
        n_left_base += n;

        // Try both NaN directions
        for nan_dir in [NanDirection::Left, NanDirection::Right] {
            // Compute left statistics including NaN if direction is left
            let (g_left, h_left, n_left) = if nan_dir.goes_left() {
                (
                    g_left_base + g_nan,
                    h_left_base + h_nan,
                    n_left_base + n_nan,
                )
            } else {
                (g_left_base, h_left_base, n_left_base)
            };

            let n_right = n_total.saturating_sub(n_left);

            // Check min_samples_leaf constraint
            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }

            let g_right = g_total - g_left;
            let h_right = h_total - h_left;

            // Check monotonicity constraint
            if !check_monotonicity_constraint(
                monotonic_constraint,
                g_left,
                h_left,
                g_right,
                h_right,
            ) {
                continue;
            }

            // Compute gain
            let gain = compute_split_gain(
                g_left, h_left, g_right, h_right, g_total, h_total, reg_lambda,
            );

            // Check min_gain threshold
            if gain <= min_gain {
                continue;
            }

            // Update best if this is better
            let dominated = if let Some(ref current_best) = best {
                gain <= current_best.gain
            } else {
                false
            };

            if !dominated {
                best = Some(SplitCandidate {
                    bin_index: bin_idx,
                    gain,
                    g_left,
                    h_left,
                    n_left,
                    nan_direction: nan_dir,
                });
            }
        }
    }

    // Convert best candidate to SplitResult
    if let Some(candidate) = best {
        let g_right = g_total - candidate.g_left;
        let h_right = h_total - candidate.h_left;
        let n_right = n_total.saturating_sub(candidate.n_left);

        Ok(Some(SplitResult::new(SplitResultConfig {
            feature_index,
            decision: SplitDecision::Threshold {
                split_bin: candidate.bin_index,
            },
            gain: candidate.gain,
            left_gradient_sum: candidate.g_left,
            left_hessian_sum: candidate.h_left,
            left_count: candidate.n_left,
            right_gradient_sum: g_right,
            right_hessian_sum: h_right,
            right_count: n_right,
            nan_direction: candidate.nan_direction,
        })))
    } else {
        Ok(None)
    }
}

/// Finds the best split across multiple features.
///
/// For each feature, builds a histogram and finds the best split, then returns
/// the split with the maximum gain across all features.
///
/// # Args
///
/// * `histograms` - Slice of histograms, one per feature.
/// * `config` - Split configuration.
/// * `n_regular_bins` - Number of regular bins per feature (excluding NaN bin).
/// * `monotonic_constraints` - Optional slice of constraints per feature.
///
/// # Returns
///
/// * `Ok(Some(SplitResult))` - Best split across all features.
/// * `Ok(None)` - No valid split found for any feature.
///
/// # Errors
///
/// Returns error if any histogram access fails.
pub fn find_best_split_across_features(
    histograms: &[HistogramBuffer],
    config: &SplitConfig,
    n_regular_bins: usize,
    monotonic_constraints: Option<&[MonotonicConstraint]>,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let mut best_split: Option<SplitResult> = None;

    for (feature_idx, histogram) in histograms.iter().enumerate() {
        let constraint = monotonic_constraints
            .and_then(|constraints| constraints.get(feature_idx).copied())
            .unwrap_or(MonotonicConstraint::None);

        let maybe_split = match find_best_split_from_histogram(
            histogram,
            feature_idx,
            config,
            n_regular_bins,
            constraint,
        ) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };

        if let Some(split) = maybe_split {
            let is_better = if let Some(ref current_best) = best_split {
                split.gain() > current_best.gain()
            } else {
                true
            };

            if is_better {
                best_split = Some(split);
            }
        }
    }

    Ok(best_split)
}
