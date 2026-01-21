//! Split finding for gradient boosting trees.
//!
//! Implements O(K) split finding over histogram bins where K is the number of bins.
//! This is the core algorithm that evaluates all possible split points and selects
//! the one with maximum gain.
//!
//! # Algorithm
//!
//! For each potential split point (bin boundary):
//! 1. Compute cumulative gradient/hessian sums for left and right children
//! 2. Evaluate split gain using the formula: `G_L^2/(H_L + λ) + G_R^2/(H_R + λ) - G^2/(H + λ)`
//! 3. Check constraints (min_samples_leaf, monotonicity)
//! 4. Track the split with maximum gain
//!
//! NaN values are handled by evaluating both NaN-goes-left and NaN-goes-right
//! scenarios for each split point.

/// Serde serialization and deserialization implementations.
mod serde_impl;

use crate::error::ClearGbmError;
use crate::types::{HistogramBuffer, SplitConfig};

#[cfg(test)]
mod tests;

/// Direction for NaN values during tree traversal.
///
/// When a feature value is NaN, this determines which child node to visit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanDirection {
    /// NaN values go to the left child.
    Left,
    /// NaN values go to the right child.
    Right,
}

impl NanDirection {
    /// Returns `true` if NaN values go left.
    #[must_use]
    pub const fn goes_left(&self) -> bool {
        match self {
            Self::Left => true,
            Self::Right => false,
        }
    }

    /// Returns `true` if NaN values go right.
    #[must_use]
    pub const fn goes_right(&self) -> bool {
        match self {
            Self::Left => false,
            Self::Right => true,
        }
    }
}

/// Configuration for creating a `SplitResult`.
///
/// Used to avoid having too many function arguments while maintaining
/// explicit, named parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitResultConfig {
    /// Feature index for the split.
    pub feature_index: usize,
    /// Bin index for the split (samples with bin <= split_bin go left).
    pub split_bin: usize,
    /// Gain from this split.
    pub gain: f64,
    /// Sum of gradients going left.
    pub left_gradient_sum: f64,
    /// Sum of hessians going left.
    pub left_hessian_sum: f64,
    /// Count of samples going left.
    pub left_count: usize,
    /// Sum of gradients going right.
    pub right_gradient_sum: f64,
    /// Sum of hessians going right.
    pub right_hessian_sum: f64,
    /// Count of samples going right.
    pub right_count: usize,
    /// Direction for NaN values.
    pub nan_direction: NanDirection,
}

/// Result of a split search.
///
/// Contains all information needed to perform the split and create child nodes.
/// This is immutable after construction.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitResult {
    /// Feature index for the split.
    feature_index: usize,
    /// Bin index for the split (samples with bin <= split_bin go left).
    split_bin: usize,
    /// Gain from this split.
    gain: f64,
    /// Sum of gradients going left.
    left_gradient_sum: f64,
    /// Sum of hessians going left.
    left_hessian_sum: f64,
    /// Count of samples going left.
    left_count: usize,
    /// Sum of gradients going right.
    right_gradient_sum: f64,
    /// Sum of hessians going right.
    right_hessian_sum: f64,
    /// Count of samples going right.
    right_count: usize,
    /// Direction for NaN values.
    nan_direction: NanDirection,
}

impl SplitResult {
    /// Creates a new `SplitResult` from configuration.
    ///
    /// # Args
    ///
    /// * `config` - Configuration containing all split parameters.
    ///
    /// # Returns
    ///
    /// A new `SplitResult`.
    #[must_use]
    pub const fn new(config: SplitResultConfig) -> Self {
        Self {
            feature_index: config.feature_index,
            split_bin: config.split_bin,
            gain: config.gain,
            left_gradient_sum: config.left_gradient_sum,
            left_hessian_sum: config.left_hessian_sum,
            left_count: config.left_count,
            right_gradient_sum: config.right_gradient_sum,
            right_hessian_sum: config.right_hessian_sum,
            right_count: config.right_count,
            nan_direction: config.nan_direction,
        }
    }

    /// Returns the feature index for the split.
    #[must_use]
    pub const fn feature_index(&self) -> usize {
        self.feature_index
    }

    /// Returns the bin index for the split.
    #[must_use]
    pub const fn split_bin(&self) -> usize {
        self.split_bin
    }

    /// Returns the gain from this split.
    #[must_use]
    pub const fn gain(&self) -> f64 {
        self.gain
    }

    /// Returns the sum of gradients going left.
    #[must_use]
    pub const fn left_gradient_sum(&self) -> f64 {
        self.left_gradient_sum
    }

    /// Returns the sum of hessians going left.
    #[must_use]
    pub const fn left_hessian_sum(&self) -> f64 {
        self.left_hessian_sum
    }

    /// Returns the count of samples going left.
    #[must_use]
    pub const fn left_count(&self) -> usize {
        self.left_count
    }

    /// Returns the sum of gradients going right.
    #[must_use]
    pub const fn right_gradient_sum(&self) -> f64 {
        self.right_gradient_sum
    }

    /// Returns the sum of hessians going right.
    #[must_use]
    pub const fn right_hessian_sum(&self) -> f64 {
        self.right_hessian_sum
    }

    /// Returns the count of samples going right.
    #[must_use]
    pub const fn right_count(&self) -> usize {
        self.right_count
    }

    /// Returns the direction for NaN values.
    #[must_use]
    pub const fn nan_direction(&self) -> NanDirection {
        self.nan_direction
    }

    /// Returns whether NaN values go left.
    #[must_use]
    pub const fn nan_goes_left(&self) -> bool {
        self.nan_direction.goes_left()
    }
}

/// Monotonicity constraint for a feature.
///
/// Specifies whether predictions must increase, decrease, or have no constraint
/// as feature values increase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotonicConstraint {
    /// No constraint on prediction direction.
    None,
    /// Predictions must increase as feature values increase.
    Increasing,
    /// Predictions must decrease as feature values increase.
    Decreasing,
}

impl MonotonicConstraint {
    /// Creates a constraint from an integer value.
    ///
    /// # Args
    ///
    /// * `value` - Constraint value: -1 (decreasing), 0 (none), +1 (increasing).
    ///
    /// # Returns
    ///
    /// The corresponding `MonotonicConstraint`.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if value is not -1, 0, or 1.
    pub fn from_int(value: i32) -> Result<Self, ClearGbmError> {
        match value {
            -1_i32 => Ok(Self::Decreasing),
            0_i32 => Ok(Self::None),
            1_i32 => Ok(Self::Increasing),
            _ => Err(ClearGbmError::InvalidParameter {
                name: "monotonic_constraint".to_string(),
                reason: format!("must be -1, 0, or 1, got {value}"),
            }),
        }
    }

    /// Returns `true` if there is no constraint.
    #[must_use]
    pub const fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            Self::Increasing => false,
            Self::Decreasing => false,
        }
    }

    /// Returns `true` if this is an increasing constraint.
    #[must_use]
    pub const fn is_increasing(&self) -> bool {
        match self {
            Self::None => false,
            Self::Increasing => true,
            Self::Decreasing => false,
        }
    }

    /// Returns `true` if this is a decreasing constraint.
    #[must_use]
    pub const fn is_decreasing(&self) -> bool {
        match self {
            Self::None => false,
            Self::Increasing => false,
            Self::Decreasing => true,
        }
    }
}

/// Epsilon value for floating-point comparisons.
const EPSILON: f64 = 1e-10_f64;

/// Computes the gain from a split with L2 regularization.
///
/// The gain formula is:
/// `Gain = G_L^2/(H_L + λ) + G_R^2/(H_R + λ) - G^2/(H + λ)`
///
/// Where:
/// - `G_L`, `H_L` are gradient/hessian sums for left child
/// - `G_R`, `H_R` are gradient/hessian sums for right child
/// - `G`, `H` are total gradient/hessian sums
/// - `λ` is the L2 regularization parameter
///
/// # Args
///
/// * `g_left` - Sum of gradients in left child.
/// * `h_left` - Sum of hessians in left child.
/// * `g_right` - Sum of gradients in right child.
/// * `h_right` - Sum of hessians in right child.
/// * `g_total` - Total sum of gradients.
/// * `h_total` - Total sum of hessians.
/// * `reg_lambda` - L2 regularization parameter.
///
/// # Returns
///
/// Split gain (higher is better). Returns 0.0 if any hessian sum is too small.
#[must_use]
pub fn compute_split_gain(
    g_left: f64,
    h_left: f64,
    g_right: f64,
    h_right: f64,
    g_total: f64,
    h_total: f64,
    reg_lambda: f64,
) -> f64 {
    let h_left_reg = h_left + reg_lambda;
    let h_right_reg = h_right + reg_lambda;
    let h_total_reg = h_total + reg_lambda;

    // Avoid division by zero
    if h_left_reg.abs() < EPSILON || h_right_reg.abs() < EPSILON || h_total_reg.abs() < EPSILON {
        return 0.0_f64;
    }

    let gain_left = (g_left * g_left) / h_left_reg;
    let gain_right = (g_right * g_right) / h_right_reg;
    let gain_total = (g_total * g_total) / h_total_reg;

    // Gain = sum of child gains - parent gain
    // This is always >= 0 for valid splits
    let gain = gain_left + gain_right - gain_total;

    if gain < 0.0_f64 {
        0.0_f64
    } else {
        gain
    }
}

/// Checks if a split satisfies monotonicity constraints.
///
/// For monotonically increasing features: left_value <= right_value
/// For monotonically decreasing features: left_value >= right_value
///
/// Values are computed as -G/H (the optimal leaf value formula without regularization).
///
/// # Args
///
/// * `constraint` - The monotonicity constraint.
/// * `g_left` - Sum of gradients in left child.
/// * `h_left` - Sum of hessians in left child.
/// * `g_right` - Sum of gradients in right child.
/// * `h_right` - Sum of hessians in right child.
///
/// # Returns
///
/// `true` if the constraint is satisfied (or if there is no constraint).
#[must_use]
pub fn check_monotonicity_constraint(
    constraint: MonotonicConstraint,
    g_left: f64,
    h_left: f64,
    g_right: f64,
    h_right: f64,
) -> bool {
    if constraint.is_none() {
        return true;
    }

    // Use EPSILON as floor value for near-zero hessians to avoid division by zero
    let h_left_safe = if h_left.abs() < EPSILON {
        EPSILON
    } else {
        h_left
    };
    let h_right_safe = if h_right.abs() < EPSILON {
        EPSILON
    } else {
        h_right
    };

    // Compute optimal leaf values: -G/H
    let left_value = -g_left / h_left_safe;
    let right_value = -g_right / h_right_safe;

    // At this point constraint is either Increasing or Decreasing (None handled above)
    if constraint.is_increasing() {
        left_value <= right_value + EPSILON
    } else {
        // Must be Decreasing
        left_value >= right_value - EPSILON
    }
}

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

    // Extract NaN bin statistics
    let (g_nan, h_nan, n_nan) = if has_nan_bin {
        let g = match histogram.gradient_sum(nan_bin_idx) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let h = match histogram.hessian_sum(nan_bin_idx) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let n = match histogram.count(nan_bin_idx) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        (g, h, n)
    } else {
        (0.0_f64, 0.0_f64, 0_usize)
    };

    // Compute totals for regular bins and cache bin data for the split search
    let mut g_regular = 0.0_f64;
    let mut h_regular = 0.0_f64;
    let mut n_regular = 0_usize;

    // Cache bin data to avoid redundant histogram access in the split search loop
    let mut bin_data: Vec<(f64, f64, usize)> = Vec::with_capacity(n_regular_bins);

    for i in 0_usize..n_regular_bins {
        let g = match histogram.gradient_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let h = match histogram.hessian_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let n = match histogram.count(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        g_regular += g;
        h_regular += h;
        n_regular += n;
        bin_data.push((g, h, n));
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
            split_bin: candidate.bin_index,
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
