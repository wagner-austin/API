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

use serde::{Deserialize, Serialize};

use crate::error::ClearGbmError;
use crate::types::{HistogramBuffer, SplitConfig};

/// Direction for NaN values during tree traversal.
///
/// When a feature value is NaN, this determines which child node to visit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
        matches!(self, Self::Left)
    }

    /// Returns `true` if NaN values go right.
    #[must_use]
    pub const fn goes_right(&self) -> bool {
        matches!(self, Self::Right)
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub fn from_int(value: i32) -> std::result::Result<Self, ClearGbmError> {
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
        matches!(self, Self::None)
    }

    /// Returns `true` if this is an increasing constraint.
    #[must_use]
    pub const fn is_increasing(&self) -> bool {
        matches!(self, Self::Increasing)
    }

    /// Returns `true` if this is a decreasing constraint.
    #[must_use]
    pub const fn is_decreasing(&self) -> bool {
        matches!(self, Self::Decreasing)
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
    // Add L2 regularization to hessian sums
    let h_left_reg = h_left + reg_lambda;
    let h_right_reg = h_right + reg_lambda;
    let h_total_reg = h_total + reg_lambda;

    // Avoid division by zero
    if h_left_reg.abs() < EPSILON || h_right_reg.abs() < EPSILON || h_total_reg.abs() < EPSILON {
        return 0.0_f64;
    }

    let score_left = (g_left * g_left) / h_left_reg;
    let score_right = (g_right * g_right) / h_right_reg;
    let score_total = (g_total * g_total) / h_total_reg;

    score_left + score_right - score_total
}

/// Checks if a split satisfies a monotonicity constraint.
///
/// For an increasing constraint, the left child's prediction must be <= right child's.
/// For a decreasing constraint, the left child's prediction must be >= right child's.
///
/// Leaf values are computed as `-G/H`, so:
/// - Left value = `-g_left / h_left`
/// - Right value = `-g_right / h_right`
///
/// # Args
///
/// * `constraint` - The monotonicity constraint to check.
/// * `g_left` - Sum of gradients in left child.
/// * `h_left` - Sum of hessians in left child.
/// * `g_right` - Sum of gradients in right child.
/// * `h_right` - Sum of hessians in right child.
///
/// # Returns
///
/// `true` if the constraint is satisfied, `false` otherwise.
#[must_use]
pub fn check_monotonicity_constraint(
    constraint: MonotonicConstraint,
    g_left: f64,
    h_left: f64,
    g_right: f64,
    h_right: f64,
) -> bool {
    // No constraint means always satisfied
    if constraint.is_none() {
        return true;
    }

    // Compute leaf values: -G/H (with epsilon for numerical stability)
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

    let left_value = -g_left / h_left_safe;
    let right_value = -g_right / h_right_safe;

    // Check constraint (None already returned above)
    constraint.is_increasing() && left_value <= right_value
        || constraint.is_decreasing() && left_value >= right_value
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
) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
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
        (
            histogram.gradient_sum(nan_bin_idx)?,
            histogram.hessian_sum(nan_bin_idx)?,
            histogram.count(nan_bin_idx)?,
        )
    } else {
        (0.0_f64, 0.0_f64, 0_usize)
    };

    // Compute totals for regular bins
    let mut g_regular = 0.0_f64;
    let mut h_regular = 0.0_f64;
    let mut n_regular = 0_usize;

    for i in 0_usize..n_regular_bins {
        g_regular += histogram.gradient_sum(i)?;
        h_regular += histogram.hessian_sum(i)?;
        n_regular += histogram.count(i)?;
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
    for bin_idx in 0_usize..(n_regular_bins.saturating_sub(1_usize)) {
        g_left_base += histogram.gradient_sum(bin_idx)?;
        h_left_base += histogram.hessian_sum(bin_idx)?;
        n_left_base += histogram.count(bin_idx)?;

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
) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
    let mut best_split: Option<SplitResult> = None;

    for (feature_idx, histogram) in histograms.iter().enumerate() {
        let constraint = monotonic_constraints
            .and_then(|constraints| constraints.get(feature_idx).copied())
            .unwrap_or(MonotonicConstraint::None);

        if let Some(split) = find_best_split_from_histogram(
            histogram,
            feature_idx,
            config,
            n_regular_bins,
            constraint,
        )? {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Boxed error type for tests with multiple error sources.
    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    // =========================================================================
    // Shared test helpers - called by both success and error tests
    // This ensures both Ok and Err branches of ? are covered
    // =========================================================================

    /// Parameters for creating a `SplitConfig` in tests.
    ///
    /// Used to avoid too_many_arguments lint in test helper functions.
    #[derive(Debug, Clone, Copy)]
    struct TestSplitParams {
        /// Minimum samples required to split a node.
        min_samples_split: usize,
        /// Minimum samples required in each leaf.
        min_samples_leaf: usize,
        /// Maximum number of bins.
        max_bins: usize,
        /// L2 regularization parameter.
        reg_lambda: f64,
        /// Minimum gain required for a split.
        min_gain: f64,
    }

    impl TestSplitParams {
        /// Creates a `SplitConfig` from these parameters.
        fn to_config(self) -> std::result::Result<SplitConfig, ClearGbmError> {
            SplitConfig::new(
                self.min_samples_split,
                self.min_samples_leaf,
                self.max_bins,
                self.reg_lambda,
                self.min_gain,
            )
        }
    }

    /// Helper for find_best_split_from_histogram tests with full config params.
    /// Called with both valid and invalid inputs to cover ? branches.
    fn helper_find_split_with_config(
        histogram: &HistogramBuffer,
        feature_index: usize,
        n_regular_bins: usize,
        constraint: MonotonicConstraint,
        params: TestSplitParams,
    ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
        let config = params.to_config()?;
        find_best_split_from_histogram(
            histogram,
            feature_index,
            &config,
            n_regular_bins,
            constraint,
        )
    }

    /// Helper for find_best_split_across_features tests with full config params.
    fn helper_find_split_across_with_config(
        histograms: &[HistogramBuffer],
        n_regular_bins: usize,
        constraints: Option<&[MonotonicConstraint]>,
        params: TestSplitParams,
    ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
        let config = params.to_config()?;
        find_best_split_across_features(histograms, &config, n_regular_bins, constraints)
    }

    // =========================================================================
    // NanDirection tests
    // =========================================================================

    #[test]
    fn test_nan_direction_left() -> TestResult {
        let dir = NanDirection::Left;
        assert!(dir.goes_left());
        assert!(!dir.goes_right());
        Ok(())
    }

    #[test]
    fn test_nan_direction_right() -> TestResult {
        let dir = NanDirection::Right;
        assert!(!dir.goes_left());
        assert!(dir.goes_right());
        Ok(())
    }

    #[test]
    fn test_nan_direction_clone() -> TestResult {
        let dir = NanDirection::Left;
        let cloned = dir;
        assert_eq!(dir, cloned);
        Ok(())
    }

    #[test]
    fn test_nan_direction_serialize_deserialize() -> TestResult {
        let dir = NanDirection::Left;
        let json_str = serde_json::to_string(&dir)?;
        let parsed: NanDirection = serde_json::from_str(&json_str)?;
        assert_eq!(parsed, dir);
        Ok(())
    }

    // =========================================================================
    // MonotonicConstraint tests
    // =========================================================================

    #[test]
    fn test_monotonic_constraint_from_int_none() -> std::result::Result<(), ClearGbmError> {
        let constraint = MonotonicConstraint::from_int(0_i32)?;
        assert!(constraint.is_none());
        assert_eq!(constraint, MonotonicConstraint::None);
        Ok(())
    }

    #[test]
    fn test_monotonic_constraint_from_int_increasing() -> std::result::Result<(), ClearGbmError> {
        let constraint = MonotonicConstraint::from_int(1_i32)?;
        assert!(!constraint.is_none());
        assert_eq!(constraint, MonotonicConstraint::Increasing);
        Ok(())
    }

    #[test]
    fn test_monotonic_constraint_from_int_decreasing() -> std::result::Result<(), ClearGbmError> {
        let constraint = MonotonicConstraint::from_int(-1_i32)?;
        assert!(!constraint.is_none());
        assert_eq!(constraint, MonotonicConstraint::Decreasing);
        Ok(())
    }

    #[test]
    fn test_monotonic_constraint_from_int_invalid() -> TestResult {
        let result = MonotonicConstraint::from_int(2_i32);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "monotonic_constraint"
        ));
        Ok(())
    }

    // =========================================================================
    // SplitResult tests
    // =========================================================================

    #[test]
    fn test_split_result_new() -> TestResult {
        let config = SplitResultConfig {
            feature_index: 2_usize,
            split_bin: 5_usize,
            gain: 0.123_f64,
            left_gradient_sum: 1.0_f64,
            left_hessian_sum: 2.0_f64,
            left_count: 50_usize,
            right_gradient_sum: 0.5_f64,
            right_hessian_sum: 1.5_f64,
            right_count: 30_usize,
            nan_direction: NanDirection::Left,
        };
        let result = SplitResult::new(config);

        assert_eq!(result.feature_index(), 2_usize);
        assert_eq!(result.split_bin(), 5_usize);
        assert!((result.gain() - 0.123_f64).abs() < EPSILON);
        assert!((result.left_gradient_sum() - 1.0_f64).abs() < EPSILON);
        assert!((result.left_hessian_sum() - 2.0_f64).abs() < EPSILON);
        assert_eq!(result.left_count(), 50_usize);
        assert!((result.right_gradient_sum() - 0.5_f64).abs() < EPSILON);
        assert!((result.right_hessian_sum() - 1.5_f64).abs() < EPSILON);
        assert_eq!(result.right_count(), 30_usize);
        assert_eq!(result.nan_direction(), NanDirection::Left);
        assert!(result.nan_goes_left());
        Ok(())
    }

    #[test]
    fn test_split_result_serialize_deserialize() -> TestResult {
        let config = SplitResultConfig {
            feature_index: 1_usize,
            split_bin: 3_usize,
            gain: 0.5_f64,
            left_gradient_sum: 2.0_f64,
            left_hessian_sum: 4.0_f64,
            left_count: 100_usize,
            right_gradient_sum: 1.0_f64,
            right_hessian_sum: 2.0_f64,
            right_count: 50_usize,
            nan_direction: NanDirection::Right,
        };
        let result = SplitResult::new(config);

        let json_str = serde_json::to_string(&result)?;
        let parsed: SplitResult = serde_json::from_str(&json_str)?;
        assert_eq!(parsed, result);
        Ok(())
    }

    // =========================================================================
    // compute_split_gain tests
    // =========================================================================

    #[test]
    fn test_compute_split_gain_basic() -> TestResult {
        // Simple case: equal split, should have positive gain
        let gain = compute_split_gain(
            1.0_f64,  // g_left
            10.0_f64, // h_left
            1.0_f64,  // g_right
            10.0_f64, // h_right
            2.0_f64,  // g_total
            20.0_f64, // h_total
            0.0_f64,  // reg_lambda
        );
        // Gain = 1^2/10 + 1^2/10 - 2^2/20 = 0.1 + 0.1 - 0.2 = 0
        assert!(gain.abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_compute_split_gain_asymmetric() -> TestResult {
        // Asymmetric split with clear gain
        let gain = compute_split_gain(
            2.0_f64,  // g_left
            10.0_f64, // h_left
            0.0_f64,  // g_right
            10.0_f64, // h_right
            2.0_f64,  // g_total
            20.0_f64, // h_total
            0.0_f64,  // reg_lambda
        );
        // Gain = 4/10 + 0/10 - 4/20 = 0.4 - 0.2 = 0.2
        assert!((gain - 0.2_f64).abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_compute_split_gain_with_regularization() -> TestResult {
        // With L2 regularization
        let gain = compute_split_gain(
            2.0_f64,  // g_left
            10.0_f64, // h_left
            0.0_f64,  // g_right
            10.0_f64, // h_right
            2.0_f64,  // g_total
            20.0_f64, // h_total
            1.0_f64,  // reg_lambda = 1.0
        );
        // Gain = 4/11 + 0/11 - 4/21 ≈ 0.3636 - 0.1905 ≈ 0.173
        assert!(gain > 0.0_f64);
        assert!(gain < 0.2_f64); // Less than without regularization
        Ok(())
    }

    #[test]
    fn test_compute_split_gain_zero_hessian() -> TestResult {
        // Zero hessian should return 0 gain
        let gain = compute_split_gain(
            1.0_f64, // g_left
            0.0_f64, // h_left = 0
            1.0_f64, // g_right
            1.0_f64, // h_right
            2.0_f64, // g_total
            1.0_f64, // h_total
            0.0_f64, // reg_lambda
        );
        assert!(gain.abs() < EPSILON);
        Ok(())
    }

    // =========================================================================
    // check_monotonicity_constraint tests
    // =========================================================================

    #[test]
    fn test_check_monotonicity_none() -> TestResult {
        let result = check_monotonicity_constraint(
            MonotonicConstraint::None,
            1.0_f64,  // g_left
            10.0_f64, // h_left
            -1.0_f64, // g_right
            10.0_f64, // h_right
        );
        assert!(result); // No constraint, always passes
        Ok(())
    }

    #[test]
    fn test_check_monotonicity_increasing_satisfied() -> TestResult {
        // Left value = -g_left/h_left = -1/10 = -0.1
        // Right value = -g_right/h_right = -(-1)/10 = 0.1
        // -0.1 <= 0.1, so increasing constraint is satisfied
        let result = check_monotonicity_constraint(
            MonotonicConstraint::Increasing,
            1.0_f64,  // g_left
            10.0_f64, // h_left
            -1.0_f64, // g_right
            10.0_f64, // h_right
        );
        assert!(result);
        Ok(())
    }

    #[test]
    fn test_check_monotonicity_increasing_violated() -> TestResult {
        // Left value = -g_left/h_left = -(-1)/10 = 0.1
        // Right value = -g_right/h_right = -(1)/10 = -0.1
        // 0.1 > -0.1, so increasing constraint is violated
        let result = check_monotonicity_constraint(
            MonotonicConstraint::Increasing,
            -1.0_f64, // g_left
            10.0_f64, // h_left
            1.0_f64,  // g_right
            10.0_f64, // h_right
        );
        assert!(!result);
        Ok(())
    }

    #[test]
    fn test_check_monotonicity_decreasing_satisfied() -> TestResult {
        // Left value = -g_left/h_left = -(-1)/10 = 0.1
        // Right value = -g_right/h_right = -(1)/10 = -0.1
        // 0.1 >= -0.1, so decreasing constraint is satisfied
        let result = check_monotonicity_constraint(
            MonotonicConstraint::Decreasing,
            -1.0_f64, // g_left
            10.0_f64, // h_left
            1.0_f64,  // g_right
            10.0_f64, // h_right
        );
        assert!(result);
        Ok(())
    }

    #[test]
    fn test_check_monotonicity_decreasing_violated() -> TestResult {
        // Left value = -g_left/h_left = -(1)/10 = -0.1
        // Right value = -g_right/h_right = -(-1)/10 = 0.1
        // -0.1 < 0.1, so decreasing constraint is violated
        let result = check_monotonicity_constraint(
            MonotonicConstraint::Decreasing,
            1.0_f64,  // g_left
            10.0_f64, // h_left
            -1.0_f64, // g_right
            10.0_f64, // h_right
        );
        assert!(!result);
        Ok(())
    }

    #[test]
    fn test_check_monotonicity_near_zero_hessian_left() -> TestResult {
        // When h_left is very small (< EPSILON), use EPSILON as safe value
        // Left value = -1.0 / EPSILON (large negative)
        // Right value = -(-1.0) / 10.0 = 0.1
        // With increasing constraint: large_neg <= 0.1 should be true
        let result = check_monotonicity_constraint(
            MonotonicConstraint::Increasing,
            1.0_f64,  // g_left
            0.0_f64,  // h_left (near zero)
            -1.0_f64, // g_right
            10.0_f64, // h_right
        );
        assert!(result);
        Ok(())
    }

    #[test]
    fn test_check_monotonicity_near_zero_hessian_right() -> TestResult {
        // When h_right is very small (< EPSILON), use EPSILON as safe value
        // Left value = -1.0 / 10.0 = -0.1
        // Right value = -1.0 / EPSILON (large negative)
        // With increasing constraint: -0.1 <= large_neg should be false
        let result = check_monotonicity_constraint(
            MonotonicConstraint::Increasing,
            1.0_f64,  // g_left
            10.0_f64, // h_left
            1.0_f64,  // g_right
            0.0_f64,  // h_right (near zero)
        );
        assert!(!result);
        Ok(())
    }

    #[test]
    fn test_check_monotonicity_both_hessians_near_zero() -> TestResult {
        // When both hessians are near zero, both use EPSILON
        // Left value = -1.0 / EPSILON, Right value = -(-1.0) / EPSILON
        // With decreasing: -1/EPSILON >= 1/EPSILON should be false
        let result = check_monotonicity_constraint(
            MonotonicConstraint::Decreasing,
            1.0_f64,  // g_left
            0.0_f64,  // h_left (near zero)
            -1.0_f64, // g_right
            0.0_f64,  // h_right (near zero)
        );
        assert!(!result);
        Ok(())
    }

    // =========================================================================
    // find_best_split_from_histogram tests
    // =========================================================================

    // =========================================================================
    // find_best_split_from_histogram tests - using inner functions for coverage
    // =========================================================================

    #[test]
    fn test_find_best_split_simple() -> TestResult {
        // Inner function for full branch coverage of ?
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut histogram = HistogramBuffer::new(4_usize);
            for _ in 0_usize..10_usize {
                histogram.accumulate(0_usize, 0.05_f64, 0.1_f64)?;
            }
            for _ in 0_usize..10_usize {
                histogram.accumulate(1_usize, 0.03_f64, 0.1_f64)?;
            }
            for _ in 0_usize..10_usize {
                histogram.accumulate(2_usize, -0.08_f64, 0.1_f64)?;
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
        let maybe_split = inner(2_usize)?;
        let split = maybe_split.ok_or("expected split")?;
        assert_eq!(split.feature_index(), 0_usize);
        assert!(split.gain() > 0.0_f64);
        assert_eq!(split.split_bin(), 1_usize);
        // Cover Err path (invalid min_samples_split)
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_with_nan_bin() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut histogram = HistogramBuffer::new(4_usize);
            for _ in 0_usize..10_usize {
                histogram.accumulate(0_usize, 0.1_f64, 0.1_f64)?;
            }
            for _ in 0_usize..10_usize {
                histogram.accumulate(1_usize, 0.1_f64, 0.1_f64)?;
            }
            for _ in 0_usize..10_usize {
                histogram.accumulate(2_usize, -0.2_f64, 0.1_f64)?;
            }
            for _ in 0_usize..5_usize {
                histogram.accumulate(3_usize, 0.05_f64, 0.1_f64)?;
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
        let maybe_split = inner(2_usize)?;
        let split = maybe_split.ok_or("expected split")?;
        assert_eq!(split.left_count() + split.right_count(), 35_usize);
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_min_samples_leaf() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut histogram = HistogramBuffer::new(3_usize);
            histogram.accumulate(0_usize, 0.1_f64, 0.1_f64)?;
            histogram.accumulate(0_usize, 0.1_f64, 0.1_f64)?;
            histogram.accumulate(1_usize, -0.1_f64, 0.1_f64)?;
            histogram.accumulate(1_usize, -0.1_f64, 0.1_f64)?;
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
        assert!(inner(2_usize)?.is_none());
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_min_gain_threshold() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut histogram = HistogramBuffer::new(3_usize);
            for _ in 0_usize..10_usize {
                histogram.accumulate(0_usize, 0.01_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                histogram.accumulate(1_usize, 0.01_f64, 1.0_f64)?;
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
        assert!(inner(2_usize)?.is_none());
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_monotonicity_constraint() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut histogram = HistogramBuffer::new(3_usize);
            for _ in 0_usize..10_usize {
                histogram.accumulate(0_usize, -0.5_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                histogram.accumulate(1_usize, 0.5_f64, 1.0_f64)?;
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
        assert!(inner(2_usize)?.is_none());
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_empty_histogram() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
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
        assert!(inner(2_usize)?.is_none());
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_n_regular_bins_exceeds_n_bins() -> TestResult {
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
    fn test_find_best_split_zero_regular_bins() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
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
        assert!(inner(2_usize)?.is_none());
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    // =========================================================================
    // find_best_split_across_features tests - using inner functions for coverage
    // =========================================================================

    #[test]
    fn test_find_best_split_across_features_single() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut histogram = HistogramBuffer::new(3_usize);
            for _ in 0_usize..10_usize {
                histogram.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                histogram.accumulate(1_usize, -0.5_f64, 1.0_f64)?;
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
        let maybe_split = inner(2_usize)?;
        let split = maybe_split.ok_or("expected split")?;
        assert_eq!(split.feature_index(), 0_usize);
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_across_features_multiple() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut hist0 = HistogramBuffer::new(3_usize);
            for _ in 0_usize..10_usize {
                hist0.accumulate(0_usize, 0.1_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                hist0.accumulate(1_usize, -0.1_f64, 1.0_f64)?;
            }
            let mut hist1 = HistogramBuffer::new(3_usize);
            for _ in 0_usize..10_usize {
                hist1.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                hist1.accumulate(1_usize, -0.5_f64, 1.0_f64)?;
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
        let split = inner(2_usize)?.ok_or("expected split")?;
        assert_eq!(split.feature_index(), 1_usize);
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_across_features_with_constraints() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut hist0 = HistogramBuffer::new(3_usize);
            for _ in 0_usize..10_usize {
                hist0.accumulate(0_usize, -0.5_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                hist0.accumulate(1_usize, 0.5_f64, 1.0_f64)?;
            }
            let mut hist1 = HistogramBuffer::new(3_usize);
            for _ in 0_usize..10_usize {
                hist1.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                hist1.accumulate(1_usize, -0.5_f64, 1.0_f64)?;
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
        let split = inner(2_usize)?.ok_or("expected split")?;
        assert_eq!(split.feature_index(), 1_usize);
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_across_features_no_valid_split() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut hist0 = HistogramBuffer::new(3_usize);
            hist0.accumulate(0_usize, 0.1_f64, 1.0_f64)?;
            hist0.accumulate(1_usize, -0.1_f64, 1.0_f64)?;
            let mut hist1 = HistogramBuffer::new(3_usize);
            hist1.accumulate(0_usize, 0.1_f64, 1.0_f64)?;
            hist1.accumulate(1_usize, -0.1_f64, 1.0_f64)?;
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
        assert!(inner(2_usize)?.is_none());
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_across_features_empty() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
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
        assert!(inner(2_usize)?.is_none());
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    // =========================================================================
    // Additional tests for edge case coverage
    // =========================================================================

    #[test]
    fn test_check_monotonicity_zero_hessian() -> TestResult {
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
    fn test_find_best_split_no_nan_bin() -> TestResult {
        fn inner(
            min_samples_split: usize,
        ) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
            let mut histogram = HistogramBuffer::new(3_usize);
            for _ in 0_usize..10_usize {
                histogram.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                histogram.accumulate(1_usize, -0.5_f64, 1.0_f64)?;
            }
            for _ in 0_usize..5_usize {
                histogram.accumulate(2_usize, 0.1_f64, 1.0_f64)?;
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
        assert!(inner(2_usize)?.is_some());
        assert!(inner(0_usize).is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_from_histogram_n_regular_bins_too_large() -> TestResult {
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
    fn test_find_best_split_across_features_error_propagation() -> TestResult {
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
    // These use inner functions with ? to cover both Ok and Err branches
    // =========================================================================

    /// Covers error propagation for find_best_split_from_histogram calls.
    #[test]
    fn test_coverage_find_split_error_propagation() -> TestResult {
        fn inner(n_bins: usize, n_regular_bins: usize) -> std::result::Result<(), ClearGbmError> {
            let mut hist = HistogramBuffer::new(n_bins);
            for _ in 0_usize..10_usize {
                hist.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                hist.accumulate(1_usize, -0.5_f64, 1.0_f64)?;
            }
            let cfg = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
            let _ = find_best_split_from_histogram(
                &hist,
                0_usize,
                &cfg,
                n_regular_bins,
                MonotonicConstraint::None,
            )?;
            Ok(())
        }
        assert!(inner(4_usize, 3_usize).is_ok());
        assert!(inner(3_usize, 100_usize).is_err());
        Ok(())
    }

    /// Covers error propagation for find_best_split_across_features calls.
    #[test]
    fn test_coverage_find_split_across_error_propagation() -> TestResult {
        fn inner(n_regular_bins: usize) -> std::result::Result<(), ClearGbmError> {
            let mut hist = HistogramBuffer::new(3_usize);
            for _ in 0_usize..10_usize {
                hist.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
            }
            for _ in 0_usize..10_usize {
                hist.accumulate(1_usize, -0.5_f64, 1.0_f64)?;
            }
            let cfg = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
            let _ = find_best_split_across_features(&[hist], &cfg, n_regular_bins, None)?;
            Ok(())
        }
        assert!(inner(2_usize).is_ok());
        assert!(inner(100_usize).is_err());
        Ok(())
    }

    /// Covers SplitConfig validation error paths.
    #[test]
    fn test_coverage_split_config_errors() -> TestResult {
        fn inner(min_split: usize) -> std::result::Result<(), ClearGbmError> {
            let _ = SplitConfig::new(min_split, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
            Ok(())
        }
        assert!(inner(2_usize).is_ok());
        assert!(inner(1_usize).is_err());
        Ok(())
    }

    /// Covers ok_or error paths for Option to Result conversion.
    #[test]
    fn test_coverage_option_to_result_conversion() -> TestResult {
        fn inner(find_split: bool) -> std::result::Result<SplitResult, ClearGbmError> {
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
            maybe.ok_or(ClearGbmError::TreeConstructionFailed {
                reason: "no split".to_string(),
            })
        }
        assert!(inner(true).is_ok());
        assert!(inner(false).is_err());
        Ok(())
    }

    /// Covers ? operator propagation on Result types.
    #[test]
    fn test_coverage_question_mark_propagation() -> TestResult {
        /// Inner function that uses ? operator for error propagation.
        fn check_result(
            result: std::result::Result<i32, ClearGbmError>,
        ) -> std::result::Result<i32, ClearGbmError> {
            result?;
            Ok(42_i32)
        }

        // Cover Ok path - ? doesn't trigger, returns Ok(42)
        let ok_input: std::result::Result<i32, ClearGbmError> = Ok(1_i32);
        assert!(check_result(ok_input).is_ok());

        // Cover Err path - ? propagates the error
        let err_input: std::result::Result<i32, ClearGbmError> =
            Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "test".to_string(),
            });
        assert!(matches!(
            check_result(err_input),
            Err(ClearGbmError::InvalidParameter { .. })
        ));

        Ok(())
    }
}
