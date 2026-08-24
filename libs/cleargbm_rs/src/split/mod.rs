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

/// The many-vs-many categorical subset search.
mod categorical;
/// The threshold (numeric) prefix-sum search.
mod threshold;
/// The threshold search over packed integer histograms.
mod threshold_quantized;

#[cfg(test)]
mod tests;

pub use categorical::{find_best_categorical_split_from_histogram, CategoryBinSet};
pub use threshold::{find_best_split_across_features, find_best_split_from_histogram};
pub use threshold_quantized::{find_best_split_from_quantized_histogram, QuantizedScanScales};

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

/// How a split partitions a node's samples.
///
/// Numeric features split on a bin boundary (an ordered threshold);
/// categorical features split on set membership over category bins. An enum
/// rather than an optional side-field so a threshold split can never carry a
/// meaningless bin set and a categorical split can never carry a fabricated
/// threshold bin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitDecision {
    /// Samples with `bin <= split_bin` go left.
    Threshold {
        /// The boundary bin.
        split_bin: usize,
    },
    /// Samples whose bin is in the set go left; everything else goes right.
    CategorySubset {
        /// The bins routed left.
        left_bins: CategoryBinSet,
    },
}

/// Configuration for creating a `SplitResult`.
///
/// Used to avoid having too many function arguments while maintaining
/// explicit, named parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitResultConfig {
    /// Feature index for the split.
    pub feature_index: usize,
    /// How the split partitions samples.
    pub decision: SplitDecision,
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
    /// How the split partitions samples.
    decision: SplitDecision,
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
            decision: config.decision,
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

    /// Returns how the split partitions samples.
    #[must_use]
    pub const fn decision(&self) -> SplitDecision {
        self.decision
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
pub(crate) const EPSILON: f64 = 1e-10_f64;

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
