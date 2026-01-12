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

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::ClearGbmError;
use crate::types::{HistogramBuffer, SplitConfig};

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

impl Serialize for NanDirection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Left => serializer.serialize_str("Left"),
            Self::Right => serializer.serialize_str("Right"),
        }
    }
}

/// Visitor for deserializing `NanDirection` from string.
struct NanDirectionVisitor;

impl<'de> Visitor<'de> for NanDirectionVisitor {
    type Value = NanDirection;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("\"Left\" or \"Right\"")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "Left" => Ok(NanDirection::Left),
            "Right" => Ok(NanDirection::Right),
            _ => Err(E::custom(format!("unknown NanDirection variant: {value}"))),
        }
    }
}

impl<'de> Deserialize<'de> for NanDirection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(NanDirectionVisitor)
    }
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

impl Serialize for SplitResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("SplitResult", 10) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("feature_index", &self.feature_index) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("split_bin", &self.split_bin) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("gain", &self.gain) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("left_gradient_sum", &self.left_gradient_sum) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("left_hessian_sum", &self.left_hessian_sum) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("left_count", &self.left_count) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("right_gradient_sum", &self.right_gradient_sum) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("right_hessian_sum", &self.right_hessian_sum) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("right_count", &self.right_count) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("nan_direction", &self.nan_direction) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `SplitResult` deserialization.
enum SplitResultField {
    /// The feature index field.
    FeatureIndex,
    /// The split bin field.
    SplitBin,
    /// The gain field.
    Gain,
    /// The left gradient sum field.
    LeftGradientSum,
    /// The left hessian sum field.
    LeftHessianSum,
    /// The left count field.
    LeftCount,
    /// The right gradient sum field.
    RightGradientSum,
    /// The right hessian sum field.
    RightHessianSum,
    /// The right count field.
    RightCount,
    /// The NaN direction field.
    NanDirection,
}

/// Visitor for deserializing `SplitResultField` from string.
struct SplitResultFieldVisitor;

impl<'de> Visitor<'de> for SplitResultFieldVisitor {
    type Value = SplitResultField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "feature_index" => Ok(SplitResultField::FeatureIndex),
            "split_bin" => Ok(SplitResultField::SplitBin),
            "gain" => Ok(SplitResultField::Gain),
            "left_gradient_sum" => Ok(SplitResultField::LeftGradientSum),
            "left_hessian_sum" => Ok(SplitResultField::LeftHessianSum),
            "left_count" => Ok(SplitResultField::LeftCount),
            "right_gradient_sum" => Ok(SplitResultField::RightGradientSum),
            "right_hessian_sum" => Ok(SplitResultField::RightHessianSum),
            "right_count" => Ok(SplitResultField::RightCount),
            "nan_direction" => Ok(SplitResultField::NanDirection),
            _ => Err(E::unknown_field(value, SPLIT_RESULT_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for SplitResultField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(SplitResultFieldVisitor)
    }
}

/// Field names for `SplitResult` serialization.
const SPLIT_RESULT_FIELDS: &[&str] = &[
    "feature_index",
    "split_bin",
    "gain",
    "left_gradient_sum",
    "left_hessian_sum",
    "left_count",
    "right_gradient_sum",
    "right_hessian_sum",
    "right_count",
    "nan_direction",
];

impl<'de> Deserialize<'de> for SplitResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SplitResultVisitor;

        impl<'de> Visitor<'de> for SplitResultVisitor {
            type Value = SplitResult;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct SplitResult")
            }

            fn visit_map<V>(self, mut map: V) -> Result<SplitResult, V::Error>
            where
                V: de::MapAccess<'de>,
            {
                let mut feature_index = None;
                let mut split_bin = None;
                let mut gain = None;
                let mut left_gradient_sum = None;
                let mut left_hessian_sum = None;
                let mut left_count = None;
                let mut right_gradient_sum = None;
                let mut right_hessian_sum = None;
                let mut right_count = None;
                let mut nan_direction = None;

                loop {
                    let key: Option<SplitResultField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        SplitResultField::FeatureIndex => {
                            feature_index = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::SplitBin => {
                            split_bin = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::Gain => {
                            gain = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::LeftGradientSum => {
                            left_gradient_sum = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::LeftHessianSum => {
                            left_hessian_sum = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::LeftCount => {
                            left_count = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::RightGradientSum => {
                            right_gradient_sum = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::RightHessianSum => {
                            right_hessian_sum = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::RightCount => {
                            right_count = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitResultField::NanDirection => {
                            nan_direction = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

                let feature_index = match feature_index {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("feature_index")),
                };
                let split_bin = match split_bin {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("split_bin")),
                };
                let gain = match gain {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("gain")),
                };
                let left_gradient_sum = match left_gradient_sum {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("left_gradient_sum")),
                };
                let left_hessian_sum = match left_hessian_sum {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("left_hessian_sum")),
                };
                let left_count = match left_count {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("left_count")),
                };
                let right_gradient_sum = match right_gradient_sum {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("right_gradient_sum")),
                };
                let right_hessian_sum = match right_hessian_sum {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("right_hessian_sum")),
                };
                let right_count = match right_count {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("right_count")),
                };
                let nan_direction = match nan_direction {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("nan_direction")),
                };

                Ok(SplitResult {
                    feature_index,
                    split_bin,
                    gain,
                    left_gradient_sum,
                    left_hessian_sum,
                    left_count,
                    right_gradient_sum,
                    right_hessian_sum,
                    right_count,
                    nan_direction,
                })
            }
        }

        deserializer.deserialize_struct("SplitResult", SPLIT_RESULT_FIELDS, SplitResultVisitor)
    }
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

    // Compute totals for regular bins
    let mut g_regular = 0.0_f64;
    let mut h_regular = 0.0_f64;
    let mut n_regular = 0_usize;

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
        let g = match histogram.gradient_sum(bin_idx) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let h = match histogram.hessian_sum(bin_idx) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let n = match histogram.count(bin_idx) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Failing serializer for testing error propagation paths.
    mod failing_serializer {
        use core::fmt::{self, Display};
        use serde::ser::{self, Serialize};

        /// Error type for failing serializer.
        #[derive(Debug)]
        pub struct FailError {
            /// Error message.
            pub message: String,
        }

        impl Display for FailError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.message)
            }
        }

        impl std::error::Error for FailError {}

        impl ser::Error for FailError {
            fn custom<T: Display>(msg: T) -> Self {
                FailError {
                    message: msg.to_string(),
                }
            }
        }

        /// Serializer that fails after N fields.
        pub struct FailAfterN {
            count: usize,
            fail_after: usize,
            fail_on_struct: bool,
        }

        impl FailAfterN {
            pub fn new(fail_after: usize) -> Self {
                FailAfterN {
                    count: 0,
                    fail_after,
                    fail_on_struct: false,
                }
            }

            pub fn fail_on_struct() -> Self {
                FailAfterN {
                    count: 0,
                    fail_after: usize::MAX,
                    fail_on_struct: true,
                }
            }
        }

        /// Struct serializer state.
        pub struct FailAfterNStruct<'a> {
            ser: &'a mut FailAfterN,
        }

        impl<'a> ser::SerializeStruct for FailAfterNStruct<'a> {
            type Ok = ();
            type Error = FailError;

            fn serialize_field<T>(
                &mut self,
                _key: &'static str,
                _value: &T,
            ) -> Result<(), Self::Error>
            where
                T: ?Sized + Serialize,
            {
                self.ser.count += 1;
                if self.ser.count > self.ser.fail_after {
                    Err(FailError {
                        message: "intentional failure".to_string(),
                    })
                } else {
                    Ok(())
                }
            }

            fn end(self) -> Result<Self::Ok, Self::Error> {
                Ok(())
            }
        }

        impl<'a> ser::Serializer for &'a mut FailAfterN {
            type Ok = ();
            type Error = FailError;
            type SerializeSeq = ser::Impossible<(), FailError>;
            type SerializeTuple = ser::Impossible<(), FailError>;
            type SerializeTupleStruct = ser::Impossible<(), FailError>;
            type SerializeTupleVariant = ser::Impossible<(), FailError>;
            type SerializeMap = ser::Impossible<(), FailError>;
            type SerializeStruct = FailAfterNStruct<'a>;
            type SerializeStructVariant = ser::Impossible<(), FailError>;

            fn serialize_bool(self, _v: bool) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_i8(self, _v: i8) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_i16(self, _v: i16) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_i32(self, _v: i32) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_i64(self, _v: i64) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_u8(self, _v: u8) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_u16(self, _v: u16) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_u32(self, _v: u32) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_u64(self, _v: u64) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_f32(self, _v: f32) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_f64(self, _v: f64) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_char(self, _v: char) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_str(self, _v: &str) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_bytes(self, _v: &[u8]) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_none(self) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_unit(self) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_unit_struct(self, _name: &'static str) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_unit_variant(
                self,
                _name: &'static str,
                _idx: u32,
                _variant: &'static str,
            ) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_newtype_struct<T: ?Sized + Serialize>(
                self,
                _name: &'static str,
                _value: &T,
            ) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_newtype_variant<T: ?Sized + Serialize>(
                self,
                _name: &'static str,
                _idx: u32,
                _variant: &'static str,
                _value: &T,
            ) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, FailError> {
                Err(FailError {
                    message: "seq not supported".to_string(),
                })
            }
            fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, FailError> {
                Err(FailError {
                    message: "tuple not supported".to_string(),
                })
            }
            fn serialize_tuple_struct(
                self,
                _name: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeTupleStruct, FailError> {
                Err(FailError {
                    message: "tuple_struct not supported".to_string(),
                })
            }
            fn serialize_tuple_variant(
                self,
                _name: &'static str,
                _idx: u32,
                _variant: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeTupleVariant, FailError> {
                Err(FailError {
                    message: "tuple_variant not supported".to_string(),
                })
            }
            fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, FailError> {
                Err(FailError {
                    message: "map not supported".to_string(),
                })
            }
            fn serialize_struct(
                self,
                _name: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeStruct, FailError> {
                if self.fail_on_struct {
                    Err(FailError {
                        message: "intentional failure on serialize_struct".to_string(),
                    })
                } else {
                    Ok(FailAfterNStruct { ser: self })
                }
            }
            fn serialize_struct_variant(
                self,
                _name: &'static str,
                _idx: u32,
                _variant: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeStructVariant, FailError> {
                Err(FailError {
                    message: "struct_variant not supported".to_string(),
                })
            }
        }
    }

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
        fn to_config(self) -> Result<SplitConfig, ClearGbmError> {
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
    ) -> Result<Option<SplitResult>, ClearGbmError> {
        let config = match params.to_config() {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
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
    ) -> Result<Option<SplitResult>, ClearGbmError> {
        let config = match params.to_config() {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        find_best_split_across_features(histograms, &config, n_regular_bins, constraints)
    }

    // =========================================================================
    // NanDirection tests
    // =========================================================================

    #[test]
    fn test_nan_direction_left() -> Result<(), ClearGbmError> {
        let dir = NanDirection::Left;
        assert!(dir.goes_left());
        assert!(!dir.goes_right());
        Ok(())
    }

    #[test]
    fn test_nan_direction_right() -> Result<(), ClearGbmError> {
        let dir = NanDirection::Right;
        assert!(!dir.goes_left());
        assert!(dir.goes_right());
        Ok(())
    }

    #[test]
    fn test_nan_direction_clone() -> Result<(), ClearGbmError> {
        let dir = NanDirection::Left;
        let cloned = dir;
        assert_eq!(dir, cloned);
        Ok(())
    }

    #[test]
    fn test_nan_direction_serialize_deserialize() -> Result<(), ClearGbmError> {
        let dir = NanDirection::Left;
        let json_str = match serde_json::to_string(&dir) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let parsed: NanDirection = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(parsed, dir);
        Ok(())
    }

    // =========================================================================
    // MonotonicConstraint tests
    // =========================================================================

    #[test]
    fn test_monotonic_constraint_from_int_none() -> Result<(), ClearGbmError> {
        let constraint = match MonotonicConstraint::from_int(0_i32) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        assert!(constraint.is_none());
        assert_eq!(constraint, MonotonicConstraint::None);
        Ok(())
    }

    #[test]
    fn test_monotonic_constraint_from_int_increasing() -> Result<(), ClearGbmError> {
        let constraint = match MonotonicConstraint::from_int(1_i32) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        assert!(!constraint.is_none());
        assert_eq!(constraint, MonotonicConstraint::Increasing);
        Ok(())
    }

    #[test]
    fn test_monotonic_constraint_from_int_decreasing() -> Result<(), ClearGbmError> {
        let constraint = match MonotonicConstraint::from_int(-1_i32) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        assert!(!constraint.is_none());
        assert_eq!(constraint, MonotonicConstraint::Decreasing);
        Ok(())
    }

    #[test]
    fn test_monotonic_constraint_from_int_invalid() -> Result<(), ClearGbmError> {
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
    fn test_split_result_new() -> Result<(), ClearGbmError> {
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
    fn test_split_result_serialize_deserialize() -> Result<(), ClearGbmError> {
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

        let json_str = match serde_json::to_string(&result) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let parsed: SplitResult = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(parsed, result);
        Ok(())
    }

    // =========================================================================
    // compute_split_gain tests
    // =========================================================================

    #[test]
    fn test_compute_split_gain_basic() -> Result<(), ClearGbmError> {
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
    fn test_compute_split_gain_asymmetric() -> Result<(), ClearGbmError> {
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
    fn test_compute_split_gain_with_regularization() -> Result<(), ClearGbmError> {
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
    fn test_compute_split_gain_zero_hessian() -> Result<(), ClearGbmError> {
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
    fn test_check_monotonicity_none() -> Result<(), ClearGbmError> {
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
    fn test_check_monotonicity_increasing_satisfied() -> Result<(), ClearGbmError> {
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
    fn test_check_monotonicity_increasing_violated() -> Result<(), ClearGbmError> {
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
    fn test_check_monotonicity_decreasing_satisfied() -> Result<(), ClearGbmError> {
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
    fn test_check_monotonicity_decreasing_violated() -> Result<(), ClearGbmError> {
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
    fn test_check_monotonicity_near_zero_hessian_left() -> Result<(), ClearGbmError> {
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
    fn test_check_monotonicity_near_zero_hessian_right() -> Result<(), ClearGbmError> {
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
    fn test_check_monotonicity_both_hessians_near_zero() -> Result<(), ClearGbmError> {
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
        assert_eq!(split.split_bin(), 1_usize);
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

    // =========================================================================
    // find_best_split_across_features tests - using inner functions for coverage
    // =========================================================================

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

    // =========================================================================
    // Additional tests for edge case coverage
    // =========================================================================

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
    // These use inner functions with ? to cover both Ok and Err branches
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

    // =========================================================================
    // Serde error path tests - NanDirection
    // =========================================================================

    #[test]
    fn test_nan_direction_deserialize_invalid_value() -> Result<(), ClearGbmError> {
        // Invalid string value
        let json = r#""Invalid""#;
        let result: Result<NanDirection, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_nan_direction_deserialize_wrong_type() -> Result<(), ClearGbmError> {
        // Number instead of string
        let json = r#"123"#;
        let result: Result<NanDirection, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_nan_direction_deserialize_left() -> Result<(), ClearGbmError> {
        let json = r#""Left""#;
        let dir: NanDirection = match serde_json::from_str(json) {
            Ok(d) => d,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert!(matches!(dir, NanDirection::Left));
        Ok(())
    }

    #[test]
    fn test_nan_direction_deserialize_right() -> Result<(), ClearGbmError> {
        let json = r#""Right""#;
        let dir: NanDirection = match serde_json::from_str(json) {
            Ok(d) => d,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert!(matches!(dir, NanDirection::Right));
        Ok(())
    }

    // =========================================================================
    // Serde error path tests - SplitResult
    // =========================================================================

    #[test]
    fn test_split_result_deserialize_missing_field() -> Result<(), ClearGbmError> {
        // Missing nan_direction field
        let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5}"#;
        let result: Result<SplitResult, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_result_deserialize_unknown_field() -> Result<(), ClearGbmError> {
        let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left","extra":123}"#;
        let result: Result<SplitResult, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_result_deserialize_wrong_type() -> Result<(), ClearGbmError> {
        // feature_index should be usize, not string
        let json = r#"{"feature_index":"wrong","split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
        let result: Result<SplitResult, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_result_deserialize_all_fields() -> Result<(), ClearGbmError> {
        let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
        let sr: SplitResult = match serde_json::from_str(json) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(sr.feature_index(), 1_usize);
        assert_eq!(sr.split_bin(), 3_usize);
        assert!((sr.gain() - 0.5_f64).abs() < 1e-10_f64);
        assert_eq!(sr.left_count(), 10_usize);
        assert_eq!(sr.right_count(), 5_usize);
        assert!(matches!(sr.nan_direction(), NanDirection::Left));
        Ok(())
    }

    #[test]
    fn test_split_result_deserialize_with_right_nan() -> Result<(), ClearGbmError> {
        let json = r#"{"feature_index":2,"split_bin":5,"gain":1.0,"left_gradient_sum":2.0,"left_hessian_sum":3.0,"left_count":20,"right_gradient_sum":1.0,"right_hessian_sum":2.0,"right_count":10,"nan_direction":"Right"}"#;
        let sr: SplitResult = match serde_json::from_str(json) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert!(matches!(sr.nan_direction(), NanDirection::Right));
        Ok(())
    }

    #[test]
    fn test_split_result_serialize_roundtrip() -> Result<(), ClearGbmError> {
        let config = SplitResultConfig {
            feature_index: 3_usize,
            split_bin: 7_usize,
            gain: 2.5_f64,
            left_gradient_sum: 10.0_f64,
            left_hessian_sum: 5.0_f64,
            left_count: 100_usize,
            right_gradient_sum: -10.0_f64,
            right_hessian_sum: 5.0_f64,
            right_count: 50_usize,
            nan_direction: NanDirection::Right,
        };
        let sr = SplitResult::new(config);

        let json_str = match serde_json::to_string(&sr) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };

        let parsed: SplitResult = match serde_json::from_str(&json_str) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };

        assert_eq!(parsed.feature_index(), sr.feature_index());
        assert_eq!(parsed.split_bin(), sr.split_bin());
        assert!((parsed.gain() - sr.gain()).abs() < 1e-10_f64);
        assert_eq!(parsed.left_count(), sr.left_count());
        assert_eq!(parsed.right_count(), sr.right_count());
        Ok(())
    }

    #[test]
    fn test_nan_direction_deserialize_from_number() -> Result<(), ClearGbmError> {
        // Try deserializing NanDirection from a number (triggers expecting method)
        let json = r#"123"#;
        let result: Result<NanDirection, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        // The error should mention expected format
        assert!(
            err_msg.contains("Left") || err_msg.contains("Right") || err_msg.contains("string")
        );
        Ok(())
    }

    #[test]
    fn test_nan_direction_deserialize_from_object() -> Result<(), ClearGbmError> {
        // Try deserializing NanDirection from an object
        let json = r#"{"value": "Left"}"#;
        let result: Result<NanDirection, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_nan_direction_deserialize_from_array() -> Result<(), ClearGbmError> {
        // Try deserializing NanDirection from an array
        let json = r#"["Left"]"#;
        let result: Result<NanDirection, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_nan_direction_deserialize_from_bool() -> Result<(), ClearGbmError> {
        // Try deserializing NanDirection from a boolean
        let json = r#"true"#;
        let result: Result<NanDirection, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_nan_direction_deserialize_from_null() -> Result<(), ClearGbmError> {
        // Try deserializing NanDirection from null
        let json = r#"null"#;
        let result: Result<NanDirection, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_result_deserialize_from_array() -> Result<(), ClearGbmError> {
        // Try deserializing SplitResult from an array (triggers expecting)
        let json = r#"[1, 2, 3]"#;
        let result: Result<SplitResult, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_result_deserialize_from_string() -> Result<(), ClearGbmError> {
        // Try deserializing SplitResult from a string
        let json = r#""not a struct""#;
        let result: Result<SplitResult, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_result_deserialize_from_number() -> Result<(), ClearGbmError> {
        // Try deserializing SplitResult from a number
        let json = r#"42"#;
        let result: Result<SplitResult, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    // Serialization error path tests using failing serializer

    #[test]
    fn test_split_result_serialize_fail_each_field() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let config = SplitResultConfig {
            feature_index: 0_usize,
            split_bin: 5_usize,
            gain: 0.5_f64,
            left_gradient_sum: -1.0_f64,
            left_hessian_sum: 2.0_f64,
            left_count: 50_usize,
            right_gradient_sum: 1.0_f64,
            right_hessian_sum: 2.0_f64,
            right_count: 50_usize,
            nan_direction: NanDirection::Left,
        };
        let sr = SplitResult::new(config);
        // SplitResult has 10 fields
        for fail_at in 0_usize..10_usize {
            let mut ser = FailAfterN::new(fail_at);
            let result = sr.serialize(&mut ser);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_split_result_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let config = SplitResultConfig {
            feature_index: 0_usize,
            split_bin: 5_usize,
            gain: 0.5_f64,
            left_gradient_sum: -1.0_f64,
            left_hessian_sum: 2.0_f64,
            left_count: 50_usize,
            right_gradient_sum: 1.0_f64,
            right_hessian_sum: 2.0_f64,
            right_count: 50_usize,
            nan_direction: NanDirection::Left,
        };
        let sr = SplitResult::new(config);
        let mut ser = FailAfterN::fail_on_struct();
        let result = sr.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_failing_serializer_coverage() -> Result<(), ClearGbmError> {
        use failing_serializer::{FailAfterN, FailError};
        use serde::ser::{Error, SerializeStruct, Serializer};

        // Test FailError Display
        let err = FailError {
            message: "test".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("test"));

        // Test FailError custom
        let custom_err = FailError::custom("custom error");
        assert!(custom_err.message.contains("custom"));

        // Test all serializer primitive methods
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_bool(true).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_i8(1_i8).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_i16(1_i16).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_i32(1_i32).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_i64(1_i64).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_u8(1_u8).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_u16(1_u16).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_u32(1_u32).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_u64(1_u64).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_f32(1.0_f32).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_f64(1.0_f64).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_char('a').is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_str("test").is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_bytes(&[1_u8, 2_u8]).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_none().is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_some(&1_u32).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_unit().is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_unit_struct("Unit").is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_unit_variant("E", 0, "V").is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_newtype_struct("N", &1_u32).is_ok());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser)
            .serialize_newtype_variant("E", 0, "V", &1_u32)
            .is_ok());

        // Test error methods
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_seq(Some(1)).is_err());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_tuple(1).is_err());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_tuple_struct("T", 1).is_err());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_tuple_variant("E", 0, "V", 1).is_err());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_map(Some(1)).is_err());
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_struct_variant("E", 0, "V", 1).is_err());

        // Test serialize_struct
        let mut ser = FailAfterN::new(100);
        let struct_ser = (&mut ser).serialize_struct("S", 1);
        assert!(struct_ser.is_ok());

        // Test struct end
        let mut ser = FailAfterN::new(100);
        let struct_ser = match (&mut ser).serialize_struct("Test", 0) {
            Ok(s) => s,
            Err(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "failed".to_string(),
                })
            }
        };
        assert!(struct_ser.end().is_ok());

        // Test struct serialize_field Ok then Err
        let mut ser = FailAfterN::new(1);
        let mut struct_ser = match (&mut ser).serialize_struct("Test", 2) {
            Ok(s) => s,
            Err(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "failed".to_string(),
                })
            }
        };
        assert!(struct_ser.serialize_field("f1", &1_u32).is_ok());
        assert!(struct_ser.serialize_field("f2", &2_u32).is_err());

        Ok(())
    }
}
