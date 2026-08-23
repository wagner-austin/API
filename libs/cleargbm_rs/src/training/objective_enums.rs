//! The two wire enums a config states: the tree growth policy and the
//! training objective. Both spell themselves identically at the Python
//! boundary and in serialized JSON.

use crate::error::ClearGbmError;

/// Tree growth policy: the order in which nodes are chosen for splitting.
///
/// This is a genuine algorithm parameter, not a fallback switch. The two
/// policies build different trees from identical data:
///
/// * [`Self::DepthWise`] expands every node at depth `d` before any node at
///   depth `d + 1`, bounded by `max_depth`.
/// * [`Self::LeafWise`] repeatedly splits whichever leaf promises the largest
///   gain, bounded by a leaf budget (best-first induction, Shi 2007).
///
/// The wire spelling is `"depth_wise"` / `"leaf_wise"` at BOTH boundaries —
/// the Python config dict and the serialized model JSON. Deliberately unlike
/// [`MonotonicConstraint`], which spells itself as ints across pyo3 and as
/// variant names in JSON; one value with two spellings is a trap, not a
/// feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStrategy {
    /// Level-order expansion bounded by `max_depth`.
    DepthWise,
    /// Best-first expansion bounded by `num_leaves`.
    LeafWise,
}

impl GrowthStrategy {
    /// Returns the wire spelling of this policy.
    ///
    /// # Returns
    ///
    /// `"depth_wise"` or `"leaf_wise"`.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::DepthWise => "depth_wise",
            Self::LeafWise => "leaf_wise",
        }
    }

    /// Parses a policy from its wire spelling.
    ///
    /// # Args
    ///
    /// * `value` - `"depth_wise"` or `"leaf_wise"`.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if `value` is neither.
    pub fn from_wire(value: &str) -> Result<Self, ClearGbmError> {
        match value {
            "depth_wise" => Ok(Self::DepthWise),
            "leaf_wise" => Ok(Self::LeafWise),
            other => Err(ClearGbmError::InvalidParameter {
                name: "growth_strategy".to_string(),
                reason: format!("expected \"depth_wise\" or \"leaf_wise\", got {other:?}"),
            }),
        }
    }
}

/// Training objective: the loss whose gradients the trees descend.
///
/// The objective decides four things behind one seam — the base score, the
/// per-round gradients and hessians, the early-stopping evaluation loss, and
/// the prediction transform:
///
/// * [`Self::BinaryLogLoss`] — binary classification. Labels are 0/1, the
///   base score is weighted log-odds, gradients come from the sigmoid of the
///   raw score, and probabilities are read through the sigmoid.
/// * [`Self::SquaredError`] — regression. Labels are continuous, the base
///   score is the label mean, `gradient = prediction - y`, `hessian = 1`,
///   and raw scores are the predictions (identity transform).
///
/// The wire spelling is `"binary_log_loss"` / `"squared_error"` at BOTH
/// boundaries — the Python config dict and the serialized model JSON — the
/// same one-value-one-spelling rule as [`GrowthStrategy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Binary classification under (optionally class-weighted) log loss.
    BinaryLogLoss,
    /// Regression under squared error.
    SquaredError,
    /// K-class classification under softmax cross-entropy: K trees per
    /// boosting round, one score column per class.
    MulticlassSoftmax,
}

impl Objective {
    /// Returns the wire spelling of this objective.
    ///
    /// # Returns
    ///
    /// `"binary_log_loss"`, `"squared_error"` or `"multiclass_softmax"`.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::BinaryLogLoss => "binary_log_loss",
            Self::SquaredError => "squared_error",
            Self::MulticlassSoftmax => "multiclass_softmax",
        }
    }

    /// Parses an objective from its wire spelling.
    ///
    /// # Args
    ///
    /// * `value` - `"binary_log_loss"`, `"squared_error"` or
    ///   `"multiclass_softmax"`.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if `value` is none of them.
    pub fn from_wire(value: &str) -> Result<Self, ClearGbmError> {
        match value {
            "binary_log_loss" => Ok(Self::BinaryLogLoss),
            "squared_error" => Ok(Self::SquaredError),
            "multiclass_softmax" => Ok(Self::MulticlassSoftmax),
            other => Err(ClearGbmError::InvalidParameter {
                name: "objective".to_string(),
                reason: format!(
                    "expected \"binary_log_loss\", \"squared_error\" or \
                     \"multiclass_softmax\", got {other:?}"
                ),
            }),
        }
    }
}
