//! Training configuration for gradient boosting.
//!
//! Validated configuration struct with accessors for all hyperparameters.

use crate::error::ClearGbmError;
use crate::split::MonotonicConstraint;

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

/// Unvalidated parameters for constructing a [`GradientBoostingConfig`].
///
/// Groups all hyperparameters into a single struct to avoid
/// `clippy::too_many_arguments`. Pass to [`GradientBoostingConfig::new`]
/// for validation.
#[derive(Debug, Clone)]
pub struct GradientBoostingConfigParams {
    /// Number of boosting iterations (>= 1).
    pub n_estimators: usize,
    /// Maximum tree depth (>= 1).
    pub max_depth: usize,
    /// Shrinkage factor applied to each tree's contribution (in (0.0, 1.0]).
    pub learning_rate: f64,
    /// Minimum samples required to create a split (>= 2).
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node (>= 1).
    pub min_samples_leaf: usize,
    /// Maximum number of histogram bins per feature (>= 2).
    pub max_bins: usize,
    /// Fraction of training samples used per iteration (in (0.0, 1.0]).
    pub subsample: f64,
    /// Random seed for reproducibility.
    pub random_state: u64,
    /// Per-feature monotonic constraints (None = no constraints).
    pub monotonic_constraints: Option<Vec<MonotonicConstraint>>,
    /// L1 regularization (>= 0.0).
    pub reg_alpha: f64,
    /// L2 regularization (>= 0.0).
    pub reg_lambda: f64,
    /// Early stopping patience (None = disabled, Some(n) where n >= 1).
    pub early_stopping_rounds: Option<usize>,
    /// Tree growth policy.
    pub growth_strategy: GrowthStrategy,
    /// Leaf budget for `LeafWise` growth. Must be `Some(n)` with `n >= 2`
    /// under `LeafWise` and `None` under `DepthWise`.
    pub num_leaves: Option<usize>,
    /// Weight applied to positive samples in the loss, its gradients and
    /// the base score (finite, > 0.0; 1.0 = unweighted).
    pub scale_pos_weight: f64,
    /// Features each split may consider (None = all; Some(k) with k >= 1).
    /// The k <= n_features bound is checked at train time where the
    /// feature count is known.
    pub max_features: Option<usize>,
}

/// Configuration for gradient boosting training.
///
/// All fields are validated at construction time. Matches the Python
/// `decode_gradient_boosting_config` function.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientBoostingConfig {
    /// Number of boosting iterations (>= 1).
    n_estimators: usize,
    /// Maximum tree depth (>= 1).
    max_depth: usize,
    /// Shrinkage factor applied to each tree's contribution (in (0.0, 1.0]).
    learning_rate: f64,
    /// Minimum samples required to create a split (>= 2).
    min_samples_split: usize,
    /// Minimum samples required in a leaf node (>= 1).
    min_samples_leaf: usize,
    /// Maximum number of histogram bins per feature (>= 2).
    max_bins: usize,
    /// Fraction of training samples used per iteration (in (0.0, 1.0]).
    subsample: f64,
    /// Random seed for reproducibility.
    random_state: u64,
    /// Per-feature monotonic constraints (None = no constraints).
    monotonic_constraints: Option<Vec<MonotonicConstraint>>,
    /// L1 regularization (>= 0.0).
    reg_alpha: f64,
    /// L2 regularization (>= 0.0).
    reg_lambda: f64,
    /// Early stopping patience (None = disabled, Some(n) where n >= 1).
    early_stopping_rounds: Option<usize>,
    /// Tree growth policy.
    growth_strategy: GrowthStrategy,
    /// Leaf budget, present exactly when `growth_strategy` is `LeafWise`.
    num_leaves: Option<usize>,
    /// Weight applied to positive samples in the loss, its gradients and
    /// the base score (finite, > 0.0; 1.0 = unweighted).
    scale_pos_weight: f64,
    /// Features each split may consider (None = all).
    max_features: Option<usize>,
}

impl GradientBoostingConfig {
    /// Creates a new validated configuration from parameters.
    ///
    /// # Args
    ///
    /// * `params` - Unvalidated hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if any field is out of range.
    pub fn new(params: GradientBoostingConfigParams) -> Result<Self, ClearGbmError> {
        let GradientBoostingConfigParams {
            n_estimators,
            max_depth,
            learning_rate,
            min_samples_split,
            min_samples_leaf,
            max_bins,
            subsample,
            random_state,
            monotonic_constraints,
            reg_alpha,
            reg_lambda,
            early_stopping_rounds,
            growth_strategy,
            num_leaves,
            scale_pos_weight,
            max_features,
        } = params;

        if n_estimators < 1_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "n_estimators".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if max_depth < 1_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "max_depth".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if learning_rate <= 0.0_f64 || learning_rate > 1.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: format!("must be in (0.0, 1.0], got {learning_rate}"),
            });
        }
        if min_samples_split < 2_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "min_samples_split".to_string(),
                reason: "must be >= 2".to_string(),
            });
        }
        if min_samples_leaf < 1_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "min_samples_leaf".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if max_bins < 2_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "max_bins".to_string(),
                reason: "must be >= 2".to_string(),
            });
        }
        // Bin indices are packed into u8 for cache-line density
        // (see FeatureBins storage layout). Enforce the u8 upper bound
        // here so downstream code can rely on it without another check.
        if max_bins > 255_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "max_bins".to_string(),
                reason: format!("must be <= 255 (u8 bin index), got {max_bins}"),
            });
        }
        if subsample <= 0.0_f64 || subsample > 1.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "subsample".to_string(),
                reason: format!("must be in (0.0, 1.0], got {subsample}"),
            });
        }
        if reg_alpha < 0.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "reg_alpha".to_string(),
                reason: format!("must be >= 0.0, got {reg_alpha}"),
            });
        }
        if reg_lambda < 0.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "reg_lambda".to_string(),
                reason: format!("must be >= 0.0, got {reg_lambda}"),
            });
        }
        if let Some(rounds) = early_stopping_rounds {
            if rounds < 1_usize {
                return Err(ClearGbmError::InvalidParameter {
                    name: "early_stopping_rounds".to_string(),
                    reason: "must be >= 1 when set".to_string(),
                });
            }
        }
        if !scale_pos_weight.is_finite() || scale_pos_weight <= 0.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "scale_pos_weight".to_string(),
                reason: format!("must be a finite positive number, got {scale_pos_weight}"),
            });
        }
        if let Some(k) = max_features {
            if k < 1_usize {
                return Err(ClearGbmError::InvalidParameter {
                    name: "max_features".to_string(),
                    reason: "must be >= 1 when set".to_string(),
                });
            }
        }
        // `num_leaves` is paired with the policy rather than merely ignored
        // under the wrong one. A leaf budget silently doing nothing under
        // depth-wise growth is the same class of defect as a missing
        // growth_strategy: the run reports a knob it did not honour.
        match (growth_strategy, num_leaves) {
            (GrowthStrategy::LeafWise, None) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "num_leaves".to_string(),
                    reason: "must be set when growth_strategy is \"leaf_wise\" — best-first \
                             growth has no depth to bound it, so the leaf budget is its only \
                             capacity limit"
                        .to_string(),
                })
            }
            (GrowthStrategy::LeafWise, Some(budget)) => {
                if budget < 2_usize {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "num_leaves".to_string(),
                        reason: format!("must be >= 2, got {budget}"),
                    });
                }
            }
            (GrowthStrategy::DepthWise, Some(budget)) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "num_leaves".to_string(),
                    reason: format!(
                        "must be unset when growth_strategy is \"depth_wise\" (got {budget}); \
                         depth-wise growth is bounded by max_depth and would ignore it"
                    ),
                })
            }
            (GrowthStrategy::DepthWise, None) => {}
        }

        Ok(Self {
            n_estimators,
            max_depth,
            learning_rate,
            min_samples_split,
            min_samples_leaf,
            max_bins,
            subsample,
            random_state,
            monotonic_constraints,
            reg_alpha,
            reg_lambda,
            early_stopping_rounds,
            growth_strategy,
            num_leaves,
            scale_pos_weight,
            max_features,
        })
    }

    /// Returns the number of boosting iterations.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    /// Returns the maximum tree depth.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns the learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the minimum samples for a split.
    #[must_use]
    pub fn min_samples_split(&self) -> usize {
        self.min_samples_split
    }

    /// Returns the minimum samples in a leaf.
    #[must_use]
    pub fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Returns the maximum number of bins.
    #[must_use]
    pub fn max_bins(&self) -> usize {
        self.max_bins
    }

    /// Returns the subsample fraction.
    #[must_use]
    pub fn subsample(&self) -> f64 {
        self.subsample
    }

    /// Returns the random seed.
    #[must_use]
    pub fn random_state(&self) -> u64 {
        self.random_state
    }

    /// Returns the monotonic constraints (if any).
    #[must_use]
    pub fn monotonic_constraints(&self) -> Option<&[MonotonicConstraint]> {
        self.monotonic_constraints.as_deref()
    }

    /// Returns the L1 regularization parameter.
    #[must_use]
    pub fn reg_alpha(&self) -> f64 {
        self.reg_alpha
    }

    /// Returns the L2 regularization parameter.
    #[must_use]
    pub fn reg_lambda(&self) -> f64 {
        self.reg_lambda
    }

    /// Returns the early stopping patience (if set).
    #[must_use]
    pub fn early_stopping_rounds(&self) -> Option<usize> {
        self.early_stopping_rounds
    }

    /// Returns the tree growth policy.
    #[must_use]
    pub fn growth_strategy(&self) -> GrowthStrategy {
        self.growth_strategy
    }

    /// Returns the positive-class weight applied to the loss and gradients.
    #[must_use]
    pub fn scale_pos_weight(&self) -> f64 {
        self.scale_pos_weight
    }

    /// Returns the per-split feature budget (None = all features).
    #[must_use]
    pub fn max_features(&self) -> Option<usize> {
        self.max_features
    }

    /// Returns the leaf budget, set exactly under `LeafWise` growth.
    #[must_use]
    pub fn num_leaves(&self) -> Option<usize> {
        self.num_leaves
    }
}
