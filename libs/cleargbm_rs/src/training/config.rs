//! Training configuration for gradient boosting.
//!
//! Validated configuration struct with accessors for all hyperparameters.

use crate::error::ClearGbmError;
use crate::split::MonotonicConstraint;

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
}
