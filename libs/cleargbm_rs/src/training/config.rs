//! Training configuration for gradient boosting.
//!
//! Validated configuration struct with accessors for all hyperparameters.

use crate::error::ClearGbmError;
use crate::split::MonotonicConstraint;

pub use super::objective_enums::{GrowthStrategy, Objective};

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
    /// Training objective.
    pub objective: Objective,
    /// Weight applied to positive samples in the loss, its gradients and
    /// the base score. Must be `Some(w)` with `w` finite and positive under
    /// `BinaryLogLoss` (`1.0` = unweighted) and `None` under `SquaredError`,
    /// which has no positive class to weight.
    pub scale_pos_weight: Option<f64>,
    /// Features each split may consider (None = all; Some(k) with k >= 1).
    /// The k <= n_features bound is checked at train time where the
    /// feature count is known.
    pub max_features: Option<usize>,
    /// Fraction of features each TREE may consider (None = all; Some(f)
    /// with 0 < f < 1). Some(1.0) is rejected: it would be a second
    /// spelling of "all features". The per-split `max_features` budget
    /// applies within the tree's sampled set.
    pub colsample_bytree: Option<f64>,
    /// Feature indices treated as categorical (None = every feature is
    /// numeric; Some(v) with v non-empty and strictly ascending — the one
    /// canonical spelling of a set). Values must be non-negative integer
    /// codes; splits partition categories by set membership rather than by
    /// threshold. The idx < n_features bound is checked at train time
    /// where the feature count is known, as is the pairing rule that a
    /// categorical feature carries no monotonic constraint.
    pub categorical_features: Option<Vec<usize>>,
    /// Number of classes, paired with the objective: must be `Some(k)` with
    /// `k >= 2` under `MulticlassSoftmax` (each round trains k trees) and
    /// `None` under every other objective, which has no class count to
    /// state.
    pub n_classes: Option<usize>,
    /// NDCG truncation position, paired with the objective: must be
    /// `Some(k)` with `k >= 1` under `LambdaRank` (it bounds the outer pair
    /// loop and the max-DCG normalizer) and `None` under every other
    /// objective, which has no ranking cutoff to state.
    pub lambdarank_truncation_level: Option<usize>,
    /// GOSS top rate: the fraction of rows kept outright by
    /// |gradient x hessian| rank. Paired with `goss_other_rate`
    /// (both-or-neither); each in (0, 1), summing to at most 1. GOSS
    /// replaces row subsampling, so it requires `subsample = 1.0`.
    pub goss_top_rate: Option<f64>,
    /// GOSS other rate: the fraction of the remaining rows sampled and
    /// reweighted by `(1 - top) / other`. See `goss_top_rate`.
    pub goss_other_rate: Option<f64>,
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
    /// Training objective.
    objective: Objective,
    /// Positive-class weight, present exactly when `objective` is
    /// `BinaryLogLoss`.
    scale_pos_weight: Option<f64>,
    /// Features each split may consider (None = all).
    max_features: Option<usize>,
    /// Fraction of features each tree may consider (None = all).
    colsample_bytree: Option<f64>,
    /// Feature indices treated as categorical (None = all numeric).
    categorical_features: Option<Vec<usize>>,
    /// Class count, present exactly when `objective` is `MulticlassSoftmax`.
    n_classes: Option<usize>,
    /// NDCG truncation position, present exactly when `objective` is
    /// `LambdaRank`.
    lambdarank_truncation_level: Option<usize>,
    /// GOSS top rate, paired with `goss_other_rate` (both-or-neither).
    goss_top_rate: Option<f64>,
    /// GOSS other rate, paired with `goss_top_rate` (both-or-neither).
    goss_other_rate: Option<f64>,
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
            objective,
            scale_pos_weight,
            max_features,
            colsample_bytree,
            categorical_features,
            n_classes,
            lambdarank_truncation_level,
            goss_top_rate,
            goss_other_rate,
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
        // `scale_pos_weight` is paired with the objective rather than merely
        // ignored under the wrong one — same rule as `num_leaves` below. A
        // class weight silently doing nothing under squared error would be a
        // config field training does not honour.
        match (objective, scale_pos_weight) {
            (Objective::BinaryLogLoss, None) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "scale_pos_weight".to_string(),
                    reason: "must be set when objective is \"binary_log_loss\"; state 1.0 \
                             explicitly for unweighted training"
                        .to_string(),
                })
            }
            (Objective::BinaryLogLoss, Some(w)) => {
                if !w.is_finite() || w <= 0.0_f64 {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "scale_pos_weight".to_string(),
                        reason: format!("must be a finite positive number, got {w}"),
                    });
                }
            }
            (Objective::SquaredError, Some(w)) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "scale_pos_weight".to_string(),
                    reason: format!(
                        "must be unset when objective is \"squared_error\" (got {w}); squared \
                         error has no positive class to weight"
                    ),
                })
            }
            (Objective::SquaredError, None) => {}
            (Objective::MulticlassSoftmax, Some(w)) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "scale_pos_weight".to_string(),
                    reason: format!(
                        "must be unset when objective is \"multiclass_softmax\" (got {w}); \
                         softmax has no single positive class to weight"
                    ),
                })
            }
            (Objective::MulticlassSoftmax, None) => {}
            (Objective::LambdaRank, Some(w)) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "scale_pos_weight".to_string(),
                    reason: format!(
                        "must be unset when objective is \"lambdarank\" (got {w}); ranking \
                         weighs rows via sample_weight, not a two-class ratio"
                    ),
                })
            }
            (Objective::LambdaRank, None) => {}
        }
        // `n_classes` is paired with the objective the same way: stated
        // exactly when softmax needs it, never carried decoratively.
        match (objective, n_classes) {
            (Objective::MulticlassSoftmax, None) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "n_classes".to_string(),
                    reason: "must be set when objective is \"multiclass_softmax\"".to_string(),
                })
            }
            (Objective::MulticlassSoftmax, Some(k)) => {
                if k < 2_usize {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "n_classes".to_string(),
                        reason: format!("must be >= 2, got {k}"),
                    });
                }
            }
            (
                Objective::BinaryLogLoss | Objective::SquaredError | Objective::LambdaRank,
                Some(k),
            ) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "n_classes".to_string(),
                    reason: format!(
                        "must be unset when objective is \"{}\" (got {k}); only \
                         \"multiclass_softmax\" states a class count",
                        objective.as_str()
                    ),
                })
            }
            (Objective::BinaryLogLoss | Objective::SquaredError | Objective::LambdaRank, None) => {}
        }
        // `lambdarank_truncation_level` is paired with the objective the
        // same way: it bounds the ranking pair loop and the max-DCG
        // normalizer, so it exists exactly when there is a ranking to cut.
        match (objective, lambdarank_truncation_level) {
            (Objective::LambdaRank, None) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "lambdarank_truncation_level".to_string(),
                    reason: "must be set when objective is \"lambdarank\"".to_string(),
                })
            }
            (Objective::LambdaRank, Some(k)) => {
                if k < 1_usize {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "lambdarank_truncation_level".to_string(),
                        reason: format!("must be >= 1, got {k}"),
                    });
                }
            }
            (
                Objective::BinaryLogLoss | Objective::SquaredError | Objective::MulticlassSoftmax,
                Some(k),
            ) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "lambdarank_truncation_level".to_string(),
                    reason: format!(
                        "must be unset when objective is \"{}\" (got {k}); only \
                         \"lambdarank\" states a truncation position",
                        objective.as_str()
                    ),
                })
            }
            (
                Objective::BinaryLogLoss | Objective::SquaredError | Objective::MulticlassSoftmax,
                None,
            ) => {}
        }
        // GOSS: both rates travel together, each in (0, 1), summing to
        // at most 1, and GOSS replaces row subsampling outright — a run
        // that stated both `subsample < 1` and GOSS would be sampling
        // rows twice under two different laws.
        match (goss_top_rate, goss_other_rate) {
            (None, None) => {}
            (Some(_), None) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "goss_other_rate".to_string(),
                    reason: "must be set when goss_top_rate is set (the GOSS rates travel \
                             together)"
                        .to_string(),
                })
            }
            (None, Some(_)) => {
                return Err(ClearGbmError::InvalidParameter {
                    name: "goss_top_rate".to_string(),
                    reason: "must be set when goss_other_rate is set (the GOSS rates travel \
                             together)"
                        .to_string(),
                })
            }
            (Some(top), Some(other)) => {
                if !top.is_finite() || top <= 0.0_f64 || top >= 1.0_f64 {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "goss_top_rate".to_string(),
                        reason: format!("must be in (0.0, 1.0) exclusive, got {top}"),
                    });
                }
                if !other.is_finite() || other <= 0.0_f64 || other >= 1.0_f64 {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "goss_other_rate".to_string(),
                        reason: format!("must be in (0.0, 1.0) exclusive, got {other}"),
                    });
                }
                if top + other > 1.0_f64 {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "goss_top_rate".to_string(),
                        reason: format!(
                            "goss_top_rate + goss_other_rate must be <= 1.0, got {}",
                            top + other
                        ),
                    });
                }
                if subsample < 1.0_f64 {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "subsample".to_string(),
                        reason: format!(
                            "must be 1.0 when GOSS is enabled (got {subsample}); GOSS \
                             replaces row subsampling"
                        ),
                    });
                }
            }
        }
        if let Some(k) = max_features {
            if k < 1_usize {
                return Err(ClearGbmError::InvalidParameter {
                    name: "max_features".to_string(),
                    reason: "must be >= 1 when set".to_string(),
                });
            }
        }
        // A fraction of exactly 1.0 is every feature, which already has a
        // spelling: null. Two spellings of one behavior is how a config
        // stops being self-describing, so the boundary is exclusive.
        if let Some(ref indices) = categorical_features {
            if indices.is_empty() {
                return Err(ClearGbmError::InvalidParameter {
                    name: "categorical_features".to_string(),
                    reason: "must be non-empty when set (null = every feature numeric)".to_string(),
                });
            }
            for pair in indices.windows(2_usize) {
                if pair[1] <= pair[0] {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "categorical_features".to_string(),
                        reason: format!(
                            "must be strictly ascending, got {} after {}",
                            pair[1], pair[0]
                        ),
                    });
                }
            }
        }
        if let Some(f) = colsample_bytree {
            if !f.is_finite() || f <= 0.0_f64 || f >= 1.0_f64 {
                return Err(ClearGbmError::InvalidParameter {
                    name: "colsample_bytree".to_string(),
                    reason: format!(
                        "must be in (0.0, 1.0) exclusive when set (null = all features), got {f}"
                    ),
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
                    reason: "must be set when growth_strategy is \"leaf_wise\"; best-first \
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
            objective,
            scale_pos_weight,
            max_features,
            colsample_bytree,
            categorical_features,
            n_classes,
            lambdarank_truncation_level,
            goss_top_rate,
            goss_other_rate,
        })
    }

    /// Returns the number of boosting iterations.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    /// Returns a copy of this config with `n_estimators` replaced.
    ///
    /// Used by continued training so the continued artifact's config
    /// states the total round budget the combined model trained under.
    /// The caller guarantees `n_estimators >= 1`; every other field is
    /// already validated, so no re-validation runs.
    #[must_use]
    pub(crate) fn with_n_estimators(&self, n_estimators: usize) -> Self {
        let mut updated = self.clone();
        updated.n_estimators = n_estimators;
        updated
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

    /// Returns the training objective.
    #[must_use]
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// Returns the positive-class weight, set exactly under `BinaryLogLoss`.
    #[must_use]
    pub fn scale_pos_weight(&self) -> Option<f64> {
        self.scale_pos_weight
    }

    /// Returns the per-split feature budget (None = all features).
    #[must_use]
    pub fn max_features(&self) -> Option<usize> {
        self.max_features
    }

    /// Returns the per-tree feature fraction (None = all features).
    #[must_use]
    pub fn colsample_bytree(&self) -> Option<f64> {
        self.colsample_bytree
    }

    /// Returns the categorical feature indices (None = all numeric).
    #[must_use]
    pub fn categorical_features(&self) -> Option<&[usize]> {
        self.categorical_features.as_deref()
    }

    /// Returns the class count (present exactly under `MulticlassSoftmax`).
    #[must_use]
    pub fn n_classes(&self) -> Option<usize> {
        self.n_classes
    }

    /// Returns the NDCG truncation position (present exactly under
    /// `LambdaRank`).
    #[must_use]
    pub fn lambdarank_truncation_level(&self) -> Option<usize> {
        self.lambdarank_truncation_level
    }

    /// Returns the GOSS top rate (None = GOSS off).
    #[must_use]
    pub fn goss_top_rate(&self) -> Option<f64> {
        self.goss_top_rate
    }

    /// Returns the GOSS other rate (None = GOSS off).
    #[must_use]
    pub fn goss_other_rate(&self) -> Option<f64> {
        self.goss_other_rate
    }

    /// Returns the leaf budget, set exactly under `LeafWise` growth.
    #[must_use]
    pub fn num_leaves(&self) -> Option<usize> {
        self.num_leaves
    }
}
