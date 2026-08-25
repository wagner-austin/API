//! Validation rules for [`super::config::GradientBoostingConfig`].
//!
//! Split from `config.rs` so the struct + accessors and the rulebook
//! each stay within the file-size discipline. Every rule here is the
//! constitution's config-honesty invariant in executable form: a field
//! is either honored by training or refused at construction — never
//! carried decoratively.

use crate::error::ClearGbmError;

use super::config::GradientBoostingConfigParams;
use super::objective_enums::{GrowthStrategy, Objective};

/// Validates the scalar hyperparameter ranges.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` naming the first field out
/// of range.
pub(super) fn validate_scalar_ranges(
    params: &GradientBoostingConfigParams,
) -> Result<(), ClearGbmError> {
    if params.n_estimators < 1_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "n_estimators".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }
    if params.max_depth < 1_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "max_depth".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }
    let learning_rate = params.learning_rate;
    if learning_rate <= 0.0_f64 || learning_rate > 1.0_f64 {
        return Err(ClearGbmError::InvalidParameter {
            name: "learning_rate".to_string(),
            reason: format!("must be in (0.0, 1.0], got {learning_rate}"),
        });
    }
    if params.min_samples_split < 2_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "min_samples_split".to_string(),
            reason: "must be >= 2".to_string(),
        });
    }
    if params.min_samples_leaf < 1_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "min_samples_leaf".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }
    if params.max_bins < 2_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "max_bins".to_string(),
            reason: "must be >= 2".to_string(),
        });
    }
    // Bin indices are packed into u8 for cache-line density
    // (see FeatureBins storage layout). Enforce the u8 upper bound
    // here so downstream code can rely on it without another check.
    let max_bins = params.max_bins;
    if max_bins > 255_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "max_bins".to_string(),
            reason: format!("must be <= 255 (u8 bin index), got {max_bins}"),
        });
    }
    let subsample = params.subsample;
    if subsample <= 0.0_f64 || subsample > 1.0_f64 {
        return Err(ClearGbmError::InvalidParameter {
            name: "subsample".to_string(),
            reason: format!("must be in (0.0, 1.0], got {subsample}"),
        });
    }
    let reg_alpha = params.reg_alpha;
    if reg_alpha < 0.0_f64 {
        return Err(ClearGbmError::InvalidParameter {
            name: "reg_alpha".to_string(),
            reason: format!("must be >= 0.0, got {reg_alpha}"),
        });
    }
    let reg_lambda = params.reg_lambda;
    if reg_lambda < 0.0_f64 {
        return Err(ClearGbmError::InvalidParameter {
            name: "reg_lambda".to_string(),
            reason: format!("must be >= 0.0, got {reg_lambda}"),
        });
    }
    if let Some(rounds) = params.early_stopping_rounds {
        if rounds < 1_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "early_stopping_rounds".to_string(),
                reason: "must be >= 1 when set".to_string(),
            });
        }
    }
    Ok(())
}

/// Validates the objective-paired fields (`scale_pos_weight`,
/// `n_classes`, `lambdarank_truncation_level`).
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` for a pairing violation.
pub(super) fn validate_objective_pairings(
    params: &GradientBoostingConfigParams,
) -> Result<(), ClearGbmError> {
    let objective = params.objective;
    // `scale_pos_weight` is paired with the objective rather than merely
    // ignored under the wrong one — same rule as `num_leaves`. A class
    // weight silently doing nothing under squared error would be a
    // config field training does not honour.
    match (objective, params.scale_pos_weight) {
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
    match (objective, params.n_classes) {
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
        (Objective::BinaryLogLoss | Objective::SquaredError | Objective::LambdaRank, Some(k)) => {
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
    match (objective, params.lambdarank_truncation_level) {
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
    Ok(())
}

/// Validates the GOSS rate pair and its exclusivity with `subsample`.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` for a pairing or range
/// violation.
pub(super) fn validate_goss(params: &GradientBoostingConfigParams) -> Result<(), ClearGbmError> {
    // GOSS: both rates travel together, each in (0, 1), summing to
    // at most 1, and GOSS replaces row subsampling outright — a run
    // that stated both `subsample < 1` and GOSS would be sampling
    // rows twice under two different laws.
    match (params.goss_top_rate, params.goss_other_rate) {
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
            let subsample = params.subsample;
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
    Ok(())
}

/// Validates the feature-axis knobs (`max_features`,
/// `categorical_features`, `colsample_bytree`).
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` for a range or shape
/// violation.
pub(super) fn validate_feature_axes(
    params: &GradientBoostingConfigParams,
) -> Result<(), ClearGbmError> {
    if let Some(k) = params.max_features {
        if k < 1_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "max_features".to_string(),
                reason: "must be >= 1 when set".to_string(),
            });
        }
    }
    if let Some(ref indices) = params.categorical_features {
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
    // A fraction of exactly 1.0 is every feature, which already has a
    // spelling: null. Two spellings of one behavior is how a config
    // stops being self-describing, so the boundary is exclusive.
    if let Some(f) = params.colsample_bytree {
        if !f.is_finite() || f <= 0.0_f64 || f >= 1.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "colsample_bytree".to_string(),
                reason: format!(
                    "must be in (0.0, 1.0) exclusive when set (null = all features), got {f}"
                ),
            });
        }
    }
    Ok(())
}

/// Validates the growth-policy pairing (`num_leaves`).
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` for a pairing violation.
pub(super) fn validate_growth(params: &GradientBoostingConfigParams) -> Result<(), ClearGbmError> {
    // `num_leaves` is paired with the policy rather than merely ignored
    // under the wrong one. A leaf budget silently doing nothing under
    // depth-wise growth is the same class of defect as a missing
    // growth_strategy: the run reports a knob it did not honour.
    match (params.growth_strategy, params.num_leaves) {
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
    Ok(())
}

/// Validates `quantized_gradient_bins` and its interactions.
///
/// The knob is honest end to end: even (gradients get exactly `bins/2`
/// per side — an odd count would silently train under `bins - 1`),
/// bounded so values pack into `int8` (hessians span `[0, bins]`),
/// limited to the single-score objectives the quantized loop implements,
/// and exclusive with categorical features, whose subset scan has no
/// packed integer form here yet.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` for a range, parity,
/// objective, or categorical violation.
pub(super) fn validate_quantized(
    params: &GradientBoostingConfigParams,
) -> Result<(), ClearGbmError> {
    let Some(bins) = params.quantized_gradient_bins else {
        return Ok(());
    };
    if !(2_usize..=126_usize).contains(&bins) {
        return Err(ClearGbmError::InvalidParameter {
            name: "quantized_gradient_bins".to_string(),
            reason: format!(
                "must be in [2, 126] (quantized values pack into int8; hessians span \
                 [0, bins]), got {bins}"
            ),
        });
    }
    if bins % 2_usize != 0_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "quantized_gradient_bins".to_string(),
            reason: format!(
                "must be even (gradients get bins/2 per side; an odd count would train \
                 under {} bins, not the {bins} the config states)",
                bins - 1_usize
            ),
        });
    }
    match params.objective {
        Objective::BinaryLogLoss | Objective::SquaredError => {}
        Objective::MulticlassSoftmax | Objective::LambdaRank => {
            return Err(ClearGbmError::InvalidParameter {
                name: "quantized_gradient_bins".to_string(),
                reason: format!(
                    "must be unset when objective is \"{}\"; quantized training is \
                     implemented for the single-score objectives (\"binary_log_loss\", \
                     \"squared_error\") only",
                    params.objective.as_str()
                ),
            })
        }
    }
    if params.categorical_features.is_some() {
        return Err(ClearGbmError::InvalidParameter {
            name: "quantized_gradient_bins".to_string(),
            reason: "must be unset when categorical_features is set; the categorical \
                     subset scan has no packed integer histogram form"
                .to_string(),
        });
    }
    Ok(())
}

/// Validates `min_data_in_bin`.
///
/// When set, the floor must be at least 2: a floor of 1 is exactly the
/// unset behavior (every distinct value may hold its own bin), and two
/// spellings of one behavior would make configs lie about themselves.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` for a floor below 2.
pub(super) fn validate_min_data_in_bin(
    params: &GradientBoostingConfigParams,
) -> Result<(), ClearGbmError> {
    let Some(floor) = params.min_data_in_bin else {
        return Ok(());
    };
    if floor < 2_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "min_data_in_bin".to_string(),
            reason: format!(
                "must be >= 2 when set (a floor of {floor} is the unset behavior; \
                 use null instead of a second spelling for it)"
            ),
        });
    }
    Ok(())
}
