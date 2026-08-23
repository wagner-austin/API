//! Training labels and objective resolution.
//!
//! Labels arrive typed by kind — binary `u8` or continuous `f64` — and the
//! configured [`Objective`] must agree with that kind. [`resolve_objective`]
//! checks the pairing once, at the training entry, and produces a
//! [`ResolvedObjective`] the boosting loop can match on without ever seeing
//! an invalid combination: a binary objective holding continuous labels (or
//! the reverse) is unrepresentable past this point.

use crate::error::ClearGbmError;
use crate::losses::validation::{
    validate_continuous_labels, validate_labels, validate_multiclass_labels,
    validate_weight_pairing,
};

use super::config::Objective;

/// Training or validation labels, typed by kind.
///
/// The caller states what its labels are; [`resolve_objective`] holds that
/// statement against the configured objective. An entry point that accepted
/// one numeric type and reinterpreted it per objective would let a mislabeled
/// dataset train silently — the typed split makes the mismatch an error
/// instead.
#[derive(Debug, Clone, Copy)]
pub enum TrainingLabels<'a> {
    /// Binary classification labels, each 0 or 1.
    Binary(&'a [u8]),
    /// Continuous regression targets, each finite.
    Continuous(&'a [f64]),
    /// Multiclass labels, each a class index `< n_classes`.
    Multiclass(&'a [u32]),
}

impl TrainingLabels<'_> {
    /// Returns the number of labels.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Binary(y) => y.len(),
            Self::Continuous(y) => y.len(),
            Self::Multiclass(y) => y.len(),
        }
    }

    /// Returns `true` if there are no labels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0_usize
    }

    /// Returns the kind name used in pairing error messages.
    #[must_use]
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Binary(_) => "binary (u8) labels",
            Self::Continuous(_) => "continuous (f64) labels",
            Self::Multiclass(_) => "multiclass (u32) labels",
        }
    }
}

/// Validation features paired with their labels and optional weights.
///
/// The group travels as one value so "features without labels" (or the
/// reverse) is unrepresentable in the core — the both-or-neither check lives
/// at the boundary that receives them as separate arguments.
#[derive(Debug, Clone, Copy)]
pub struct ValidationData<'a> {
    /// Validation feature matrix `[n_val_samples][n_features]`.
    pub x: &'a [&'a [f64]],
    /// Validation labels, same kind as the training labels.
    pub y: TrainingLabels<'a>,
    /// Optional per-row weights for the evaluation loss; `None` weighs
    /// every row 1.
    pub weight: Option<&'a [f64]>,
}

/// A resolved validation split: features, labels of the objective's kind,
/// and optional per-row evaluation weights.
pub(crate) struct ResolvedValidation<'a, Y> {
    /// Validation feature matrix `[n_val_samples][n_features]`.
    pub x: &'a [&'a [f64]],
    /// Validation labels, already narrowed to the objective's kind.
    pub y: &'a [Y],
    /// Optional per-row evaluation weights; `None` weighs every row 1.
    pub weight: Option<&'a [f64]>,
}

/// A resolved multiclass training task: labels, class count, weights and
/// the optional validation split, content-validated.
pub(crate) struct ResolvedMulticlass<'a> {
    /// Training labels, each `< n_classes`.
    pub y_train: &'a [u32],
    /// The configured class count.
    pub n_classes: usize,
    /// Optional per-row training weights; `None` weighs every row 1.
    pub weights: Option<&'a [f64]>,
    /// The validation split, when provided.
    pub val: Option<ResolvedValidation<'a, u32>>,
}

/// The resolved training task: either one of the single-score objectives
/// (one score per row, one tree per round) or the multiclass task (K
/// scores per row, K trees per round). The split keeps the single-score
/// boosting loop free of unreachable multiclass arms and vice versa.
pub(crate) enum ResolvedTraining<'a> {
    /// A single-score objective (binary log loss or squared error).
    SingleScore(ResolvedObjective<'a>),
    /// The multiclass softmax task.
    Multiclass(ResolvedMulticlass<'a>),
}

/// The objective with its labels and objective-specific parameters, resolved
/// and content-validated.
///
/// Constructed only by [`resolve_objective`]; every variant carries exactly
/// the data its loss needs — including validation features paired with
/// already-narrowed validation labels — so the boosting loop dispatches with
/// a total match: no unreachable arms, no per-round re-checking.
pub(crate) enum ResolvedObjective<'a> {
    /// Binary log loss over 0/1 labels with a positive-class weight.
    Binary {
        /// Training labels, each 0 or 1.
        y_train: &'a [u8],
        /// Optional per-row training weights; `None` weighs every row 1.
        weights: Option<&'a [f64]>,
        /// The validation split, when provided.
        val: Option<ResolvedValidation<'a, u8>>,
        /// Positive-class weight (1.0 = unweighted).
        scale_pos_weight: f64,
    },
    /// Squared error over continuous targets.
    SquaredError {
        /// Training targets, each finite.
        y_train: &'a [f64],
        /// Optional per-row training weights; `None` weighs every row 1.
        weights: Option<&'a [f64]>,
        /// The validation split, when provided.
        val: Option<ResolvedValidation<'a, f64>>,
    },
}

impl<'a> ResolvedObjective<'a> {
    /// Returns the validation feature matrix, when validation data was
    /// provided — used to size the running validation predictions before
    /// the boosting loop starts.
    pub(crate) fn val_features(&self) -> Option<&'a [&'a [f64]]> {
        match self {
            Self::Binary { val, .. } => val.as_ref().map(|v| v.x),
            Self::SquaredError { val, .. } => val.as_ref().map(|v| v.x),
        }
    }
}

/// Builds the objective/label pairing error for one label argument.
fn pairing_error(
    name: &str,
    objective: Objective,
    expected: &str,
    got: TrainingLabels<'_>,
) -> ClearGbmError {
    ClearGbmError::InvalidParameter {
        name: name.to_string(),
        reason: format!(
            "objective \"{}\" requires {expected}, got {}",
            objective.as_str(),
            got.kind()
        ),
    }
}

/// Resolves the configured objective against typed labels and weights.
///
/// Checks that the label kind matches the objective for both training and
/// validation labels, that the objective-paired `scale_pos_weight` is
/// present exactly under `BinaryLogLoss` (the config constructor enforces
/// the same rule; this function is total over its inputs rather than
/// trusting the caller), that label contents are valid for their kind
/// (0/1 for binary, finite for continuous), and that any per-row weights
/// pair with their labels (matching length; finite, strictly positive
/// values).
///
/// # Args
///
/// * `objective` - The configured training objective.
/// * `scale_pos_weight` - The configured positive-class weight.
/// * `y_train` - Typed training labels.
/// * `sample_weight` - Optional per-row training weights.
/// * `validation` - Validation features paired with typed labels and
///   optional weights, when provided.
///
/// # Errors
///
/// * `ClearGbmError::InvalidParameter` on any objective/label-kind mismatch,
///   on a weight/objective pairing violation, on a non-finite continuous
///   label, or on an invalid sample weight.
/// * `ClearGbmError::ShapeMismatch` if a weight slice's length differs from
///   its labels'.
/// * `ClearGbmError::InvalidLabel` if a binary label is not 0 or 1.
pub(crate) fn resolve_objective<'a>(
    objective: Objective,
    scale_pos_weight: Option<f64>,
    n_classes: Option<usize>,
    y_train: TrainingLabels<'a>,
    sample_weight: Option<&'a [f64]>,
    validation: Option<ValidationData<'a>>,
) -> Result<ResolvedTraining<'a>, ClearGbmError> {
    // Weight pairing is objective-independent: lengths against the labels
    // they accompany, values finite and strictly positive.
    if let Some(w) = sample_weight {
        propagate!(validate_weight_pairing(y_train.len(), w, "sample_weight"));
    }
    if let Some(v) = validation {
        if let Some(w) = v.weight {
            propagate!(validate_weight_pairing(v.y.len(), w, "val_sample_weight"));
        }
    }

    match objective {
        Objective::BinaryLogLoss => {
            let yt = match y_train {
                TrainingLabels::Binary(y) => y,
                TrainingLabels::Continuous(_) | TrainingLabels::Multiclass(_) => {
                    return Err(pairing_error(
                        "y_train",
                        objective,
                        "binary (u8) labels",
                        y_train,
                    ))
                }
            };
            let val = match validation {
                None => None,
                Some(ValidationData {
                    x,
                    y: TrainingLabels::Binary(y),
                    weight,
                }) => Some(ResolvedValidation { x, y, weight }),
                Some(ValidationData {
                    y: labels @ (TrainingLabels::Continuous(_) | TrainingLabels::Multiclass(_)),
                    ..
                }) => {
                    return Err(pairing_error(
                        "y_val",
                        objective,
                        "binary (u8) labels",
                        labels,
                    ))
                }
            };
            let w = match scale_pos_weight {
                Some(w) => w,
                None => {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "scale_pos_weight".to_string(),
                        reason: "must be set when objective is \"binary_log_loss\"".to_string(),
                    })
                }
            };
            propagate!(validate_labels(yt));
            if let Some(v) = &val {
                propagate!(validate_labels(v.y));
            }
            Ok(ResolvedTraining::SingleScore(ResolvedObjective::Binary {
                y_train: yt,
                weights: sample_weight,
                val,
                scale_pos_weight: w,
            }))
        }
        Objective::MulticlassSoftmax => {
            let yt = match y_train {
                TrainingLabels::Multiclass(y) => y,
                TrainingLabels::Binary(_) | TrainingLabels::Continuous(_) => {
                    return Err(pairing_error(
                        "y_train",
                        objective,
                        "multiclass (u32) labels",
                        y_train,
                    ))
                }
            };
            let val = match validation {
                None => None,
                Some(ValidationData {
                    x,
                    y: TrainingLabels::Multiclass(y),
                    weight,
                }) => Some(ResolvedValidation { x, y, weight }),
                Some(ValidationData {
                    y: labels @ (TrainingLabels::Binary(_) | TrainingLabels::Continuous(_)),
                    ..
                }) => {
                    return Err(pairing_error(
                        "y_val",
                        objective,
                        "multiclass (u32) labels",
                        labels,
                    ))
                }
            };
            if let Some(w) = scale_pos_weight {
                return Err(ClearGbmError::InvalidParameter {
                    name: "scale_pos_weight".to_string(),
                    reason: format!(
                        "must be unset when objective is \"multiclass_softmax\", got {w}"
                    ),
                });
            }
            let k = match n_classes {
                Some(k) => k,
                None => {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "n_classes".to_string(),
                        reason: "must be set when objective is \"multiclass_softmax\"".to_string(),
                    })
                }
            };
            propagate!(validate_multiclass_labels(yt, k, "y_train"));
            if let Some(v) = &val {
                propagate!(validate_multiclass_labels(v.y, k, "y_val"));
            }
            Ok(ResolvedTraining::Multiclass(ResolvedMulticlass {
                y_train: yt,
                n_classes: k,
                weights: sample_weight,
                val,
            }))
        }
        Objective::SquaredError => {
            let yt = match y_train {
                TrainingLabels::Continuous(y) => y,
                TrainingLabels::Binary(_) | TrainingLabels::Multiclass(_) => {
                    return Err(pairing_error(
                        "y_train",
                        objective,
                        "continuous (f64) labels",
                        y_train,
                    ))
                }
            };
            let val = match validation {
                None => None,
                Some(ValidationData {
                    x,
                    y: TrainingLabels::Continuous(y),
                    weight,
                }) => Some(ResolvedValidation { x, y, weight }),
                Some(ValidationData {
                    y: labels @ (TrainingLabels::Binary(_) | TrainingLabels::Multiclass(_)),
                    ..
                }) => {
                    return Err(pairing_error(
                        "y_val",
                        objective,
                        "continuous (f64) labels",
                        labels,
                    ))
                }
            };
            if let Some(w) = scale_pos_weight {
                return Err(ClearGbmError::InvalidParameter {
                    name: "scale_pos_weight".to_string(),
                    reason: format!("must be unset when objective is \"squared_error\", got {w}"),
                });
            }
            propagate!(validate_continuous_labels(yt, "y_train"));
            if let Some(v) = &val {
                propagate!(validate_continuous_labels(v.y, "y_val"));
            }
            Ok(ResolvedTraining::SingleScore(
                ResolvedObjective::SquaredError {
                    y_train: yt,
                    weights: sample_weight,
                    val,
                },
            ))
        }
    }
}
