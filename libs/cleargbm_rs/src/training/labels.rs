//! Training labels and objective resolution.
//!
//! Labels arrive typed by kind — binary `u8` or continuous `f64` — and the
//! configured [`Objective`] must agree with that kind. [`resolve_objective`]
//! checks the pairing once, at the training entry, and produces a
//! [`ResolvedObjective`] the boosting loop can match on without ever seeing
//! an invalid combination: a binary objective holding continuous labels (or
//! the reverse) is unrepresentable past this point.

use crate::error::ClearGbmError;
use crate::losses::validation::{validate_continuous_labels, validate_labels};

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
}

impl TrainingLabels<'_> {
    /// Returns the number of labels.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Binary(y) => y.len(),
            Self::Continuous(y) => y.len(),
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
        }
    }
}

/// Validation features paired with their labels.
///
/// The pair travels as one value so "features without labels" (or the
/// reverse) is unrepresentable in the core — the both-or-neither check lives
/// at the boundary that receives them as separate arguments.
#[derive(Debug, Clone, Copy)]
pub struct ValidationData<'a> {
    /// Validation feature matrix `[n_val_samples][n_features]`.
    pub x: &'a [&'a [f64]],
    /// Validation labels, same kind as the training labels.
    pub y: TrainingLabels<'a>,
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
        /// Validation features and labels, when provided.
        val: Option<(&'a [&'a [f64]], &'a [u8])>,
        /// Positive-class weight (1.0 = unweighted).
        scale_pos_weight: f64,
    },
    /// Squared error over continuous targets.
    SquaredError {
        /// Training targets, each finite.
        y_train: &'a [f64],
        /// Validation features and targets, when provided.
        val: Option<(&'a [&'a [f64]], &'a [f64])>,
    },
}

impl<'a> ResolvedObjective<'a> {
    /// Returns the validation feature matrix, when validation data was
    /// provided — used to size the running validation predictions before
    /// the boosting loop starts.
    pub(crate) fn val_features(&self) -> Option<&'a [&'a [f64]]> {
        match self {
            Self::Binary { val, .. } => val.map(|(x, _)| x),
            Self::SquaredError { val, .. } => val.map(|(x, _)| x),
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

/// Resolves the configured objective against typed labels.
///
/// Checks that the label kind matches the objective for both training and
/// validation labels, that the objective-paired `scale_pos_weight` is
/// present exactly under `BinaryLogLoss` (the config constructor enforces
/// the same rule; this function is total over its inputs rather than
/// trusting the caller), and that label contents are valid for their kind
/// (0/1 for binary, finite for continuous).
///
/// # Args
///
/// * `objective` - The configured training objective.
/// * `scale_pos_weight` - The configured positive-class weight.
/// * `y_train` - Typed training labels.
/// * `validation` - Validation features paired with typed labels, when
///   provided.
///
/// # Errors
///
/// * `ClearGbmError::InvalidParameter` on any objective/label-kind mismatch,
///   on a weight/objective pairing violation, or on a non-finite continuous
///   label.
/// * `ClearGbmError::InvalidLabel` if a binary label is not 0 or 1.
pub(crate) fn resolve_objective<'a>(
    objective: Objective,
    scale_pos_weight: Option<f64>,
    y_train: TrainingLabels<'a>,
    validation: Option<ValidationData<'a>>,
) -> Result<ResolvedObjective<'a>, ClearGbmError> {
    match objective {
        Objective::BinaryLogLoss => {
            let yt = match y_train {
                TrainingLabels::Binary(y) => y,
                TrainingLabels::Continuous(_) => {
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
                }) => Some((x, y)),
                Some(ValidationData {
                    y: labels @ TrainingLabels::Continuous(_),
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
            if let Some((_, y)) = val {
                propagate!(validate_labels(y));
            }
            Ok(ResolvedObjective::Binary {
                y_train: yt,
                val,
                scale_pos_weight: w,
            })
        }
        Objective::SquaredError => {
            let yt = match y_train {
                TrainingLabels::Continuous(y) => y,
                TrainingLabels::Binary(_) => {
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
                }) => Some((x, y)),
                Some(ValidationData {
                    y: labels @ TrainingLabels::Binary(_),
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
            if let Some((_, y)) = val {
                propagate!(validate_continuous_labels(y, "y_val"));
            }
            Ok(ResolvedObjective::SquaredError { y_train: yt, val })
        }
    }
}
