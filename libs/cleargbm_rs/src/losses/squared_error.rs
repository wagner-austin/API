//! Squared-error loss and base score for regression.
//!
//! The squared-error objective is `L = sum[w_i * (y - pred)^2] / sum[w_i]`
//! with optional per-row sample weights (`None` weighs every row 1). Its
//! raw-score derivatives are `gradient = w * (pred - y)` and `hessian = w`,
//! computed inline in the training loop; this module owns the base score
//! and the early-stopping evaluation loss.

use crate::error::ClearGbmError;

use super::validation::{validate_continuous_labels, validate_weight_pairing};

/// Computes the initial prediction for squared-error regression: the
/// weighted mean of the targets.
///
/// The weighted mean is the constant score minimizing weighted squared
/// error, the regression analogue of the log-odds base score for binary
/// log loss. With `sample_weight = None` the numerator adds bare targets
/// and the denominator counts in exact integer-valued `f64` increments —
/// bit-identical to the unweighted mean.
///
/// # Args
///
/// * `y_true` - Continuous regression targets.
/// * `sample_weight` - Optional per-row weights (finite, > 0); `None`
///   weighs every row 1.
///
/// # Returns
///
/// The weighted target mean.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `y_true` is empty.
/// * `ClearGbmError::ShapeMismatch` if `sample_weight` length differs.
/// * `ClearGbmError::InvalidParameter` if any label is not finite or any
///   weight is not finite and positive.
pub fn squared_error_initial_prediction(
    y_true: &[f64],
    sample_weight: Option<&[f64]>,
) -> Result<f64, ClearGbmError> {
    if y_true.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "y_true must not be empty".to_string(),
        });
    }
    match validate_continuous_labels(y_true, "y_true") {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    if let Some(w) = sample_weight {
        propagate!(validate_weight_pairing(y_true.len(), w, "sample_weight"));
    }

    let mut sum = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for (i, &label) in y_true.iter().enumerate() {
        match sample_weight {
            Some(w) => {
                sum += w[i] * label;
                weight_sum += w[i];
            }
            None => {
                sum += label;
                weight_sum += 1.0_f64;
            }
        }
    }
    Ok(sum / weight_sum)
}

/// Computes weighted mean squared error between targets and raw predictions.
///
/// `loss = sum[w_i * (y - pred)^2] / sum[w_i]`
///
/// Raw scores are the predictions under the squared-error objective
/// (identity transform), so this evaluates them directly — no sigmoid, no
/// clipping. With `sample_weight = None` each squared error adds bare and
/// the denominator counts exactly — bit-identical to the unweighted MSE.
///
/// # Args
///
/// * `y_true` - Continuous regression targets.
/// * `y_pred` - Raw predictions.
/// * `sample_weight` - Optional per-row weights (finite, > 0); `None`
///   weighs every row 1.
///
/// # Returns
///
/// Weighted mean squared error.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `y_true` is empty.
/// * `ClearGbmError::ShapeMismatch` if lengths differ.
/// * `ClearGbmError::InvalidParameter` if any label is not finite or any
///   weight is not finite and positive.
pub fn squared_error_loss(
    y_true: &[f64],
    y_pred: &[f64],
    sample_weight: Option<&[f64]>,
) -> Result<f64, ClearGbmError> {
    if y_true.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "y_true must not be empty".to_string(),
        });
    }
    if y_true.len() != y_pred.len() {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("y_pred length {}", y_true.len()),
            got: format!("y_pred length {}", y_pred.len()),
        });
    }
    match validate_continuous_labels(y_true, "y_true") {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    if let Some(w) = sample_weight {
        propagate!(validate_weight_pairing(y_true.len(), w, "sample_weight"));
    }

    let mut total = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for (i, (&label, &pred)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        let diff = label - pred;
        match sample_weight {
            Some(w) => {
                total += w[i] * (diff * diff);
                weight_sum += w[i];
            }
            None => {
                total += diff * diff;
                weight_sum += 1.0_f64;
            }
        }
    }
    Ok(total / weight_sum)
}
