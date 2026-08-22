//! Initial prediction (log-odds) computation for binary classification.

use crate::error::ClearGbmError;

use super::validation::{validate_labels, validate_scale_pos_weight, CLIP_EPS};

/// Computes the initial prediction (log-odds of weighted positive rate).
///
/// `initial = log(p_positive / (1 - p_positive))`
///
/// where `p_positive = (w * n_positive) / (w * n_positive + n_negative)`
/// and `w` is `scale_pos_weight` — the constant score minimizing the
/// weighted log loss the training gradients descend, LightGBM's
/// boost-from-average under class weights. At `w = 1.0` the weighted
/// positive count multiplies by exactly `1.0` and both counts are
/// integer-valued `f64`s, so the result is bit-identical to the historical
/// unweighted log-odds.
///
/// # Args
///
/// * `y_true` - True binary labels (0 or 1).
/// * `scale_pos_weight` - Weight applied to positive samples; must be
///   finite and positive.
///
/// # Returns
///
/// Initial prediction in log-odds space.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `y_true` is empty.
/// * `ClearGbmError::InvalidLabel` if any label is not 0 or 1.
/// * `ClearGbmError::InvalidParameter` if all labels are 0, all labels are
///   1, or `scale_pos_weight` is not a finite positive number.
pub fn binary_log_loss_initial_prediction(
    y_true: &[u8],
    scale_pos_weight: f64,
) -> Result<f64, ClearGbmError> {
    if y_true.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "y_true must not be empty".to_string(),
        });
    }
    match validate_labels(y_true) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match validate_scale_pos_weight(scale_pos_weight) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut n_positive_f64 = 0.0_f64;
    let mut n_negative_f64 = 0.0_f64;
    for &label in y_true {
        if label == 1_u8 {
            n_positive_f64 += 1.0_f64;
        } else {
            n_negative_f64 += 1.0_f64;
        }
    }

    let weighted_positive = scale_pos_weight * n_positive_f64;
    let p_positive = weighted_positive / (weighted_positive + n_negative_f64);

    if p_positive < CLIP_EPS {
        return Err(ClearGbmError::InvalidParameter {
            name: "y_true".to_string(),
            reason: "cannot compute initial prediction: all labels are 0".to_string(),
        });
    }
    if p_positive > 1.0_f64 - CLIP_EPS {
        return Err(ClearGbmError::InvalidParameter {
            name: "y_true".to_string(),
            reason: "cannot compute initial prediction: all labels are 1".to_string(),
        });
    }

    Ok((p_positive / (1.0_f64 - p_positive)).ln())
}
