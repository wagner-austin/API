//! Initial prediction (log-odds) computation for binary classification.

use crate::error::ClearGbmError;

use super::validation::{
    validate_labels, validate_scale_pos_weight, validate_weight_pairing, CLIP_EPS,
};

/// Computes the initial prediction (log-odds of weighted positive rate).
///
/// `initial = log(p_positive / (1 - p_positive))`
///
/// where `p_positive = (spw * W_pos) / (spw * W_pos + W_neg)`, `spw` is
/// `scale_pos_weight`, and `W_pos` / `W_neg` are the per-row sample-weight
/// sums over positives and negatives — the constant score minimizing the
/// weighted log loss the training gradients descend, LightGBM's
/// boost-from-average under weights.
///
/// The class factor deliberately stays a single closed-form multiply
/// (`spw * W_pos`), never distributed into the per-row accumulation: with
/// `sample_weight = None` the sums count in exact integer-valued `f64`
/// increments and the expression is bit-identical to the historical
/// class-weighted (and, at `spw = 1.0`, unweighted) log-odds.
///
/// # Args
///
/// * `y_true` - True binary labels (0 or 1).
/// * `scale_pos_weight` - Weight applied to positive samples; must be
///   finite and positive.
/// * `sample_weight` - Optional per-row weights (finite, > 0); `None`
///   weighs every row 1.
///
/// # Returns
///
/// Initial prediction in log-odds space.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `y_true` is empty.
/// * `ClearGbmError::ShapeMismatch` if `sample_weight` length differs.
/// * `ClearGbmError::InvalidLabel` if any label is not 0 or 1.
/// * `ClearGbmError::InvalidParameter` if all labels are 0, all labels are
///   1, `scale_pos_weight` is not a finite positive number, or any sample
///   weight is not finite and positive.
pub fn binary_log_loss_initial_prediction(
    y_true: &[u8],
    scale_pos_weight: f64,
    sample_weight: Option<&[f64]>,
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
    if let Some(w) = sample_weight {
        propagate!(validate_weight_pairing(y_true.len(), w, "sample_weight"));
    }

    let mut n_positive_f64 = 0.0_f64;
    let mut n_negative_f64 = 0.0_f64;
    for (i, &label) in y_true.iter().enumerate() {
        let row_weight = match sample_weight {
            Some(w) => w[i],
            None => 1.0_f64,
        };
        if label == 1_u8 {
            n_positive_f64 += row_weight;
        } else {
            n_negative_f64 += row_weight;
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
