//! Binary cross-entropy (log loss) computation.

use crate::error::ClearGbmError;

use super::validation::{
    validate_labels, validate_lengths, validate_scale_pos_weight, validate_weight_pairing, CLIP_EPS,
};

/// Computes weighted mean binary cross-entropy (log loss).
///
/// `loss = -sum[w_i * (y * log(p) + (1-y) * log(1-p))] / sum[w_i]`
///
/// where the effective row weight `w_i` is the product of the class term
/// (`scale_pos_weight` for positives, `1` for negatives) and the optional
/// per-row sample weight — the same weighting the training gradients
/// carry, so the early-stopping criterion optimizes the objective actually
/// being trained. With `sample_weight = None` the class term stands alone,
/// and at `scale_pos_weight = 1.0` every multiply is by exactly `1.0` with
/// the weight sum accumulating exactly like the old count, so the result
/// is bit-identical to the historical loss at each specialization.
///
/// Predictions are clipped to `[1e-15, 1 - 1e-15]` to avoid `log(0)`.
///
/// # Args
///
/// * `y_true` - True binary labels (0 or 1).
/// * `y_pred` - Predicted probabilities in (0, 1).
/// * `scale_pos_weight` - Weight applied to positive samples; must be
///   finite and positive.
/// * `sample_weight` - Optional per-row weights (finite, > 0); `None`
///   weighs every row 1.
///
/// # Returns
///
/// Weighted mean binary cross-entropy loss.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `y_true` is empty.
/// * `ClearGbmError::ShapeMismatch` if lengths differ.
/// * `ClearGbmError::InvalidLabel` if any label is not 0 or 1.
/// * `ClearGbmError::InvalidParameter` if `scale_pos_weight` or any sample
///   weight is not a finite positive number.
pub fn binary_log_loss(
    y_true: &[u8],
    y_pred: &[f64],
    scale_pos_weight: f64,
    sample_weight: Option<&[f64]>,
) -> Result<f64, ClearGbmError> {
    if y_true.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "y_true must not be empty".to_string(),
        });
    }
    match validate_lengths(y_true, y_pred) {
        Ok(()) => {}
        Err(e) => return Err(e),
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

    let mut total_loss = 0.0_f64;
    let mut weight_sum = 0.0_f64;

    for (i, (&label, &pred)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        let y = f64::from(label);
        let class_term = if label == 1_u8 {
            scale_pos_weight
        } else {
            1.0_f64
        };
        // With no sample weights the class term stands alone rather than
        // being multiplied by a synthesized 1.0 — same bits either way
        // (IEEE multiply by 1.0 is an identity), stated explicitly so the
        // bit-identity claim reads from the code.
        let w = match sample_weight {
            Some(ws) => class_term * ws[i],
            None => class_term,
        };
        let p = pred.clamp(CLIP_EPS, 1.0_f64 - CLIP_EPS);
        let sample_loss = -(y * p.ln() + (1.0_f64 - y) * (1.0_f64 - p).ln());
        total_loss += w * sample_loss;
        weight_sum += w;
    }

    Ok(total_loss / weight_sum)
}
