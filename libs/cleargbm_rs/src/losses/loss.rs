//! Binary cross-entropy (log loss) computation.

use crate::error::ClearGbmError;

use super::validation::{validate_labels, validate_lengths, validate_scale_pos_weight, CLIP_EPS};

/// Computes weighted mean binary cross-entropy (log loss).
///
/// `loss = -sum[w_i * (y * log(p) + (1-y) * log(1-p))] / sum[w_i]`
///
/// where `w_i` is `scale_pos_weight` for positive samples and `1` for
/// negatives — the same weighting the training gradients carry, so the
/// early-stopping criterion optimizes the objective actually being trained.
/// At `scale_pos_weight = 1.0` every multiply is by exactly `1.0` and the
/// weight sum accumulates exactly like the old count, so the result is
/// bit-identical to the historical unweighted loss.
///
/// Predictions are clipped to `[1e-15, 1 - 1e-15]` to avoid `log(0)`.
///
/// # Args
///
/// * `y_true` - True binary labels (0 or 1).
/// * `y_pred` - Predicted probabilities in (0, 1).
/// * `scale_pos_weight` - Weight applied to positive samples; must be
///   finite and positive.
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
/// * `ClearGbmError::InvalidParameter` if `scale_pos_weight` is not a
///   finite positive number.
pub fn binary_log_loss(
    y_true: &[u8],
    y_pred: &[f64],
    scale_pos_weight: f64,
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

    let mut total_loss = 0.0_f64;
    let mut weight_sum = 0.0_f64;

    for (&label, &pred) in y_true.iter().zip(y_pred.iter()) {
        let y = f64::from(label);
        let w = if label == 1_u8 {
            scale_pos_weight
        } else {
            1.0_f64
        };
        let p = pred.clamp(CLIP_EPS, 1.0_f64 - CLIP_EPS);
        let sample_loss = -(y * p.ln() + (1.0_f64 - y) * (1.0_f64 - p).ln());
        total_loss += w * sample_loss;
        weight_sum += w;
    }

    Ok(total_loss / weight_sum)
}
