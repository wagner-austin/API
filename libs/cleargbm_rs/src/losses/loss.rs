//! Binary cross-entropy (log loss) computation.

use crate::error::ClearGbmError;

use super::validation::{validate_labels, validate_lengths, CLIP_EPS};

/// Computes mean binary cross-entropy (log loss).
///
/// `loss = -mean[y * log(p) + (1-y) * log(1-p)]`
///
/// Predictions are clipped to `[1e-15, 1 - 1e-15]` to avoid `log(0)`.
///
/// # Args
///
/// * `y_true` - True binary labels (0 or 1).
/// * `y_pred` - Predicted probabilities in (0, 1).
///
/// # Returns
///
/// Mean binary cross-entropy loss.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `y_true` is empty.
/// * `ClearGbmError::ShapeMismatch` if lengths differ.
/// * `ClearGbmError::InvalidLabel` if any label is not 0 or 1.
pub fn binary_log_loss(y_true: &[u8], y_pred: &[f64]) -> Result<f64, ClearGbmError> {
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

    let mut total_loss = 0.0_f64;
    let mut n_f64 = 0.0_f64;

    for (&label, &pred) in y_true.iter().zip(y_pred.iter()) {
        let y = f64::from(label);
        let p = pred.clamp(CLIP_EPS, 1.0_f64 - CLIP_EPS);
        let sample_loss = -(y * p.ln() + (1.0_f64 - y) * (1.0_f64 - p).ln());
        total_loss += sample_loss;
        n_f64 += 1.0_f64;
    }

    Ok(total_loss / n_f64)
}
