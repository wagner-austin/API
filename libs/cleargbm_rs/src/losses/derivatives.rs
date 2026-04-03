//! Gradient and hessian computation for binary log loss.
//!
//! These first and second derivatives feed the tree building loop
//! at each boosting round.

use crate::error::ClearGbmError;

use super::validation::{validate_labels, validate_lengths, CLIP_EPS};

/// Computes gradients of binary log loss (first derivative).
///
/// `gradient = p - y`
///
/// The gradient of log loss with respect to the raw prediction
/// (before sigmoid) is simply `p - y`, where `p` is the predicted probability.
///
/// # Args
///
/// * `y_true` - True binary labels (0 or 1).
/// * `y_pred` - Predicted probabilities in (0, 1).
///
/// # Returns
///
/// Gradient for each sample.
///
/// # Errors
///
/// * `ClearGbmError::ShapeMismatch` if lengths differ.
/// * `ClearGbmError::InvalidLabel` if any label is not 0 or 1.
pub fn binary_log_loss_gradients(y_true: &[u8], y_pred: &[f64]) -> Result<Vec<f64>, ClearGbmError> {
    match validate_lengths(y_true, y_pred) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match validate_labels(y_true) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut result = Vec::with_capacity(y_true.len());
    for (&label, &pred) in y_true.iter().zip(y_pred.iter()) {
        let y = f64::from(label);
        result.push(pred - y);
    }
    Ok(result)
}

/// Computes hessians of binary log loss (second derivative).
///
/// `hessian = p * (1 - p)`
///
/// Predictions are clipped to `[1e-15, 1 - 1e-15]` to avoid
/// numerical issues at boundaries.
///
/// # Args
///
/// * `y_true` - True binary labels (0 or 1).
/// * `y_pred` - Predicted probabilities in (0, 1).
///
/// # Returns
///
/// Hessian for each sample.
///
/// # Errors
///
/// * `ClearGbmError::ShapeMismatch` if lengths differ.
/// * `ClearGbmError::InvalidLabel` if any label is not 0 or 1.
pub fn binary_log_loss_hessians(y_true: &[u8], y_pred: &[f64]) -> Result<Vec<f64>, ClearGbmError> {
    match validate_lengths(y_true, y_pred) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match validate_labels(y_true) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut result = Vec::with_capacity(y_true.len());
    for &pred in y_pred {
        let p = pred.clamp(CLIP_EPS, 1.0_f64 - CLIP_EPS);
        result.push(p * (1.0_f64 - p));
    }
    Ok(result)
}
