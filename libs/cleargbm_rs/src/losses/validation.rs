//! Validation helpers for loss function inputs.
//!
//! Provides label validation, length checking, and safe integer-to-float
//! conversion for use by all loss functions in this module.

use crate::error::ClearGbmError;

/// Epsilon for clipping predictions to avoid `log(0)`.
///
/// Matches the Python `cleargbm` constant exactly for reproducibility.
pub(crate) const CLIP_EPS: f64 = 1e-15_f64;

/// Validates that all labels are 0 or 1.
///
/// # Args
///
/// * `y_true` - Slice of binary labels.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidLabel` if any label is not 0 or 1.
pub(crate) fn validate_labels(y_true: &[u8]) -> Result<(), ClearGbmError> {
    for (i, &label) in y_true.iter().enumerate() {
        if label > 1_u8 {
            return Err(ClearGbmError::InvalidLabel {
                value: label,
                index: i,
            });
        }
    }
    Ok(())
}

/// Validates that `y_true` and `y_pred` have the same length.
///
/// # Args
///
/// * `y_true` - Slice of binary labels.
/// * `y_pred` - Slice of predicted probabilities.
///
/// # Errors
///
/// Returns `ClearGbmError::ShapeMismatch` if lengths differ.
pub(crate) fn validate_lengths(y_true: &[u8], y_pred: &[f64]) -> Result<(), ClearGbmError> {
    if y_true.len() != y_pred.len() {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("y_pred length {}", y_true.len()),
            got: format!("y_pred length {}", y_pred.len()),
        });
    }
    Ok(())
}

/// Validates that a positive-class weight is a finite positive number.
///
/// # Args
///
/// * `scale_pos_weight` - Weight applied to positive samples.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` if the weight is NaN,
/// infinite, zero or negative.
pub(crate) fn validate_scale_pos_weight(scale_pos_weight: f64) -> Result<(), ClearGbmError> {
    if !scale_pos_weight.is_finite() || scale_pos_weight <= 0.0_f64 {
        return Err(ClearGbmError::InvalidParameter {
            name: "scale_pos_weight".to_string(),
            reason: format!("must be a finite positive number, got {scale_pos_weight}"),
        });
    }
    Ok(())
}

/// Converts a `usize` count to `f64` for arithmetic.
///
/// Uses `u32` as an intermediate to guarantee lossless conversion
/// (`f64` can represent all `u32` values exactly).
///
/// # Args
///
/// * `n` - The `usize` value to convert.
/// * `context` - Description of what is being converted (for error messages).
///
/// # Errors
///
/// Returns [`ClearGbmError::IntegerConversion`] if `n` exceeds `u32::MAX`.
#[cfg(test)]
pub(crate) fn usize_to_f64(n: usize, context: &str) -> Result<f64, ClearGbmError> {
    match u32::try_from(n) {
        Ok(v) => Ok(f64::from(v)),
        Err(_) => Err(ClearGbmError::IntegerConversion {
            context: format!("{context}: {n} exceeds u32::MAX"),
        }),
    }
}
