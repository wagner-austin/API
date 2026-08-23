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

/// Validates that all continuous regression targets are finite.
///
/// A NaN or infinite label would poison the base score, every gradient it
/// touches, and the evaluation loss, so it is rejected at the boundary with
/// its index rather than surfacing later as a NaN model.
///
/// # Args
///
/// * `y_true` - Slice of continuous regression targets.
/// * `name` - The caller's argument name (`"y_true"`, `"y_train"`,
///   `"y_val"`), used in the error so it points at the offending input.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` naming the first non-finite
/// label and its index.
pub(crate) fn validate_continuous_labels(y_true: &[f64], name: &str) -> Result<(), ClearGbmError> {
    for (i, &label) in y_true.iter().enumerate() {
        if !label.is_finite() {
            return Err(ClearGbmError::InvalidParameter {
                name: name.to_string(),
                reason: format!("label at index {i} must be finite, got {label}"),
            });
        }
    }
    Ok(())
}

/// Validates per-row sample weights: finite and strictly positive.
///
/// Zero weights are rejected rather than allowed: a row weighted 0 can
/// empty a leaf's hessian sum, and with `reg_lambda = 0` that turns the
/// leaf value into 0/0. A caller who wants a row to contribute nothing
/// should drop the row — an explicit act, not a weight.
///
/// # Args
///
/// * `weights` - Per-row sample weights.
/// * `name` - The caller's argument name (`"sample_weight"`,
///   `"val_sample_weight"`), used in the error.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` naming the first offending
/// weight and its index.
pub(crate) fn validate_sample_weights(weights: &[f64], name: &str) -> Result<(), ClearGbmError> {
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w <= 0.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: name.to_string(),
                reason: format!("weight at index {i} must be finite and > 0, got {w}"),
            });
        }
    }
    Ok(())
}

/// Validates multiclass labels: every label must be `< n_classes`.
///
/// # Args
///
/// * `y` - Class labels.
/// * `n_classes` - The configured class count.
/// * `name` - Argument name, used in the error message.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` naming the first offending
/// index and value.
pub(crate) fn validate_multiclass_labels(
    y: &[u32],
    n_classes: usize,
    name: &str,
) -> Result<(), ClearGbmError> {
    for (i, &label) in y.iter().enumerate() {
        if crate::narrow::index_widen(label) >= n_classes {
            return Err(ClearGbmError::InvalidParameter {
                name: name.to_string(),
                reason: format!(
                    "labels must be < n_classes ({n_classes}), got {label} at index {i}"
                ),
            });
        }
    }
    Ok(())
}

/// Validates a sample-weight slice against its labels: length must match
/// and every weight must be finite and strictly positive.
///
/// # Args
///
/// * `n_labels` - Number of labels the weights accompany.
/// * `weights` - Per-row sample weights.
/// * `weight_name` - The caller's weight argument name, used in errors.
///
/// # Errors
///
/// * `ClearGbmError::ShapeMismatch` if the lengths differ.
/// * `ClearGbmError::InvalidParameter` on a non-finite or non-positive
///   weight.
pub(crate) fn validate_weight_pairing(
    n_labels: usize,
    weights: &[f64],
    weight_name: &str,
) -> Result<(), ClearGbmError> {
    if weights.len() != n_labels {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("{weight_name} length {n_labels}"),
            got: format!("{weight_name} length {}", weights.len()),
        });
    }
    validate_sample_weights(weights, weight_name)
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
