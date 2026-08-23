//! Squared-error loss and base score for regression.
//!
//! The squared-error objective is `L = mean[(y - pred)^2]`. Its raw-score
//! derivatives are `gradient = pred - y` and `hessian = 1`, computed inline
//! in the training loop; this module owns the base score and the
//! early-stopping evaluation loss.

use crate::error::ClearGbmError;

use super::validation::validate_continuous_labels;

/// Computes the initial prediction for squared-error regression: `mean(y)`.
///
/// The label mean is the constant score minimizing squared error, the
/// regression analogue of the log-odds base score for binary log loss.
/// The mean is accumulated as `sum / n` with `n` counted in integer-valued
/// `f64` increments, the same exact-count pattern the binary base score
/// uses.
///
/// # Args
///
/// * `y_true` - Continuous regression targets.
///
/// # Returns
///
/// The label mean.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `y_true` is empty.
/// * `ClearGbmError::InvalidParameter` if any label is not finite.
pub fn squared_error_initial_prediction(y_true: &[f64]) -> Result<f64, ClearGbmError> {
    if y_true.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "y_true must not be empty".to_string(),
        });
    }
    match validate_continuous_labels(y_true, "y_true") {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut sum = 0.0_f64;
    let mut count = 0.0_f64;
    for &label in y_true {
        sum += label;
        count += 1.0_f64;
    }
    Ok(sum / count)
}

/// Computes mean squared error between targets and raw predictions.
///
/// `loss = mean[(y - pred)^2]`
///
/// Raw scores are the predictions under the squared-error objective (identity
/// transform), so this evaluates them directly — no sigmoid, no clipping.
///
/// # Args
///
/// * `y_true` - Continuous regression targets.
/// * `y_pred` - Raw predictions.
///
/// # Returns
///
/// Mean squared error.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `y_true` is empty.
/// * `ClearGbmError::ShapeMismatch` if lengths differ.
/// * `ClearGbmError::InvalidParameter` if any label is not finite.
pub fn squared_error_loss(y_true: &[f64], y_pred: &[f64]) -> Result<f64, ClearGbmError> {
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

    let mut total = 0.0_f64;
    let mut count = 0.0_f64;
    for (&label, &pred) in y_true.iter().zip(y_pred.iter()) {
        let diff = label - pred;
        total += diff * diff;
        count += 1.0_f64;
    }
    Ok(total / count)
}
