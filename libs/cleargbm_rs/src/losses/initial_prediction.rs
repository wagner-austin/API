//! Initial prediction (log-odds) computation for binary classification.

use crate::error::ClearGbmError;

use super::validation::{validate_labels, CLIP_EPS};

/// Computes the initial prediction (log-odds of positive class rate).
///
/// `initial = log(p_positive / (1 - p_positive))`
///
/// where `p_positive = count(y == 1) / count(y)`.
///
/// # Args
///
/// * `y_true` - True binary labels (0 or 1).
///
/// # Returns
///
/// Initial prediction in log-odds space.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `y_true` is empty.
/// * `ClearGbmError::InvalidLabel` if any label is not 0 or 1.
/// * `ClearGbmError::InvalidParameter` if all labels are 0 or all labels are 1.
pub fn binary_log_loss_initial_prediction(y_true: &[u8]) -> Result<f64, ClearGbmError> {
    if y_true.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "y_true must not be empty".to_string(),
        });
    }
    match validate_labels(y_true) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut n_positive_f64 = 0.0_f64;
    let mut n_total_f64 = 0.0_f64;
    for &label in y_true {
        if label == 1_u8 {
            n_positive_f64 += 1.0_f64;
        }
        n_total_f64 += 1.0_f64;
    }

    let p_positive = n_positive_f64 / n_total_f64;

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
