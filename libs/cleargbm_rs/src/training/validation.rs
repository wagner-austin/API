//! Input validation for gradient boosting training.
//!
//! Validates feature-matrix shapes, label counts, and feature names before
//! training. Label *content* (0/1 for binary, finite for continuous) is
//! validated by [`super::labels::resolve_objective`], which knows the label
//! kind; these functions own everything label-kind-agnostic.

use crate::error::ClearGbmError;

/// Validates training input shapes and returns the number of features.
///
/// # Args
///
/// * `x_train` - Training feature matrix `[n_samples][n_features]`.
/// * `n_labels` - Number of training labels.
/// * `feature_names` - Feature names (one per feature).
///
/// # Returns
///
/// The number of features.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `x_train` is empty or has zero features.
/// * `ClearGbmError::ShapeMismatch` if `x_train` rows differ in length,
///   or the label count doesn't match `x_train`, or `feature_names` length
///   doesn't match.
pub(crate) fn validate_training_inputs(
    x_train: &[&[f64]],
    n_labels: usize,
    feature_names: &[String],
) -> Result<usize, ClearGbmError> {
    if x_train.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "x_train must not be empty".to_string(),
        });
    }
    let n_features = x_train[0].len();
    if n_features == 0_usize {
        return Err(ClearGbmError::EmptyInput {
            context: "x_train has zero features".to_string(),
        });
    }
    let n_samples = x_train.len();

    // Validate row consistency
    for (i, row) in x_train.iter().enumerate() {
        if row.len() != n_features {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("all rows with {n_features} features"),
                got: format!("row {i} has {} features", row.len()),
            });
        }
    }

    // Validate label count matches x_train
    if n_labels != n_samples {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("{n_samples} labels"),
            got: format!("{n_labels} labels"),
        });
    }

    // Validate feature_names length
    if feature_names.len() != n_features {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("{n_features} feature names"),
            got: format!("{} feature names", feature_names.len()),
        });
    }

    Ok(n_features)
}

/// Validates optional validation-input shapes against the training dimensions.
///
/// # Args
///
/// * `x_val` - Validation feature matrix `[n_val_samples][n_features]`.
/// * `n_labels` - Number of validation labels.
/// * `n_features` - Expected number of features (from training data).
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `x_val` is empty.
/// * `ClearGbmError::ShapeMismatch` if dimensions don't match.
pub(crate) fn validate_validation_inputs(
    x_val: &[&[f64]],
    n_labels: usize,
    n_features: usize,
) -> Result<(), ClearGbmError> {
    if x_val.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "x_val must not be empty".to_string(),
        });
    }
    let n_val_features = x_val[0].len();
    if n_val_features != n_features {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("{n_features} features"),
            got: format!("{n_val_features} features in x_val"),
        });
    }

    // Validate row consistency
    for (i, row) in x_val.iter().enumerate() {
        if row.len() != n_features {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("all validation rows with {n_features} features"),
                got: format!("row {i} has {} features", row.len()),
            });
        }
    }

    // Validate label count matches x_val
    if n_labels != x_val.len() {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("{} validation labels", x_val.len()),
            got: format!("{n_labels} validation labels"),
        });
    }

    Ok(())
}
