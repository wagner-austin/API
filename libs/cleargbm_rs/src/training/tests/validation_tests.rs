//! Tests for training input shape validation.
//!
//! Label *content* validation (0/1 for binary, finite for continuous) moved
//! to objective resolution — see `labels_tests`.

use crate::error::ClearGbmError;
use crate::training::validation::{validate_training_inputs, validate_validation_inputs};

#[test]
fn test_valid_training_inputs() -> Result<(), ClearGbmError> {
    let row0: Vec<f64> = vec![1.0_f64, 2.0_f64, 3.0_f64];
    let row1: Vec<f64> = vec![4.0_f64, 5.0_f64, 6.0_f64];
    let x_train: Vec<&[f64]> = vec![row0.as_slice(), row1.as_slice()];
    let feature_names: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let n_features = match validate_training_inputs(&x_train, 2_usize, &feature_names) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };
    assert_eq!(n_features, 3_usize);
    Ok(())
}

#[test]
fn test_empty_x_train() -> Result<(), ClearGbmError> {
    let x_train: Vec<&[f64]> = vec![];
    let feature_names: Vec<String> = vec![];
    let result = validate_training_inputs(&x_train, 0_usize, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for empty x_train".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_zero_features() -> Result<(), ClearGbmError> {
    let row0: Vec<f64> = vec![];
    let x_train: Vec<&[f64]> = vec![row0.as_slice()];
    let feature_names: Vec<String> = vec![];
    let result = validate_training_inputs(&x_train, 1_usize, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for zero features".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_inconsistent_row_lengths() -> Result<(), ClearGbmError> {
    let row0: Vec<f64> = vec![1.0_f64, 2.0_f64];
    let row1: Vec<f64> = vec![3.0_f64, 4.0_f64, 5.0_f64];
    let x_train: Vec<&[f64]> = vec![row0.as_slice(), row1.as_slice()];
    let feature_names: Vec<String> = vec!["a".to_string(), "b".to_string()];
    let result = validate_training_inputs(&x_train, 2_usize, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for inconsistent rows".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_y_train_length_mismatch() -> Result<(), ClearGbmError> {
    let row0: Vec<f64> = vec![1.0_f64, 2.0_f64];
    let row1: Vec<f64> = vec![3.0_f64, 4.0_f64];
    let x_train: Vec<&[f64]> = vec![row0.as_slice(), row1.as_slice()];
    let feature_names: Vec<String> = vec!["a".to_string(), "b".to_string()];
    // only 1 label for 2 samples
    let result = validate_training_inputs(&x_train, 1_usize, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for y_train length mismatch".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_feature_names_length_mismatch() -> Result<(), ClearGbmError> {
    let row0: Vec<f64> = vec![1.0_f64, 2.0_f64];
    let x_train: Vec<&[f64]> = vec![row0.as_slice()];
    let feature_names: Vec<String> = vec!["a".to_string()]; // 1 name for 2 features
    let result = validate_training_inputs(&x_train, 1_usize, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for feature_names length".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

// --- Validation set validation ---

#[test]
fn test_valid_validation_inputs() -> Result<(), ClearGbmError> {
    let row0: Vec<f64> = vec![1.0_f64, 2.0_f64];
    let row1: Vec<f64> = vec![3.0_f64, 4.0_f64];
    let x_val: Vec<&[f64]> = vec![row0.as_slice(), row1.as_slice()];
    match validate_validation_inputs(&x_val, 2_usize, 2_usize) {
        Ok(()) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_empty_x_val() -> Result<(), ClearGbmError> {
    let x_val: Vec<&[f64]> = vec![];
    let result = validate_validation_inputs(&x_val, 0_usize, 2_usize);
    match result {
        Ok(()) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for empty x_val".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_val_feature_count_mismatch() -> Result<(), ClearGbmError> {
    let row0: Vec<f64> = vec![1.0_f64, 2.0_f64, 3.0_f64];
    let x_val: Vec<&[f64]> = vec![row0.as_slice()];
    // Training had 2 features but val has 3
    let result = validate_validation_inputs(&x_val, 1_usize, 2_usize);
    match result {
        Ok(()) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for feature count mismatch".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_val_row_inconsistent() -> Result<(), ClearGbmError> {
    let row0: Vec<f64> = vec![1.0_f64, 2.0_f64];
    let row1: Vec<f64> = vec![3.0_f64, 4.0_f64, 5.0_f64];
    let x_val: Vec<&[f64]> = vec![row0.as_slice(), row1.as_slice()];
    let result = validate_validation_inputs(&x_val, 2_usize, 2_usize);
    match result {
        Ok(()) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for inconsistent val rows".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_val_y_length_mismatch() -> Result<(), ClearGbmError> {
    let row0: Vec<f64> = vec![1.0_f64, 2.0_f64];
    let row1: Vec<f64> = vec![3.0_f64, 4.0_f64];
    let x_val: Vec<&[f64]> = vec![row0.as_slice(), row1.as_slice()];
    // 1 label for 2 samples
    let result = validate_validation_inputs(&x_val, 1_usize, 2_usize);
    match result {
        Ok(()) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for y_val length mismatch".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}
