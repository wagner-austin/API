//! Tests for the squared-error base score and evaluation loss.

use crate::error::ClearGbmError;
use crate::losses::{squared_error_initial_prediction, squared_error_loss};

// =============================================================================
// squared_error_initial_prediction
// =============================================================================

#[test]
fn test_initial_prediction_is_label_mean() -> Result<(), ClearGbmError> {
    let y = [1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64];
    let base = propagate!(squared_error_initial_prediction(&y));
    assert!((base - 2.5_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_initial_prediction_single_label() -> Result<(), ClearGbmError> {
    let y = [-7.25_f64];
    let base = propagate!(squared_error_initial_prediction(&y));
    assert!((base - -7.25_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_initial_prediction_negative_labels() -> Result<(), ClearGbmError> {
    let y = [-3.0_f64, -1.0_f64];
    let base = propagate!(squared_error_initial_prediction(&y));
    assert!((base - -2.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_initial_prediction_empty_is_error() -> Result<(), ClearGbmError> {
    let y: [f64; 0] = [];
    match squared_error_initial_prediction(&y) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "empty targets must be rejected".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert!(context.contains("y_true"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_initial_prediction_nan_label_is_error() -> Result<(), ClearGbmError> {
    let y = [1.0_f64, f64::NAN, 3.0_f64];
    match squared_error_initial_prediction(&y) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a NaN target must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_true");
            assert!(reason.contains("index 1"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_initial_prediction_infinite_label_is_error() -> Result<(), ClearGbmError> {
    let y = [f64::INFINITY];
    match squared_error_initial_prediction(&y) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an infinite target must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_true");
            assert!(reason.contains("index 0"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

// =============================================================================
// squared_error_loss
// =============================================================================

#[test]
fn test_loss_zero_at_perfect_prediction() -> Result<(), ClearGbmError> {
    let y = [1.0_f64, -2.0_f64, 0.5_f64];
    let loss = propagate!(squared_error_loss(&y, &y));
    assert!((loss - 0.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_loss_is_mean_squared_error() -> Result<(), ClearGbmError> {
    let y = [0.0_f64, 0.0_f64];
    let pred = [1.0_f64, 3.0_f64];
    // (1 + 9) / 2 = 5
    let loss = propagate!(squared_error_loss(&y, &pred));
    assert!((loss - 5.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_loss_symmetric_in_error_sign() -> Result<(), ClearGbmError> {
    let y = [2.0_f64];
    let over = propagate!(squared_error_loss(&y, &[3.0_f64]));
    let under = propagate!(squared_error_loss(&y, &[1.0_f64]));
    assert!((over - under).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_loss_empty_is_error() -> Result<(), ClearGbmError> {
    let y: [f64; 0] = [];
    let pred: [f64; 0] = [];
    match squared_error_loss(&y, &pred) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "empty targets must be rejected".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert!(context.contains("y_true"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_loss_length_mismatch_is_error() -> Result<(), ClearGbmError> {
    let y = [1.0_f64, 2.0_f64];
    let pred = [1.0_f64];
    match squared_error_loss(&y, &pred) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a length mismatch must be rejected".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { expected, got }) => {
            assert!(expected.contains('2'));
            assert!(got.contains('1'));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_loss_nan_label_is_error() -> Result<(), ClearGbmError> {
    let y = [f64::NAN];
    let pred = [0.0_f64];
    match squared_error_loss(&y, &pred) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a NaN target must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "y_true");
            Ok(())
        }
        Err(e) => Err(e),
    }
}
