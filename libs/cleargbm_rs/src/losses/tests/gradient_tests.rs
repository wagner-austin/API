//! Tests for gradient (first derivative) computation.

use crate::error::ClearGbmError;
use crate::losses::binary_log_loss_gradients;

// --- Correctness ---

#[test]
fn test_gradients_perfect_positive() -> Result<(), ClearGbmError> {
    // y=1, p=1.0 → gradient = 1.0 - 1 = 0.0
    let grads = match binary_log_loss_gradients(&[1_u8], &[1.0_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(grads.len(), 1_usize);
    assert!((grads[0_usize] - 0.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_gradients_perfect_negative() -> Result<(), ClearGbmError> {
    // y=0, p=0.0 → gradient = 0.0 - 0 = 0.0
    let grads = match binary_log_loss_gradients(&[0_u8], &[0.0_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(grads.len(), 1_usize);
    assert!((grads[0_usize] - 0.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_gradients_formula_p_minus_y() -> Result<(), ClearGbmError> {
    // gradient = p - y
    let y_true = [1_u8, 0_u8, 1_u8, 0_u8];
    let y_pred = [0.7_f64, 0.3_f64, 0.9_f64, 0.1_f64];
    let grads = match binary_log_loss_gradients(&y_true, &y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(grads.len(), 4_usize);
    // y=1, p=0.7 → 0.7 - 1.0 = -0.3
    assert!((grads[0_usize] - (-0.3_f64)).abs() < 1e-15_f64);
    // y=0, p=0.3 → 0.3 - 0.0 = 0.3
    assert!((grads[1_usize] - 0.3_f64).abs() < 1e-15_f64);
    // y=1, p=0.9 → 0.9 - 1.0 = -0.1
    assert!((grads[2_usize] - (-0.1_f64)).abs() < 1e-15_f64);
    // y=0, p=0.1 → 0.1 - 0.0 = 0.1
    assert!((grads[3_usize] - 0.1_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_gradients_positive_when_overconfident_negative() -> Result<(), ClearGbmError> {
    // y=0 but p=0.9 → gradient = 0.9 (positive = push prediction down)
    let grads = match binary_log_loss_gradients(&[0_u8], &[0.9_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(grads[0_usize] > 0.0_f64);
    assert!((grads[0_usize] - 0.9_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_gradients_negative_when_overconfident_positive() -> Result<(), ClearGbmError> {
    // y=1 but p=0.1 → gradient = 0.1 - 1.0 = -0.9 (negative = push prediction up)
    let grads = match binary_log_loss_gradients(&[1_u8], &[0.1_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(grads[0_usize] < 0.0_f64);
    assert!((grads[0_usize] - (-0.9_f64)).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_gradients_at_half() -> Result<(), ClearGbmError> {
    // y=1, p=0.5 → gradient = -0.5
    // y=0, p=0.5 → gradient = 0.5
    let grads = match binary_log_loss_gradients(&[1_u8, 0_u8], &[0.5_f64, 0.5_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((grads[0_usize] - (-0.5_f64)).abs() < 1e-15_f64);
    assert!((grads[1_usize] - 0.5_f64).abs() < 1e-15_f64);
    Ok(())
}

// --- Error paths ---

#[test]
fn test_gradients_length_mismatch() -> Result<(), ClearGbmError> {
    let result = binary_log_loss_gradients(&[0_u8, 1_u8, 0_u8], &[0.5_f64]);
    match result {
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected ShapeMismatch, got {other:?}"),
        }),
    }
}

#[test]
fn test_gradients_invalid_label() -> Result<(), ClearGbmError> {
    let result = binary_log_loss_gradients(&[3_u8], &[0.5_f64]);
    match result {
        Err(ClearGbmError::InvalidLabel { value, .. }) => {
            assert_eq!(value, 3_u8);
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidLabel, got {other:?}"),
        }),
    }
}

#[test]
fn test_gradients_empty_inputs() -> Result<(), ClearGbmError> {
    // Empty inputs are valid for gradients (lengths match at 0)
    let grads = match binary_log_loss_gradients(&[], &[]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(grads.is_empty());
    Ok(())
}
