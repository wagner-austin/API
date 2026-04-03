//! Tests for hessian (second derivative) computation.

use crate::error::ClearGbmError;
use crate::losses::binary_log_loss_hessians;

// --- Correctness ---

#[test]
fn test_hessians_formula_p_times_one_minus_p() -> Result<(), ClearGbmError> {
    // hessian = p * (1-p)
    let y_true = [1_u8, 0_u8, 1_u8, 0_u8];
    let y_pred = [0.7_f64, 0.3_f64, 0.9_f64, 0.1_f64];
    let hess = match binary_log_loss_hessians(&y_true, &y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(hess.len(), 4_usize);
    // p=0.7: 0.7 * 0.3 = 0.21
    assert!((hess[0_usize] - 0.21_f64).abs() < 1e-15_f64);
    // p=0.3: 0.3 * 0.7 = 0.21
    assert!((hess[1_usize] - 0.21_f64).abs() < 1e-15_f64);
    // p=0.9: 0.9 * 0.1 = 0.09
    assert!((hess[2_usize] - 0.09_f64).abs() < 1e-15_f64);
    // p=0.1: 0.1 * 0.9 = 0.09
    assert!((hess[3_usize] - 0.09_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_hessians_max_at_half() -> Result<(), ClearGbmError> {
    // Hessian is maximized at p=0.5: 0.5 * 0.5 = 0.25
    let hess = match binary_log_loss_hessians(&[1_u8], &[0.5_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((hess[0_usize] - 0.25_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_hessians_always_positive() -> Result<(), ClearGbmError> {
    // Hessians should always be positive (p*(1-p) > 0 for p in (0,1))
    let preds = [
        0.01_f64, 0.1_f64, 0.3_f64, 0.5_f64, 0.7_f64, 0.9_f64, 0.99_f64,
    ];
    let labels = [1_u8, 0_u8, 1_u8, 0_u8, 1_u8, 0_u8, 1_u8];
    let hess = match binary_log_loss_hessians(&labels, &preds) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    for (idx, &h) in hess.iter().enumerate() {
        assert!(
            h > 0.0_f64,
            "hessian at index {idx} should be positive, got {h}"
        );
    }
    Ok(())
}

#[test]
fn test_hessians_symmetric() -> Result<(), ClearGbmError> {
    // p*(1-p) = (1-p)*p, so hessian at p and at (1-p) should be equal
    let hess = match binary_log_loss_hessians(&[1_u8, 1_u8], &[0.3_f64, 0.7_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((hess[0_usize] - hess[1_usize]).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_hessians_clips_extreme_predictions() -> Result<(), ClearGbmError> {
    // p=0.0 and p=1.0 should be clipped, producing finite positive hessians
    let hess = match binary_log_loss_hessians(&[1_u8, 0_u8], &[0.0_f64, 1.0_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    for (idx, &h) in hess.iter().enumerate() {
        assert!(
            h > 0.0_f64,
            "clipped hessian at index {idx} should be positive, got {h}"
        );
        assert!(
            h.is_finite(),
            "clipped hessian at index {idx} should be finite, got {h}"
        );
    }
    Ok(())
}

#[test]
fn test_hessians_independent_of_labels() -> Result<(), ClearGbmError> {
    // Hessian depends only on p, not on y
    let hess_pos = match binary_log_loss_hessians(&[1_u8], &[0.6_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let hess_neg = match binary_log_loss_hessians(&[0_u8], &[0.6_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((hess_pos[0_usize] - hess_neg[0_usize]).abs() < 1e-15_f64);
    Ok(())
}

// --- Error paths ---

#[test]
fn test_hessians_length_mismatch() -> Result<(), ClearGbmError> {
    let result = binary_log_loss_hessians(&[0_u8, 1_u8], &[0.5_f64]);
    match result {
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected ShapeMismatch, got {other:?}"),
        }),
    }
}

#[test]
fn test_hessians_invalid_label() -> Result<(), ClearGbmError> {
    let result = binary_log_loss_hessians(&[7_u8], &[0.5_f64]);
    match result {
        Err(ClearGbmError::InvalidLabel { value, .. }) => {
            assert_eq!(value, 7_u8);
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidLabel, got {other:?}"),
        }),
    }
}

#[test]
fn test_hessians_empty_inputs() -> Result<(), ClearGbmError> {
    let hess = match binary_log_loss_hessians(&[], &[]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(hess.is_empty());
    Ok(())
}
