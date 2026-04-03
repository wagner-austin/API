//! Tests for binary cross-entropy (log loss) computation.

use crate::error::ClearGbmError;
use crate::losses::binary_log_loss;

// --- Correctness ---

#[test]
fn test_binary_log_loss_perfect_predictions() -> Result<(), ClearGbmError> {
    // Near-perfect predictions should give near-zero loss
    let y_true = [1_u8, 0_u8, 1_u8, 0_u8];
    let y_pred = [0.999_f64, 0.001_f64, 0.999_f64, 0.001_f64];
    let loss = match binary_log_loss(&y_true, &y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        loss < 0.01_f64,
        "perfect prediction loss should be near 0, got {loss}"
    );
    assert!(loss > 0.0_f64, "loss must be positive");
    Ok(())
}

#[test]
fn test_binary_log_loss_worst_predictions() -> Result<(), ClearGbmError> {
    // Inverted predictions should give high loss
    let y_true = [1_u8, 0_u8, 1_u8, 0_u8];
    let y_pred = [0.001_f64, 0.999_f64, 0.001_f64, 0.999_f64];
    let loss = match binary_log_loss(&y_true, &y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        loss > 5.0_f64,
        "worst predictions loss should be high, got {loss}"
    );
    Ok(())
}

#[test]
fn test_binary_log_loss_uniform_half() -> Result<(), ClearGbmError> {
    // Predicting 0.5 for everything should give loss = ln(2) ≈ 0.6931...
    let y_true = [1_u8, 0_u8];
    let y_pred = [0.5_f64, 0.5_f64];
    let loss = match binary_log_loss(&y_true, &y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let expected = 2.0_f64.ln(); // ln(2) ≈ 0.6931
    assert!(
        (loss - expected).abs() < 1e-10_f64,
        "uniform 0.5 loss should be ln(2), got {loss}"
    );
    Ok(())
}

#[test]
fn test_binary_log_loss_single_positive() -> Result<(), ClearGbmError> {
    // Single sample: y=1, p=0.8 → loss = -ln(0.8) ≈ 0.2231
    let y_true = [1_u8];
    let y_pred = [0.8_f64];
    let loss = match binary_log_loss(&y_true, &y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let expected = -(0.8_f64.ln());
    assert!(
        (loss - expected).abs() < 1e-10_f64,
        "expected {expected}, got {loss}"
    );
    Ok(())
}

#[test]
fn test_binary_log_loss_single_negative() -> Result<(), ClearGbmError> {
    // Single sample: y=0, p=0.3 → loss = -ln(1-0.3) = -ln(0.7) ≈ 0.3567
    let y_true = [0_u8];
    let y_pred = [0.3_f64];
    let loss = match binary_log_loss(&y_true, &y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let expected = -(0.7_f64.ln());
    assert!(
        (loss - expected).abs() < 1e-10_f64,
        "expected {expected}, got {loss}"
    );
    Ok(())
}

#[test]
fn test_binary_log_loss_clips_extreme_predictions() -> Result<(), ClearGbmError> {
    // Predictions at 0.0 and 1.0 should be clipped, not produce NaN/Inf
    let y_true = [1_u8, 0_u8];
    let y_pred = [0.0_f64, 1.0_f64];
    let loss = match binary_log_loss(&y_true, &y_pred) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(loss.is_finite(), "loss must be finite, got {loss}");
    assert!(loss > 0.0_f64, "loss must be positive");
    Ok(())
}

#[test]
fn test_binary_log_loss_symmetric() -> Result<(), ClearGbmError> {
    // Loss for (y=1, p=0.8) and (y=0, p=0.2) should be the same
    let loss_pos = match binary_log_loss(&[1_u8], &[0.8_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let loss_neg = match binary_log_loss(&[0_u8], &[0.2_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        (loss_pos - loss_neg).abs() < 1e-10_f64,
        "should be symmetric: {loss_pos} vs {loss_neg}"
    );
    Ok(())
}

// --- Error paths ---

#[test]
fn test_binary_log_loss_empty_input() -> Result<(), ClearGbmError> {
    let result = binary_log_loss(&[], &[]);
    match result {
        Err(ClearGbmError::EmptyInput { context }) => {
            assert!(context.contains("y_true"));
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected EmptyInput, got {other:?}"),
        }),
    }
}

#[test]
fn test_binary_log_loss_length_mismatch() -> Result<(), ClearGbmError> {
    let result = binary_log_loss(&[0_u8, 1_u8], &[0.5_f64]);
    match result {
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected ShapeMismatch, got {other:?}"),
        }),
    }
}

#[test]
fn test_binary_log_loss_invalid_label() -> Result<(), ClearGbmError> {
    let result = binary_log_loss(&[0_u8, 2_u8], &[0.5_f64, 0.5_f64]);
    match result {
        Err(ClearGbmError::InvalidLabel { value, index }) => {
            assert_eq!(value, 2_u8);
            assert_eq!(index, 1_usize);
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidLabel, got {other:?}"),
        }),
    }
}
