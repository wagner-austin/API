//! Tests for log-odds initial prediction.

use crate::error::ClearGbmError;
use crate::losses::binary_log_loss_initial_prediction;

// --- Correctness ---

#[test]
fn test_initial_prediction_balanced() -> Result<(), ClearGbmError> {
    // 50% positive → log(0.5/0.5) = log(1) = 0.0
    let result = match binary_log_loss_initial_prediction(&[0_u8, 1_u8]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        result.abs() < 1e-10_f64,
        "balanced labels should give log-odds ≈ 0.0, got {result}"
    );
    Ok(())
}

#[test]
fn test_initial_prediction_balanced_larger() -> Result<(), ClearGbmError> {
    // 50% positive with more samples
    let labels = [0_u8, 1_u8, 0_u8, 1_u8, 0_u8, 1_u8];
    let result = match binary_log_loss_initial_prediction(&labels) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        result.abs() < 1e-10_f64,
        "balanced labels should give log-odds ≈ 0.0, got {result}"
    );
    Ok(())
}

#[test]
fn test_initial_prediction_mostly_positive() -> Result<(), ClearGbmError> {
    // 75% positive → log(0.75/0.25) = log(3) ≈ 1.0986
    let labels = [1_u8, 1_u8, 1_u8, 0_u8];
    let result = match binary_log_loss_initial_prediction(&labels) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let expected = 3.0_f64.ln();
    assert!(
        (result - expected).abs() < 1e-10_f64,
        "expected {expected}, got {result}"
    );
    Ok(())
}

#[test]
fn test_initial_prediction_mostly_negative() -> Result<(), ClearGbmError> {
    // 25% positive → log(0.25/0.75) = log(1/3) ≈ -1.0986
    let labels = [0_u8, 0_u8, 0_u8, 1_u8];
    let result = match binary_log_loss_initial_prediction(&labels) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let expected = (1.0_f64 / 3.0_f64).ln();
    assert!(
        (result - expected).abs() < 1e-10_f64,
        "expected {expected}, got {result}"
    );
    Ok(())
}

#[test]
fn test_initial_prediction_positive_when_majority_positive() -> Result<(), ClearGbmError> {
    // More positives → positive log-odds
    let labels = [1_u8, 1_u8, 1_u8, 1_u8, 0_u8];
    let result = match binary_log_loss_initial_prediction(&labels) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        result > 0.0_f64,
        "majority positive should give positive log-odds, got {result}"
    );
    Ok(())
}

#[test]
fn test_initial_prediction_negative_when_majority_negative() -> Result<(), ClearGbmError> {
    // More negatives → negative log-odds
    let labels = [0_u8, 0_u8, 0_u8, 0_u8, 1_u8];
    let result = match binary_log_loss_initial_prediction(&labels) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        result < 0.0_f64,
        "majority negative should give negative log-odds, got {result}"
    );
    Ok(())
}

#[test]
fn test_initial_prediction_antisymmetric() -> Result<(), ClearGbmError> {
    // log(p/(1-p)) and log((1-p)/p) should be negatives of each other
    let mostly_pos = [1_u8, 1_u8, 1_u8, 0_u8];
    let mostly_neg = [0_u8, 0_u8, 0_u8, 1_u8];
    let result_pos = match binary_log_loss_initial_prediction(&mostly_pos) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let result_neg = match binary_log_loss_initial_prediction(&mostly_neg) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!(
        (result_pos + result_neg).abs() < 1e-10_f64,
        "should be antisymmetric: {result_pos} + {result_neg} should be ≈ 0"
    );
    Ok(())
}

// --- Error paths ---

#[test]
fn test_initial_prediction_empty() -> Result<(), ClearGbmError> {
    let result = binary_log_loss_initial_prediction(&[]);
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
fn test_initial_prediction_all_zeros() -> Result<(), ClearGbmError> {
    let result = binary_log_loss_initial_prediction(&[0_u8, 0_u8, 0_u8]);
    match result {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_true");
            assert!(reason.contains("all labels are 0"));
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidParameter for all zeros, got {other:?}"),
        }),
    }
}

#[test]
fn test_initial_prediction_all_ones() -> Result<(), ClearGbmError> {
    let result = binary_log_loss_initial_prediction(&[1_u8, 1_u8, 1_u8]);
    match result {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_true");
            assert!(reason.contains("all labels are 1"));
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidParameter for all ones, got {other:?}"),
        }),
    }
}

#[test]
fn test_initial_prediction_invalid_label() -> Result<(), ClearGbmError> {
    let result = binary_log_loss_initial_prediction(&[0_u8, 1_u8, 5_u8]);
    match result {
        Err(ClearGbmError::InvalidLabel { value, index }) => {
            assert_eq!(value, 5_u8);
            assert_eq!(index, 2_usize);
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidLabel, got {other:?}"),
        }),
    }
}
