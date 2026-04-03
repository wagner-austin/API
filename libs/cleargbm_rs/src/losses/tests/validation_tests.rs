//! Tests for label/length validation and usize-to-f64 conversion.

use crate::error::ClearGbmError;
use crate::losses::validation::{usize_to_f64, validate_labels, validate_lengths};

// --- validate_labels ---

#[test]
fn test_validate_labels_all_zeros() -> Result<(), ClearGbmError> {
    validate_labels(&[0_u8, 0_u8, 0_u8])
}

#[test]
fn test_validate_labels_all_ones() -> Result<(), ClearGbmError> {
    validate_labels(&[1_u8, 1_u8, 1_u8])
}

#[test]
fn test_validate_labels_mixed() -> Result<(), ClearGbmError> {
    validate_labels(&[0_u8, 1_u8, 0_u8, 1_u8])
}

#[test]
fn test_validate_labels_empty() -> Result<(), ClearGbmError> {
    validate_labels(&[])
}

#[test]
fn test_validate_labels_invalid_two() -> Result<(), ClearGbmError> {
    let result = validate_labels(&[0_u8, 1_u8, 2_u8]);
    match result {
        Err(ClearGbmError::InvalidLabel { value, index }) => {
            assert_eq!(value, 2_u8);
            assert_eq!(index, 2_usize);
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidLabel, got {other:?}"),
        }),
    }
}

#[test]
fn test_validate_labels_invalid_255() -> Result<(), ClearGbmError> {
    let result = validate_labels(&[255_u8]);
    match result {
        Err(ClearGbmError::InvalidLabel { value, index }) => {
            assert_eq!(value, 255_u8);
            assert_eq!(index, 0_usize);
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidLabel, got {other:?}"),
        }),
    }
}

#[test]
fn test_validate_labels_first_invalid() -> Result<(), ClearGbmError> {
    let result = validate_labels(&[5_u8, 0_u8, 1_u8]);
    match result {
        Err(ClearGbmError::InvalidLabel { value, index }) => {
            assert_eq!(value, 5_u8);
            assert_eq!(index, 0_usize);
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidLabel, got {other:?}"),
        }),
    }
}

// --- validate_lengths ---

#[test]
fn test_validate_lengths_equal() -> Result<(), ClearGbmError> {
    validate_lengths(&[0_u8, 1_u8], &[0.5_f64, 0.5_f64])
}

#[test]
fn test_validate_lengths_both_empty() -> Result<(), ClearGbmError> {
    validate_lengths(&[], &[])
}

#[test]
fn test_validate_lengths_mismatch_true_longer() -> Result<(), ClearGbmError> {
    let result = validate_lengths(&[0_u8, 1_u8, 0_u8], &[0.5_f64]);
    match result {
        Err(ClearGbmError::ShapeMismatch { expected, got }) => {
            assert!(expected.contains("3"));
            assert!(got.contains("1"));
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected ShapeMismatch, got {other:?}"),
        }),
    }
}

#[test]
fn test_validate_lengths_mismatch_pred_longer() -> Result<(), ClearGbmError> {
    let result = validate_lengths(&[0_u8], &[0.5_f64, 0.5_f64, 0.5_f64]);
    match result {
        Err(ClearGbmError::ShapeMismatch { expected, got }) => {
            assert!(expected.contains("1"));
            assert!(got.contains("3"));
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected ShapeMismatch, got {other:?}"),
        }),
    }
}

// --- usize_to_f64 ---

#[test]
fn test_usize_to_f64_zero() -> Result<(), ClearGbmError> {
    let result = match usize_to_f64(0_usize, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((result - 0.0_f64).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn test_usize_to_f64_small() -> Result<(), ClearGbmError> {
    let result = match usize_to_f64(42_usize, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((result - 42.0_f64).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn test_usize_to_f64_u32_max() -> Result<(), ClearGbmError> {
    let max_u32 = 4_294_967_295_usize; // u32::MAX
    let result = match usize_to_f64(max_u32, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((result - 4_294_967_295.0_f64).abs() < 1.0_f64);
    Ok(())
}

#[test]
fn test_usize_to_f64_exceeds_u32_max() -> Result<(), ClearGbmError> {
    let too_large = 4_294_967_296_usize; // u32::MAX + 1
    let result = usize_to_f64(too_large, "test context");
    match result {
        Err(ClearGbmError::IntegerConversion { context }) => {
            assert!(context.contains("test context"));
            assert!(context.contains("exceeds u32::MAX"));
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected IntegerConversion, got {other:?}"),
        }),
    }
}

#[test]
fn test_usize_to_f64_one() -> Result<(), ClearGbmError> {
    let result = match usize_to_f64(1_usize, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((result - 1.0_f64).abs() < f64::EPSILON);
    Ok(())
}
