//! Tests for SplitResult type.

use super::helpers::EPSILON;
use crate::error::ClearGbmError;
use crate::split::{NanDirection, SplitDecision, SplitResult, SplitResultConfig};

#[test]
fn test_split_result_new() -> Result<(), ClearGbmError> {
    let config = SplitResultConfig {
        feature_index: 2_usize,
        decision: SplitDecision::Threshold { split_bin: 5_usize },
        gain: 0.123_f64,
        left_gradient_sum: 1.0_f64,
        left_hessian_sum: 2.0_f64,
        left_count: 50_usize,
        right_gradient_sum: 0.5_f64,
        right_hessian_sum: 1.5_f64,
        right_count: 30_usize,
        nan_direction: NanDirection::Left,
    };
    let result = SplitResult::new(config);

    assert_eq!(result.feature_index(), 2_usize);
    assert_eq!(
        result.decision(),
        SplitDecision::Threshold { split_bin: 5_usize }
    );
    assert!((result.gain() - 0.123_f64).abs() < EPSILON);
    assert!((result.left_gradient_sum() - 1.0_f64).abs() < EPSILON);
    assert!((result.left_hessian_sum() - 2.0_f64).abs() < EPSILON);
    assert_eq!(result.left_count(), 50_usize);
    assert!((result.right_gradient_sum() - 0.5_f64).abs() < EPSILON);
    assert!((result.right_hessian_sum() - 1.5_f64).abs() < EPSILON);
    assert_eq!(result.right_count(), 30_usize);
    assert_eq!(result.nan_direction(), NanDirection::Left);
    assert!(result.nan_goes_left());
    Ok(())
}

#[test]
fn test_split_result_serialize_deserialize() -> Result<(), ClearGbmError> {
    let config = SplitResultConfig {
        feature_index: 1_usize,
        decision: SplitDecision::Threshold { split_bin: 3_usize },
        gain: 0.5_f64,
        left_gradient_sum: 2.0_f64,
        left_hessian_sum: 4.0_f64,
        left_count: 100_usize,
        right_gradient_sum: 1.0_f64,
        right_hessian_sum: 2.0_f64,
        right_count: 50_usize,
        nan_direction: NanDirection::Right,
    };
    let result = SplitResult::new(config);

    let json_str = match serde_json::to_string(&result) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let parsed: SplitResult = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(parsed, result);
    Ok(())
}
