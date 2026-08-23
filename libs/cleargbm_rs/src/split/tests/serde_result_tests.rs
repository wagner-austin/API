//! Serde tests for `SplitResult`: round-trips, the decision's two
//! spellings, per-field wrong-value/error/missing coverage, and the
//! failing-serializer battery.

use crate::error::ClearGbmError;
use crate::split::{NanDirection, SplitDecision, SplitResult, SplitResultConfig};

// =========================================================================
// Serde error path tests - SplitResult
// =========================================================================

#[test]
fn test_split_result_deserialize_missing_field() -> Result<(), ClearGbmError> {
    // Missing nan_direction field
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_unknown_field() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left","extra":123}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    // feature_index should be usize, not string
    let json = r#"{"feature_index":"wrong","split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_all_fields() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let sr: SplitResult = match serde_json::from_str(json) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(sr.feature_index(), 1_usize);
    assert_eq!(
        sr.decision(),
        SplitDecision::Threshold { split_bin: 3_usize }
    );
    assert!((sr.gain() - 0.5_f64).abs() < 1e-10_f64);
    assert_eq!(sr.left_count(), 10_usize);
    assert_eq!(sr.right_count(), 5_usize);
    assert!(matches!(sr.nan_direction(), NanDirection::Left));
    Ok(())
}

#[test]
fn test_split_result_deserialize_with_right_nan() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":2,"split_bin":5,"categories_left_bins":null,"gain":1.0,"left_gradient_sum":2.0,"left_hessian_sum":3.0,"left_count":20,"right_gradient_sum":1.0,"right_hessian_sum":2.0,"right_count":10,"nan_direction":"Right"}"#;
    let sr: SplitResult = match serde_json::from_str(json) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(sr.nan_direction(), NanDirection::Right));
    Ok(())
}

#[test]
fn test_split_result_serialize_roundtrip() -> Result<(), ClearGbmError> {
    let config = SplitResultConfig {
        feature_index: 3_usize,
        decision: SplitDecision::Threshold { split_bin: 7_usize },
        gain: 2.5_f64,
        left_gradient_sum: 10.0_f64,
        left_hessian_sum: 5.0_f64,
        left_count: 100_usize,
        right_gradient_sum: -10.0_f64,
        right_hessian_sum: 5.0_f64,
        right_count: 50_usize,
        nan_direction: NanDirection::Right,
    };
    let sr = SplitResult::new(config);

    let json_str = match serde_json::to_string(&sr) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };

    let parsed: SplitResult = match serde_json::from_str(&json_str) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };

    assert_eq!(parsed.feature_index(), sr.feature_index());
    assert_eq!(parsed.decision(), sr.decision());
    assert!((parsed.gain() - sr.gain()).abs() < 1e-10_f64);
    assert_eq!(parsed.left_count(), sr.left_count());
    assert_eq!(parsed.right_count(), sr.right_count());
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_from_number() -> Result<(), ClearGbmError> {
    // Try deserializing NanDirection from a number (triggers expecting method)
    let json = r#"123"#;
    let result: Result<NanDirection, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    // The error should mention expected format
    assert!(err_msg.contains("Left") || err_msg.contains("Right") || err_msg.contains("string"));
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_from_object() -> Result<(), ClearGbmError> {
    // Try deserializing NanDirection from an object
    let json = r#"{"value": "Left"}"#;
    let result: Result<NanDirection, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_from_array() -> Result<(), ClearGbmError> {
    // Try deserializing NanDirection from an array
    let json = r#"["Left"]"#;
    let result: Result<NanDirection, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_from_bool() -> Result<(), ClearGbmError> {
    // Try deserializing NanDirection from a boolean
    let json = r#"true"#;
    let result: Result<NanDirection, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_from_null() -> Result<(), ClearGbmError> {
    // Try deserializing NanDirection from null
    let json = r#"null"#;
    let result: Result<NanDirection, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_from_array() -> Result<(), ClearGbmError> {
    // Try deserializing SplitResult from an array (triggers expecting)
    let json = r#"[1, 2, 3]"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_from_string() -> Result<(), ClearGbmError> {
    // Try deserializing SplitResult from a string
    let json = r#""not a struct""#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_from_number() -> Result<(), ClearGbmError> {
    // Try deserializing SplitResult from a number
    let json = r#"42"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// Serialization error path tests using failing serializer

#[test]
fn test_split_result_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = SplitResultConfig {
        feature_index: 0_usize,
        decision: SplitDecision::Threshold { split_bin: 5_usize },
        gain: 0.5_f64,
        left_gradient_sum: -1.0_f64,
        left_hessian_sum: 2.0_f64,
        left_count: 50_usize,
        right_gradient_sum: 1.0_f64,
        right_hessian_sum: 2.0_f64,
        right_count: 50_usize,
        nan_direction: NanDirection::Left,
    };
    let sr = SplitResult::new(config);
    // SplitResult has 11 fields
    for fail_at in 0_usize..11_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = sr.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_split_result_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = SplitResultConfig {
        feature_index: 0_usize,
        decision: SplitDecision::Threshold { split_bin: 5_usize },
        gain: 0.5_f64,
        left_gradient_sum: -1.0_f64,
        left_hessian_sum: 2.0_f64,
        left_count: 50_usize,
        right_gradient_sum: 1.0_f64,
        right_hessian_sum: 2.0_f64,
        right_count: 50_usize,
        nan_direction: NanDirection::Left,
    };
    let sr = SplitResult::new(config);
    let mut ser = FailingSerializer::fail_on_struct();
    let result = sr.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = SplitResultConfig {
        feature_index: 0_usize,
        decision: SplitDecision::Threshold { split_bin: 5_usize },
        gain: 0.5_f64,
        left_gradient_sum: -1.0_f64,
        left_hessian_sum: 2.0_f64,
        left_count: 50_usize,
        right_gradient_sum: 1.0_f64,
        right_hessian_sum: 2.0_f64,
        right_count: 50_usize,
        nan_direction: NanDirection::Left,
    };
    let sr = SplitResult::new(config);
    let mut ser = FailingSerializer::fail_on_end();
    let result = sr.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}
