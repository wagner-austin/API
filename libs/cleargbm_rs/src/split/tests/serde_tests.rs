//! Serde error path tests for split types.

use crate::error::ClearGbmError;
use crate::split::{MonotonicConstraint, NanDirection, SplitResult, SplitResultConfig};

// =========================================================================
// Serde error path tests - NanDirection
// =========================================================================

#[test]
fn test_nan_direction_deserialize_invalid_value() -> Result<(), ClearGbmError> {
    // Invalid string value
    let json = r#""Invalid""#;
    let result: Result<NanDirection, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    // Number instead of string
    let json = r#"123"#;
    let result: Result<NanDirection, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_left() -> Result<(), ClearGbmError> {
    let json = r#""Left""#;
    let dir: NanDirection = match serde_json::from_str(json) {
        Ok(d) => d,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(dir, NanDirection::Left));
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_right() -> Result<(), ClearGbmError> {
    let json = r#""Right""#;
    let dir: NanDirection = match serde_json::from_str(json) {
        Ok(d) => d,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(dir, NanDirection::Right));
    Ok(())
}

// =========================================================================
// Direct NanDirection deserialize via testkit deserializer
// =========================================================================

#[test]
fn test_nan_direction_via_minimal_value_deserializer() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::MinimalValueDeserializer;
    use serde::Deserialize;
    // MinimalValueDeserializer::deserialize_str returns "Right"
    let result = NanDirection::deserialize(MinimalValueDeserializer);
    assert!(result.is_ok());
    let dir = match result {
        Ok(d) => d,
        Err(_) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: "failed".to_string(),
            })
        }
    };
    assert!(matches!(dir, NanDirection::Right));
    Ok(())
}

#[test]
fn test_nan_direction_via_minimal_struct_deserializer() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::MinimalStructDeserializer;
    use serde::Deserialize;
    // MinimalStructDeserializer::deserialize_str returns "Left"
    let result = NanDirection::deserialize(MinimalStructDeserializer);
    assert!(result.is_ok());
    let dir = match result {
        Ok(d) => d,
        Err(_) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: "failed".to_string(),
            })
        }
    };
    assert!(matches!(dir, NanDirection::Left));
    Ok(())
}

// =========================================================================
// Serde tests - MonotonicConstraint
// =========================================================================

#[test]
fn test_monotonic_constraint_serialize_none() -> Result<(), ClearGbmError> {
    let json = match serde_json::to_string(&MonotonicConstraint::None) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(json, r#""None""#);
    Ok(())
}

#[test]
fn test_monotonic_constraint_serialize_increasing() -> Result<(), ClearGbmError> {
    let json = match serde_json::to_string(&MonotonicConstraint::Increasing) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(json, r#""Increasing""#);
    Ok(())
}

#[test]
fn test_monotonic_constraint_serialize_decreasing() -> Result<(), ClearGbmError> {
    let json = match serde_json::to_string(&MonotonicConstraint::Decreasing) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(json, r#""Decreasing""#);
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_none() -> Result<(), ClearGbmError> {
    let json = r#""None""#;
    let c: MonotonicConstraint = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(c, MonotonicConstraint::None));
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_increasing() -> Result<(), ClearGbmError> {
    let json = r#""Increasing""#;
    let c: MonotonicConstraint = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(c, MonotonicConstraint::Increasing));
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_decreasing() -> Result<(), ClearGbmError> {
    let json = r#""Decreasing""#;
    let c: MonotonicConstraint = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(c, MonotonicConstraint::Decreasing));
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_invalid_string() -> Result<(), ClearGbmError> {
    let json = r#""Bogus""#;
    let result: Result<MonotonicConstraint, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<MonotonicConstraint, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_monotonic_constraint_roundtrip_all_variants() -> Result<(), ClearGbmError> {
    for original in [
        MonotonicConstraint::None,
        MonotonicConstraint::Increasing,
        MonotonicConstraint::Decreasing,
    ] {
        let json = match serde_json::to_string(&original) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let decoded: MonotonicConstraint = match serde_json::from_str(&json) {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(decoded, original);
    }
    Ok(())
}

// =========================================================================
// Serde error path tests - SplitResult
// =========================================================================

#[test]
fn test_split_result_deserialize_missing_field() -> Result<(), ClearGbmError> {
    // Missing nan_direction field
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_unknown_field() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left","extra":123}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    // feature_index should be usize, not string
    let json = r#"{"feature_index":"wrong","split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_all_fields() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let sr: SplitResult = match serde_json::from_str(json) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(sr.feature_index(), 1_usize);
    assert_eq!(sr.split_bin(), 3_usize);
    assert!((sr.gain() - 0.5_f64).abs() < 1e-10_f64);
    assert_eq!(sr.left_count(), 10_usize);
    assert_eq!(sr.right_count(), 5_usize);
    assert!(matches!(sr.nan_direction(), NanDirection::Left));
    Ok(())
}

#[test]
fn test_split_result_deserialize_with_right_nan() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":2,"split_bin":5,"gain":1.0,"left_gradient_sum":2.0,"left_hessian_sum":3.0,"left_count":20,"right_gradient_sum":1.0,"right_hessian_sum":2.0,"right_count":10,"nan_direction":"Right"}"#;
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
        split_bin: 7_usize,
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
    assert_eq!(parsed.split_bin(), sr.split_bin());
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
        split_bin: 5_usize,
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
    // SplitResult has 10 fields
    for fail_at in 0_usize..10_usize {
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
        split_bin: 5_usize,
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
        split_bin: 5_usize,
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

// =========================================================================
// Deserialization error path tests using specialized deserializers
// =========================================================================

#[test]
fn test_split_result_deserialize_with_integer_key() -> Result<(), ClearGbmError> {
    // Triggers the expecting() method on SplitResultFieldVisitor
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;

    let deser = IntegerKeyDeserializer;
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_feature_index() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("feature_index");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_split_bin() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("split_bin");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_gain() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("gain");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_left_gradient_sum() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("left_gradient_sum");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_left_hessian_sum() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("left_hessian_sum");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_left_count() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("left_count");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_right_gradient_sum() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("right_gradient_sum");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_right_hessian_sum() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("right_hessian_sum");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_right_count() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("right_count");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_nan_direction() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("nan_direction");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_with_integer() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerDeserializer;
    use serde::Deserialize;

    let deser = IntegerDeserializer;
    let result = NanDirection::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_key() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnKeyDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnKeyDeserializer;
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_feature_index() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("feature_index");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_split_bin() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("split_bin");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_gain() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("gain");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_left_gradient_sum() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("left_gradient_sum");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_left_hessian_sum() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("left_hessian_sum");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_left_count() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("left_count");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_right_gradient_sum() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("right_gradient_sum");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_right_hessian_sum() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("right_hessian_sum");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_right_count() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("right_count");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_error_on_value_nan_direction() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("nan_direction");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests to exercise DuplicateFieldMapAccess and StructDuplicateFieldMapAccess
// These types don't check for duplicates, but calling these exercises the code paths.
// =========================================================================

#[test]
fn test_split_result_deserialize_duplicate_field_map_access() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;

    // DuplicateFieldMapAccess returns the same field twice with integer values.
    // SplitResult doesn't check for duplicates, but this fails because only one field is provided.
    let deser = DuplicateFieldDeserializer::new("feature_index");
    let result = SplitResult::deserialize(deser);
    // Should fail due to missing other required fields
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_struct_duplicate_field_map_access() -> Result<(), ClearGbmError> {
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;

    // StructDuplicateFieldMapAccess returns the same field twice with struct/seq values.
    // SplitResult expects specific field types, so this will fail on type mismatch.
    let deser = StructDuplicateFieldDeserializer::new("feature_index");
    let result = SplitResult::deserialize(deser);
    // Should fail due to type mismatch or missing fields
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Additional missing field tests for full coverage
// =========================================================================

#[test]
fn test_split_result_missing_feature_index() -> Result<(), ClearGbmError> {
    let json = r#"{"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_split_bin() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_gain() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_left_gradient_sum() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_left_hessian_sum() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_left_count() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_right_gradient_sum() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_right_hessian_sum() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_right_count() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_failing_serializer_coverage() -> Result<(), ClearGbmError> {
    use crate::testkit::{FailingSerializer, SerError};
    use serde::ser::{Error, SerializeStruct, Serializer};

    // Test SerError Display
    let err = SerError {
        message: "test".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("test"));

    // Test SerError custom
    let custom_err = SerError::custom("custom error");
    assert!(custom_err.message.contains("custom"));

    // Test all serializer primitive methods
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_bool(true).is_ok());

    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_i8(1_i8).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_i16(1_i16).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_i32(1_i32).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_i64(1_i64).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_u8(1_u8).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_u16(1_u16).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_u32(1_u32).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_u64(1_u64).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_f32(1.0_f32).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_f64(1.0_f64).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_char('a').is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_str("test").is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_bytes(&[1_u8, 2_u8]).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_none().is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_some(&1_u32).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_unit().is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_unit_struct("Unit").is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_unit_variant("E", 0, "V").is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_newtype_struct("N", &1_u32).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser)
        .serialize_newtype_variant("E", 0, "V", &1_u32)
        .is_ok());

    // Test error methods
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_seq(Some(1)).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_tuple(1).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_tuple_struct("T", 1).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_tuple_variant("E", 0, "V", 1).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_map(Some(1)).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_struct_variant("E", 0, "V", 1).is_err());

    // Test serialize_struct
    let mut ser = FailingSerializer::fail_after(100);
    let struct_ser = (&mut ser).serialize_struct("S", 1);
    assert!(struct_ser.is_ok());

    // Test struct end
    let mut ser = FailingSerializer::fail_after(100);
    let struct_ser = match (&mut ser).serialize_struct("Test", 0) {
        Ok(s) => s,
        Err(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "failed".to_string(),
            })
        }
    };
    assert!(struct_ser.end().is_ok());

    // Test struct serialize_field Ok then Err
    let mut ser = FailingSerializer::fail_after(1);
    let mut struct_ser = match (&mut ser).serialize_struct("Test", 2) {
        Ok(s) => s,
        Err(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "failed".to_string(),
            })
        }
    };
    assert!(struct_ser.serialize_field("f1", &1_u32).is_ok());
    assert!(struct_ser.serialize_field("f2", &2_u32).is_err());

    Ok(())
}
