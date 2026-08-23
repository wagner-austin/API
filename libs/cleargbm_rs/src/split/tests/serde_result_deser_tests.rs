//! `SplitResult` deserialization error paths through the specialized
//! testkit deserializers: per-field wrong values, injected value
//! errors, and duplicate-field map access.

use crate::error::ClearGbmError;
use crate::split::{CategoryBinSet, NanDirection, SplitDecision, SplitResult, SplitResultConfig};

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
    let json = r#"{"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_split_bin() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_gain() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_left_gradient_sum() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_left_hessian_sum() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_left_count() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_right_gradient_sum() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_right_hessian_sum() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_right_count() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"nan_direction":"Left"}"#;
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

#[test]
fn test_split_result_roundtrips_a_categorical_decision() -> Result<(), ClearGbmError> {
    let mut left_bins = CategoryBinSet::new();
    left_bins.insert(0_usize);
    left_bins.insert(5_usize);
    let original = SplitResult::new(SplitResultConfig {
        feature_index: 2_usize,
        decision: SplitDecision::CategorySubset { left_bins },
        gain: 0.75_f64,
        left_gradient_sum: -2.0_f64,
        left_hessian_sum: 3.0_f64,
        left_count: 12_usize,
        right_gradient_sum: 2.0_f64,
        right_hessian_sum: 3.0_f64,
        right_count: 12_usize,
        nan_direction: NanDirection::Right,
    });
    let json = match serde_json::to_string(&original) {
        Ok(j) => j,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(json.contains(r#""split_bin":null"#));
    assert!(json.contains(r#""categories_left_bins":[0,5]"#));
    let decoded: SplitResult = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(decoded, original);
    Ok(())
}

#[test]
fn test_split_result_rejects_both_decisions_set() -> Result<(), ClearGbmError> {
    // A payload claiming to be both a threshold and a categorical split is
    // rejected, as is one claiming to be neither.
    let both = r#"{"feature_index":1,"split_bin":3,"categories_left_bins":[1],"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(both);
    let err_text = match result {
        Ok(_) => String::new(),
        Err(e) => e.to_string(),
    };
    assert!(err_text.contains("exactly one"), "got: {err_text:?}");

    let neither = r#"{"feature_index":1,"split_bin":null,"categories_left_bins":null,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(neither);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_deserialize_wrong_value_categories() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("categories_left_bins");
    let result = SplitResult::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_result_missing_categories_left_bins() -> Result<(), ClearGbmError> {
    // Ten complete fields but no categories_left_bins: pre-categorical
    // payloads are rejected, not silently read as threshold splits.
    let json = r#"{"feature_index":1,"split_bin":3,"gain":0.5,"left_gradient_sum":1.0,"left_hessian_sum":2.0,"left_count":10,"right_gradient_sum":0.5,"right_hessian_sum":1.0,"right_count":5,"nan_direction":"Left"}"#;
    let result: Result<SplitResult, _> = serde_json::from_str(json);
    let err_text = match result {
        Ok(_) => String::new(),
        Err(e) => e.to_string(),
    };
    assert!(
        err_text.contains("categories_left_bins"),
        "got: {err_text:?}"
    );
    Ok(())
}
