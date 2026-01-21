//! Serde error path tests for SplitConfig.

use crate::error::ClearGbmError;
use crate::types::SplitConfig;

#[test]
fn test_split_config_deserialize_missing_min_gain() -> Result<(), ClearGbmError> {
    let json = r#"{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("min_gain"));
    Ok(())
}

#[test]
fn test_split_config_deserialize_missing_min_samples_split() -> Result<(), ClearGbmError> {
    let json = r#"{"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("min_samples_split"));
    Ok(())
}

#[test]
fn test_split_config_deserialize_missing_min_samples_leaf() -> Result<(), ClearGbmError> {
    let json = r#"{"min_samples_split":2,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("min_samples_leaf"));
    Ok(())
}

#[test]
fn test_split_config_deserialize_missing_max_bins() -> Result<(), ClearGbmError> {
    let json = r#"{"min_samples_split":2,"min_samples_leaf":1,"reg_lambda":0.0,"min_gain":0.0}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("max_bins"));
    Ok(())
}

#[test]
fn test_split_config_deserialize_missing_reg_lambda() -> Result<(), ClearGbmError> {
    let json = r#"{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"min_gain":0.0}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("reg_lambda"));
    Ok(())
}

#[test]
fn test_split_config_deserialize_unknown_field() -> Result<(), ClearGbmError> {
    let json = r#"{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0,"extra":999}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    let json = r#"{"min_samples_split":true,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_all_fields() -> Result<(), ClearGbmError> {
    let json = r#"{"min_samples_split":10,"min_samples_leaf":5,"max_bins":128,"reg_lambda":0.5,"min_gain":0.01}"#;
    let config: SplitConfig = match serde_json::from_str(json) {
        Ok(c) => c,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(config.min_samples_split(), 10_usize);
    assert_eq!(config.min_samples_leaf(), 5_usize);
    assert_eq!(config.max_bins(), 128_usize);
    assert!((config.reg_lambda() - 0.5_f64).abs() < 1e-10_f64);
    assert!((config.min_gain() - 0.01_f64).abs() < 1e-10_f64);
    Ok(())
}

// =========================================================================
// serde_json invalid type tests (covers visit_map error paths)
// =========================================================================

#[test]
fn test_split_config_serde_json_invalid_min_gain_type() -> Result<(), ClearGbmError> {
    let json = r#"{"min_gain":"not_a_number","min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_serde_json_invalid_min_samples_leaf_type() -> Result<(), ClearGbmError> {
    let json = r#"{"min_gain":0.0,"min_samples_split":2,"min_samples_leaf":"not_a_number","max_bins":64,"reg_lambda":0.0}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_serde_json_invalid_max_bins_type() -> Result<(), ClearGbmError> {
    let json = r#"{"min_gain":0.0,"min_samples_split":2,"min_samples_leaf":1,"max_bins":"not_a_number","reg_lambda":0.0}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_serde_json_invalid_reg_lambda_type() -> Result<(), ClearGbmError> {
    let json = r#"{"min_gain":0.0,"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":"not_a_number"}"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// FailingDeserializer tests for SplitConfig
// =========================================================================

#[test]
fn test_split_config_deserialize_with_integer_key() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;

    let deser = IntegerKeyDeserializer;
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_wrong_value_min_samples_split() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("min_samples_split");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_wrong_value_min_samples_leaf() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("min_samples_leaf");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_wrong_value_max_bins() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("max_bins");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_wrong_value_reg_lambda() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("reg_lambda");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_wrong_value_min_gain() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("min_gain");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_error_on_key() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnKeyDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnKeyDeserializer;
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_error_on_value_min_samples_split() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("min_samples_split");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_error_on_value_min_samples_leaf() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("min_samples_leaf");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_error_on_value_max_bins() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("max_bins");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_error_on_value_reg_lambda() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("reg_lambda");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_error_on_value_min_gain() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("min_gain");
    let result = SplitConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests to exercise DuplicateFieldMapAccess and StructDuplicateFieldMapAccess
// These types don't check for duplicates, but calling these exercises the code paths.
// =========================================================================

#[test]
fn test_split_config_deserialize_duplicate_field_map_access() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;

    // DuplicateFieldMapAccess returns the same field twice with integer values.
    // SplitConfig doesn't check for duplicates, but this fails because only one field is provided.
    let deser = DuplicateFieldDeserializer::new("min_gain");
    let result = SplitConfig::deserialize(deser);
    // Should fail due to missing other required fields
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_struct_duplicate_field_map_access() -> Result<(), ClearGbmError> {
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;

    // StructDuplicateFieldMapAccess returns the same field twice with struct/seq values.
    // SplitConfig expects numeric fields, so this will fail on type mismatch.
    let deser = StructDuplicateFieldDeserializer::new("min_gain");
    let result = SplitConfig::deserialize(deser);
    // Should fail due to type mismatch or missing fields
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// FailingSerializer tests for SplitConfig
// =========================================================================

#[test]
fn test_split_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_struct();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    // SplitConfig has 5 fields
    for fail_at in 0_usize..5_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_split_config_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_end();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}
