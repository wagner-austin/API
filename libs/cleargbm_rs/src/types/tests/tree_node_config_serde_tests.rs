//! Serde error path tests for TreeNodeConfig.

use crate::error::ClearGbmError;
use crate::types::TreeNodeConfig;

#[test]
fn test_tree_node_config_deserialize_missing_nan_goes_left() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_missing_node_id() -> Result<(), ClearGbmError> {
    let json = r#"{"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_missing_feature_index() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_missing_threshold() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"feature_index":2,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_missing_value() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_missing_n_samples() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"value":1.0,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_missing_left_child() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"right_child":3,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_missing_right_child() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_unknown_field() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true,"unknown_field":42}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":"not_a_number","feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_all_fields() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
    let config: TreeNodeConfig = match serde_json::from_str(json) {
        Ok(c) => c,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(config.node_id, 1_usize);
    assert_eq!(config.feature_index, 2_usize);
    assert!((config.threshold - 0.5_f64).abs() < 1e-10_f64);
    assert!((config.value - 1.0_f64).abs() < 1e-10_f64);
    assert_eq!(config.n_samples, 100_usize);
    assert_eq!(config.left_child, 2_usize);
    assert_eq!(config.right_child, 3_usize);
    assert!(config.nan_goes_left);
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_duplicate_field() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":1,"node_id":99,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
    let config: TreeNodeConfig = match serde_json::from_str(json) {
        Ok(c) => c,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(config.node_id, 99_usize);
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_roundtrip() -> Result<(), ClearGbmError> {
    let config = TreeNodeConfig {
        node_id: 5_usize,
        feature_index: 2_usize,
        threshold: 0.75_f64,
        value: 0.123_f64,
        n_samples: 500_usize,
        left_child: 10_usize,
        right_child: 11_usize,
        nan_goes_left: false,
    };
    let json_str = match serde_json::to_string(&config) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let parsed: TreeNodeConfig = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(parsed.node_id, config.node_id);
    assert_eq!(parsed.feature_index, config.feature_index);
    assert!((parsed.threshold - config.threshold).abs() < 1e-10_f64);
    assert!((parsed.value - config.value).abs() < 1e-10_f64);
    assert_eq!(parsed.n_samples, config.n_samples);
    assert_eq!(parsed.left_child, config.left_child);
    assert_eq!(parsed.right_child, config.right_child);
    assert_eq!(parsed.nan_goes_left, config.nan_goes_left);
    Ok(())
}

// =========================================================================
// serde_json invalid type tests (covers visit_map error paths)
// =========================================================================

#[test]
fn test_tree_node_config_serde_json_invalid_node_id_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":"not_a_number","feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serde_json_invalid_value_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"feature_index":null,"threshold":null,"value":"not_a_number","n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serde_json_invalid_n_samples_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"feature_index":null,"threshold":null,"value":0.5,"n_samples":"not_a_number","left_child":null,"right_child":null,"nan_goes_left":true}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serde_json_invalid_nan_goes_left_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":"not_a_bool"}"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// FailingDeserializer tests for TreeNodeConfig
// =========================================================================

#[test]
fn test_tree_node_config_deserialize_with_integer_key() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;

    let deser = IntegerKeyDeserializer;
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_wrong_value_node_id() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("node_id");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_wrong_value_feature_index() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("feature_index");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_wrong_value_threshold() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("threshold");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_wrong_value_value() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("value");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_wrong_value_n_samples() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("n_samples");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_wrong_value_left_child() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("left_child");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_wrong_value_right_child() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("right_child");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_wrong_value_nan_goes_left() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("nan_goes_left");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_error_on_key() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnKeyDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnKeyDeserializer;
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_error_on_value_node_id() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("node_id");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_error_on_value_feature_index() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("feature_index");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_error_on_value_threshold() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("threshold");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_error_on_value_value() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("value");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_error_on_value_n_samples() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("n_samples");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_error_on_value_left_child() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("left_child");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_error_on_value_right_child() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("right_child");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_error_on_value_nan_goes_left() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("nan_goes_left");
    let result = TreeNodeConfig::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests to exercise DuplicateFieldMapAccess and StructDuplicateFieldMapAccess
// These types don't check for duplicates, but calling these exercises the code paths.
// =========================================================================

#[test]
fn test_tree_node_config_deserialize_duplicate_field_map_access() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;

    // DuplicateFieldMapAccess returns the same field twice with integer values.
    // TreeNodeConfig doesn't check for duplicates, but this fails because only one field is provided.
    let deser = DuplicateFieldDeserializer::new("node_id");
    let result = TreeNodeConfig::deserialize(deser);
    // Should fail due to missing other required fields
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_struct_duplicate_field_map_access() -> Result<(), ClearGbmError>
{
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;

    // StructDuplicateFieldMapAccess returns the same field twice with struct/seq values.
    // TreeNodeConfig expects specific field types, so this will fail on type mismatch.
    let deser = StructDuplicateFieldDeserializer::new("node_id");
    let result = TreeNodeConfig::deserialize(deser);
    // Should fail due to type mismatch or missing fields
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// FailingSerializer tests for TreeNodeConfig
// =========================================================================

#[test]
fn test_tree_node_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 100_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_on_struct();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 100_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    // TreeNodeConfig has 8 fields
    for fail_at in 0_usize..8_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 100_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_on_end();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}
