//! Serde error path tests for Tree and TreeBuildConfig.

use crate::error::ClearGbmError;
use crate::tree::{Tree, TreeBuildConfig};
use crate::types::{SplitConfig, TreeNode};

// =========================================================================
// TreeBuildConfig serde tests
// =========================================================================

#[test]
fn test_tree_build_config_deserialize_missing_max_depth() -> Result<(), ClearGbmError> {
    let json = r#"{"max_leaves":8,"reg_alpha":0.0,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("max_depth"));
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_missing_max_leaves() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":6,"reg_alpha":0.0,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("max_leaves"));
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_missing_reg_alpha() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":6,"max_leaves":8,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("reg_alpha"));
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_missing_reg_lambda() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":6,"max_leaves":8,"reg_alpha":0.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("reg_lambda"));
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_missing_split_config() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":6,"max_leaves":8,"reg_alpha":0.0,"reg_lambda":1.0}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("split_config"));
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_unknown_field() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":6,"max_leaves":8,"reg_alpha":0.0,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0},"unknown_field":"value"}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("unknown field"));
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":"six","max_leaves":8,"reg_alpha":0.0,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_all_fields() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":6,"max_leaves":8,"reg_alpha":0.1,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
    let config: TreeBuildConfig = match serde_json::from_str(json) {
        Ok(c) => c,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(config.max_depth(), 6_usize);
    assert_eq!(config.max_leaves(), 8_usize);
    assert!((config.reg_alpha() - 0.1_f64).abs() < 1e-10_f64);
    assert!((config.reg_lambda() - 1.0_f64).abs() < 1e-10_f64);
    Ok(())
}

#[test]
fn test_tree_build_config_serialize_roundtrip() -> Result<(), ClearGbmError> {
    let split_config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = match TreeBuildConfig::new(6_usize, 8_usize, 0.1_f64, 1.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = match serde_json::to_string(&config) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let deserialized: TreeBuildConfig = match serde_json::from_str(&json) {
        Ok(c) => c,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(config, deserialized);
    Ok(())
}

// =========================================================================
// Tree serde tests
// =========================================================================

#[test]
fn test_tree_deserialize_missing_nodes() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":3,"n_leaves":4}"#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("nodes"));
    Ok(())
}

#[test]
fn test_tree_deserialize_missing_max_depth() -> Result<(), ClearGbmError> {
    let json = r#"{"nodes":[],"n_leaves":0}"#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("max_depth"));
    Ok(())
}

#[test]
fn test_tree_deserialize_missing_n_leaves() -> Result<(), ClearGbmError> {
    let json = r#"{"nodes":[],"max_depth":0}"#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("n_leaves"));
    Ok(())
}

#[test]
fn test_tree_deserialize_unknown_field() -> Result<(), ClearGbmError> {
    let json = r#"{"nodes":[],"max_depth":0,"n_leaves":0,"unknown":"value"}"#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("unknown field"));
    Ok(())
}

#[test]
fn test_tree_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    let json = r#"{"nodes":"not_an_array","max_depth":0,"n_leaves":0}"#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_deserialize_all_fields_empty() -> Result<(), ClearGbmError> {
    let json = r#"{"nodes":[],"max_depth":0,"n_leaves":0}"#;
    let tree: Tree = match serde_json::from_str(json) {
        Ok(t) => t,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(tree.n_nodes(), 0_usize);
    assert_eq!(tree.max_depth(), 0_usize);
    assert_eq!(tree.n_leaves(), 0_usize);
    Ok(())
}

#[test]
fn test_tree_serialize_roundtrip() -> Result<(), ClearGbmError> {
    let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
    let json = match serde_json::to_string(&tree) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let deserialized: Tree = match serde_json::from_str(&json) {
        Ok(t) => t,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(tree.n_nodes(), deserialized.n_nodes());
    assert_eq!(tree.max_depth(), deserialized.max_depth());
    assert_eq!(tree.n_leaves(), deserialized.n_leaves());
    Ok(())
}

#[test]
fn test_tree_deserialize_with_nodes() -> Result<(), ClearGbmError> {
    // Create a proper JSON with a leaf node
    let json = r#"{"nodes":[{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":10,"left_child":null,"right_child":null,"nan_goes_left":true}],"max_depth":0,"n_leaves":1}"#;
    let tree: Tree = match serde_json::from_str(json) {
        Ok(t) => t,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(tree.n_nodes(), 1_usize);
    assert_eq!(tree.max_depth(), 0_usize);
    assert_eq!(tree.n_leaves(), 1_usize);
    Ok(())
}

// =========================================================================
// Type mismatch tests to trigger expecting() methods
// =========================================================================

#[test]
fn test_tree_build_config_deserialize_from_array() -> Result<(), ClearGbmError> {
    let json = r#"[1, 2, 3]"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_from_string() -> Result<(), ClearGbmError> {
    let json = r#""not a struct""#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_from_number() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_deserialize_from_array() -> Result<(), ClearGbmError> {
    let json = r#"[1, 2, 3]"#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_deserialize_from_string() -> Result<(), ClearGbmError> {
    let json = r#""not a struct""#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_deserialize_from_number() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Serialization error path tests using failing serializer
// =========================================================================

#[test]
fn test_tree_build_config_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let split_config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = match TreeBuildConfig::new(6_usize, 8_usize, 0.1_f64, 1.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    // TreeBuildConfig has 5 fields
    for fail_at in 0_usize..5_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_tree_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
    // Tree has 3 fields
    for fail_at in 0_usize..3_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = tree.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_tree_build_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let split_config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = match TreeBuildConfig::new(6_usize, 8_usize, 0.0_f64, 1.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_struct();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
    let mut ser = FailingSerializer::fail_on_struct();
    let result = tree.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let split_config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = match TreeBuildConfig::new(6_usize, 8_usize, 0.0_f64, 1.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_end();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
    let mut ser = FailingSerializer::fail_on_end();
    let result = tree.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_failing_serializer_coverage() -> Result<(), ClearGbmError> {
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
    assert!((&mut ser).serialize_f64(1.0_f64).is_ok());
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

// =========================================================================
// Failing deserializer tests
// =========================================================================

#[test]
fn test_tree_build_config_field_expecting() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(IntegerKeyDeserializer);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_field_expecting() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(IntegerKeyDeserializer);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Wrong value tests for TreeBuildConfig
// =========================================================================

#[test]
fn test_tree_build_config_wrong_value_max_depth() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("max_depth"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_wrong_value_max_leaves() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("max_leaves"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_wrong_value_reg_alpha() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("reg_alpha"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_wrong_value_reg_lambda() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("reg_lambda"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_wrong_value_split_config() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("split_config"));
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Wrong value tests for Tree
// =========================================================================

#[test]
fn test_tree_wrong_value_n_features() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(WrongValueDeserializer::new("n_features"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_wrong_value_nodes() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(WrongValueDeserializer::new("nodes"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_failing_deserializer_coverage() -> Result<(), ClearGbmError> {
    use crate::testkit::{DeError, IntegerDeserializer, IntegerKeyMapAccess};
    use serde::de::{Deserializer, Error, MapAccess};

    // Test DeError Display
    let err = DeError {
        message: "test".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("test"));

    // Test DeError custom
    let custom_err = DeError::custom("custom");
    assert!(custom_err.message.contains("custom"));

    // Test IntegerDeserializer
    struct I64Visitor;
    impl<'de> serde::de::Visitor<'de> for I64Visitor {
        type Value = i64;
        fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let de = IntegerDeserializer;
    let result = de.deserialize_any(I64Visitor);
    assert!(result.is_ok());

    // Test IntegerKeyMapAccess done state
    let mut map_access = IntegerKeyMapAccess { done: true };
    let key_result: Result<Option<String>, _> = map_access.next_key();
    assert!(matches!(key_result, Ok(None)));

    // Test IntegerKeyMapAccess next_value (returns integer successfully)
    let mut map_access2 = IntegerKeyMapAccess { done: false };
    let value_result: Result<i64, _> = map_access2.next_value();
    assert!(value_result.is_ok());

    Ok(())
}

// =========================================================================
// Duplicate field tests for TreeBuildConfig
// =========================================================================

#[test]
fn test_tree_build_config_duplicate_max_depth() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(DuplicateFieldDeserializer::new("max_depth"));
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("duplicate"));
    Ok(())
}

#[test]
fn test_tree_build_config_json_duplicate_max_depth() -> Result<(), ClearGbmError> {
    // JSON with duplicate max_depth field - our manual deserializer should reject it
    let json = r#"{"max_depth":3,"max_leaves":0,"reg_alpha":0.0,"reg_lambda":0.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0},"max_depth":5}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => "".to_string(),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("duplicate"));
    Ok(())
}

#[test]
fn test_tree_build_config_duplicate_max_leaves() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(DuplicateFieldDeserializer::new("max_leaves"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_json_duplicate_max_leaves() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":3,"max_leaves":0,"max_leaves":5,"reg_alpha":0.0,"reg_lambda":0.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0}}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_json_duplicate_reg_alpha() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":3,"max_leaves":0,"reg_alpha":0.0,"reg_alpha":0.5,"reg_lambda":0.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0}}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_json_duplicate_reg_lambda() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":3,"max_leaves":0,"reg_alpha":0.0,"reg_lambda":0.0,"reg_lambda":0.5,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0}}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_json_duplicate_split_config() -> Result<(), ClearGbmError> {
    let json = r#"{"max_depth":3,"max_leaves":0,"reg_alpha":0.0,"reg_lambda":0.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0},"split_config":{"min_samples_split":3,"min_samples_leaf":2,"max_bins":32,"reg_lambda":0.1,"min_gain":0.01}}"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_duplicate_reg_alpha() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(DuplicateFieldDeserializer::new("reg_alpha"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_duplicate_reg_lambda() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(DuplicateFieldDeserializer::new("reg_lambda"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_duplicate_split_config() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(DuplicateFieldDeserializer::new("split_config"));
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Duplicate field tests for Tree
// =========================================================================

#[test]
fn test_tree_duplicate_nodes() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(DuplicateFieldDeserializer::new("nodes"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_duplicate_max_depth() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(DuplicateFieldDeserializer::new("max_depth"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_json_duplicate_max_depth() -> Result<(), ClearGbmError> {
    let node = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true}"#;
    let json = format!(r#"{{"nodes":[{node}],"max_depth":2,"max_depth":3,"n_leaves":1}}"#);
    let result: Result<Tree, _> = serde_json::from_str(&json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_duplicate_n_leaves() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(DuplicateFieldDeserializer::new("n_leaves"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_json_duplicate_n_leaves() -> Result<(), ClearGbmError> {
    let node = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true}"#;
    let json = format!(r#"{{"nodes":[{node}],"max_depth":2,"n_leaves":1,"n_leaves":2}}"#);
    let result: Result<Tree, _> = serde_json::from_str(&json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_json_duplicate_nodes() -> Result<(), ClearGbmError> {
    let node = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true}"#;
    let json = format!(r#"{{"nodes":[{node}],"nodes":[{node}],"max_depth":2,"n_leaves":1}}"#);
    let result: Result<Tree, _> = serde_json::from_str(&json);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Additional wrong value tests for Tree
// =========================================================================

#[test]
fn test_tree_wrong_value_max_depth() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(WrongValueDeserializer::new("max_depth"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_wrong_value_n_leaves() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(WrongValueDeserializer::new("n_leaves"));
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Struct duplicate field tests (for complex types like split_config, nodes)
// =========================================================================

#[test]
fn test_tree_build_config_struct_duplicate_split_config() -> Result<(), ClearGbmError> {
    // Test line 179: duplicate_field("split_config") for struct type
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;
    let result =
        TreeBuildConfig::deserialize(StructDuplicateFieldDeserializer::new("split_config"));
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for duplicate split_config".to_string(),
            })
        }
        Err(e) => format!("{e:?}"),
    };
    assert!(
        err_msg.contains("duplicate") || err_msg.contains("split_config"),
        "Expected duplicate field error, got: {err_msg}"
    );
    Ok(())
}

#[test]
fn test_tree_struct_duplicate_nodes() -> Result<(), ClearGbmError> {
    // Test line 337: duplicate_field("nodes") for sequence type
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(StructDuplicateFieldDeserializer::new("nodes"));
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error for duplicate nodes".to_string(),
            })
        }
        Err(e) => format!("{e:?}"),
    };
    assert!(
        err_msg.contains("duplicate") || err_msg.contains("nodes"),
        "Expected duplicate field error, got: {err_msg}"
    );
    Ok(())
}

// =========================================================================
// Error propagation tests (covers Err(e) => return Err(e) paths)
// =========================================================================

#[test]
fn test_tree_build_config_error_on_key() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnKeyDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(ErrorOnKeyDeserializer);
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
    };
    assert!(err_msg.contains("next_key"));
    Ok(())
}

#[test]
fn test_tree_build_config_error_on_value() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(ErrorOnValueDeserializer::new("max_depth"));
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
    };
    assert!(err_msg.contains("next_value"));
    Ok(())
}

#[test]
fn test_tree_build_config_error_on_value_max_leaves() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(ErrorOnValueDeserializer::new("max_leaves"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_error_on_value_reg_alpha() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(ErrorOnValueDeserializer::new("reg_alpha"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_error_on_value_reg_lambda() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(ErrorOnValueDeserializer::new("reg_lambda"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_error_on_value_split_config() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(ErrorOnValueDeserializer::new("split_config"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_error_on_key() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnKeyDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(ErrorOnKeyDeserializer);
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
    };
    assert!(err_msg.contains("next_key"));
    Ok(())
}

#[test]
fn test_tree_error_on_value() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(ErrorOnValueDeserializer::new("nodes"));
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected error".to_string(),
            })
        }
    };
    assert!(err_msg.contains("next_value"));
    Ok(())
}

#[test]
fn test_tree_error_on_value_max_depth() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(ErrorOnValueDeserializer::new("max_depth"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_error_on_value_n_leaves() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(ErrorOnValueDeserializer::new("n_leaves"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_error_on_value_n_features() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(ErrorOnValueDeserializer::new("n_features"));
    assert!(result.is_err());
    Ok(())
}
