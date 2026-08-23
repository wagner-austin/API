//! Tree deserialization error paths: failing and specialized
//! deserializers driving every per-field error arm.

use crate::error::ClearGbmError;
use crate::tree::{Tree, TreeBuildConfig};

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
    let node = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
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
    let node = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let json = format!(r#"{{"nodes":[{node}],"max_depth":2,"n_leaves":1,"n_leaves":2}}"#);
    let result: Result<Tree, _> = serde_json::from_str(&json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_json_duplicate_nodes() -> Result<(), ClearGbmError> {
    let node = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
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
