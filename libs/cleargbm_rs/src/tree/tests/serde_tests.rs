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
    let json = r#"{"nodes":[{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":10,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}],"max_depth":0,"n_leaves":1}"#;
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
