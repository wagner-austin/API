//! Tests for TreeNode type.

use crate::error::ClearGbmError;
use crate::types::{TreeNode, TreeNodeConfig};

#[test]
fn test_tree_node_new_leaf() -> Result<(), ClearGbmError> {
    let node = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
    assert_eq!(node.node_id(), 0_usize);
    assert!(node.is_leaf());
    assert_eq!(node.feature_index(), None);
    assert_eq!(node.threshold(), None);
    assert!((node.value() - 0.5_f64).abs() < f64::EPSILON);
    assert_eq!(node.n_samples(), 100_usize);
    assert_eq!(node.left_child(), None);
    assert_eq!(node.right_child(), None);
    assert!(node.nan_goes_left());
    Ok(())
}

#[test]
fn test_tree_node_new_internal() -> Result<(), ClearGbmError> {
    let config = TreeNodeConfig {
        node_id: 1_usize,
        feature_index: 3_usize,
        threshold: 0.25_f64,
        value: 0.1_f64,
        n_samples: 50_usize,
        left_child: 2_usize,
        right_child: 3_usize,
        nan_goes_left: false,
    };
    let node = TreeNode::new_internal(config);
    assert_eq!(node.node_id(), 1_usize);
    assert!(!node.is_leaf());
    assert_eq!(node.feature_index(), Some(3_usize));
    assert_eq!(node.threshold(), Some(0.25_f64));
    assert!((node.value() - 0.1_f64).abs() < f64::EPSILON);
    assert_eq!(node.n_samples(), 50_usize);
    assert_eq!(node.left_child(), Some(2_usize));
    assert_eq!(node.right_child(), Some(3_usize));
    assert!(!node.nan_goes_left());
    Ok(())
}

#[test]
fn test_tree_node_clone() -> Result<(), ClearGbmError> {
    let node = TreeNode::new_leaf(5_usize, 1.0_f64, 10_usize);
    let cloned = node.clone();
    assert_eq!(node, cloned);
    Ok(())
}

#[test]
fn test_tree_node_debug() -> Result<(), ClearGbmError> {
    let node = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
    let debug_str = format!("{node:?}");
    assert!(debug_str.contains("TreeNode"));
    assert!(debug_str.contains("node_id: 0"));
    Ok(())
}

#[test]
fn test_tree_node_serialize_deserialize() -> Result<(), ClearGbmError> {
    let config = TreeNodeConfig {
        node_id: 1_usize,
        feature_index: 2_usize,
        threshold: 0.5_f64,
        value: 0.3_f64,
        n_samples: 200_usize,
        left_child: 3_usize,
        right_child: 4_usize,
        nan_goes_left: true,
    };
    let node = TreeNode::new_internal(config);
    let json_str = match serde_json::to_string(&node) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let parsed: TreeNode = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(parsed, node);
    Ok(())
}
