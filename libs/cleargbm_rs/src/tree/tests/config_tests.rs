//! Tests for TreeBuildConfig and Tree structs.

use crate::error::ClearGbmError;
use crate::tree::nodes::EPSILON;
use crate::tree::{Tree, TreeBuildConfig};
use crate::types::{SplitConfig, TreeNode};

// =========================================================================
// TreeBuildConfig tests
// =========================================================================

#[test]
fn test_tree_build_config_new_valid() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let c = match TreeBuildConfig::new(5_usize, 10_usize, 0.0_f64, 1.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    assert_eq!(c.max_depth(), 5_usize);
    assert_eq!(c.max_leaves(), 10_usize);
    assert!(c.reg_alpha().abs() < EPSILON);
    assert!((c.reg_lambda() - 1.0_f64).abs() < EPSILON);
    Ok(())
}

#[test]
fn test_tree_build_config_negative_reg_alpha() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = TreeBuildConfig::new(5_usize, 10_usize, -0.1_f64, 1.0_f64, sc);

    assert!(config.is_err());
    assert!(matches!(
        config.err(),
        Some(ClearGbmError::InvalidParameter { name, .. }) if name == "reg_alpha"
    ));
    Ok(())
}

#[test]
fn test_tree_build_config_negative_reg_lambda() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = TreeBuildConfig::new(5_usize, 10_usize, 0.0_f64, -1.0_f64, sc);

    assert!(config.is_err());
    assert!(matches!(
        config.err(),
        Some(ClearGbmError::InvalidParameter { name, .. }) if name == "reg_lambda"
    ));
    Ok(())
}

#[test]
fn test_tree_build_config_getters() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.5_f64, 0.01_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let c = match TreeBuildConfig::new(5_usize, 10_usize, 0.1_f64, 0.5_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    assert_eq!(c.max_depth(), 5_usize);
    assert_eq!(c.max_leaves(), 10_usize);
    assert!((c.reg_alpha() - 0.1_f64).abs() < EPSILON);
    assert!((c.reg_lambda() - 0.5_f64).abs() < EPSILON);
    assert_eq!(c.split_config().min_samples_split(), 2_usize);
    Ok(())
}

#[test]
fn test_tree_build_config_serialize_deserialize() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let c = match TreeBuildConfig::new(5_usize, 10_usize, 0.1_f64, 0.5_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let json_str = match serde_json::to_string(&c) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let p: TreeBuildConfig = match serde_json::from_str(&json_str) {
        Ok(c) => c,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };

    assert_eq!(p.max_depth(), 5_usize);
    assert_eq!(p.max_leaves(), 10_usize);
    Ok(())
}

// =========================================================================
// Tree tests
// =========================================================================

#[test]
fn test_tree_new() -> Result<(), ClearGbmError> {
    let leaf = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
    let tree = Tree::new(vec![leaf], 0_usize, 1_usize);

    assert_eq!(tree.n_nodes(), 1_usize);
    assert_eq!(tree.n_leaves(), 1_usize);
    assert_eq!(tree.max_depth(), 0_usize);
    Ok(())
}

#[test]
fn test_tree_nodes_accessor() -> Result<(), ClearGbmError> {
    let leaf1 = TreeNode::new_leaf(0_usize, 0.5_f64, 50_usize);
    let leaf2 = TreeNode::new_leaf(1_usize, -0.5_f64, 50_usize);
    let tree = Tree::new(vec![leaf1, leaf2], 0_usize, 2_usize);

    let nodes = tree.nodes();
    assert_eq!(nodes.len(), 2_usize);
    assert_eq!(nodes[0_usize].node_id(), 0_usize);
    assert_eq!(nodes[1_usize].node_id(), 1_usize);
    Ok(())
}

#[test]
fn test_tree_node_access() -> Result<(), ClearGbmError> {
    let leaf = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
    let tree = Tree::new(vec![leaf.clone()], 0_usize, 1_usize);

    let node = match tree.root() {
        Ok(n) => n,
        Err(e) => return Err(e),
    };
    assert_eq!(node.node_id(), 0_usize);
    assert!(node.is_leaf());

    let node0 = match tree.node(0_usize) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };
    assert_eq!(node0.node_id(), 0_usize);

    let missing = tree.node(99_usize);
    assert!(missing.is_err());
    assert!(matches!(
        missing.err(),
        Some(ClearGbmError::NodeNotFound { node_id: 99_usize })
    ));
    Ok(())
}

#[test]
fn test_tree_empty_root_error() -> Result<(), ClearGbmError> {
    let tree = Tree::new(vec![], 0_usize, 0_usize);
    let root = tree.root();
    assert!(root.is_err());
    assert!(matches!(
        root.err(),
        Some(ClearGbmError::NodeNotFound { node_id: 0_usize })
    ));
    Ok(())
}

#[test]
fn test_tree_serialize_deserialize() -> Result<(), ClearGbmError> {
    let leaf = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
    let tree = Tree::new(vec![leaf], 1_usize, 1_usize);

    let json_str = match serde_json::to_string(&tree) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let p: Tree = match serde_json::from_str(&json_str) {
        Ok(t) => t,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };

    assert_eq!(p.n_nodes(), 1_usize);
    assert_eq!(p.max_depth(), 1_usize);
    assert_eq!(p.n_leaves(), 1_usize);
    Ok(())
}
