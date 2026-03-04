//! Tests for single-sample tree prediction.

use crate::error::ClearGbmError;
use crate::predict::predict_single;
use crate::tree::Tree;
use crate::types::{TreeNode, TreeNodeConfig};

/// Helper to build a simple 3-node tree: root splits on feature 0 at threshold 0.5.
/// Left leaf (node 1) value = -1.0, right leaf (node 2) value = 1.0.
fn make_simple_tree() -> Tree {
    let root = TreeNode::new_internal(TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 0_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    });
    let left_leaf = TreeNode::new_leaf(1_usize, -1.0_f64, 5_usize);
    let right_leaf = TreeNode::new_leaf(2_usize, 1.0_f64, 5_usize);
    Tree::new(vec![root, left_leaf, right_leaf], 1_usize, 2_usize)
}

/// Helper to build a depth-2 tree with 4 leaves.
/// Root: feature 0, threshold 0.5
///   Left (node 1): feature 1, threshold 0.3 (nan_goes_left=false)
///     Left leaf (node 3): value = -2.0
///     Right leaf (node 4): value = -0.5
///   Right (node 2): feature 1, threshold 0.7 (nan_goes_left=true)
///     Left leaf (node 5): value = 0.5
///     Right leaf (node 6): value = 2.0
fn make_deep_tree() -> Tree {
    let root = TreeNode::new_internal(TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 0_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 100_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    });
    let node1 = TreeNode::new_internal(TreeNodeConfig {
        node_id: 1_usize,
        feature_index: 1_usize,
        threshold: 0.3_f64,
        value: 0.0_f64,
        n_samples: 50_usize,
        left_child: 3_usize,
        right_child: 4_usize,
        nan_goes_left: false,
    });
    let node2 = TreeNode::new_internal(TreeNodeConfig {
        node_id: 2_usize,
        feature_index: 1_usize,
        threshold: 0.7_f64,
        value: 0.0_f64,
        n_samples: 50_usize,
        left_child: 5_usize,
        right_child: 6_usize,
        nan_goes_left: true,
    });
    let leaf3 = TreeNode::new_leaf(3_usize, -2.0_f64, 25_usize);
    let leaf4 = TreeNode::new_leaf(4_usize, -0.5_f64, 25_usize);
    let leaf5 = TreeNode::new_leaf(5_usize, 0.5_f64, 25_usize);
    let leaf6 = TreeNode::new_leaf(6_usize, 2.0_f64, 25_usize);
    Tree::new(
        vec![root, node1, node2, leaf3, leaf4, leaf5, leaf6],
        2_usize,
        4_usize,
    )
}

#[test]
fn test_predict_single_leaf_only_tree() -> Result<(), ClearGbmError> {
    let tree = Tree::new(
        vec![TreeNode::new_leaf(0_usize, 0.42_f64, 100_usize)],
        0_usize,
        1_usize,
    );
    let features = [1.0_f64, 2.0_f64, 3.0_f64];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 0.42_f64).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_goes_left() -> Result<(), ClearGbmError> {
    let tree = make_simple_tree();
    // feature[0] = 0.3 <= 0.5, goes left -> leaf value -1.0
    let features = [0.3_f64];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - (-1.0_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_goes_right() -> Result<(), ClearGbmError> {
    let tree = make_simple_tree();
    // feature[0] = 0.8 > 0.5, goes right -> leaf value 1.0
    let features = [0.8_f64];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 1.0_f64).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_exact_threshold_goes_left() -> Result<(), ClearGbmError> {
    let tree = make_simple_tree();
    // feature[0] = 0.5 <= 0.5, goes left -> leaf value -1.0
    let features = [0.5_f64];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - (-1.0_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_nan_goes_left() -> Result<(), ClearGbmError> {
    let tree = make_simple_tree(); // nan_goes_left = true at root
    let features = [f64::NAN];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - (-1.0_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_nan_goes_right() -> Result<(), ClearGbmError> {
    // Build tree with nan_goes_left = false
    let root = TreeNode::new_internal(TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 0_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: false,
    });
    let left_leaf = TreeNode::new_leaf(1_usize, -1.0_f64, 5_usize);
    let right_leaf = TreeNode::new_leaf(2_usize, 1.0_f64, 5_usize);
    let tree = Tree::new(vec![root, left_leaf, right_leaf], 1_usize, 2_usize);

    let features = [f64::NAN];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 1.0_f64).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_deep_tree_left_left() -> Result<(), ClearGbmError> {
    let tree = make_deep_tree();
    // feature[0]=0.2 <= 0.5 -> node1, feature[1]=0.1 <= 0.3 -> leaf3: -2.0
    let features = [0.2_f64, 0.1_f64];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - (-2.0_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_deep_tree_left_right() -> Result<(), ClearGbmError> {
    let tree = make_deep_tree();
    // feature[0]=0.2 <= 0.5 -> node1, feature[1]=0.5 > 0.3 -> leaf4: -0.5
    let features = [0.2_f64, 0.5_f64];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - (-0.5_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_deep_tree_right_left() -> Result<(), ClearGbmError> {
    let tree = make_deep_tree();
    // feature[0]=0.8 > 0.5 -> node2, feature[1]=0.5 <= 0.7 -> leaf5: 0.5
    let features = [0.8_f64, 0.5_f64];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 0.5_f64).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_deep_tree_right_right() -> Result<(), ClearGbmError> {
    let tree = make_deep_tree();
    // feature[0]=0.8 > 0.5 -> node2, feature[1]=0.9 > 0.7 -> leaf6: 2.0
    let features = [0.8_f64, 0.9_f64];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 2.0_f64).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_deep_tree_nan_at_depth1() -> Result<(), ClearGbmError> {
    let tree = make_deep_tree();
    // feature[0]=0.2 <= 0.5 -> node1 (nan_goes_left=false at node1)
    // feature[1]=NaN -> goes right -> leaf4: -0.5
    let features = [0.2_f64, f64::NAN];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - (-0.5_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_deep_tree_nan_at_depth1_goes_left() -> Result<(), ClearGbmError> {
    let tree = make_deep_tree();
    // feature[0]=0.8 > 0.5 -> node2 (nan_goes_left=true at node2)
    // feature[1]=NaN -> goes left -> leaf5: 0.5
    let features = [0.8_f64, f64::NAN];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 0.5_f64).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_single_extra_features_ignored() -> Result<(), ClearGbmError> {
    let tree = make_simple_tree();
    // Only feature 0 is used; extra features are ignored
    let features = [0.3_f64, 99.0_f64, 100.0_f64];
    let result = predict_single(&tree, &features);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - (-1.0_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}
