//! Tests for error conditions in prediction functions.

use crate::error::ClearGbmError;
use crate::predict::{predict_ensemble, predict_single, predict_tree, PredictEnsembleConfig};
use crate::tree::Tree;
use crate::types::{TreeNode, TreeNodeConfig};

// --- predict_single error tests ---

#[test]
fn test_predict_single_empty_tree() -> Result<(), ClearGbmError> {
    let tree = Tree::new(vec![], 0_usize, 0_usize);
    let features = [1.0_f64];
    let result = predict_single(&tree, &features);
    assert!(matches!(
        result,
        Err(ClearGbmError::NodeNotFound { node_id: 0_usize })
    ));
    Ok(())
}

#[test]
fn test_predict_single_feature_index_out_of_bounds() -> Result<(), ClearGbmError> {
    // Internal node references feature 5, but only 2 features provided
    let root = TreeNode::new_internal(TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 5_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    });
    let left_leaf = TreeNode::new_leaf(1_usize, -1.0_f64, 5_usize);
    let right_leaf = TreeNode::new_leaf(2_usize, 1.0_f64, 5_usize);
    let tree = Tree::new(vec![root, left_leaf, right_leaf], 1_usize, 2_usize);

    let features = [0.3_f64, 0.7_f64];
    let result = predict_single(&tree, &features);
    assert!(matches!(
        result,
        Err(ClearGbmError::FeatureIndexOutOfBounds {
            index: 5_usize,
            n_features: 2_usize
        })
    ));
    Ok(())
}

#[test]
fn test_predict_single_missing_feature_index() -> Result<(), ClearGbmError> {
    // Malformed: internal node (is_leaf=false) but feature_index=None
    let bad_node = TreeNode {
        node_id: 0_usize,
        is_leaf: false,
        feature_index: None,
        threshold: Some(0.5_f64),
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: Some(1_usize),
        right_child: Some(2_usize),
        nan_goes_left: true,
        categories_goes_left: None,
    };
    let tree = Tree::new(vec![bad_node], 0_usize, 0_usize);
    let features = [0.3_f64];
    let result = predict_single(&tree, &features);
    assert!(matches!(
        result,
        Err(ClearGbmError::TreeConstructionFailed { .. })
    ));
    if let Err(ClearGbmError::TreeConstructionFailed { reason }) = &result {
        assert!(reason.contains("missing feature_index"));
    }
    Ok(())
}

#[test]
fn test_predict_single_missing_threshold() -> Result<(), ClearGbmError> {
    let bad_node = TreeNode {
        node_id: 0_usize,
        is_leaf: false,
        feature_index: Some(0_usize),
        threshold: None,
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: Some(1_usize),
        right_child: Some(2_usize),
        nan_goes_left: true,
        categories_goes_left: None,
    };
    let tree = Tree::new(vec![bad_node], 0_usize, 0_usize);
    let features = [0.3_f64];
    let result = predict_single(&tree, &features);
    assert!(matches!(
        result,
        Err(ClearGbmError::TreeConstructionFailed { .. })
    ));
    if let Err(ClearGbmError::TreeConstructionFailed { reason }) = &result {
        assert!(reason.contains("missing threshold"));
    }
    Ok(())
}

#[test]
fn test_predict_single_missing_left_child() -> Result<(), ClearGbmError> {
    let bad_node = TreeNode {
        node_id: 0_usize,
        is_leaf: false,
        feature_index: Some(0_usize),
        threshold: Some(0.5_f64),
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: None,
        right_child: Some(2_usize),
        nan_goes_left: true,
        categories_goes_left: None,
    };
    let tree = Tree::new(vec![bad_node], 0_usize, 0_usize);
    let features = [0.3_f64];
    let result = predict_single(&tree, &features);
    assert!(matches!(
        result,
        Err(ClearGbmError::TreeConstructionFailed { .. })
    ));
    if let Err(ClearGbmError::TreeConstructionFailed { reason }) = &result {
        assert!(reason.contains("missing left_child"));
    }
    Ok(())
}

#[test]
fn test_predict_single_missing_right_child() -> Result<(), ClearGbmError> {
    let bad_node = TreeNode {
        node_id: 0_usize,
        is_leaf: false,
        feature_index: Some(0_usize),
        threshold: Some(0.5_f64),
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: Some(1_usize),
        right_child: None,
        nan_goes_left: true,
        categories_goes_left: None,
    };
    let leaf = TreeNode::new_leaf(1_usize, -1.0_f64, 5_usize);
    let tree = Tree::new(vec![bad_node, leaf], 0_usize, 0_usize);
    let features = [0.3_f64];
    let result = predict_single(&tree, &features);
    assert!(matches!(
        result,
        Err(ClearGbmError::TreeConstructionFailed { .. })
    ));
    if let Err(ClearGbmError::TreeConstructionFailed { reason }) = &result {
        assert!(reason.contains("missing right_child"));
    }
    Ok(())
}

#[test]
fn test_predict_single_child_node_not_found() -> Result<(), ClearGbmError> {
    // Internal node references child node_id 99 which does not exist
    let root = TreeNode::new_internal(TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 0_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: 99_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    });
    let tree = Tree::new(vec![root], 0_usize, 0_usize);
    let features = [0.3_f64]; // <= 0.5, goes left to node 99
    let result = predict_single(&tree, &features);
    assert!(matches!(
        result,
        Err(ClearGbmError::NodeNotFound { node_id: 99_usize })
    ));
    Ok(())
}

#[test]
fn test_predict_single_cycle_guard() -> Result<(), ClearGbmError> {
    // Create a cycle: node 0 -> node 1 -> node 0 (via child pointers)
    let node0 = TreeNode::new_internal(TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 0_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 1_usize,
        nan_goes_left: true,
    });
    let node1 = TreeNode::new_internal(TreeNodeConfig {
        node_id: 1_usize,
        feature_index: 0_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: 0_usize,
        right_child: 0_usize,
        nan_goes_left: true,
    });
    let tree = Tree::new(vec![node0, node1], 1_usize, 0_usize);
    let features = [0.3_f64];
    let result = predict_single(&tree, &features);
    assert!(matches!(
        result,
        Err(ClearGbmError::TreeConstructionFailed { .. })
    ));
    if let Err(ClearGbmError::TreeConstructionFailed { reason }) = &result {
        assert!(reason.contains("exceeded maximum iterations"));
    }
    Ok(())
}

// --- predict_tree error tests ---

#[test]
fn test_predict_tree_empty_features() -> Result<(), ClearGbmError> {
    let tree = Tree::new(
        vec![TreeNode::new_leaf(0_usize, 1.0_f64, 10_usize)],
        0_usize,
        1_usize,
    );
    let features: &[&[f64]] = &[];
    let result = predict_tree(&tree, features);
    assert!(matches!(result, Err(ClearGbmError::EmptyInput { .. })));
    if let Err(ClearGbmError::EmptyInput { context }) = &result {
        assert!(context.contains("predict_tree"));
    }
    Ok(())
}

#[test]
fn test_predict_tree_propagates_predict_single_error() -> Result<(), ClearGbmError> {
    // Tree that will fail on predict_single (empty tree)
    let tree = Tree::new(vec![], 0_usize, 0_usize);
    let row: &[f64] = &[1.0_f64];
    let features: &[&[f64]] = &[row];
    let result = predict_tree(&tree, features);
    assert!(matches!(result, Err(ClearGbmError::NodeNotFound { .. })));
    Ok(())
}

// --- predict_ensemble error tests ---

#[test]
fn test_predict_ensemble_empty_features() -> Result<(), ClearGbmError> {
    let tree = Tree::new(
        vec![TreeNode::new_leaf(0_usize, 1.0_f64, 10_usize)],
        0_usize,
        1_usize,
    );
    let config = match PredictEnsembleConfig::new(0.0_f64, 0.1_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let features: &[&[f64]] = &[];
    let result = predict_ensemble(&[tree], features, &config);
    assert!(matches!(result, Err(ClearGbmError::EmptyInput { .. })));
    if let Err(ClearGbmError::EmptyInput { context }) = &result {
        assert!(context.contains("predict_ensemble"));
    }
    Ok(())
}

#[test]
fn test_predict_ensemble_empty_trees() -> Result<(), ClearGbmError> {
    let config = match PredictEnsembleConfig::new(0.0_f64, 0.1_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let row: &[f64] = &[1.0_f64];
    let features: &[&[f64]] = &[row];
    let result = predict_ensemble(&[], features, &config);
    assert!(matches!(result, Err(ClearGbmError::EmptyInput { .. })));
    if let Err(ClearGbmError::EmptyInput { context }) = &result {
        assert!(context.contains("trees for predict_ensemble"));
    }
    Ok(())
}

#[test]
fn test_predict_ensemble_propagates_predict_single_error() -> Result<(), ClearGbmError> {
    // Tree with OOB feature index
    let root = TreeNode::new_internal(TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 10_usize,
        threshold: 0.5_f64,
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    });
    let left_leaf = TreeNode::new_leaf(1_usize, -1.0_f64, 5_usize);
    let right_leaf = TreeNode::new_leaf(2_usize, 1.0_f64, 5_usize);
    let tree = Tree::new(vec![root, left_leaf, right_leaf], 1_usize, 2_usize);

    let config = match PredictEnsembleConfig::new(0.0_f64, 0.1_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let row: &[f64] = &[0.3_f64];
    let features: &[&[f64]] = &[row];
    let result = predict_ensemble(&[tree], features, &config);
    assert!(matches!(
        result,
        Err(ClearGbmError::FeatureIndexOutOfBounds { .. })
    ));
    Ok(())
}

// --- PredictEnsembleConfig error tests ---

#[test]
fn test_predict_ensemble_config_zero_lr() -> Result<(), ClearGbmError> {
    let result = PredictEnsembleConfig::new(0.0_f64, 0.0_f64);
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { .. })
    ));
    if let Err(ClearGbmError::InvalidParameter { name, reason }) = &result {
        assert_eq!(name, "learning_rate");
        assert!(reason.contains("(0.0, 1.0]"));
    }
    Ok(())
}

#[test]
fn test_predict_ensemble_config_negative_lr() -> Result<(), ClearGbmError> {
    let result = PredictEnsembleConfig::new(0.0_f64, -0.1_f64);
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { .. })
    ));
    Ok(())
}

#[test]
fn test_predict_ensemble_config_lr_above_one() -> Result<(), ClearGbmError> {
    let result = PredictEnsembleConfig::new(0.0_f64, 1.5_f64);
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { .. })
    ));
    Ok(())
}
