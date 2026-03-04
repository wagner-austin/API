//! Tests for batch tree prediction.

use crate::error::ClearGbmError;
use crate::predict::{predict_single, predict_tree};
use crate::tree::Tree;
use crate::types::{TreeNode, TreeNodeConfig};

/// Helper to build a simple 3-node tree.
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

#[test]
fn test_predict_tree_single_sample() -> Result<(), ClearGbmError> {
    let tree = make_simple_tree();
    let row: &[f64] = &[0.3_f64];
    let features: &[&[f64]] = &[row];
    let result = predict_tree(&tree, features);
    assert!(result.is_ok());
    if let Ok(preds) = result {
        assert_eq!(preds.len(), 1_usize);
        assert!((preds[0_usize] - (-1.0_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_tree_multiple_samples() -> Result<(), ClearGbmError> {
    let tree = make_simple_tree();
    let row1: &[f64] = &[0.3_f64]; // left -> -1.0
    let row2: &[f64] = &[0.8_f64]; // right -> 1.0
    let row3: &[f64] = &[0.5_f64]; // left (<=) -> -1.0
    let features: &[&[f64]] = &[row1, row2, row3];
    let result = predict_tree(&tree, features);
    assert!(result.is_ok());
    if let Ok(preds) = result {
        assert_eq!(preds.len(), 3_usize);
        assert!((preds[0_usize] - (-1.0_f64)).abs() < 1e-15_f64);
        assert!((preds[1_usize] - 1.0_f64).abs() < 1e-15_f64);
        assert!((preds[2_usize] - (-1.0_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_tree_matches_predict_single() -> Result<(), ClearGbmError> {
    let tree = make_simple_tree();
    let rows: Vec<Vec<f64>> = vec![vec![0.1_f64], vec![0.4_f64], vec![0.6_f64], vec![0.9_f64]];
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let batch_result = predict_tree(&tree, &row_slices);
    assert!(batch_result.is_ok());

    if let Ok(batch_preds) = batch_result {
        for (i, row) in rows.iter().enumerate() {
            let single_result = predict_single(&tree, row);
            assert!(single_result.is_ok());
            if let Ok(single_pred) = single_result {
                assert!(
                    (batch_preds[i] - single_pred).abs() < 1e-15_f64,
                    "mismatch at index {i}: batch={}, single={single_pred}",
                    batch_preds[i]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_predict_tree_different_paths() -> Result<(), ClearGbmError> {
    let tree = make_simple_tree();
    let row_left: &[f64] = &[0.2_f64]; // left -> -1.0
    let row_right: &[f64] = &[0.7_f64]; // right -> 1.0
    let row_nan: &[f64] = &[f64::NAN]; // nan_goes_left -> -1.0
    let features: &[&[f64]] = &[row_left, row_right, row_nan];
    let result = predict_tree(&tree, features);
    assert!(result.is_ok());
    if let Ok(preds) = result {
        assert!((preds[0_usize] - (-1.0_f64)).abs() < 1e-15_f64);
        assert!((preds[1_usize] - 1.0_f64).abs() < 1e-15_f64);
        assert!((preds[2_usize] - (-1.0_f64)).abs() < 1e-15_f64);
    }
    Ok(())
}
