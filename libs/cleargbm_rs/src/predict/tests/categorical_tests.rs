//! Tests for categorical set-split routing during prediction.
//!
//! Membership in a node's left-routed codes sends a sample left; every
//! other non-missing value — other codes, unseen codes, non-integer
//! values — routes right; missing values follow `nan_goes_left`.

use crate::error::ClearGbmError;
use crate::predict::predict_single;
use crate::tree::Tree;
use crate::types::{CategoricalNodeConfig, TreeNode};

/// A 3-node tree whose root routes codes {2, 7} left on feature 0.
/// Left leaf (node 1) value = -1.0, right leaf (node 2) value = 1.0.
fn make_categorical_tree(nan_goes_left: bool) -> Tree {
    let root = TreeNode::new_categorical_internal(CategoricalNodeConfig {
        node_id: 0_usize,
        feature_index: 0_usize,
        categories_goes_left: vec![2.0_f64, 7.0_f64],
        value: 0.0_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left,
    });
    let left_leaf = TreeNode::new_leaf(1_usize, -1.0_f64, 5_usize);
    let right_leaf = TreeNode::new_leaf(2_usize, 1.0_f64, 5_usize);
    Tree::new(vec![root, left_leaf, right_leaf], 1_usize, 2_usize)
}

#[test]
fn test_member_codes_route_left() -> Result<(), ClearGbmError> {
    let tree = make_categorical_tree(true);
    for code in [2.0_f64, 7.0_f64] {
        let pred = propagate!(predict_single(&tree, &[code]));
        assert!((pred - -1.0_f64).abs() < 1e-15_f64, "code {code} not left");
    }
    Ok(())
}

#[test]
fn test_non_member_and_unseen_codes_route_right() -> Result<(), ClearGbmError> {
    let tree = make_categorical_tree(true);
    // 0 and 5 are other categories; 99 was never seen in training; 2.5 is
    // not an integer code at all. None is a member, so all go right.
    for value in [0.0_f64, 5.0_f64, 99.0_f64, 2.5_f64] {
        let pred = propagate!(predict_single(&tree, &[value]));
        assert!(
            (pred - 1.0_f64).abs() < 1e-15_f64,
            "value {value} not right"
        );
    }
    Ok(())
}

#[test]
fn test_missing_follows_the_nan_direction() -> Result<(), ClearGbmError> {
    let left_tree = make_categorical_tree(true);
    let pred_left = propagate!(predict_single(&left_tree, &[f64::NAN]));
    assert!((pred_left - -1.0_f64).abs() < 1e-15_f64);

    let right_tree = make_categorical_tree(false);
    let pred_right = propagate!(predict_single(&right_tree, &[f64::NAN]));
    assert!((pred_right - 1.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_negative_zero_matches_the_zero_code() -> Result<(), ClearGbmError> {
    // Codes are normalized to 0.0 at binning; the traversal normalizes the
    // incoming value the same way, so -0.0 is the 0 category, not "unseen".
    let root = TreeNode::new_categorical_internal(CategoricalNodeConfig {
        node_id: 0_usize,
        feature_index: 0_usize,
        categories_goes_left: vec![0.0_f64],
        value: 0.0_f64,
        n_samples: 4_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: false,
    });
    let tree = Tree::new(
        vec![
            root,
            TreeNode::new_leaf(1_usize, -1.0_f64, 2_usize),
            TreeNode::new_leaf(2_usize, 1.0_f64, 2_usize),
        ],
        1_usize,
        2_usize,
    );
    let pred = propagate!(predict_single(&tree, &[-0.0_f64]));
    assert!((pred - -1.0_f64).abs() < 1e-15_f64);
    Ok(())
}
