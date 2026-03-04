//! Tests for ensemble prediction and probability conversion.

use crate::error::ClearGbmError;
use crate::predict::{predict_ensemble, predict_proba, sigmoid, PredictEnsembleConfig};
use crate::tree::Tree;
use crate::types::{TreeNode, TreeNodeConfig};

/// Helper to build a simple 3-node tree with given leaf values.
fn make_tree_with_values(left_value: f64, right_value: f64) -> Tree {
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
    let left_leaf = TreeNode::new_leaf(1_usize, left_value, 5_usize);
    let right_leaf = TreeNode::new_leaf(2_usize, right_value, 5_usize);
    Tree::new(vec![root, left_leaf, right_leaf], 1_usize, 2_usize)
}

#[test]
fn test_predict_ensemble_config_valid() -> Result<(), ClearGbmError> {
    let config = match PredictEnsembleConfig::new(0.5_f64, 0.1_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert!((config.base_prediction() - 0.5_f64).abs() < 1e-15_f64);
    assert!((config.learning_rate() - 0.1_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_ensemble_config_lr_one_valid() -> Result<(), ClearGbmError> {
    let config = match PredictEnsembleConfig::new(0.0_f64, 1.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert!((config.learning_rate() - 1.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_ensemble_config_negative_base() -> Result<(), ClearGbmError> {
    // Negative base_prediction is valid (e.g., log-odds for rare class)
    let config = match PredictEnsembleConfig::new(-2.0_f64, 0.5_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert!((config.base_prediction() - (-2.0_f64)).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_ensemble_single_tree() -> Result<(), ClearGbmError> {
    let tree = make_tree_with_values(-1.0_f64, 1.0_f64);
    let config = match PredictEnsembleConfig::new(0.5_f64, 0.1_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let row: &[f64] = &[0.3_f64]; // left -> -1.0
    let features: &[&[f64]] = &[row];
    let preds = match predict_ensemble(&[tree], features, &config) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    // 0.5 + 0.1 * (-1.0) = 0.4
    assert!((preds[0_usize] - 0.4_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_ensemble_multiple_trees() -> Result<(), ClearGbmError> {
    let tree1 = make_tree_with_values(-1.0_f64, 1.0_f64);
    let tree2 = make_tree_with_values(-0.5_f64, 0.5_f64);
    let config = match PredictEnsembleConfig::new(0.0_f64, 0.1_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let row: &[f64] = &[0.3_f64]; // left in both trees
    let features: &[&[f64]] = &[row];
    let preds = match predict_ensemble(&[tree1, tree2], features, &config) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    // 0.0 + 0.1 * (-1.0) + 0.1 * (-0.5) = -0.15
    assert!((preds[0_usize] - (-0.15_f64)).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_ensemble_base_prediction_applied() -> Result<(), ClearGbmError> {
    let tree = make_tree_with_values(0.0_f64, 0.0_f64);
    let config = match PredictEnsembleConfig::new(2.5_f64, 0.1_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let row: &[f64] = &[0.3_f64]; // left -> 0.0
    let features: &[&[f64]] = &[row];
    let preds = match predict_ensemble(&[tree], features, &config) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    // 2.5 + 0.1 * 0.0 = 2.5
    assert!((preds[0_usize] - 2.5_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_ensemble_learning_rate_scaling() -> Result<(), ClearGbmError> {
    let tree = make_tree_with_values(-2.0_f64, 2.0_f64);
    let config_small = match PredictEnsembleConfig::new(0.0_f64, 0.01_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config_large = match PredictEnsembleConfig::new(0.0_f64, 1.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let row: &[f64] = &[0.3_f64]; // left -> -2.0
    let features: &[&[f64]] = &[row];

    let preds_small = match predict_ensemble(std::slice::from_ref(&tree), features, &config_small) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_large = match predict_ensemble(&[tree], features, &config_large) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    // 0.01 * (-2.0) = -0.02
    assert!((preds_small[0_usize] - (-0.02_f64)).abs() < 1e-15_f64);
    // 1.0 * (-2.0) = -2.0
    assert!((preds_large[0_usize] - (-2.0_f64)).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_ensemble_multiple_samples() -> Result<(), ClearGbmError> {
    let tree = make_tree_with_values(-1.0_f64, 1.0_f64);
    let config = match PredictEnsembleConfig::new(0.0_f64, 0.5_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let row_left: &[f64] = &[0.3_f64]; // left -> -1.0
    let row_right: &[f64] = &[0.8_f64]; // right -> 1.0
    let features: &[&[f64]] = &[row_left, row_right];
    let preds = match predict_ensemble(&[tree], features, &config) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(preds.len(), 2_usize);
    // 0.0 + 0.5 * (-1.0) = -0.5
    assert!((preds[0_usize] - (-0.5_f64)).abs() < 1e-15_f64);
    // 0.0 + 0.5 * 1.0 = 0.5
    assert!((preds[1_usize] - 0.5_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_proba_from_zero() -> Result<(), ClearGbmError> {
    let result = predict_proba(&[0.0_f64]);
    assert_eq!(result.len(), 1_usize);
    assert!((result[0_usize].0 - 0.5_f64).abs() < 1e-15_f64);
    assert!((result[0_usize].1 - 0.5_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_proba_positive() -> Result<(), ClearGbmError> {
    let result = predict_proba(&[2.0_f64]);
    assert_eq!(result.len(), 1_usize);
    let (prob_0, prob_1) = result[0_usize];
    assert!(prob_1 > 0.5_f64);
    assert!(prob_0 < 0.5_f64);
    assert!((prob_0 + prob_1 - 1.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_proba_negative() -> Result<(), ClearGbmError> {
    let result = predict_proba(&[-2.0_f64]);
    assert_eq!(result.len(), 1_usize);
    let (prob_0, prob_1) = result[0_usize];
    assert!(prob_1 < 0.5_f64);
    assert!(prob_0 > 0.5_f64);
    assert!((prob_0 + prob_1 - 1.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_predict_proba_empty_returns_empty() -> Result<(), ClearGbmError> {
    let result = predict_proba(&[]);
    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_predict_proba_multiple_samples() -> Result<(), ClearGbmError> {
    let raws = [-5.0_f64, 0.0_f64, 5.0_f64];
    let result = predict_proba(&raws);
    assert_eq!(result.len(), 3_usize);
    for (prob_0, prob_1) in &result {
        assert!(*prob_0 >= 0.0_f64);
        assert!(*prob_0 <= 1.0_f64);
        assert!(*prob_1 >= 0.0_f64);
        assert!(*prob_1 <= 1.0_f64);
        assert!((prob_0 + prob_1 - 1.0_f64).abs() < 1e-15_f64);
    }
    // Monotonicity: higher raw -> higher prob_1
    assert!(result[2_usize].1 > result[1_usize].1);
    assert!(result[1_usize].1 > result[0_usize].1);
    Ok(())
}

#[test]
fn test_predict_proba_matches_sigmoid() -> Result<(), ClearGbmError> {
    let raws = [1.0_f64, -1.0_f64, 3.0_f64];
    let result = predict_proba(&raws);
    for (i, &raw) in raws.iter().enumerate() {
        let expected_prob_1 = sigmoid(raw);
        assert!((result[i].1 - expected_prob_1).abs() < 1e-15_f64);
        assert!((result[i].0 - (1.0_f64 - expected_prob_1)).abs() < 1e-15_f64);
    }
    Ok(())
}

#[test]
fn test_predict_ensemble_config_debug() -> Result<(), ClearGbmError> {
    let config = match PredictEnsembleConfig::new(1.0_f64, 0.5_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("PredictEnsembleConfig"));
    assert!(debug_str.contains("base_prediction"));
    assert!(debug_str.contains("learning_rate"));
    Ok(())
}

#[test]
fn test_predict_ensemble_config_clone() -> Result<(), ClearGbmError> {
    let config = match PredictEnsembleConfig::new(1.0_f64, 0.5_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cloned = config.clone();
    assert_eq!(config, cloned);
    Ok(())
}
