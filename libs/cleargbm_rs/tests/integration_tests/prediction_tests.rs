//! Prediction integration tests.
//!
//! Tests the full predict pipeline: build a tree then predict (single, batch,
//! ensemble), verify probability bounds, and check sigmoid numerical stability.

use cleargbm_rs::{
    build_tree, predict_ensemble, predict_proba, predict_single, predict_tree, sigmoid,
    BuildTreeInput, ClearGbmError, Hooks, PredictEnsembleConfig, SplitConfig, TreeBuildConfig,
};

use super::EPSILON;

/// Test end-to-end: build a tree from data, then predict on the same data
#[test]
fn test_end_to_end_build_then_predict() -> std::result::Result<(), ClearGbmError> {
    let sample_indices: Vec<usize> = (0_usize..6_usize).collect();
    let gradients = vec![-1.0_f64, -1.0_f64, -1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];

    let bins = vec![
        vec![0_usize],
        vec![0_usize],
        vec![0_usize],
        vec![1_usize],
        vec![1_usize],
        vec![1_usize],
    ];
    let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

    let split_config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let tree_config = match TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 2_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Predict on samples that should go left (feature < threshold)
    let left_features = [0.3_f64];
    let left_pred = match predict_single(&tree, &left_features) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    // Left leaf: G=-3, H=3, value = -(-3)/3 = 1.0
    assert!(
        (left_pred - 1.0_f64).abs() < EPSILON,
        "Left prediction should be 1.0, got {left_pred}"
    );

    // Predict on samples that should go right (feature > threshold)
    let right_features = [0.8_f64];
    let right_pred = match predict_single(&tree, &right_features) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    // Right leaf: G=3, H=3, value = -(3)/3 = -1.0
    assert!(
        (right_pred - (-1.0_f64)).abs() < EPSILON,
        "Right prediction should be -1.0, got {right_pred}"
    );

    Ok(())
}

/// Test that batch prediction matches individual predictions
#[test]
fn test_prediction_batch_matches_individual() -> std::result::Result<(), ClearGbmError> {
    let sample_indices: Vec<usize> = (0_usize..6_usize).collect();
    let gradients = vec![-1.0_f64, -1.0_f64, -1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];

    let bins = vec![
        vec![0_usize],
        vec![0_usize],
        vec![0_usize],
        vec![1_usize],
        vec![1_usize],
        vec![1_usize],
    ];
    let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

    let split_config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let tree_config = match TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 2_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    let rows: Vec<Vec<f64>> = vec![vec![0.2_f64], vec![0.5_f64], vec![0.8_f64]];
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let batch_preds = match predict_tree(&tree, &row_slices) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };

    for (i, row) in rows.iter().enumerate() {
        let single_pred = match predict_single(&tree, row) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!(
            (batch_preds[i] - single_pred).abs() < EPSILON,
            "Batch and single prediction mismatch at index {i}"
        );
    }

    Ok(())
}

/// Test ensemble prediction mathematical correctness
#[test]
fn test_ensemble_prediction_mathematical_correctness() -> std::result::Result<(), ClearGbmError> {
    let sample_indices: Vec<usize> = (0_usize..6_usize).collect();
    let gradients = vec![-1.0_f64, -1.0_f64, -1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];

    let bins = vec![
        vec![0_usize],
        vec![0_usize],
        vec![0_usize],
        vec![1_usize],
        vec![1_usize],
        vec![1_usize],
    ];
    let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

    let split_config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let tree_config = match TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins: &bins,
        n_regular_bins: 2_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    let base_prediction = 0.5_f64;
    let learning_rate = 0.1_f64;
    let config = match PredictEnsembleConfig::new(base_prediction, learning_rate) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let row: &[f64] = &[0.3_f64]; // left -> leaf value 1.0
    let features: &[&[f64]] = &[row];

    let ensemble_pred = match predict_ensemble(std::slice::from_ref(&tree), features, &config) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };

    let tree_pred = match predict_single(&tree, row) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // expected = base_prediction + learning_rate * tree_pred
    let expected = base_prediction + learning_rate * tree_pred;
    assert!(
        (ensemble_pred[0_usize] - expected).abs() < EPSILON,
        "Ensemble prediction {0} should equal {expected}",
        ensemble_pred[0_usize]
    );

    Ok(())
}

/// Test that predict_proba produces valid probabilities
#[test]
fn test_predict_proba_probability_bounds() -> std::result::Result<(), ClearGbmError> {
    let raw_preds = [-10.0_f64, -1.0_f64, 0.0_f64, 1.0_f64, 10.0_f64];
    let probas = predict_proba(&raw_preds);

    assert_eq!(probas.len(), 5_usize);
    for (prob_0, prob_1) in &probas {
        assert!(
            (0.0_f64..=1.0_f64).contains(prob_0),
            "prob_0 out of [0,1]: {prob_0}"
        );
        assert!(
            (0.0_f64..=1.0_f64).contains(prob_1),
            "prob_1 out of [0,1]: {prob_1}"
        );
        assert!(
            (prob_0 + prob_1 - 1.0_f64).abs() < 1e-15_f64,
            "prob_0 + prob_1 should equal 1.0"
        );
    }

    Ok(())
}

/// Test sigmoid at extreme values for numerical stability
#[test]
fn test_sigmoid_numerical_stability() -> std::result::Result<(), ClearGbmError> {
    // These should not produce NaN, Inf, or values outside [0, 1]
    let extreme_values = [
        -1000.0_f64,
        -500.0_f64,
        -100.0_f64,
        0.0_f64,
        100.0_f64,
        500.0_f64,
        1000.0_f64,
        f64::MAX,
        f64::MIN,
    ];
    for x in extreme_values {
        let result = sigmoid(x);
        assert!(
            result.is_finite(),
            "sigmoid({x}) produced non-finite: {result}"
        );
        assert!(
            (0.0_f64..=1.0_f64).contains(&result),
            "sigmoid({x}) out of [0,1]: {result}"
        );
    }

    Ok(())
}
