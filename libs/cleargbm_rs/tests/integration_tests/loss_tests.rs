//! Loss function integration tests.
//!
//! Verifies that loss, gradient, and hessian form a mathematically consistent
//! system, and that the loss module integrates correctly with tree building.

use cleargbm_rs::{
    binary_log_loss, binary_log_loss_gradients, binary_log_loss_hessians,
    binary_log_loss_initial_prediction, build_tree, predict_single, sigmoid, sigmoid_array,
    BuildTreeInput, ClearGbmError, Hooks, SplitConfig, TreeBuildConfig,
};

use super::EPSILON;

/// Test that loss, gradient, and hessian form a mathematically consistent system.
///
/// For binary log loss:
/// - Loss decreases when predictions move toward true labels
/// - Gradients point in the correct direction (p - y)
/// - Hessians are always positive (convex loss)
#[test]
fn test_loss_gradient_hessian_consistency() -> std::result::Result<(), ClearGbmError> {
    let y_true = [1_u8, 0_u8, 1_u8, 0_u8, 1_u8];

    // Good predictions (close to true labels)
    let good_preds = [0.9_f64, 0.1_f64, 0.8_f64, 0.2_f64, 0.95_f64];
    // Bad predictions (far from true labels)
    let bad_preds = [0.2_f64, 0.8_f64, 0.3_f64, 0.7_f64, 0.1_f64];

    let good_loss = match binary_log_loss(&y_true, &good_preds) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let bad_loss = match binary_log_loss(&y_true, &bad_preds) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // Better predictions → lower loss
    assert!(
        good_loss < bad_loss,
        "good predictions should have lower loss: {good_loss} vs {bad_loss}"
    );

    // Gradients should point toward correction
    let grads = match binary_log_loss_gradients(&y_true, &good_preds) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    // For y=1, p=0.9: gradient = 0.9 - 1.0 = -0.1 (negative = push up = correct direction)
    assert!(
        grads[0_usize] < 0.0_f64,
        "gradient for y=1 should be negative"
    );
    // For y=0, p=0.1: gradient = 0.1 - 0.0 = 0.1 (positive = push down = correct direction)
    assert!(
        grads[1_usize] > 0.0_f64,
        "gradient for y=0 should be positive"
    );

    // Hessians must be positive (convex loss surface)
    let hess = match binary_log_loss_hessians(&y_true, &good_preds) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    for (idx, &h) in hess.iter().enumerate() {
        assert!(h > 0.0_f64, "hessian at {idx} must be positive, got {h}");
    }

    Ok(())
}

/// Test that initial prediction correctly inverts through sigmoid.
///
/// sigmoid(initial_prediction) should approximately equal the positive class rate.
#[test]
fn test_initial_prediction_inverts_through_sigmoid() -> std::result::Result<(), ClearGbmError> {
    // 40% positive rate
    let y_true = [1_u8, 1_u8, 0_u8, 0_u8, 0_u8];
    let init_pred = match binary_log_loss_initial_prediction(&y_true) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // sigmoid(log-odds) should give back the positive rate
    let recovered_rate = sigmoid(init_pred);
    let actual_rate = 0.4_f64; // 2 out of 5

    assert!(
        (recovered_rate - actual_rate).abs() < EPSILON,
        "sigmoid(init_pred) should recover positive rate: expected {actual_rate}, got {recovered_rate}"
    );

    Ok(())
}

/// Test that sigmoid_array matches scalar sigmoid exactly (integration-level).
#[test]
fn test_sigmoid_array_matches_scalar_integration() -> std::result::Result<(), ClearGbmError> {
    let inputs = [
        -50.0_f64, -5.0_f64, -0.5_f64, 0.0_f64, 0.5_f64, 5.0_f64, 50.0_f64,
    ];
    let array_result = sigmoid_array(&inputs);

    for (idx, &x) in inputs.iter().enumerate() {
        let scalar_result = sigmoid(x);
        assert!(
            (array_result[idx] - scalar_result).abs() < 1e-15_f64,
            "mismatch at {idx}: array={}, scalar={scalar_result}",
            array_result[idx]
        );
    }

    Ok(())
}

/// Test end-to-end: compute gradients/hessians from loss, feed into tree building.
///
/// This verifies the losses module integrates correctly with the tree building module.
#[test]
fn test_loss_feeds_tree_building() -> std::result::Result<(), ClearGbmError> {
    // Simulate binary classification with clear separation
    let y_true = [1_u8, 1_u8, 1_u8, 0_u8, 0_u8, 0_u8];

    // Start with initial prediction (log-odds)
    let init_pred = match binary_log_loss_initial_prediction(&y_true) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // Convert to probabilities via sigmoid
    let raw_preds = vec![init_pred; 6_usize];
    let probs = sigmoid_array(&raw_preds);

    // Compute gradients and hessians
    let gradients = match binary_log_loss_gradients(&y_true, &probs) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let hessians = match binary_log_loss_hessians(&y_true, &probs) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // Build a tree using the computed gradients/hessians
    let sample_indices: Vec<usize> = (0_usize..6_usize).collect();
    // Feature: first 3 samples have low values, last 3 have high values
    // 6 samples, 1 feature, column-major flat.
    let bins: Vec<u8> = vec![0_u8, 0_u8, 0_u8, 1_u8, 1_u8, 1_u8];
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
        n_samples: 6_usize,
        n_features: 1_usize,
        n_regular_bins: 2_usize,
        bin_thresholds: &bin_thresholds,
        config: &tree_config,
        monotonic_constraints: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // The tree should have found a useful split (labels have clear separation)
    assert!(
        tree.n_leaves() >= 2_usize,
        "tree should split when labels have clear separation, got {} leaves",
        tree.n_leaves()
    );

    // Predictions for "positive" features should push probability up
    // and "negative" features should push probability down
    let left_pred = match predict_single(&tree, &[0.3_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let right_pred = match predict_single(&tree, &[0.8_f64]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // y=1 samples are in the left bin → gradient is negative → leaf correction is positive
    // y=0 samples are in the right bin → gradient is positive → leaf correction is negative
    assert!(
        left_pred > right_pred,
        "positive-label side should have higher prediction: left={left_pred}, right={right_pred}"
    );

    Ok(())
}
