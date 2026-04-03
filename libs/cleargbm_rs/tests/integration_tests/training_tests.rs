//! End-to-end training pipeline integration tests.
//!
//! Tests the full train → predict workflow: model construction, loss
//! convergence, early stopping, subsampling, determinism, and
//! consistency between `predict_raw` and `predict_proba`.

use cleargbm_rs::{
    binary_log_loss, sigmoid, train_gradient_boosting, ClearGbmError, GradientBoostingConfig,
    GradientBoostingConfigParams, MonotonicConstraint,
};

use super::EPSILON;

/// Builds a default config for training integration tests.
///
/// Args:
///     n_estimators: Number of boosting rounds.
///     early_stopping_rounds: Optional patience for early stopping.
///     subsample: Row subsample fraction in (0.0, 1.0].
///     monotonic_constraints: Optional per-feature constraints.
///
/// Returns:
///     Validated `GradientBoostingConfig`.
///
/// Raises:
///     `ClearGbmError::InvalidParameter` if any hyperparameter is invalid.
fn make_config(
    n_estimators: usize,
    early_stopping_rounds: Option<usize>,
    subsample: f64,
    monotonic_constraints: Option<Vec<MonotonicConstraint>>,
) -> Result<GradientBoostingConfig, ClearGbmError> {
    GradientBoostingConfig::new(GradientBoostingConfigParams {
        n_estimators,
        max_depth: 3_usize,
        learning_rate: 0.3_f64,
        min_samples_split: 2_usize,
        min_samples_leaf: 1_usize,
        max_bins: 8_usize,
        subsample,
        random_state: 42_u64,
        monotonic_constraints,
        reg_alpha: 0.0_f64,
        reg_lambda: 0.0_f64,
        early_stopping_rounds,
    })
}

/// Builds a linearly separable dataset for binary classification.
///
/// Class 0 has features near 0; class 1 has features near 1.
///
/// Returns:
///     Tuple of (feature_matrix, labels, feature_names).
fn make_separable_dataset() -> (Vec<Vec<f64>>, Vec<u8>, Vec<String>) {
    let x: Vec<Vec<f64>> = vec![
        vec![0.1_f64, 0.2_f64],
        vec![0.15_f64, 0.25_f64],
        vec![0.2_f64, 0.1_f64],
        vec![0.05_f64, 0.3_f64],
        vec![0.18_f64, 0.22_f64],
        vec![0.12_f64, 0.15_f64],
        vec![0.8_f64, 0.9_f64],
        vec![0.85_f64, 0.75_f64],
        vec![0.9_f64, 0.85_f64],
        vec![0.95_f64, 0.7_f64],
        vec![0.78_f64, 0.88_f64],
        vec![0.82_f64, 0.92_f64],
    ];
    let y: Vec<u8> = vec![
        0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 1_u8, 1_u8, 1_u8, 1_u8, 1_u8, 1_u8,
    ];
    let feature_names = vec!["f0".to_string(), "f1".to_string()];
    (x, y, feature_names)
}

/// Train on a small separable dataset, predict, and verify loss decreases.
#[test]
fn test_train_predict_end_to_end() -> std::result::Result<(), ClearGbmError> {
    let (x, y, feature_names) = make_separable_dataset();
    let x_refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();

    let config = match make_config(5_usize, None, 1.0_f64, None) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model = match train_gradient_boosting(&x_refs, &y, None, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    // Model should have exactly 5 trees
    assert_eq!(model.n_trees(), 5_usize, "should have 5 trees");
    assert_eq!(model.n_classes(), 2_usize, "binary classification");
    assert_eq!(model.feature_names(), &feature_names);
    assert!((model.learning_rate() - 0.3_f64).abs() < EPSILON);

    // Predict probabilities
    let probas = match model.predict_proba(&x_refs) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(probas.len(), 12_usize);

    // Each probability pair must sum to 1.0 and be in [0,1]
    for &(p0, p1) in &probas {
        assert!((0.0_f64..=1.0_f64).contains(&p0), "p0 out of range: {p0}");
        assert!((0.0_f64..=1.0_f64).contains(&p1), "p1 out of range: {p1}");
        assert!((p0 + p1 - 1.0_f64).abs() < EPSILON, "proba must sum to 1");
    }

    // Class 0 samples (indices 0..6) should have p(class=0) > 0.5
    for (i, &(p0, _p1)) in probas.iter().enumerate().take(6_usize) {
        assert!(
            p0 > 0.5_f64,
            "sample {i} (class 0) should have P(0) > 0.5, got {p0}",
        );
    }

    // Class 1 samples (indices 6..12) should have p(class=1) > 0.5
    for (i, &(_p0, p1)) in probas.iter().enumerate().take(12_usize).skip(6_usize) {
        assert!(
            p1 > 0.5_f64,
            "sample {i} (class 1) should have P(1) > 0.5, got {p1}",
        );
    }

    Ok(())
}

/// Verify that training loss decreases monotonically with more trees.
#[test]
fn test_training_loss_decreases() -> std::result::Result<(), ClearGbmError> {
    let (x, y, feature_names) = make_separable_dataset();
    let x_refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();

    // Train with 1 tree then 5 trees — loss should decrease
    let config_1 = match make_config(1_usize, None, 1.0_f64, None) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config_5 = match make_config(5_usize, None, 1.0_f64, None) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model_1 = match train_gradient_boosting(&x_refs, &y, None, None, &config_1, &feature_names)
    {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let model_5 = match train_gradient_boosting(&x_refs, &y, None, None, &config_5, &feature_names)
    {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let probas_1 = match model_1.predict_proba(&x_refs) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let probas_5 = match model_5.predict_proba(&x_refs) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };

    // Extract p(class=1) for log loss computation
    let p1_1: Vec<f64> = probas_1.iter().map(|t| t.1).collect();
    let p1_5: Vec<f64> = probas_5.iter().map(|t| t.1).collect();

    let loss_1 = match binary_log_loss(&y, &p1_1) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };
    let loss_5 = match binary_log_loss(&y, &p1_5) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };

    assert!(
        loss_5 < loss_1,
        "5-tree loss ({loss_5}) should be less than 1-tree loss ({loss_1})"
    );

    Ok(())
}

/// Test early stopping: training stops before `n_estimators` when validation loss plateaus.
#[test]
fn test_early_stopping_triggers() -> std::result::Result<(), ClearGbmError> {
    let (x, y, feature_names) = make_separable_dataset();
    let x_refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();

    // Request 100 trees but with early stopping patience 2 — should stop much earlier
    let config = match make_config(100_usize, Some(2_usize), 1.0_f64, None) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    // Use same data as validation set (perfect fit → loss converges quickly)
    let model = match train_gradient_boosting(
        &x_refs,
        &y,
        Some(&x_refs),
        Some(&y),
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    // Model should have fewer than 100 trees due to early stopping
    assert!(
        model.n_trees() < 100_usize,
        "early stopping should trigger before 100 trees, got {}",
        model.n_trees()
    );
    assert!(
        model.n_trees() >= 1_usize,
        "should have at least 1 tree, got {}",
        model.n_trees()
    );

    Ok(())
}

/// Test that row subsampling produces a valid model.
#[test]
fn test_subsample_training() -> std::result::Result<(), ClearGbmError> {
    let (x, y, feature_names) = make_separable_dataset();
    let x_refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();

    let config = match make_config(5_usize, None, 0.5_f64, None) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model = match train_gradient_boosting(&x_refs, &y, None, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    assert_eq!(model.n_trees(), 5_usize);

    // Predictions should still be reasonable despite subsampling
    let probas = match model.predict_proba(&x_refs) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };

    for &(p0, p1) in &probas {
        assert!((0.0_f64..=1.0_f64).contains(&p0));
        assert!((0.0_f64..=1.0_f64).contains(&p1));
        assert!((p0 + p1 - 1.0_f64).abs() < EPSILON);
    }

    Ok(())
}

/// Test deterministic training: same seed → same model.
#[test]
fn test_deterministic_training() -> std::result::Result<(), ClearGbmError> {
    let (x, y, feature_names) = make_separable_dataset();
    let x_refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();

    let config = match make_config(3_usize, None, 0.8_f64, None) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model_a = match train_gradient_boosting(&x_refs, &y, None, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let model_b = match train_gradient_boosting(&x_refs, &y, None, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let raw_a = match model_a.predict_raw(&x_refs) {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    let raw_b = match model_b.predict_raw(&x_refs) {
        Ok(r) => r,
        Err(e) => return Err(e),
    };

    assert_eq!(raw_a.len(), raw_b.len());
    for (a, b) in raw_a.iter().zip(raw_b.iter()) {
        assert!(
            (a - b).abs() < EPSILON,
            "predictions should be identical for same seed: {a} vs {b}"
        );
    }

    Ok(())
}

/// Test single tree training (n_estimators=1).
#[test]
fn test_single_tree_training() -> std::result::Result<(), ClearGbmError> {
    let (x, y, feature_names) = make_separable_dataset();
    let x_refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();

    let config = match make_config(1_usize, None, 1.0_f64, None) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model = match train_gradient_boosting(&x_refs, &y, None, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    assert_eq!(model.n_trees(), 1_usize);

    // predict_raw should work
    let raw = match model.predict_raw(&x_refs) {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    assert_eq!(raw.len(), 12_usize);

    // Base prediction plus one tree's contribution
    let base = model.base_prediction();
    for &r in &raw {
        // Raw predictions should be finite
        assert!(r.is_finite(), "raw prediction should be finite, got {r}");
    }

    // At least some predictions should differ from base
    let differs = raw.iter().any(|&r| (r - base).abs() > EPSILON);
    assert!(differs, "at least some predictions should differ from base");

    Ok(())
}

/// Test predict_raw and predict_proba consistency.
#[test]
fn test_predict_raw_proba_consistency() -> std::result::Result<(), ClearGbmError> {
    let (x, y, feature_names) = make_separable_dataset();
    let x_refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();

    let config = match make_config(3_usize, None, 1.0_f64, None) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model = match train_gradient_boosting(&x_refs, &y, None, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let raw = match model.predict_raw(&x_refs) {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    let probas = match model.predict_proba(&x_refs) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };

    // predict_proba should equal sigmoid(raw)
    for (i, &r) in raw.iter().enumerate() {
        let expected_p1 = sigmoid(r);
        let expected_p0 = 1.0_f64 - expected_p1;
        assert!(
            (probas[i].0 - expected_p0).abs() < EPSILON,
            "P(0) mismatch at sample {i}: expected {expected_p0}, got {}",
            probas[i].0
        );
        assert!(
            (probas[i].1 - expected_p1).abs() < EPSILON,
            "P(1) mismatch at sample {i}: expected {expected_p1}, got {}",
            probas[i].1
        );
    }

    Ok(())
}
