//! Tests for the gradient boosting training loop (binary objective).
//!
//! Knob-sensitivity tests live in `train_knob_tests`; regression-objective
//! tests live in `train_regression_tests`; shared fixtures in
//! `train_helpers`.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::losses::{binary_log_loss, sigmoid_array};
use crate::split::MonotonicConstraint;
use crate::training::{
    train_gradient_boosting, GradientBoostingConfig, Objective, TrainingLabels, ValidationData,
};
use crate::training::{Parallelism, TrainingRuntime};

use super::train_helpers::{default_params, make_config, make_simple_dataset, train_binary};

#[test]
fn test_basic_training() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 5_usize);
    assert_eq!(model.config().objective(), Objective::BinaryLogLoss);
    Ok(())
}

#[test]
fn test_single_tree() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(1_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 1_usize);
    Ok(())
}

#[test]
fn test_training_loss_decreases() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Train 1 tree
    let config_1 = match make_config(1_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model_1 = match train_binary(&x_train, &y_train, None, &config_1, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    // Train 10 trees
    let config_10 = match make_config(10_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model_10 = match train_binary(&x_train, &y_train, None, &config_10, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    // Compute losses on training data
    let raw_1 = match model_1.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let probas_1 = sigmoid_array(&raw_1);
    let loss_1 = match binary_log_loss(&y_train, &probas_1, 1.0_f64, None) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };

    let raw_10 = match model_10.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let probas_10 = sigmoid_array(&raw_10);
    let loss_10 = match binary_log_loss(&y_train, &probas_10, 1.0_f64, None) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };

    // More trees should yield lower training loss
    assert!(loss_10 < loss_1);
    Ok(())
}

#[test]
fn test_probabilities_valid() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let probas = match model.predict_proba(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    for &(p0, p1) in &probas {
        assert!((0.0_f64..=1.0_f64).contains(&p0));
        assert!((0.0_f64..=1.0_f64).contains(&p1));
        assert!((p0 + p1 - 1.0_f64).abs() < 1e-10_f64);
    }
    Ok(())
}

#[test]
fn test_with_subsampling() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mut params = default_params();
    params.subsample = 0.5_f64;
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 5_usize);
    Ok(())
}

#[test]
fn test_with_validation_set() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Use same data as validation (for testing purposes)
    let val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64, 0.2_f64], vec![0.9_f64, 0.7_f64]];
    let x_val: Vec<&[f64]> = val_rows.iter().map(Vec::as_slice).collect();
    let y_val: Vec<u8> = vec![0_u8, 1_u8];

    let config = match make_config(5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(
        &x_train,
        &y_train,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Binary(&y_val),
            weight: None,
        }),
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    // Without early stopping, should train all 5 trees
    assert_eq!(model.n_trees(), 5_usize);
    Ok(())
}

#[test]
fn test_early_stopping_triggers() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Mislabeled validation set — forces val loss to increase as model fits training data
    let val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64, 0.2_f64], vec![0.9_f64, 0.7_f64]];
    let x_val: Vec<&[f64]> = val_rows.iter().map(Vec::as_slice).collect();
    let y_val: Vec<u8> = vec![1_u8, 0_u8];

    let mut params = default_params();
    params.n_estimators = 100_usize;
    params.early_stopping_rounds = Some(3_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(
        &x_train,
        &y_train,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Binary(&y_val),
            weight: None,
        }),
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    // Early stopping should stop before 100 trees
    assert!(model.n_trees() < 100_usize);
    assert!(model.n_trees() >= 1_usize);
    Ok(())
}

#[test]
fn test_early_stopping_without_validation_ignored() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Early stopping configured but no validation set — should train all rounds
    let mut params = default_params();
    params.early_stopping_rounds = Some(2_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 5_usize);
    Ok(())
}

#[test]
fn test_empty_training_data_error() -> Result<(), ClearGbmError> {
    let x_train: Vec<&[f64]> = vec![];
    let y_train: Vec<u8> = vec![];
    let feature_names: Vec<String> = vec![];

    let config = match make_config(3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let result = train_binary(&x_train, &y_train, None, &config, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for empty training data".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_monotonic_constraints_wrong_length() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Dataset has 2 features, provide 3 constraints
    let mc = vec![
        MonotonicConstraint::None,
        MonotonicConstraint::None,
        MonotonicConstraint::None,
    ];
    let mut params = default_params();
    params.n_estimators = 3_usize;
    params.monotonic_constraints = Some(mc);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let result = train_binary(&x_train, &y_train, None, &config, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for wrong constraint count".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_monotonic_constraints_correct_length() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let mc = vec![MonotonicConstraint::None, MonotonicConstraint::None];
    let mut params = default_params();
    params.n_estimators = 3_usize;
    params.monotonic_constraints = Some(mc);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 3_usize);
    Ok(())
}

#[test]
fn test_deterministic_training() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let config = match make_config(3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model1 = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let model2 = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    // Same config + same data → same raw predictions
    let raw1 = match model1.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let raw2 = match model2.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(raw1.len(), raw2.len());
    for i in 0_usize..raw1.len() {
        assert!((raw1[i] - raw2[i]).abs() < 1e-12_f64);
    }
    Ok(())
}

#[test]
fn test_with_regularization() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let mut params = default_params();
    params.reg_alpha = 0.5_f64;
    params.reg_lambda = 2.0_f64;
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 5_usize);
    // Should still produce valid probabilities
    let probas = match model.predict_proba(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    for &(p0, p1) in &probas {
        assert!((0.0_f64..=1.0_f64).contains(&p0));
        assert!((0.0_f64..=1.0_f64).contains(&p1));
    }
    Ok(())
}

#[test]
fn test_validation_set_wrong_features() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Validation with 3 features instead of 2
    let val_rows: Vec<Vec<f64>> = vec![vec![0.5_f64, 0.5_f64, 0.5_f64]];
    let x_val: Vec<&[f64]> = val_rows.iter().map(Vec::as_slice).collect();
    let y_val: Vec<u8> = vec![0_u8];

    let config = match make_config(3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let result = train_binary(
        &x_train,
        &y_train,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Binary(&y_val),
            weight: None,
        }),
        &config,
        &feature_names,
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for wrong feature count in val".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_all_same_labels_error() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![
        vec![0.1_f64, 0.2_f64],
        vec![0.3_f64, 0.4_f64],
        vec![0.5_f64, 0.6_f64],
    ];
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_train: Vec<u8> = vec![0_u8, 0_u8, 0_u8];
    let feature_names: Vec<String> = vec!["f0".to_string(), "f1".to_string()];
    let config = match make_config(3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let result = train_binary(&x_train, &y_train, None, &config, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for all-same labels".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "y_true");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_training_reports_a_worker_pool_that_cannot_be_built() -> Result<(), ClearGbmError> {
    // No caller input can make rayon refuse a pool — `n_jobs` saturates rather
    // than failing, and a huge thread count is clamped, not rejected. The
    // failure is still real (the OS can refuse a thread), so it is reached the
    // way the crate reaches every other unprovokable path: through `Hooks`.
    //
    // The injected builder returns a genuine `ThreadPoolBuildError` produced by
    // a real failing `build()`, not a fabricated one, so this exercises the
    // same error text production would surface.
    fn failing_pool(
        _threads: core::num::NonZeroUsize,
    ) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1_usize)
            .stack_size(usize::MAX)
            .build()
    }

    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64, 0.1_f64],
        vec![0.1_f64, 0.0_f64],
        vec![0.9_f64, 1.0_f64],
        vec![1.0_f64, 0.9_f64],
    ];
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_train: Vec<u8> = vec![0_u8, 0_u8, 1_u8, 1_u8];
    let feature_names: Vec<String> = vec!["f0".to_string(), "f1".to_string()];

    let config = match make_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    match train_gradient_boosting(
        &x_train,
        TrainingLabels::Binary(&y_train),
        None,
        None,
        &config,
        &feature_names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::with_pool_builder(failing_pool),
        },
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "training must fail when the worker pool cannot be built".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "n_jobs");
            assert!(
                reason.contains("could not build a worker pool"),
                "error should name the pool failure, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_default_pool_builder_produces_a_usable_pool() -> Result<(), ClearGbmError> {
    // The default hook is what production runs; assert it honours the
    // requested worker count rather than only that it returns something.
    let hooks = Hooks::default();
    let requested = match core::num::NonZeroUsize::new(2_usize) {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "2 is nonzero".to_string(),
            })
        }
    };
    let pool = match (hooks.build_pool)(requested) {
        Ok(p) => p,
        Err(e) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "build_pool".to_string(),
                reason: format!("default builder failed: {e}"),
            })
        }
    };
    assert_eq!(pool.current_num_threads(), 2_usize);
    Ok(())
}
