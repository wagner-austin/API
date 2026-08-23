//! Rejection, composition and infrastructure-failure tests for the
//! multiclass task, over the shared cluster fixture in
//! [`super::train_multiclass_tests`].

use crate::error::ClearGbmError;
use crate::training::labels::{TrainingLabels, ValidationData};
use crate::training::{
    train_gradient_boosting, GradientBoostingConfig, GradientBoostingModel, Objective, Parallelism,
    TrainingRuntime,
};

use super::train_helpers::default_params;
use super::train_multiclass_tests::make_multiclass_dataset;

#[test]
fn test_rejects_labels_at_or_beyond_n_classes() -> Result<(), ClearGbmError> {
    let (rows, mut y) = make_multiclass_dataset();
    y[5] = 3_u32;
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["position".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.max_bins = 16_usize;
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    params.n_classes = Some(3_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let runtime = TrainingRuntime {
        parallelism: Parallelism::Single,
        hooks: &crate::hooks::Hooks::default(),
    };
    match train_gradient_boosting(
        &x_train,
        TrainingLabels::Multiclass(&y),
        None,
        None,
        &config,
        &feature_names,
        &runtime,
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a label >= n_classes must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_train");
            assert!(reason.contains("got 3 at index 5"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_rejects_a_class_count_outside_multiclass() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.n_classes = Some(3_usize);
    match GradientBoostingConfig::new(params) {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "n_classes under binary must be rejected".to_string(),
            })
        }
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_classes");
        }
        Err(e) => return Err(e),
    }
    // And the multiclass objective demands one.
    let mut params = default_params();
    params.max_bins = 16_usize;
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    match GradientBoostingConfig::new(params) {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "multiclass without n_classes must be rejected".to_string(),
            })
        }
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_classes");
        }
        Err(e) => return Err(e),
    }
    // K = 1 cannot describe a classification task.
    let mut params = default_params();
    params.max_bins = 16_usize;
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    params.n_classes = Some(1_usize);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "n_classes = 1 must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "n_classes");
            assert!(reason.contains(">= 2"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_rejects_a_class_weight_under_multiclass() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.max_bins = 16_usize;
    params.objective = Objective::MulticlassSoftmax;
    params.n_classes = Some(3_usize);
    params.scale_pos_weight = Some(2.0_f64);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "scale_pos_weight under multiclass must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "scale_pos_weight");
            assert!(reason.contains("multiclass_softmax"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_subsampled_multiclass_covers_the_fallback_path() -> Result<(), ClearGbmError> {
    let (rows, y) = make_multiclass_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["position".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.n_estimators = 3_usize;
    params.max_bins = 16_usize;
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    params.n_classes = Some(3_usize);
    params.subsample = 0.75_f64;
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let runtime = TrainingRuntime {
        parallelism: Parallelism::Single,
        hooks: &crate::hooks::Hooks::default(),
    };
    let model = propagate!(train_gradient_boosting(
        &x_train,
        TrainingLabels::Multiclass(&y),
        None,
        None,
        &config,
        &feature_names,
        &runtime,
    ));
    assert_eq!(model.n_trees(), 9_usize);
    Ok(())
}

#[test]
fn test_multiclass_early_stopping_fires_on_a_worsening_validation() -> Result<(), ClearGbmError> {
    // Validation labels are the training labels rotated by one class, so
    // every round that fits the training clusters better makes the
    // validation loss worse: early stopping must fire and truncate to a
    // whole number of rounds well short of the maximum.
    let (rows, y) = make_multiclass_dataset();
    let y_rotated: Vec<u32> = y.iter().map(|&l| (l + 1_u32) % 3_u32).collect();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["position".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.n_estimators = 40_usize;
    params.max_bins = 16_usize;
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    params.n_classes = Some(3_usize);
    params.early_stopping_rounds = Some(2_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let runtime = TrainingRuntime {
        parallelism: Parallelism::Single,
        hooks: &crate::hooks::Hooks::default(),
    };
    let model = propagate!(train_gradient_boosting(
        &x_train,
        TrainingLabels::Multiclass(&y),
        None,
        Some(ValidationData {
            x: &x_train,
            y: TrainingLabels::Multiclass(&y_rotated),
            weight: None,
        }),
        &config,
        &feature_names,
        &runtime,
    ));
    assert_eq!(model.n_trees() % 3_usize, 0_usize);
    assert!(
        model.n_trees() < 120_usize,
        "early stopping never fired: {} trees",
        model.n_trees()
    );
    Ok(())
}

#[test]
fn test_multiclass_composes_with_the_feature_sampling_axes() -> Result<(), ClearGbmError> {
    // max_features and colsample_bytree both apply per (round, class) tree;
    // turning them on must change the model and stay deterministic.
    let (rows, y) = make_multiclass_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["position".to_string(), "constant".to_string()];
    let build = |sampled: bool| -> Result<GradientBoostingModel, ClearGbmError> {
        let mut params = default_params();
        params.n_estimators = 4_usize;
        params.max_bins = 16_usize;
        params.objective = Objective::MulticlassSoftmax;
        params.scale_pos_weight = None;
        params.n_classes = Some(3_usize);
        if sampled {
            params.max_features = Some(1_usize);
            params.colsample_bytree = Some(0.5_f64);
        }
        let config = match GradientBoostingConfig::new(params) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let runtime = TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &crate::hooks::Hooks::default(),
        };
        train_gradient_boosting(
            &x_train,
            TrainingLabels::Multiclass(&y),
            None,
            None,
            &config,
            &feature_names,
            &runtime,
        )
    };
    let plain = match build(false) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let sampled = match build(true) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let sampled_again = match build(true) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let preds_plain = propagate!(plain.predict_raw_multiclass(&x_train));
    let preds_sampled = propagate!(sampled.predict_raw_multiclass(&x_train));
    let preds_again = propagate!(sampled_again.predict_raw_multiclass(&x_train));
    assert_ne!(preds_plain, preds_sampled);
    assert_eq!(preds_sampled, preds_again);
    Ok(())
}

#[test]
fn test_multiclass_rejects_a_class_count_beyond_u32() -> Result<(), ClearGbmError> {
    let (rows, y) = make_multiclass_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["position".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    params.n_classes = Some(usize::MAX);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let runtime = TrainingRuntime {
        parallelism: Parallelism::Single,
        hooks: &crate::hooks::Hooks::default(),
    };
    match train_gradient_boosting(
        &x_train,
        TrainingLabels::Multiclass(&y),
        None,
        None,
        &config,
        &feature_names,
        &runtime,
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a class count beyond u32::MAX must be rejected".to_string(),
        }),
        Err(ClearGbmError::IntegerConversion { context }) => {
            assert!(context.contains("u32::MAX"), "got: {context}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_multiclass_surfaces_a_pool_construction_failure() -> Result<(), ClearGbmError> {
    fn failing_pool(
        _threads: core::num::NonZeroUsize,
    ) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1_usize)
            .stack_size(usize::MAX)
            .build()
    }

    let (rows, y) = make_multiclass_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["position".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    params.n_classes = Some(3_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    match train_gradient_boosting(
        &x_train,
        TrainingLabels::Multiclass(&y),
        None,
        None,
        &config,
        &feature_names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &crate::hooks::Hooks::with_pool_builder(failing_pool),
        },
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "training must fail when the worker pool cannot be built".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "n_jobs");
            assert!(reason.contains("could not build a worker pool"), "{reason}");
            Ok(())
        }
        Err(other) => Err(other),
    }
}
