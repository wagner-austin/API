//! Shared fixtures and call helpers for the training-loop tests.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::training::{
    train_gradient_boosting, GradientBoostingConfig, GradientBoostingConfigParams,
    GradientBoostingModel, GrowthStrategy, Objective, TrainingLabels, ValidationData,
};
use crate::training::{Parallelism, TrainingRuntime};

/// Helper: simple linearly separable dataset.
pub(super) fn make_simple_dataset() -> (Vec<Vec<f64>>, Vec<u8>, Vec<String>) {
    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64, 0.1_f64],
        vec![0.1_f64, 0.0_f64],
        vec![0.2_f64, 0.2_f64],
        vec![0.3_f64, 0.1_f64],
        vec![0.8_f64, 0.9_f64],
        vec![0.9_f64, 0.8_f64],
        vec![1.0_f64, 1.0_f64],
        vec![0.7_f64, 0.9_f64],
    ];
    let y = vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8, 1_u8, 1_u8, 1_u8];
    let names = vec!["f0".to_string(), "f1".to_string()];
    (rows, y, names)
}

/// A dataset whose structure survives the first split.
///
/// Feature 0 separates a pure right half from a mixed left half, and feature 1
/// separates the left half. Depth-wise growth at depth 2 therefore reaches
/// three leaves, while a leaf budget of 2 stops at one split — which is what
/// lets the two policies be told apart by tree shape. On the linearly
/// separable `make_simple_dataset` they cannot: one split makes both sides
/// pure, so every policy stops at two leaves.
pub(super) fn make_nested_dataset() -> (Vec<Vec<f64>>, Vec<u8>, Vec<String>) {
    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64, 0.0_f64],
        vec![0.1_f64, 0.1_f64],
        vec![0.0_f64, 1.0_f64],
        vec![0.1_f64, 0.9_f64],
        vec![1.0_f64, 0.0_f64],
        vec![0.9_f64, 0.1_f64],
        vec![1.0_f64, 1.0_f64],
        vec![0.9_f64, 0.9_f64],
    ];
    let y = vec![0_u8, 0_u8, 1_u8, 1_u8, 1_u8, 1_u8, 1_u8, 1_u8];
    let names = vec!["f0".to_string(), "f1".to_string()];
    (rows, y, names)
}

/// Creates default valid binary-classification params for reuse in tests.
pub(super) fn default_params() -> GradientBoostingConfigParams {
    GradientBoostingConfigParams {
        n_estimators: 5_usize,
        max_depth: 2_usize,
        learning_rate: 0.3_f64,
        min_samples_split: 2_usize,
        min_samples_leaf: 1_usize,
        max_bins: 4_usize,
        subsample: 1.0_f64,
        random_state: 42_u64,
        monotonic_constraints: None,
        reg_alpha: 0.0_f64,
        reg_lambda: 1.0_f64,
        early_stopping_rounds: None,
        growth_strategy: GrowthStrategy::DepthWise,
        num_leaves: None,
        objective: Objective::BinaryLogLoss,
        scale_pos_weight: Some(1.0_f64),
        max_features: None,
        colsample_bytree: None,
        categorical_features: None,
        n_classes: None,
        lambdarank_truncation_level: None,
        goss_top_rate: None,
        goss_other_rate: None,
        quantized_gradient_bins: None,
        min_data_in_bin: None,
    }
}

/// Creates default valid squared-error params for reuse in tests.
pub(super) fn default_regression_params() -> GradientBoostingConfigParams {
    let mut params = default_params();
    params.objective = Objective::SquaredError;
    params.scale_pos_weight = None;
    params
}

/// Helper: default training config with custom n_estimators.
pub(super) fn make_config(n_estimators: usize) -> Result<GradientBoostingConfig, ClearGbmError> {
    let mut params = default_params();
    params.n_estimators = n_estimators;
    GradientBoostingConfig::new(params)
}

/// Helper: default regression config with custom n_estimators.
pub(super) fn make_regression_config(
    n_estimators: usize,
) -> Result<GradientBoostingConfig, ClearGbmError> {
    let mut params = default_regression_params();
    params.n_estimators = n_estimators;
    GradientBoostingConfig::new(params)
}

/// Helper: leaf-wise training config with the given leaf budget.
pub(super) fn make_leaf_wise_config(
    n_estimators: usize,
    num_leaves: usize,
) -> Result<GradientBoostingConfig, ClearGbmError> {
    let mut params = default_params();
    params.n_estimators = n_estimators;
    params.growth_strategy = GrowthStrategy::LeafWise;
    params.num_leaves = Some(num_leaves);
    GradientBoostingConfig::new(params)
}

/// Trains with binary labels on a single-threaded default-hooks runtime.
pub(super) fn train_binary(
    x_train: &[&[f64]],
    y_train: &[u8],
    validation: Option<ValidationData<'_>>,
    config: &GradientBoostingConfig,
    feature_names: &[String],
) -> Result<GradientBoostingModel, ClearGbmError> {
    train_binary_weighted(x_train, y_train, None, validation, config, feature_names)
}

/// Trains with binary labels and optional per-row weights.
pub(super) fn train_binary_weighted(
    x_train: &[&[f64]],
    y_train: &[u8],
    sample_weight: Option<&[f64]>,
    validation: Option<ValidationData<'_>>,
    config: &GradientBoostingConfig,
    feature_names: &[String],
) -> Result<GradientBoostingModel, ClearGbmError> {
    train_gradient_boosting(
        x_train,
        TrainingLabels::Binary(y_train),
        sample_weight,
        validation,
        config,
        feature_names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    )
}

/// Trains with continuous targets on a single-threaded default-hooks runtime.
pub(super) fn train_regression(
    x_train: &[&[f64]],
    y_train: &[f64],
    validation: Option<ValidationData<'_>>,
    config: &GradientBoostingConfig,
    feature_names: &[String],
) -> Result<GradientBoostingModel, ClearGbmError> {
    train_regression_weighted(x_train, y_train, None, validation, config, feature_names)
}

/// Trains with continuous targets and optional per-row weights.
pub(super) fn train_regression_weighted(
    x_train: &[&[f64]],
    y_train: &[f64],
    sample_weight: Option<&[f64]>,
    validation: Option<ValidationData<'_>>,
    config: &GradientBoostingConfig,
    feature_names: &[String],
) -> Result<GradientBoostingModel, ClearGbmError> {
    train_gradient_boosting(
        x_train,
        TrainingLabels::Continuous(y_train),
        sample_weight,
        validation,
        config,
        feature_names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    )
}
