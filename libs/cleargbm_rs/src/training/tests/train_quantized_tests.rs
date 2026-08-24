//! End-to-end training tests for `quantized_gradient_bins`.
//!
//! The knob's contract at the training surface: deterministic per
//! config, honored (a coarse quantization changes the model), composed
//! correctly with GOSS / subsampling / early stopping / leaf-wise
//! growth, and exact under split training — the continuation's rounding
//! randoms and rotation offsets are pure functions of the config, so
//! 3 + 3 rounds equals a fresh 6-round run bit for bit.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::training::{
    continue_gradient_boosting, train_gradient_boosting, GradientBoostingConfig, Objective,
    Parallelism, TrainingLabels, TrainingRuntime, ValidationData,
};

use super::train_helpers::{default_params, make_simple_dataset};

/// Serializes a model, translating the serde error into the crate's.
fn model_json(model: &crate::training::GradientBoostingModel) -> Result<String, ClearGbmError> {
    match serde_json::to_string(model) {
        Ok(json) => Ok(json),
        Err(e) => Err(ClearGbmError::TreeConstructionFailed {
            reason: e.to_string(),
        }),
    }
}

/// Builds the single-threaded default runtime over borrowed hooks.
fn runtime(hooks: &Hooks) -> TrainingRuntime<'_> {
    TrainingRuntime {
        parallelism: Parallelism::Single,
        hooks,
    }
}

/// A binary config with quantized training at the given bin count.
fn quantized_config(
    n_estimators: usize,
    bins: Option<usize>,
) -> Result<GradientBoostingConfig, ClearGbmError> {
    let mut params = default_params();
    params.n_estimators = n_estimators;
    params.max_depth = 3_usize;
    params.quantized_gradient_bins = bins;
    GradientBoostingConfig::new(params)
}

/// A 40-row single-feature dataset with an irregular label pattern, so
/// split gains are close enough that gradient coarsening can reorder
/// them.
fn irregular_dataset() -> (Vec<Vec<f64>>, Vec<u8>, Vec<String>) {
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(40_usize);
    let mut y: Vec<u8> = Vec::with_capacity(40_usize);
    for i in 0_usize..40_usize {
        let scrambled = (i * 7919_usize) % 13_usize;
        let i_f = f64::from(u32::try_from(i).unwrap_or(u32::MAX));
        let s_f = f64::from(u32::try_from(scrambled).unwrap_or(u32::MAX));
        rows.push(vec![i_f, s_f]);
        y.push(u8::from(scrambled.is_multiple_of(3_usize) || i >= 30_usize));
    }
    let names = vec!["f0".to_string(), "f1".to_string()];
    (rows, y, names)
}

#[test]
fn test_the_knob_is_honored_coarse_bins_change_the_model() -> Result<(), ClearGbmError> {
    // Config honesty: 2-bin quantization must not silently train the
    // float path. On the irregular dataset the coarsened gains reorder
    // at least one split across 5 rounds.
    let hooks = Hooks::default();
    let (rows, y, names) = irregular_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let off = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(quantized_config(5_usize, None)),
        &names,
        &runtime(&hooks),
    ));
    let on = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(quantized_config(5_usize, Some(2_usize))),
        &names,
        &runtime(&hooks),
    ));
    let preds_off = propagate!(off.predict_raw(&x));
    let preds_on = propagate!(on.predict_raw(&x));
    assert!(
        preds_off != preds_on,
        "2-bin quantization must change the trained model"
    );
    Ok(())
}

#[test]
fn test_quantized_training_is_deterministic_per_config() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    let (rows, y, names) = irregular_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(quantized_config(4_usize, Some(4_usize)));
    let first = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &config,
        &names,
        &runtime(&hooks),
    ));
    let second = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &config,
        &names,
        &runtime(&hooks),
    ));
    let json_first = propagate!(model_json(&first));
    let json_second = propagate!(model_json(&second));
    assert_eq!(json_first, json_second, "same config + data = same model");
    Ok(())
}

#[test]
fn test_quantized_continuation_is_exact() -> Result<(), ClearGbmError> {
    // The rounding randoms are a pure function of (random_state, rows)
    // and the rotation offset of (random_state, GLOBAL round), so the
    // continuation's rounds 3..6 discretize exactly as the fresh run's
    // did — split training stays exact under quantization.
    let hooks = Hooks::default();
    let (rows, y, names) = irregular_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let base = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(quantized_config(3_usize, Some(4_usize))),
        &names,
        &runtime(&hooks),
    ));
    let continued = propagate!(continue_gradient_boosting(
        &base,
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        3_usize,
        &runtime(&hooks),
    ));
    let fresh = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(quantized_config(6_usize, Some(4_usize))),
        &names,
        &runtime(&hooks),
    ));
    let json_continued = propagate!(model_json(&continued));
    let json_fresh = propagate!(model_json(&fresh));
    assert_eq!(
        json_continued, json_fresh,
        "quantized split training must be exact"
    );
    Ok(())
}

#[test]
fn test_quantized_composes_with_goss() -> Result<(), ClearGbmError> {
    // GOSS reweights gradients in place, then the discretizer scans the
    // post-GOSS arrays — the composition trains and is deterministic.
    let hooks = Hooks::default();
    let (rows, y, names) = irregular_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mut params = default_params();
    params.n_estimators = 6_usize;
    params.learning_rate = 0.5_f64;
    params.quantized_gradient_bins = Some(4_usize);
    params.goss_top_rate = Some(0.3_f64);
    params.goss_other_rate = Some(0.3_f64);
    let config = propagate!(GradientBoostingConfig::new(params));
    let first = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &config,
        &names,
        &runtime(&hooks),
    ));
    let second = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &config,
        &names,
        &runtime(&hooks),
    ));
    assert_eq!(first.n_trees(), 6_usize);
    let json_first = propagate!(model_json(&first));
    let json_second = propagate!(model_json(&second));
    assert_eq!(json_first, json_second);
    Ok(())
}

#[test]
fn test_quantized_composes_with_row_subsampling() -> Result<(), ClearGbmError> {
    // subsample < 1 leaves some rows out of the tree; they fall back to
    // the tree walk while the discretizer still covers every row.
    let hooks = Hooks::default();
    let (rows, y, names) = irregular_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mut params = default_params();
    params.n_estimators = 4_usize;
    params.subsample = 0.6_f64;
    params.quantized_gradient_bins = Some(4_usize);
    let config = propagate!(GradientBoostingConfig::new(params));
    let model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &config,
        &names,
        &runtime(&hooks),
    ));
    assert_eq!(model.n_trees(), 4_usize);
    Ok(())
}

#[test]
fn test_quantized_composes_with_leaf_wise_growth() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    let (rows, y, names) = irregular_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mut params = default_params();
    params.n_estimators = 3_usize;
    params.growth_strategy = crate::training::GrowthStrategy::LeafWise;
    params.num_leaves = Some(4_usize);
    params.quantized_gradient_bins = Some(4_usize);
    let config = propagate!(GradientBoostingConfig::new(params));
    let model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &config,
        &names,
        &runtime(&hooks),
    ));
    assert_eq!(model.n_trees(), 3_usize);
    Ok(())
}

#[test]
fn test_quantized_early_stopping_still_fires() -> Result<(), ClearGbmError> {
    // A validation split whose labels contradict training makes the
    // validation loss rise; patience 1 must truncate the ensemble.
    let hooks = Hooks::default();
    let (rows, y, names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let flipped: Vec<u8> = y.iter().map(|&label| 1_u8 - label).collect();
    let mut params = default_params();
    params.n_estimators = 20_usize;
    params.early_stopping_rounds = Some(1_usize);
    params.quantized_gradient_bins = Some(4_usize);
    let config = propagate!(GradientBoostingConfig::new(params));
    let model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        Some(ValidationData {
            x: &x,
            y: TrainingLabels::Binary(&flipped),
            weight: None,
        }),
        &config,
        &names,
        &runtime(&hooks),
    ));
    assert!(model.n_trees() < 20_usize);
    Ok(())
}

#[test]
fn test_constant_label_regression_survives_zero_gradients() -> Result<(), ClearGbmError> {
    // Squared error on constant labels: the base prediction is the
    // label, every round-0 gradient is exactly zero, and the zero-max
    // scale guard runs inside real training.
    let hooks = Hooks::default();
    let rows: Vec<Vec<f64>> = (0_u32..8_u32).map(|i| vec![f64::from(i)]).collect();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y = vec![2.5_f64; 8_usize];
    let names = vec!["f0".to_string()];
    let mut params = default_params();
    params.objective = Objective::SquaredError;
    params.scale_pos_weight = None;
    params.n_estimators = 2_usize;
    params.quantized_gradient_bins = Some(4_usize);
    let config = propagate!(GradientBoostingConfig::new(params));
    let model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Continuous(&y),
        None,
        None,
        &config,
        &names,
        &runtime(&hooks),
    ));
    let preds = propagate!(model.predict_raw(&x));
    for &p in &preds {
        assert!((p - 2.5_f64).abs() < 1e-12_f64);
    }
    Ok(())
}

#[test]
fn test_quantized_model_round_trips_through_serde() -> Result<(), ClearGbmError> {
    // The artifact embeds the config; field 24 must survive the wire
    // and the reloaded model must predict identically.
    let hooks = Hooks::default();
    let (rows, y, names) = irregular_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(quantized_config(3_usize, Some(4_usize)));
    let model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &config,
        &names,
        &runtime(&hooks),
    ));
    let json = propagate!(model_json(&model));
    assert!(json.contains("\"quantized_gradient_bins\":4"));
    let reloaded: crate::training::GradientBoostingModel = match serde_json::from_str(&json) {
        Ok(m) => m,
        Err(e) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(reloaded.config().quantized_gradient_bins(), Some(4_usize));
    let preds_model = propagate!(model.predict_raw(&x));
    let preds_reloaded = propagate!(reloaded.predict_raw(&x));
    assert_eq!(preds_model, preds_reloaded);
    Ok(())
}
