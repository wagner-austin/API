//! Behavior tests for continued training.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::training::{
    continue_gradient_boosting, train_gradient_boosting, GradientBoostingConfig, Objective,
    Parallelism, TrainingLabels, TrainingRuntime, ValidationData,
};

use super::train_helpers::{default_params, make_simple_dataset};

/// Builds the single-threaded default runtime over borrowed hooks.
fn runtime(hooks: &Hooks) -> TrainingRuntime<'_> {
    TrainingRuntime {
        parallelism: Parallelism::Single,
        hooks,
    }
}

/// A binary config with the given round budget and otherwise-deterministic
/// knobs (full subsample, no feature sampling).
fn binary_config(n_estimators: usize) -> Result<GradientBoostingConfig, ClearGbmError> {
    let mut params = default_params();
    params.n_estimators = n_estimators;
    GradientBoostingConfig::new(params)
}

// =============================================================================
// The continuation identity: split training equals one run
// =============================================================================

#[test]
fn test_continuing_three_plus_three_equals_a_fresh_six_round_run() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    // With full subsample and no feature sampling the loop is
    // deterministic and stateless across rounds, so training 3 rounds and
    // continuing 3 more on the SAME data must reproduce a fresh 6-round
    // run bit for bit: the continuation's starting scores are exactly the
    // running scores the fresh run had after round 3.
    let (rows, y, names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let base_model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(binary_config(3_usize)),
        &names,
        &runtime(&hooks),
    ));
    let continued = propagate!(continue_gradient_boosting(
        &base_model,
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
        &propagate!(binary_config(6_usize)),
        &names,
        &runtime(&hooks),
    ));

    assert_eq!(continued.n_trees(), 6_usize);
    let preds_continued = propagate!(continued.predict_raw(&x));
    let preds_fresh = propagate!(fresh.predict_raw(&x));
    assert_eq!(preds_continued, preds_fresh, "split training must be exact");
    Ok(())
}

#[test]
fn test_regression_continuation_matches_a_fresh_run() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    let rows: Vec<Vec<f64>> = (0_u32..16_u32)
        .map(|i| vec![f64::from(i), 0.5_f64])
        .collect();
    let y: Vec<f64> = (0_u32..16_u32).map(|i| f64::from(i) * 2.0_f64).collect();
    let names = vec!["f0".to_string(), "f1".to_string()];
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mut params = default_params();
    params.objective = Objective::SquaredError;
    params.scale_pos_weight = None;
    params.n_estimators = 2_usize;
    let config2 = propagate!(GradientBoostingConfig::new(params.clone()));
    params.n_estimators = 4_usize;
    let config4 = propagate!(GradientBoostingConfig::new(params));

    let base_model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Continuous(&y),
        None,
        None,
        &config2,
        &names,
        &runtime(&hooks),
    ));
    let continued = propagate!(continue_gradient_boosting(
        &base_model,
        &x,
        TrainingLabels::Continuous(&y),
        None,
        None,
        2_usize,
        &runtime(&hooks),
    ));
    let fresh = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Continuous(&y),
        None,
        None,
        &config4,
        &names,
        &runtime(&hooks),
    ));
    assert_eq!(
        propagate!(continued.predict_raw(&x)),
        propagate!(fresh.predict_raw(&x)),
    );
    Ok(())
}

// =============================================================================
// The continued artifact
// =============================================================================

#[test]
fn test_continuation_updates_the_budget_and_leaves_the_original_alone() -> Result<(), ClearGbmError>
{
    let hooks = Hooks::default();
    let (rows, y, names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let base_model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(binary_config(3_usize)),
        &names,
        &runtime(&hooks),
    ));
    let continued = propagate!(continue_gradient_boosting(
        &base_model,
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        2_usize,
        &runtime(&hooks),
    ));
    // The continued artifact states its combined budget; the input model
    // is untouched.
    assert_eq!(continued.config().n_estimators(), 5_usize);
    assert_eq!(continued.n_trees(), 5_usize);
    assert_eq!(base_model.n_trees(), 3_usize);
    assert_eq!(base_model.config().n_estimators(), 3_usize);
    assert_eq!(continued.base_prediction(), base_model.base_prediction());
    assert_eq!(continued.feature_names(), base_model.feature_names());
    Ok(())
}

#[test]
fn test_continuation_on_new_data_changes_predictions() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    let (rows, y, names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let base_model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(binary_config(3_usize)),
        &names,
        &runtime(&hooks),
    ));
    // New rows near the decision boundary with flipped labels: the
    // continuation must move the scores.
    let new_rows: Vec<Vec<f64>> = vec![
        vec![0.45_f64, 0.5_f64],
        vec![0.5_f64, 0.45_f64],
        vec![0.55_f64, 0.5_f64],
        vec![0.5_f64, 0.55_f64],
    ];
    let new_y = vec![1_u8, 1_u8, 0_u8, 0_u8];
    let new_x: Vec<&[f64]> = new_rows.iter().map(Vec::as_slice).collect();
    let continued = propagate!(continue_gradient_boosting(
        &base_model,
        &new_x,
        TrainingLabels::Binary(&new_y),
        None,
        None,
        3_usize,
        &runtime(&hooks),
    ));
    let before = propagate!(base_model.predict_raw(&new_x));
    let after = propagate!(continued.predict_raw(&new_x));
    assert_ne!(before, after, "continuation on new data must move scores");
    Ok(())
}

#[test]
fn test_continuation_early_stops_on_a_contradicting_validation_split() -> Result<(), ClearGbmError>
{
    let hooks = Hooks::default();
    let (rows, y, names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let flipped: Vec<u8> = y.iter().map(|&l| 1_u8 - l).collect();
    let mut params = default_params();
    params.n_estimators = 3_usize;
    params.early_stopping_rounds = Some(2_usize);
    let config = propagate!(GradientBoostingConfig::new(params));
    let base_model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        Some(ValidationData {
            x: &x,
            y: TrainingLabels::Binary(&y),
            weight: None,
        }),
        &config,
        &names,
        &runtime(&hooks),
    ));
    // Continue against a validation split whose labels contradict the
    // training labels: validation loss worsens every round, so the
    // config's patience truncates the continuation early.
    let continued = propagate!(continue_gradient_boosting(
        &base_model,
        &x,
        TrainingLabels::Binary(&y),
        None,
        Some(ValidationData {
            x: &x,
            y: TrainingLabels::Binary(&flipped),
            weight: None,
        }),
        20_usize,
        &runtime(&hooks),
    ));
    assert!(
        continued.n_trees() < base_model.n_trees() + 20_usize,
        "early stopping should truncate the continuation, got {} trees",
        continued.n_trees()
    );
    Ok(())
}

// =============================================================================
// Rejections
// =============================================================================

#[test]
fn test_zero_additional_rounds_is_rejected() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    let (rows, y, names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let base_model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(binary_config(2_usize)),
        &names,
        &runtime(&hooks),
    ));
    match continue_gradient_boosting(
        &base_model,
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        0_usize,
        &runtime(&hooks),
    ) {
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "additional_rounds");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the rounds refusal, got {other:?}"),
        }),
    }
}

#[test]
fn test_multiclass_and_ranking_models_are_refused() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    use crate::training::{train_gradient_boosting_ranking, RankingTrainingData};
    // Multiclass model.
    let mc_rows: Vec<Vec<f64>> = (0_u32..9_u32)
        .map(|i| vec![f64::from(10_u32 * (i / 3_u32) + i % 3_u32), 0.0_f64])
        .collect();
    let mc_y: Vec<u32> = (0_u32..9_u32).map(|i| i / 3_u32).collect();
    let mc_x: Vec<&[f64]> = mc_rows.iter().map(Vec::as_slice).collect();
    let names = vec!["f0".to_string(), "f1".to_string()];
    let mut params = default_params();
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    params.n_classes = Some(3_usize);
    params.max_bins = 16_usize;
    let mc_config = propagate!(GradientBoostingConfig::new(params));
    let mc_model = propagate!(train_gradient_boosting(
        &mc_x,
        TrainingLabels::Multiclass(&mc_y),
        None,
        None,
        &mc_config,
        &names,
        &runtime(&hooks),
    ));
    match continue_gradient_boosting(
        &mc_model,
        &mc_x,
        TrainingLabels::Multiclass(&mc_y),
        None,
        None,
        1_usize,
        &runtime(&hooks),
    ) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "objective");
            assert!(reason.contains("multiclass_softmax"), "{reason}");
        }
        other => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected the multiclass refusal, got {other:?}"),
            })
        }
    }

    // Ranking model.
    let rank_rows: Vec<Vec<f64>> = (0_u32..8_u32)
        .map(|i| vec![f64::from(i % 4_u32), 0.5_f64])
        .collect();
    let rank_y: Vec<u32> = (0_u32..8_u32).map(|i| i % 4_u32).collect();
    let rank_x: Vec<&[f64]> = rank_rows.iter().map(Vec::as_slice).collect();
    let groups = vec![4_usize, 4_usize];
    let mut params = default_params();
    params.objective = Objective::LambdaRank;
    params.scale_pos_weight = None;
    params.lambdarank_truncation_level = Some(4_usize);
    params.max_bins = 16_usize;
    let rank_config = propagate!(GradientBoostingConfig::new(params));
    let rank_model = propagate!(train_gradient_boosting_ranking(
        &rank_x,
        &RankingTrainingData {
            y: &rank_y,
            groups: &groups,
            weight: None,
        },
        None,
        &rank_config,
        &names,
        &runtime(&hooks),
    ));
    match continue_gradient_boosting(
        &rank_model,
        &rank_x,
        TrainingLabels::Multiclass(&rank_y),
        None,
        None,
        1_usize,
        &runtime(&hooks),
    ) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "objective");
            assert!(reason.contains("lambdarank"), "{reason}");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the ranking refusal, got {other:?}"),
        }),
    }
}

#[test]
fn test_wrong_label_kind_is_rejected_by_the_pairing() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    let (rows, y, names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let base_model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(binary_config(2_usize)),
        &names,
        &runtime(&hooks),
    ));
    let wrong: Vec<f64> = y.iter().map(|&l| f64::from(l)).collect();
    match continue_gradient_boosting(
        &base_model,
        &x,
        TrainingLabels::Continuous(&wrong),
        None,
        None,
        1_usize,
        &runtime(&hooks),
    ) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_train");
            assert!(reason.contains("binary (u8) labels"), "{reason}");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the pairing refusal, got {other:?}"),
        }),
    }
}

#[test]
fn test_feature_count_must_match_the_model() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    let (rows, y, names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let base_model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(binary_config(2_usize)),
        &names,
        &runtime(&hooks),
    ));
    let narrow_rows: Vec<Vec<f64>> = rows.iter().map(|r| vec![r[0]]).collect();
    let narrow_x: Vec<&[f64]> = narrow_rows.iter().map(Vec::as_slice).collect();
    match continue_gradient_boosting(
        &base_model,
        &narrow_x,
        TrainingLabels::Binary(&y),
        None,
        None,
        1_usize,
        &runtime(&hooks),
    ) {
        Err(_) => Ok(()),
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a feature-count mismatch must be rejected".to_string(),
        }),
    }
}

#[test]
fn test_continuation_surfaces_a_pool_construction_failure() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();
    fn failing_pool(
        _threads: core::num::NonZeroUsize,
    ) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1_usize)
            .stack_size(usize::MAX)
            .build()
    }

    let (rows, y, names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let base_model = propagate!(train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &propagate!(binary_config(2_usize)),
        &names,
        &runtime(&hooks),
    ));
    match continue_gradient_boosting(
        &base_model,
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        1_usize,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::with_pool_builder(failing_pool),
        },
    ) {
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_jobs");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the pool refusal, got {other:?}"),
        }),
    }
}
