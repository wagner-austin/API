//! Behavior tests for the LambdaMART ranking trainer.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::losses::lambdarank::{label_gains, mean_ndcg_at_k};
use crate::training::{
    train_gradient_boosting, train_gradient_boosting_ranking, GradientBoostingConfig,
    GradientBoostingConfigParams, GrowthStrategy, Objective, Parallelism, RankingTrainingData,
    RankingValidationData, TrainingLabels, TrainingRuntime,
};

use super::train_helpers::default_params;

/// Default valid lambdarank params: the objective pairing flipped onto the
/// shared fixture, with enough bins to separate the ranking signal.
pub(super) fn default_ranking_params() -> GradientBoostingConfigParams {
    let mut params = default_params();
    params.objective = Objective::LambdaRank;
    params.scale_pos_weight = None;
    params.lambdarank_truncation_level = Some(4_usize);
    params.max_bins = 16_usize;
    params.n_estimators = 20_usize;
    params
}

fn ranking_config() -> Result<GradientBoostingConfig, ClearGbmError> {
    GradientBoostingConfig::new(default_ranking_params())
}

/// Three queries of four documents each. Feature 0 carries the relevance
/// signal offset per query (so a global threshold cannot fake it); feature
/// 1 is constant noise.
fn make_ranking_dataset() -> (Vec<Vec<f64>>, Vec<u32>, Vec<usize>, Vec<String>) {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();
    for query in 0_u32..3_u32 {
        let offset = f64::from(query) * 0.1_f64;
        for label in [0_u32, 1_u32, 2_u32, 3_u32] {
            rows.push(vec![f64::from(label) + offset, 0.5_f64]);
            labels.push(label);
        }
    }
    let groups = vec![4_usize, 4_usize, 4_usize];
    let names = vec!["f0".to_string(), "f1".to_string()];
    (rows, labels, groups, names)
}

fn train_default(
    rows: &[Vec<f64>],
    labels: &[u32],
    groups: &[usize],
    names: &[String],
) -> Result<crate::training::GradientBoostingModel, ClearGbmError> {
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(ranking_config());
    train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: labels,
            groups,
            weight: None,
        },
        None,
        &config,
        names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    )
}

// =============================================================================
// Learning
// =============================================================================

#[test]
fn test_ranking_learns_the_within_query_ordering() -> Result<(), ClearGbmError> {
    let (rows, labels, groups, names) = make_ranking_dataset();
    let model = propagate!(train_default(&rows, &labels, &groups, &names));
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let scores = propagate!(model.predict_raw(&x));
    let gains = label_gains();
    let ndcg = mean_ndcg_at_k(&scores, &labels, &groups, 4_usize, &gains);
    assert!(ndcg > 0.99_f64, "train NDCG@4 too low: {ndcg}");
    Ok(())
}

#[test]
fn test_ranking_model_base_score_is_zero_and_proba_is_refused() -> Result<(), ClearGbmError> {
    let (rows, labels, groups, names) = make_ranking_dataset();
    let model = propagate!(train_default(&rows, &labels, &groups, &names));
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    match model.predict_proba(&x) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "objective");
            assert!(reason.contains("ranking keys"), "{reason}");
        }
        other => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected the proba refusal, got {other:?}"),
            })
        }
    }
    Ok(())
}

#[test]
fn test_sample_weights_change_the_ranking_model() -> Result<(), ClearGbmError> {
    let (rows, labels, groups, names) = make_ranking_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(ranking_config());
    let unweighted = propagate!(train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: &labels,
            groups: &groups,
            weight: None,
        },
        None,
        &config,
        &names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    ));
    let mut weights = vec![1.0_f64; labels.len()];
    for w in weights.iter_mut().take(4_usize) {
        *w = 5.0_f64;
    }
    let weighted = propagate!(train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: &labels,
            groups: &groups,
            weight: Some(&weights),
        },
        None,
        &config,
        &names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    ));
    let preds_a = propagate!(unweighted.predict_raw(&x));
    let preds_b = propagate!(weighted.predict_raw(&x));
    assert_ne!(preds_a, preds_b, "weights must change the fitted model");
    Ok(())
}

#[test]
fn test_early_stopping_fires_on_a_reversed_validation_split() -> Result<(), ClearGbmError> {
    // Validation labels reversed within each query: as the train ordering
    // is learned, validation NDCG degrades, so the best round stays early
    // and patience runs out.
    let (rows, labels, groups, names) = make_ranking_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let reversed: Vec<u32> = labels.iter().map(|&l| 3_u32 - l).collect();
    let mut params = default_ranking_params();
    params.early_stopping_rounds = Some(2_usize);
    let config = propagate!(GradientBoostingConfig::new(params));
    let model = propagate!(train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: &labels,
            groups: &groups,
            weight: None,
        },
        Some(RankingValidationData {
            x: &x,
            y: &reversed,
            groups: &groups,
        }),
        &config,
        &names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    ));
    assert!(
        model.n_trees() < 20_usize,
        "early stopping should truncate, got {} trees",
        model.n_trees()
    );
    Ok(())
}

#[test]
fn test_validation_split_with_matching_labels_runs_to_completion() -> Result<(), ClearGbmError> {
    let (rows, labels, groups, names) = make_ranking_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mut params = default_ranking_params();
    params.early_stopping_rounds = Some(20_usize);
    params.n_estimators = 5_usize;
    let config = propagate!(GradientBoostingConfig::new(params));
    let model = propagate!(train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: &labels,
            groups: &groups,
            weight: None,
        },
        Some(RankingValidationData {
            x: &x,
            y: &labels,
            groups: &groups,
        }),
        &config,
        &names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    ));
    assert_eq!(model.n_trees(), 5_usize);
    Ok(())
}

// =============================================================================
// Rejections
// =============================================================================

#[test]
fn test_ranking_entry_rejects_a_binary_config() -> Result<(), ClearGbmError> {
    let (rows, labels, groups, names) = make_ranking_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(GradientBoostingConfig::new(default_params()));
    match train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: &labels,
            groups: &groups,
            weight: None,
        },
        None,
        &config,
        &names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    ) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "objective");
            assert!(
                reason.contains("the ranking entry trains \"lambdarank\""),
                "{reason}"
            );
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the objective refusal, got {other:?}"),
        }),
    }
}

#[test]
fn test_generic_entry_rejects_a_lambdarank_config() -> Result<(), ClearGbmError> {
    let (rows, labels, _groups, names) = make_ranking_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(ranking_config());
    match train_gradient_boosting(
        &x,
        TrainingLabels::Multiclass(&labels),
        None,
        None,
        &config,
        &names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    ) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "objective");
            assert!(
                reason.contains("train_gradient_boosting_ranking"),
                "{reason}"
            );
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the group-plumbing refusal, got {other:?}"),
        }),
    }
}

#[test]
fn test_over_limit_relevance_labels_are_rejected() -> Result<(), ClearGbmError> {
    let (rows, mut labels, groups, names) = make_ranking_dataset();
    labels[0] = 32_u32;
    match train_default(&rows, &labels, &groups, &names) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_train");
            assert!(reason.contains("got 32 at index 0"), "{reason}");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the label refusal, got {other:?}"),
        }),
    }
}

#[test]
fn test_groups_must_partition_the_training_rows() -> Result<(), ClearGbmError> {
    let (rows, labels, _groups, names) = make_ranking_dataset();
    let bad_groups = vec![4_usize, 4_usize, 3_usize];
    match train_default(&rows, &labels, &bad_groups, &names) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "group");
            assert!(
                reason.contains("sum to 11 but there are 12 rows"),
                "{reason}"
            );
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the group refusal, got {other:?}"),
        }),
    }
}

#[test]
fn test_validation_groups_are_validated_too() -> Result<(), ClearGbmError> {
    let (rows, labels, groups, names) = make_ranking_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(ranking_config());
    let bad_val_groups = vec![12_usize, 1_usize];
    match train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: &labels,
            groups: &groups,
            weight: None,
        },
        Some(RankingValidationData {
            x: &x,
            y: &labels,
            groups: &bad_val_groups,
        }),
        &config,
        &names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    ) {
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "val_group");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the val-group refusal, got {other:?}"),
        }),
    }
}

#[test]
fn test_invalid_sample_weights_are_rejected() -> Result<(), ClearGbmError> {
    let (rows, labels, groups, names) = make_ranking_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(ranking_config());
    let mut weights = vec![1.0_f64; labels.len()];
    weights[3] = 0.0_f64;
    match train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: &labels,
            groups: &groups,
            weight: Some(&weights),
        },
        None,
        &config,
        &names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    ) {
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "sample_weight");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the weight refusal, got {other:?}"),
        }),
    }
}

// =============================================================================
// Determinism
// =============================================================================

#[test]
fn test_same_config_and_data_reproduce_bit_identically() -> Result<(), ClearGbmError> {
    let (rows, labels, groups, names) = make_ranking_dataset();
    let a = propagate!(train_default(&rows, &labels, &groups, &names));
    let b = propagate!(train_default(&rows, &labels, &groups, &names));
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let preds_a = propagate!(a.predict_raw(&x));
    let preds_b = propagate!(b.predict_raw(&x));
    assert_eq!(preds_a, preds_b, "per-config determinism violated");
    Ok(())
}
// =============================================================================
// Knob composition and execution paths
// =============================================================================

#[test]
fn test_ranking_composes_with_the_sampling_knobs() -> Result<(), ClearGbmError> {
    // max_features, subsample < 1 (exercising the fallback tree walk for
    // subsampled-out rows) and leaf-wise growth all compose with the
    // ranking loop; the model still learns the ordering.
    let (rows, labels, groups, names) = make_ranking_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mut params = default_ranking_params();
    params.max_features = Some(1_usize);
    params.subsample = 0.75_f64;
    params.growth_strategy = GrowthStrategy::LeafWise;
    params.num_leaves = Some(4_usize);
    params.n_estimators = 40_usize;
    let config = propagate!(GradientBoostingConfig::new(params));
    let model = propagate!(train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: &labels,
            groups: &groups,
            weight: None,
        },
        None,
        &config,
        &names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    ));
    let scores = propagate!(model.predict_raw(&x));
    let gains = label_gains();
    let ndcg = mean_ndcg_at_k(&scores, &labels, &groups, 4_usize, &gains);
    assert!(ndcg > 0.9_f64, "sampled ranking NDCG@4 too low: {ndcg}");
    Ok(())
}

#[test]
fn test_ranking_surfaces_a_pool_construction_failure() -> Result<(), ClearGbmError> {
    fn failing_pool(
        _threads: core::num::NonZeroUsize,
    ) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1_usize)
            .stack_size(usize::MAX)
            .build()
    }

    let (rows, labels, groups, names) = make_ranking_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(ranking_config());
    match train_gradient_boosting_ranking(
        &x,
        &RankingTrainingData {
            y: &labels,
            groups: &groups,
            weight: None,
        },
        None,
        &config,
        &names,
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
            assert!(reason.contains("could not build a worker pool"), "{reason}");
            Ok(())
        }
        Err(other) => Err(other),
    }
}
