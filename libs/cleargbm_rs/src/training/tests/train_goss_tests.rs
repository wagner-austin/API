//! Behavior tests for GOSS (gradient-based one-side sampling).

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::training::{
    train_gradient_boosting, train_gradient_boosting_ranking, GradientBoostingConfig,
    GradientBoostingConfigParams, Objective, Parallelism, RankingTrainingData, TrainingLabels,
    TrainingRuntime,
};

use super::train_helpers::default_params;

/// Builds the single-threaded default runtime over borrowed hooks.
fn runtime(hooks: &Hooks) -> TrainingRuntime<'_> {
    TrainingRuntime {
        parallelism: Parallelism::Single,
        hooks,
    }
}

/// Forty rows on two features with a noisy separable signal — large
/// enough that GOSS's top/other split is non-degenerate.
fn make_goss_dataset() -> (Vec<Vec<f64>>, Vec<u8>, Vec<String>) {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    for i in 0_u32..40_u32 {
        let a = f64::from(i) / 40.0_f64;
        let b = f64::from((i * 7_u32) % 40_u32) / 40.0_f64;
        rows.push(vec![a, b]);
        labels.push(if a + 0.1_f64 * b > 0.5_f64 {
            1_u8
        } else {
            0_u8
        });
    }
    let names = vec!["f0".to_string(), "f1".to_string()];
    (rows, labels, names)
}

/// Default valid GOSS params over the shared binary fixture.
fn goss_params(top_rate: f64, other_rate: f64) -> GradientBoostingConfigParams {
    let mut params = default_params();
    params.n_estimators = 8_usize;
    params.goss_top_rate = Some(top_rate);
    params.goss_other_rate = Some(other_rate);
    params
}

fn train(
    params: GradientBoostingConfigParams,
) -> Result<crate::training::GradientBoostingModel, ClearGbmError> {
    let (rows, y, names) = make_goss_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = propagate!(GradientBoostingConfig::new(params));
    let hooks = Hooks::default();
    train_gradient_boosting(
        &x,
        TrainingLabels::Binary(&y),
        None,
        None,
        &config,
        &names,
        &runtime(&hooks),
    )
}

// =============================================================================
// Behavior
// =============================================================================

#[test]
fn test_goss_changes_the_model_after_warmup() -> Result<(), ClearGbmError> {
    // default_params has learning_rate 0.3, so the warmup is 3 rounds;
    // with 8 rounds the last 5 sample, and the model must differ from
    // the GOSS-off run.
    let with_goss = propagate!(train(goss_params(0.2_f64, 0.1_f64)));
    let mut off = default_params();
    off.n_estimators = 8_usize;
    let without = propagate!(train(off));
    let (rows, _, _) = make_goss_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    assert_ne!(
        propagate!(with_goss.predict_raw(&x)),
        propagate!(without.predict_raw(&x)),
        "GOSS must change the fitted model"
    );
    Ok(())
}

#[test]
fn test_goss_is_deterministic_per_config() -> Result<(), ClearGbmError> {
    let a = propagate!(train(goss_params(0.2_f64, 0.1_f64)));
    let b = propagate!(train(goss_params(0.2_f64, 0.1_f64)));
    let (rows, _, _) = make_goss_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    assert_eq!(
        propagate!(a.predict_raw(&x)),
        propagate!(b.predict_raw(&x)),
        "per-config determinism violated"
    );
    Ok(())
}

#[test]
fn test_goss_is_inert_during_the_warmup_rounds() -> Result<(), ClearGbmError> {
    // With learning_rate 0.3 the warmup is trunc(1/0.3) = 3 rounds; a
    // 3-round run therefore never samples, and GOSS on must be
    // bit-identical to GOSS off.
    let mut on = goss_params(0.2_f64, 0.1_f64);
    on.n_estimators = 3_usize;
    let with_goss = propagate!(train(on));
    let mut off = default_params();
    off.n_estimators = 3_usize;
    let without = propagate!(train(off));
    let (rows, _, _) = make_goss_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    assert_eq!(
        propagate!(with_goss.predict_raw(&x)),
        propagate!(without.predict_raw(&x)),
        "warmup rounds must be bit-identical to GOSS off"
    );
    Ok(())
}

#[test]
fn test_goss_composes_with_the_feature_sampling_knobs() -> Result<(), ClearGbmError> {
    let mut params = goss_params(0.3_f64, 0.2_f64);
    params.max_features = Some(1_usize);
    params.n_estimators = 10_usize;
    let model = propagate!(train(params));
    assert_eq!(model.n_trees(), 10_usize);
    Ok(())
}

// =============================================================================
// Config pairing
// =============================================================================

#[test]
fn test_goss_rates_travel_together() -> Result<(), ClearGbmError> {
    let mut lone_top = default_params();
    lone_top.goss_top_rate = Some(0.2_f64);
    match GradientBoostingConfig::new(lone_top) {
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "goss_other_rate");
        }
        other => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected the lone-top refusal, got {other:?}"),
            })
        }
    }
    let mut lone_other = default_params();
    lone_other.goss_other_rate = Some(0.1_f64);
    match GradientBoostingConfig::new(lone_other) {
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "goss_top_rate");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the lone-other refusal, got {other:?}"),
        }),
    }
}

#[test]
fn test_goss_rates_must_be_open_unit_and_sum_at_most_one() -> Result<(), ClearGbmError> {
    for (top, other, expected_name) in [
        (0.0_f64, 0.1_f64, "goss_top_rate"),
        (1.0_f64, 0.1_f64, "goss_top_rate"),
        (0.2_f64, 0.0_f64, "goss_other_rate"),
        (0.2_f64, 1.0_f64, "goss_other_rate"),
        (0.7_f64, 0.4_f64, "goss_top_rate"),
    ] {
        let params = goss_params(top, other);
        match GradientBoostingConfig::new(params) {
            Err(ClearGbmError::InvalidParameter { name, .. }) => {
                assert_eq!(name, expected_name, "top={top} other={other}");
            }
            other_result => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!(
                        "expected refusal for top={top} other={other}, got {other_result:?}"
                    ),
                })
            }
        }
    }
    Ok(())
}

#[test]
fn test_goss_excludes_row_subsampling() -> Result<(), ClearGbmError> {
    let mut params = goss_params(0.2_f64, 0.1_f64);
    params.subsample = 0.8_f64;
    match GradientBoostingConfig::new(params) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "subsample");
            assert!(reason.contains("GOSS replaces row subsampling"), "{reason}");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the subsample refusal, got {other:?}"),
        }),
    }
}

// =============================================================================
// Scope: single-score only, stated
// =============================================================================

#[test]
fn test_multiclass_and_ranking_refuse_goss() -> Result<(), ClearGbmError> {
    let hooks = Hooks::default();

    let mc_rows: Vec<Vec<f64>> = (0_u32..9_u32)
        .map(|i| vec![f64::from(10_u32 * (i / 3_u32) + i % 3_u32), 0.0_f64])
        .collect();
    let mc_y: Vec<u32> = (0_u32..9_u32).map(|i| i / 3_u32).collect();
    let mc_x: Vec<&[f64]> = mc_rows.iter().map(Vec::as_slice).collect();
    let names = vec!["f0".to_string(), "f1".to_string()];
    let mut params = goss_params(0.2_f64, 0.1_f64);
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    params.n_classes = Some(3_usize);
    params.max_bins = 16_usize;
    let mc_config = propagate!(GradientBoostingConfig::new(params));
    match train_gradient_boosting(
        &mc_x,
        TrainingLabels::Multiclass(&mc_y),
        None,
        None,
        &mc_config,
        &names,
        &runtime(&hooks),
    ) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "goss_top_rate");
            assert!(reason.contains("multiclass_softmax"), "{reason}");
        }
        other => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: format!("expected the multiclass GOSS refusal, got {other:?}"),
            })
        }
    }

    let rank_rows: Vec<Vec<f64>> = (0_u32..8_u32)
        .map(|i| vec![f64::from(i % 4_u32), 0.5_f64])
        .collect();
    let rank_y: Vec<u32> = (0_u32..8_u32).map(|i| i % 4_u32).collect();
    let rank_x: Vec<&[f64]> = rank_rows.iter().map(Vec::as_slice).collect();
    let groups = vec![4_usize, 4_usize];
    let mut params = goss_params(0.2_f64, 0.1_f64);
    params.objective = Objective::LambdaRank;
    params.scale_pos_weight = None;
    params.lambdarank_truncation_level = Some(4_usize);
    params.max_bins = 16_usize;
    let rank_config = propagate!(GradientBoostingConfig::new(params));
    match train_gradient_boosting_ranking(
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
    ) {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "goss_top_rate");
            assert!(reason.contains("lambdarank"), "{reason}");
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected the ranking GOSS refusal, got {other:?}"),
        }),
    }
}
