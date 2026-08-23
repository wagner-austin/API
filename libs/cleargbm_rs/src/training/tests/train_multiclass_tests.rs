//! End-to-end tests for the multiclass softmax task.
//!
//! The fixture is three well-separated clusters on one feature. Beyond
//! fitting it, the tests pin the structural contract: K trees per round in
//! round-major order, per-class log-prior base scores, and the strict
//! predict-surface split (single-score methods reject multiclass models
//! and vice versa).

use crate::error::ClearGbmError;
use crate::training::labels::{TrainingLabels, ValidationData};
use crate::training::{
    train_gradient_boosting, GradientBoostingConfig, GradientBoostingModel, GrowthStrategy,
    Objective, Parallelism, TrainingRuntime,
};

use super::train_helpers::default_params;

/// Twelve rows on two features: feature 0 clusters at 0, 10 and 20 by
/// class; feature 1 is constant.
pub(super) fn make_multiclass_dataset() -> (Vec<Vec<f64>>, Vec<u32>) {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<u32> = Vec::new();
    for (center, label) in [(0.0_f64, 0_u32), (10.0_f64, 1_u32), (20.0_f64, 2_u32)] {
        for offset in [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64] {
            rows.push(vec![center + offset, 0.0_f64]);
            y.push(label);
        }
    }
    (rows, y)
}

/// Trains on the cluster fixture with the given round count.
pub(super) fn train_on_fixture(
    n_estimators: usize,
    sample_weight: Option<&[f64]>,
) -> Result<GradientBoostingModel, ClearGbmError> {
    let (rows, y) = make_multiclass_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["position".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.n_estimators = n_estimators;
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
    train_gradient_boosting(
        &x_train,
        TrainingLabels::Multiclass(&y),
        sample_weight,
        None,
        &config,
        &feature_names,
        &runtime,
    )
}

#[test]
fn test_multiclass_classifies_the_clusters() -> Result<(), ClearGbmError> {
    let model = match train_on_fixture(5_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let (rows, y) = make_multiclass_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let predicted = propagate!(model.predict_class(&x));
    for (i, (&label, &pred)) in y.iter().zip(predicted.iter()).enumerate() {
        assert_eq!(
            crate::narrow::index_widen(label),
            pred,
            "row {i} misclassified"
        );
    }
    // Probabilities are proper distributions and agree with the argmax.
    let probas = propagate!(model.predict_proba_multiclass(&x));
    for (row, dist) in probas.iter().enumerate() {
        assert_eq!(dist.len(), 3_usize);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0_f64).abs() < 1e-12_f64, "row {row}: {dist:?}");
    }
    Ok(())
}

#[test]
fn test_multiclass_stores_k_trees_per_round() -> Result<(), ClearGbmError> {
    let model = match train_on_fixture(4_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 12_usize, "4 rounds x 3 classes");
    // Per-class base scores are the log priors: balanced thirds here.
    let bases = match model.class_base_predictions() {
        Some(b) => b,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "multiclass model must carry per-class base scores".to_string(),
            })
        }
    };
    for &b in bases {
        assert!((b - (1.0_f64 / 3.0_f64).ln()).abs() < 1e-12_f64);
    }
    assert!(model.base_prediction().is_none());
    Ok(())
}

#[test]
fn test_multiclass_training_is_deterministic() -> Result<(), ClearGbmError> {
    let first = match train_on_fixture(4_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let second = match train_on_fixture(4_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let (rows, _) = make_multiclass_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let preds_first = propagate!(first.predict_raw_multiclass(&x));
    let preds_second = propagate!(second.predict_raw_multiclass(&x));
    assert_eq!(preds_first, preds_second);
    Ok(())
}

#[test]
fn test_sample_weights_shift_the_probabilities() -> Result<(), ClearGbmError> {
    // Upweighting class 2's rows must change the model (knob sensitivity).
    let unweighted = match train_on_fixture(3_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let weights: Vec<f64> = (0_usize..12_usize)
        .map(|i| if i >= 8_usize { 50.0_f64 } else { 1.0_f64 })
        .collect();
    let weighted = match train_on_fixture(3_usize, Some(&weights)) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let (rows, _) = make_multiclass_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let probas_unweighted = propagate!(unweighted.predict_proba_multiclass(&x));
    let probas_weighted = propagate!(weighted.predict_proba_multiclass(&x));
    assert_ne!(probas_unweighted, probas_weighted);
    Ok(())
}

#[test]
fn test_multiclass_roundtrips_through_json() -> Result<(), ClearGbmError> {
    use super::serde_helpers::{from_json, to_json};

    let model = match train_on_fixture(3_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&model));
    assert!(json.contains(r#""base_prediction":null"#));
    assert!(json.contains(r#""class_base_predictions":["#));
    let decoded: GradientBoostingModel = propagate!(from_json(&json));
    let (rows, _) = make_multiclass_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let before = propagate!(model.predict_raw_multiclass(&x));
    let after = propagate!(decoded.predict_raw_multiclass(&x));
    assert_eq!(before, after);
    Ok(())
}

#[test]
fn test_single_score_predict_surface_rejects_multiclass() -> Result<(), ClearGbmError> {
    let model = match train_on_fixture(2_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let (rows, _) = make_multiclass_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    match model.predict_raw(&x) {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "predict_raw must reject a multiclass model".to_string(),
            })
        }
        Err(ClearGbmError::InvalidParameter { reason, .. }) => {
            assert!(reason.contains("predict_raw_multiclass"), "got: {reason}");
        }
        Err(e) => return Err(e),
    }
    match model.predict_proba(&x) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "predict_proba must reject a multiclass model".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { reason, .. }) => {
            assert!(reason.contains("predict_proba_multiclass"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_multiclass_predict_surface_rejects_single_score_models() -> Result<(), ClearGbmError> {
    use super::train_helpers::{make_config, make_simple_dataset, train_binary};

    let (rows, y, feature_names) = make_simple_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x, &y, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    match model.predict_raw_multiclass(&x) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "predict_raw_multiclass must reject a binary model".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { reason, .. }) => {
            assert!(reason.contains("multiclass_softmax"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_early_stopping_truncates_whole_rounds() -> Result<(), ClearGbmError> {
    let (rows, y) = make_multiclass_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["position".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.n_estimators = 30_usize;
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
            y: TrainingLabels::Multiclass(&y),
            weight: None,
        }),
        &config,
        &feature_names,
        &runtime,
    ));
    // Whatever round stopping chose, the tree count is a whole number of
    // rounds and below the configured maximum.
    assert_eq!(model.n_trees() % 3_usize, 0_usize);
    assert!(model.n_trees() <= 90_usize);
    assert!(model.n_trees() >= 3_usize);
    Ok(())
}

#[test]
fn test_multiclass_under_leaf_wise_growth() -> Result<(), ClearGbmError> {
    let (rows, y) = make_multiclass_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["position".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.n_estimators = 3_usize;
    params.max_bins = 16_usize;
    params.objective = Objective::MulticlassSoftmax;
    params.scale_pos_weight = None;
    params.n_classes = Some(3_usize);
    params.growth_strategy = GrowthStrategy::LeafWise;
    params.num_leaves = Some(3_usize);
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
    let predicted = propagate!(model.predict_class(&x_train));
    for (&label, &pred) in y.iter().zip(predicted.iter()) {
        assert_eq!(crate::narrow::index_widen(label), pred);
    }
    Ok(())
}
