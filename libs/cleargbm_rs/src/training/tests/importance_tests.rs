//! Tests for feature_importances.

use crate::error::ClearGbmError;
use crate::training::{
    feature_importances, train_gradient_boosting, GradientBoostingConfig,
    GradientBoostingConfigParams,
};

/// Trains a small model where feature 0 is the only informative feature.
fn train_single_informative_feature_model() -> Result<
    crate::training::GradientBoostingModel,
    ClearGbmError,
> {
    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64, 0.5_f64, 0.5_f64],
        vec![0.0_f64, 0.4_f64, 0.6_f64],
        vec![0.1_f64, 0.6_f64, 0.4_f64],
        vec![0.1_f64, 0.5_f64, 0.5_f64],
        vec![0.9_f64, 0.5_f64, 0.5_f64],
        vec![0.9_f64, 0.6_f64, 0.4_f64],
        vec![1.0_f64, 0.4_f64, 0.6_f64],
        vec![1.0_f64, 0.5_f64, 0.5_f64],
    ];
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_train: Vec<u8> = vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8, 1_u8, 1_u8, 1_u8];
    let feature_names: Vec<String> =
        vec!["informative".to_string(), "noise_a".to_string(), "noise_b".to_string()];

    let config = match GradientBoostingConfig::new(GradientBoostingConfigParams {
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
    }) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    train_gradient_boosting(&x_train, &y_train, None, None, &config, &feature_names)
}

#[test]
fn test_importances_len_matches_feature_count() -> Result<(), ClearGbmError> {
    let model = match train_single_informative_feature_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let imp = feature_importances(&model);
    assert_eq!(imp.len(), 3_usize);
    Ok(())
}

#[test]
fn test_importances_feature_names_match_model() -> Result<(), ClearGbmError> {
    let model = match train_single_informative_feature_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let imp = feature_importances(&model);
    assert_eq!(imp[0_usize].0, "informative");
    assert_eq!(imp[1_usize].0, "noise_a");
    assert_eq!(imp[2_usize].0, "noise_b");
    Ok(())
}

#[test]
fn test_importances_sum_to_one_when_splits_exist() -> Result<(), ClearGbmError> {
    let model = match train_single_informative_feature_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let imp = feature_importances(&model);
    let total: f64 = imp.iter().map(|(_, v)| *v).sum();
    assert!(
        (total - 1.0_f64).abs() < 1e-12_f64,
        "importances must sum to 1.0, got {total}"
    );
    Ok(())
}

#[test]
fn test_importances_informative_feature_dominates() -> Result<(), ClearGbmError> {
    let model = match train_single_informative_feature_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let imp = feature_importances(&model);
    // On this data, the only feature that separates the classes is index 0.
    // Every internal node must split on it, so its importance is exactly 1.0.
    assert!(
        (imp[0_usize].1 - 1.0_f64).abs() < 1e-12_f64,
        "expected feature 0 importance = 1.0, got {}",
        imp[0_usize].1
    );
    assert!(
        imp[1_usize].1.abs() < 1e-12_f64,
        "expected feature 1 importance = 0.0, got {}",
        imp[1_usize].1
    );
    assert!(
        imp[2_usize].1.abs() < 1e-12_f64,
        "expected feature 2 importance = 0.0, got {}",
        imp[2_usize].1
    );
    Ok(())
}

#[test]
fn test_importances_all_zero_when_only_root_leaves() -> Result<(), ClearGbmError> {
    // Single-class data ⇒ constant-prediction trees ⇒ every root is a leaf.
    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64, 0.0_f64],
        vec![0.5_f64, 0.5_f64],
        vec![1.0_f64, 1.0_f64],
    ];
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    // min_samples_split = 100 forces every root node to stay as a leaf on this
    // 3-row dataset — no internal nodes, so no splits at all.
    let y_train: Vec<u8> = vec![0_u8, 1_u8, 0_u8];
    let feature_names: Vec<String> = vec!["a".to_string(), "b".to_string()];
    let config = match GradientBoostingConfig::new(GradientBoostingConfigParams {
        n_estimators: 3_usize,
        max_depth: 2_usize,
        learning_rate: 0.3_f64,
        min_samples_split: 100_usize,
        min_samples_leaf: 1_usize,
        max_bins: 4_usize,
        subsample: 1.0_f64,
        random_state: 42_u64,
        monotonic_constraints: None,
        reg_alpha: 0.0_f64,
        reg_lambda: 1.0_f64,
        early_stopping_rounds: None,
    }) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model =
        match train_gradient_boosting(&x_train, &y_train, None, None, &config, &feature_names) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
    let imp = feature_importances(&model);
    assert_eq!(imp.len(), 2_usize);
    for (name, v) in imp {
        assert!(
            v.abs() < 1e-15_f64,
            "expected all zeros when no splits (feature {name}), got {v}"
        );
    }
    Ok(())
}
