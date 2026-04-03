//! Tests for GradientBoostingModel.

use crate::error::ClearGbmError;
use crate::training::{
    train_gradient_boosting, GradientBoostingConfig, GradientBoostingConfigParams,
    GradientBoostingModel,
};

/// Builds a small trained model for testing.
fn make_test_model() -> Result<GradientBoostingModel, ClearGbmError> {
    // Simple linearly separable dataset: class 0 has low values, class 1 has high values
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
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_train: Vec<u8> = vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8, 1_u8, 1_u8, 1_u8];
    let feature_names: Vec<String> = vec!["f0".to_string(), "f1".to_string()];

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
fn test_model_accessors() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 5_usize);
    assert_eq!(model.trees().len(), 5_usize);
    assert_eq!(model.n_classes(), 2_usize);
    assert!((model.learning_rate() - 0.3_f64).abs() < 1e-15_f64);
    assert_eq!(model.feature_names(), &["f0".to_string(), "f1".to_string()]);
    // base_prediction is the log-odds of the training labels (50% positive → ~0.0)
    assert!(model.base_prediction().is_finite());
    Ok(())
}

#[test]
fn test_model_config_accessor() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let config = model.config();
    assert_eq!(config.n_estimators(), 5_usize);
    assert_eq!(config.max_depth(), 2_usize);
    Ok(())
}

#[test]
fn test_predict_raw() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let rows: Vec<Vec<f64>> = vec![vec![0.0_f64, 0.0_f64], vec![1.0_f64, 1.0_f64]];
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let raw_preds = match model.predict_raw(&x) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(raw_preds.len(), 2_usize);
    // Class 0 sample should have lower raw prediction (more negative)
    // Class 1 sample should have higher raw prediction (more positive)
    assert!(raw_preds[0_usize] < raw_preds[1_usize]);
    Ok(())
}

#[test]
fn test_predict_proba() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let rows: Vec<Vec<f64>> = vec![vec![0.0_f64, 0.0_f64], vec![1.0_f64, 1.0_f64]];
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let probas = match model.predict_proba(&x) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(probas.len(), 2_usize);
    // Probabilities should sum to ~1.0
    let (p0_0, p1_0) = probas[0_usize];
    assert!((p0_0 + p1_0 - 1.0_f64).abs() < 1e-10_f64);
    let (p0_1, p1_1) = probas[1_usize];
    assert!((p0_1 + p1_1 - 1.0_f64).abs() < 1e-10_f64);
    // All probabilities in [0, 1]
    assert!((0.0_f64..=1.0_f64).contains(&p0_0));
    assert!((0.0_f64..=1.0_f64).contains(&p1_0));
    assert!((0.0_f64..=1.0_f64).contains(&p0_1));
    assert!((0.0_f64..=1.0_f64).contains(&p1_1));
    // Class 0 sample should predict class 0 with higher probability
    assert!(p0_0 > p1_0);
    // Class 1 sample should predict class 1 with higher probability
    assert!(p1_1 > p0_1);
    Ok(())
}

#[test]
fn test_predict_raw_empty() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let x: Vec<&[f64]> = vec![];
    let result = model.predict_raw(&x);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for empty input".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_predict_proba_empty() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let x: Vec<&[f64]> = vec![];
    let result = model.predict_proba(&x);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for empty input".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_predict_raw_invalid_learning_rate() -> Result<(), ClearGbmError> {
    let valid_model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    // Construct model with learning_rate = 0.0 to trigger PredictEnsembleConfig error
    let bad_model = GradientBoostingModel::new(
        valid_model.trees().to_vec(),
        valid_model.base_prediction(),
        0.0_f64,
        valid_model.feature_names().to_vec(),
        valid_model.n_classes(),
        valid_model.config().clone(),
    );
    let rows: Vec<Vec<f64>> = vec![vec![0.5_f64, 0.5_f64]];
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let result = bad_model.predict_raw(&x);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for learning_rate=0.0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "learning_rate");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_model_clone_and_eq() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let cloned = model.clone();
    assert_eq!(model, cloned);
    Ok(())
}

#[test]
fn test_model_debug() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let debug_str = format!("{model:?}");
    assert!(debug_str.contains("GradientBoostingModel"));
    Ok(())
}
