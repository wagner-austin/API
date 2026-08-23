//! Tests for the squared-error (regression) objective in the training loop.

use crate::error::ClearGbmError;
use crate::losses::squared_error_loss;
use crate::training::{Objective, TrainingLabels, ValidationData};

use super::train_helpers::{
    default_regression_params, make_config, make_regression_config, train_binary, train_regression,
};
use crate::training::GradientBoostingConfig;

/// Helper: dataset with a continuous target `y = 2*x0 + x1`.
fn make_regression_dataset() -> (Vec<Vec<f64>>, Vec<f64>, Vec<String>) {
    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64, 0.1_f64],
        vec![0.1_f64, 0.0_f64],
        vec![0.2_f64, 0.4_f64],
        vec![0.3_f64, 0.2_f64],
        vec![0.5_f64, 0.6_f64],
        vec![0.7_f64, 0.3_f64],
        vec![0.9_f64, 0.8_f64],
        vec![1.0_f64, 1.0_f64],
    ];
    let y: Vec<f64> = rows
        .iter()
        .map(|r| 2.0_f64 * r[0_usize] + r[1_usize])
        .collect();
    let names = vec!["f0".to_string(), "f1".to_string()];
    (rows, y, names)
}

#[test]
fn test_regression_training_runs_and_reports_objective() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_regression_config(5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_regression(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 5_usize);
    assert_eq!(model.config().objective(), Objective::SquaredError);
    assert_eq!(model.config().scale_pos_weight(), None);
    Ok(())
}

#[test]
fn test_regression_base_score_is_label_mean() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_regression_config(1_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_regression(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let mean: f64 = y_train.iter().sum::<f64>() / 8.0_f64;
    assert!(
        (model.base_prediction() - mean).abs() < 1e-15_f64,
        "regression base score must be the label mean: got {}, want {mean}",
        model.base_prediction()
    );
    Ok(())
}

#[test]
fn test_regression_training_mse_decreases() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let config_1 = match make_regression_config(1_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config_20 = match make_regression_config(20_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model_1 = match train_regression(&x_train, &y_train, None, &config_1, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let model_20 = match train_regression(&x_train, &y_train, None, &config_20, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let preds_1 = match model_1.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_20 = match model_20.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let mse_1 = propagate!(squared_error_loss(&y_train, &preds_1));
    let mse_20 = propagate!(squared_error_loss(&y_train, &preds_20));
    assert!(
        mse_20 < mse_1,
        "20 trees must fit the target better than 1: {mse_20} vs {mse_1}"
    );
    // On this noiseless target the ensemble should be genuinely close.
    assert!(mse_20 < 0.05_f64, "expected a tight fit, got MSE {mse_20}");
    Ok(())
}

#[test]
fn test_regression_deterministic_training() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_regression_config(4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let first = match train_regression(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let second = match train_regression(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let preds_first = match first.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_second = match second.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(preds_first, preds_second);
    Ok(())
}

#[test]
fn test_objective_changes_the_trained_model() -> Result<(), ClearGbmError> {
    // Knob-sensitivity for the objective axis itself: the same 0/1 target
    // trained as binary log loss and as squared error must produce different
    // raw predictions — log-odds and direct values do not coincide.
    let (rows, y_binary, feature_names) = super::train_helpers::make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_continuous: Vec<f64> = y_binary.iter().map(|&v| f64::from(v)).collect();

    let binary_config = match make_config(3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let regression_config = match make_regression_config(3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let binary_model = match train_binary(&x_train, &y_binary, None, &binary_config, &feature_names)
    {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let regression_model = match train_regression(
        &x_train,
        &y_continuous,
        None,
        &regression_config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let preds_binary = match binary_model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_regression = match regression_model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(
        preds_binary != preds_regression,
        "the two objectives produced identical raw predictions"
    );
    // Regression predictions of a 0/1 target stay in value space.
    for &p in &preds_regression {
        assert!(
            (-0.5_f64..=1.5_f64).contains(&p),
            "prediction {p} left value space"
        );
    }
    Ok(())
}

#[test]
fn test_regression_model_rejects_predict_proba() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_regression_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_regression(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    match model.predict_proba(&x_train) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "predict_proba must be rejected for a squared_error model".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "objective");
            assert!(
                reason.contains("predict_raw"),
                "rejection should point at predict_raw, got: {reason}"
            );
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_regression_early_stopping_triggers() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Anti-correlated validation targets — val MSE rises as train fit improves.
    let val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64, 0.1_f64], vec![0.9_f64, 0.9_f64]];
    let x_val: Vec<&[f64]> = val_rows.iter().map(Vec::as_slice).collect();
    let y_val: Vec<f64> = vec![10.0_f64, -10.0_f64];

    let mut params = default_regression_params();
    params.n_estimators = 100_usize;
    params.early_stopping_rounds = Some(3_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_regression(
        &x_train,
        &y_train,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Continuous(&y_val),
        }),
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert!(model.n_trees() < 100_usize);
    assert!(model.n_trees() >= 1_usize);
    Ok(())
}

#[test]
fn test_regression_validation_without_early_stopping() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let val_rows: Vec<Vec<f64>> = vec![vec![0.4_f64, 0.4_f64]];
    let x_val: Vec<&[f64]> = val_rows.iter().map(Vec::as_slice).collect();
    let y_val: Vec<f64> = vec![1.2_f64];

    let config = match make_regression_config(4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_regression(
        &x_train,
        &y_train,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Continuous(&y_val),
        }),
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 4_usize);
    Ok(())
}

#[test]
fn test_regression_rejects_binary_labels() -> Result<(), ClearGbmError> {
    let (rows, _, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_binary: Vec<u8> = vec![0_u8; rows.len()];

    let config = match make_regression_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let result = train_binary(&x_train, &y_binary, None, &config, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "squared_error must reject binary labels".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_train");
            assert!(
                reason.contains("squared_error"),
                "error should name the objective, got: {reason}"
            );
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_regression_rejects_binary_val_labels() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let val_rows: Vec<Vec<f64>> = vec![vec![0.4_f64, 0.4_f64]];
    let x_val: Vec<&[f64]> = val_rows.iter().map(Vec::as_slice).collect();
    let y_val: Vec<u8> = vec![1_u8];

    let config = match make_regression_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let result = train_regression(
        &x_train,
        &y_train,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Binary(&y_val),
        }),
        &config,
        &feature_names,
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "squared_error must reject binary validation labels".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "y_val");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_regression_rejects_non_finite_target() -> Result<(), ClearGbmError> {
    let (rows, mut y_train, feature_names) = make_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    y_train[3_usize] = f64::NAN;

    let config = match make_regression_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let result = train_regression(&x_train, &y_train, None, &config, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a NaN target must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_train");
            assert!(reason.contains("index 3"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_regression_negative_targets_train() -> Result<(), ClearGbmError> {
    // Regression targets are not probabilities; the whole real line is valid.
    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64, 0.0_f64],
        vec![0.2_f64, 0.1_f64],
        vec![0.6_f64, 0.5_f64],
        vec![1.0_f64, 0.9_f64],
    ];
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_train: Vec<f64> = vec![-5.0_f64, -1.0_f64, 3.0_f64, 7.0_f64];
    let feature_names = vec!["f0".to_string(), "f1".to_string()];

    let config = match make_regression_config(10_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_regression(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let preds = match model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    // The fit must move toward the negative targets, which no sigmoid-based
    // path could produce.
    assert!(
        preds[0_usize] < 0.0_f64,
        "expected a negative prediction, got {}",
        preds[0_usize]
    );
    assert!(preds[3_usize] > 0.0_f64);
    Ok(())
}
