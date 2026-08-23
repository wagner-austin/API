//! Tests for per-row sample weights in the training loop.
//!
//! Three properties anchor this file, in the order the charter demands:
//! knob sensitivity (weights must change the model), the all-ones identity
//! (`Some(&[1.0; n])` is bit-identical to `None`), and the derivation gate
//! (`scale_pos_weight` is the special case `w_i = spw` for positives — and
//! at an integer-valued weight, provably bit-identical through either
//! route, because integer-valued `f64` sums are exact under both the
//! closed-form multiply and per-row accumulation).

use crate::error::ClearGbmError;
use crate::training::{GradientBoostingConfig, TrainingLabels, ValidationData};

use super::train_helpers::{
    default_params, make_config, make_nested_dataset, make_regression_config,
    train_binary_weighted, train_regression_weighted,
};

/// Helper: regression dataset with a continuous target `y = 2*x0 + x1`.
fn make_weighted_regression_dataset() -> (Vec<Vec<f64>>, Vec<f64>, Vec<String>) {
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

// =============================================================================
// Knob sensitivity
// =============================================================================

#[test]
fn test_binary_sample_weights_change_the_trained_model() -> Result<(), ClearGbmError> {
    // The mandatory knob-sensitivity check: a weight vector that is not
    // all-ones must produce a different model, or the knob is decorative.
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let unweighted =
        match train_binary_weighted(&x_train, &y_train, None, None, &config, &feature_names) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
    let weights: Vec<f64> = (0_usize..y_train.len())
        .map(|i| if i % 2 == 0 { 4.0_f64 } else { 0.5_f64 })
        .collect();
    let weighted = match train_binary_weighted(
        &x_train,
        &y_train,
        Some(&weights),
        None,
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let preds_unweighted = match unweighted.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_weighted = match weighted.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(
        preds_unweighted != preds_weighted,
        "a non-uniform weight vector produced the same predictions as unweighted"
    );
    Ok(())
}

#[test]
fn test_regression_sample_weights_change_the_trained_model() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_weighted_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_regression_config(5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let unweighted =
        match train_regression_weighted(&x_train, &y_train, None, None, &config, &feature_names) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
    let weights: Vec<f64> = (0_usize..y_train.len())
        .map(|i| if i < 4 { 10.0_f64 } else { 0.1_f64 })
        .collect();
    let weighted = match train_regression_weighted(
        &x_train,
        &y_train,
        Some(&weights),
        None,
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let preds_unweighted = match unweighted.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_weighted = match weighted.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(
        preds_unweighted != preds_weighted,
        "a non-uniform weight vector produced the same regression predictions"
    );
    Ok(())
}

// =============================================================================
// All-ones identity
// =============================================================================

#[test]
fn test_all_ones_weights_are_bit_identical_to_none_binary() -> Result<(), ClearGbmError> {
    // `Some(&[1.0; n])` must reproduce `None` exactly: every weighted
    // expression multiplies by 1.0 (an IEEE identity) and the weight sums
    // accumulate 1.0 in exact integer-valued increments.
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let none_model =
        match train_binary_weighted(&x_train, &y_train, None, None, &config, &feature_names) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
    let ones = vec![1.0_f64; y_train.len()];
    let ones_model = match train_binary_weighted(
        &x_train,
        &y_train,
        Some(&ones),
        None,
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    assert_eq!(
        none_model.base_prediction().to_bits(),
        ones_model.base_prediction().to_bits(),
        "base scores must agree bit for bit"
    );
    let preds_none = match none_model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_ones = match ones_model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    for (a, b) in preds_none.iter().zip(preds_ones.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "predictions must agree bit for bit"
        );
    }
    Ok(())
}

#[test]
fn test_all_ones_weights_are_bit_identical_to_none_regression() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_weighted_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_regression_config(4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let none_model =
        match train_regression_weighted(&x_train, &y_train, None, None, &config, &feature_names) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
    let ones = vec![1.0_f64; y_train.len()];
    let ones_model = match train_regression_weighted(
        &x_train,
        &y_train,
        Some(&ones),
        None,
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    assert_eq!(
        none_model.base_prediction().to_bits(),
        ones_model.base_prediction().to_bits()
    );
    let preds_none = match none_model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_ones = match ones_model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    for (a, b) in preds_none.iter().zip(preds_ones.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    Ok(())
}

// =============================================================================
// The derivation gate: scale_pos_weight as the special case
// =============================================================================

#[test]
fn test_scale_pos_weight_is_the_derived_special_case() -> Result<(), ClearGbmError> {
    // spw = 3.0 through the config must equal w_i = 3.0-for-positives through
    // the sample-weight path with spw = 1.0, bit for bit. The weight is
    // integer-valued on purpose: 3.0 summed n times equals 3.0 * n exactly
    // in f64, so the closed-form base score (spw * count) and the per-row
    // accumulation (sum of weights) are provably the same bits. Effective
    // gradient weights agree because the class multiply by 1.0 is an IEEE
    // identity in one route and absent in the other.
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let mut spw_params = default_params();
    spw_params.n_estimators = 4_usize;
    spw_params.scale_pos_weight = Some(3.0_f64);
    let spw_config = match GradientBoostingConfig::new(spw_params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let unit_config = match make_config(4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let spw_model =
        match train_binary_weighted(&x_train, &y_train, None, None, &spw_config, &feature_names) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
    let derived_weights: Vec<f64> = y_train
        .iter()
        .map(|&y| if y == 1_u8 { 3.0_f64 } else { 1.0_f64 })
        .collect();
    let derived_model = match train_binary_weighted(
        &x_train,
        &y_train,
        Some(&derived_weights),
        None,
        &unit_config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    assert_eq!(
        spw_model.base_prediction().to_bits(),
        derived_model.base_prediction().to_bits(),
        "base scores must agree bit for bit between the two routes"
    );
    let preds_spw = match spw_model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_derived = match derived_model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    for (a, b) in preds_spw.iter().zip(preds_derived.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "the derived special case must reproduce the spw path bit for bit"
        );
    }
    Ok(())
}

// =============================================================================
// Weighted behavior and validation weights
// =============================================================================

#[test]
fn test_upweighted_regression_rows_fit_closer() -> Result<(), ClearGbmError> {
    // The weight must MEAN something: rows carrying 50x the weight should
    // end closer to their targets than the downweighted rest.
    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64],
        vec![0.2_f64],
        vec![0.4_f64],
        vec![0.6_f64],
        vec![0.8_f64],
        vec![1.0_f64],
    ];
    // A target the model cannot fit exactly at depth 1, forcing a tradeoff
    // the weights must decide.
    let y_train: Vec<f64> = vec![0.0_f64, 1.0_f64, 0.0_f64, 1.0_f64, 0.0_f64, 1.0_f64];
    let feature_names = vec!["f0".to_string()];
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let mut params = super::train_helpers::default_regression_params();
    params.n_estimators = 20_usize;
    params.max_depth = 1_usize;
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let weights: Vec<f64> = vec![50.0_f64, 1.0_f64, 50.0_f64, 1.0_f64, 50.0_f64, 1.0_f64];
    let model = match train_regression_weighted(
        &x_train,
        &y_train,
        Some(&weights),
        None,
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let preds = match model.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let heavy_err: f64 = [0_usize, 2, 4]
        .iter()
        .map(|&i| (preds[i] - y_train[i]).abs())
        .sum();
    let light_err: f64 = [1_usize, 3, 5]
        .iter()
        .map(|&i| (preds[i] - y_train[i]).abs())
        .sum();
    assert!(
        heavy_err < light_err,
        "50x-weighted rows must fit closer: heavy {heavy_err} vs light {light_err}"
    );
    Ok(())
}

#[test]
fn test_weighted_validation_early_stopping_runs() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_weighted_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64, 0.1_f64], vec![0.9_f64, 0.9_f64]];
    let x_val: Vec<&[f64]> = val_rows.iter().map(Vec::as_slice).collect();
    let y_val: Vec<f64> = vec![10.0_f64, -10.0_f64];
    let val_weights: Vec<f64> = vec![2.0_f64, 1.0_f64];

    let mut params = super::train_helpers::default_regression_params();
    params.n_estimators = 100_usize;
    params.early_stopping_rounds = Some(3_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_regression_weighted(
        &x_train,
        &y_train,
        None,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Continuous(&y_val),
            weight: Some(&val_weights),
        }),
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert!(model.n_trees() < 100_usize);
    Ok(())
}

#[test]
fn test_rejects_wrong_length_weights() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let short = vec![1.0_f64; y_train.len() - 1];
    match train_binary_weighted(
        &x_train,
        &y_train,
        Some(&short),
        None,
        &config,
        &feature_names,
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a short weight vector must be rejected".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { expected, .. }) => {
            assert!(expected.contains("sample_weight"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_rejects_zero_weight() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut weights = vec![1.0_f64; y_train.len()];
    weights[3_usize] = 0.0_f64;
    match train_binary_weighted(
        &x_train,
        &y_train,
        Some(&weights),
        None,
        &config,
        &feature_names,
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a zero weight must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "sample_weight");
            assert!(reason.contains("index 3"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_rejects_nan_val_weight() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_weighted_regression_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let val_rows: Vec<Vec<f64>> = vec![vec![0.4_f64, 0.4_f64]];
    let x_val: Vec<&[f64]> = val_rows.iter().map(Vec::as_slice).collect();
    let y_val: Vec<f64> = vec![1.2_f64];
    let val_weights: Vec<f64> = vec![f64::NAN];

    let config = match make_regression_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    match train_regression_weighted(
        &x_train,
        &y_train,
        None,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Continuous(&y_val),
            weight: Some(&val_weights),
        }),
        &config,
        &feature_names,
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a NaN validation weight must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "val_sample_weight");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_weighted_training_is_deterministic() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let weights: Vec<f64> = (0_usize..y_train.len())
        .map(|i| if i % 2 == 0 { 1.5_f64 } else { 2.0_f64 })
        .collect();
    let first = match train_binary_weighted(
        &x_train,
        &y_train,
        Some(&weights),
        None,
        &config,
        &feature_names,
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let second = match train_binary_weighted(
        &x_train,
        &y_train,
        Some(&weights),
        None,
        &config,
        &feature_names,
    ) {
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
