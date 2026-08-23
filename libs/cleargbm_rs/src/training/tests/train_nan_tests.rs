//! Tests proving the NaN routing direction is LEARNED per split, not fixed.
//!
//! The split search tries both NaN directions per candidate and keeps the
//! higher gain, so which side missing values fall on is decided by the
//! training data. The discriminating construction is a depth-1 stump on a
//! dataset whose only pure split must carry the missing rows to one
//! specific side of a threshold: routing NaN with the positives demands
//! `nan_goes_left = false` there, and mirroring the missing rows' labels
//! demands `true` — no fixed always-left or always-right policy can fit
//! both, and a stump has no second split to compensate with.

use crate::error::ClearGbmError;
use crate::training::{GradientBoostingConfig, GradientBoostingModel};

use super::train_helpers::{default_params, train_binary};

/// Eight rows on two features: feature 0 carries the finite values 1..4
/// labelled `[0, 0, 1, 1]` plus four NaNs, feature 1 is constant (no split
/// candidates). The only pure single split is the threshold between 2 and 3
/// with the missing rows routed to the side sharing their label.
fn make_nan_dataset(nan_rows_positive: bool) -> (Vec<Vec<f64>>, Vec<u8>) {
    let rows = vec![
        vec![1.0_f64, 0.0_f64],
        vec![2.0_f64, 0.0_f64],
        vec![3.0_f64, 0.0_f64],
        vec![4.0_f64, 0.0_f64],
        vec![f64::NAN, 0.0_f64],
        vec![f64::NAN, 0.0_f64],
        vec![f64::NAN, 0.0_f64],
        vec![f64::NAN, 0.0_f64],
    ];
    let nan_label: u8 = u8::from(nan_rows_positive);
    let y_train: Vec<u8> = vec![0, 0, 1, 1, nan_label, nan_label, nan_label, nan_label];
    (rows, y_train)
}

/// Trains a single depth-1 stump on the NaN dataset.
fn train_stump(nan_rows_positive: bool) -> Result<GradientBoostingModel, ClearGbmError> {
    let (rows, y_train) = make_nan_dataset(nan_rows_positive);
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["informative".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.n_estimators = 1_usize;
    params.max_depth = 1_usize;
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    train_binary(&x_train, &y_train, None, &config, &feature_names)
}

/// Collects the learned NaN directions from every split node in a model.
fn nan_directions(model: &GradientBoostingModel) -> Vec<bool> {
    let mut flags: Vec<bool> = Vec::new();
    for tree in model.trees() {
        for node in tree.nodes() {
            if !node.is_leaf() {
                flags.push(node.nan_goes_left());
            }
        }
    }
    flags
}

/// Positive-class probabilities for one missing row and the finite rows
/// 1.0 (a training negative) and 4.0 (a training positive).
fn probas(model: &GradientBoostingModel) -> Result<(f64, f64, f64), ClearGbmError> {
    let nan_row: Vec<f64> = vec![f64::NAN, 0.0_f64];
    let low_row: Vec<f64> = vec![1.0_f64, 0.0_f64];
    let high_row: Vec<f64> = vec![4.0_f64, 0.0_f64];
    let query: Vec<&[f64]> = vec![nan_row.as_slice(), low_row.as_slice(), high_row.as_slice()];
    let all = match model.predict_proba(&query) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    match (all.first(), all.get(1_usize), all.get(2_usize)) {
        (Some(&(_, nan_p)), Some(&(_, low_p)), Some(&(_, high_p))) => Ok((nan_p, low_p, high_p)),
        _ => Err(ClearGbmError::TreeConstructionFailed {
            reason: "predict_proba returned fewer than three rows".to_string(),
        }),
    }
}

#[test]
fn test_stump_routes_nan_toward_the_positive_side() -> Result<(), ClearGbmError> {
    // Missing rows are positives: the pure split is 2|3 with NaN going
    // RIGHT (to the positive side), so the learned flag must be false and
    // the stump must already separate the missing rows from the negatives.
    let model = match train_stump(true) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(
        nan_directions(&model),
        vec![false],
        "the stump should route NaN right, with the positives"
    );
    let (nan_p, low_p, high_p) = match probas(&model) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(nan_p > low_p, "NaN ({nan_p}) should outscore 1.0 ({low_p})");
    assert!(
        (nan_p - high_p).abs() < 1e-12_f64,
        "NaN ({nan_p}) should land in the positive leaf with 4.0 ({high_p})"
    );
    Ok(())
}

#[test]
fn test_stump_nan_routing_flips_when_the_missing_labels_are_mirrored() -> Result<(), ClearGbmError>
{
    // The same feature layout with the missing rows relabelled negative:
    // the pure split is still 2|3 but NaN must now go LEFT. A stump has no
    // second split to absorb a wrong fixed direction, so together with the
    // test above this rules out any fixed policy.
    let model = match train_stump(false) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(
        nan_directions(&model),
        vec![true],
        "the stump should route NaN left, with the negatives"
    );
    let (nan_p, low_p, high_p) = match probas(&model) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(
        nan_p < high_p,
        "NaN ({nan_p}) should undercut 4.0 ({high_p})"
    );
    assert!(
        (nan_p - low_p).abs() < 1e-12_f64,
        "NaN ({nan_p}) should land in the negative leaf with 1.0 ({low_p})"
    );
    Ok(())
}

#[test]
fn test_deeper_model_classifies_missing_rows_by_their_learned_side() -> Result<(), ClearGbmError> {
    // End to end at real depth: several boosting rounds drive the missing
    // rows' probability to the correct side of one half on both datasets.
    let build = |nan_rows_positive: bool| -> Result<f64, ClearGbmError> {
        let (rows, y_train) = make_nan_dataset(nan_rows_positive);
        let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let feature_names = vec!["informative".to_string(), "constant".to_string()];
        let config = match GradientBoostingConfig::new(default_params()) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };
        let (nan_p, _, _) = match probas(&model) {
            Ok(p) => p,
            Err(e) => return Err(e),
        };
        Ok(nan_p)
    };
    let positive_nan_proba = match build(true) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let mirrored_nan_proba = match build(false) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(
        positive_nan_proba > 0.5_f64,
        "positive missing rows should classify positive, got {positive_nan_proba}"
    );
    assert!(
        mirrored_nan_proba < 0.5_f64,
        "negative missing rows should classify negative, got {mirrored_nan_proba}"
    );
    Ok(())
}
