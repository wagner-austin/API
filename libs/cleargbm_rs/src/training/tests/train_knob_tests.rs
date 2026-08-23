//! Knob-sensitivity tests for the training loop.
//!
//! Every config knob must visibly change the trained model when moved — a
//! knob that types, coverage, and completion tests all validate while its
//! value goes nowhere is the decorative-knob defect class this crate's
//! history documents. Each test here trains twice and asserts the models
//! differ (or agree, for determinism).

use crate::error::ClearGbmError;
use crate::training::GradientBoostingConfig;

use super::train_helpers::{
    default_params, make_config, make_leaf_wise_config, make_nested_dataset, train_binary,
};

#[test]
fn test_training_dispatches_to_leaf_wise_growth() -> Result<(), ClearGbmError> {
    // The dispatch in `train` is the only thing that routes a config's policy
    // to a builder. A regression there would silently train depth-wise, which
    // is exactly the mislabelled-arm failure the axis exists to prevent — so
    // this asserts a shape only the leaf budget can produce.
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_leaf_wise_config(3_usize, 2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 3_usize);
    // A budget of 2 permits exactly one split per tree. Depth-wise growth on
    // this dataset at max_depth 2 reaches three leaves, so the counts separate
    // the two policies rather than merely confirming training ran.
    for tree in model.trees() {
        assert_eq!(tree.n_leaves(), 2_usize);
    }
    Ok(())
}

#[test]
fn test_leaf_wise_and_depth_wise_produce_different_trees() -> Result<(), ClearGbmError> {
    // Guards against a dispatch that compiles but routes both policies to the
    // same builder: with a binding budget the two must not agree.
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let depth_config = match make_config(1_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let leaf_config = match make_leaf_wise_config(1_usize, 2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let depth_model = match train_binary(&x_train, &y_train, None, &depth_config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let leaf_model = match train_binary(&x_train, &y_train, None, &leaf_config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let depth_leaves: Vec<usize> = depth_model
        .trees()
        .iter()
        .map(crate::tree::Tree::n_leaves)
        .collect();
    let leaf_leaves: Vec<usize> = leaf_model
        .trees()
        .iter()
        .map(crate::tree::Tree::n_leaves)
        .collect();
    assert_ne!(
        depth_leaves, leaf_leaves,
        "a binding leaf budget must change the tree shape"
    );
    Ok(())
}

#[test]
fn test_scale_pos_weight_changes_the_trained_model() -> Result<(), ClearGbmError> {
    // The knob-sensitivity check this crate's history demands: a weighted
    // config must produce a different model than the unweighted one, or the
    // knob is decorative. A weight of 5 shifts the base score and every
    // positive gradient, so raw predictions cannot coincide.
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let unweighted = match make_config(3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut weighted_params = default_params();
    weighted_params.n_estimators = 3_usize;
    weighted_params.scale_pos_weight = Some(5.0_f64);
    let weighted = match GradientBoostingConfig::new(weighted_params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model_unweighted = match train_binary(&x_train, &y_train, None, &unweighted, &feature_names)
    {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let model_weighted = match train_binary(&x_train, &y_train, None, &weighted, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let preds_unweighted = match model_unweighted.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_weighted = match model_weighted.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(
        preds_unweighted != preds_weighted,
        "scale_pos_weight=5 produced the same predictions as unweighted"
    );
    // The weighted base score is higher: positives count five-fold in the
    // prevalence, so every raw prediction starts from larger log-odds.
    assert!(preds_weighted.iter().sum::<f64>() > preds_unweighted.iter().sum::<f64>());
    Ok(())
}

#[test]
fn test_max_features_changes_the_trained_model() -> Result<(), ClearGbmError> {
    // Knob-sensitivity: restricting every split to one of the two features
    // must alter which splits win somewhere across the trees.
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let unrestricted = match make_config(5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut restricted_params = default_params();
    restricted_params.n_estimators = 5_usize;
    restricted_params.max_features = Some(1_usize);
    let restricted = match GradientBoostingConfig::new(restricted_params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model_all = match train_binary(&x_train, &y_train, None, &unrestricted, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let model_one = match train_binary(&x_train, &y_train, None, &restricted, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };

    let preds_all = match model_all.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_one = match model_one.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(
        preds_all != preds_one,
        "max_features=1 produced the same predictions as all-features"
    );
    Ok(())
}

#[test]
fn test_max_features_deterministic_across_runs() -> Result<(), ClearGbmError> {
    // The subset derivation is a pure function of (seed, round, node), so
    // two identical runs must agree bit for bit.
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let mut params = default_params();
    params.n_estimators = 4_usize;
    params.max_features = Some(1_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let first = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let second = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
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
fn test_max_features_above_feature_count_is_rejected() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let mut params = default_params();
    params.max_features = Some(3_usize); // dataset has 2 features
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let result = train_binary(&x_train, &y_train, None, &config, &feature_names);
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected rejection of max_features > n_features".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "max_features");
            assert!(reason.contains("n_features"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_max_features_applies_under_leaf_wise_growth() -> Result<(), ClearGbmError> {
    // Both growers must consult the same per-node subset derivation; this
    // drives the leaf-wise path's mask construction.
    let (rows, y_train, feature_names) = make_nested_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    let mut restricted_params = default_params();
    restricted_params.n_estimators = 3_usize;
    restricted_params.growth_strategy = crate::training::GrowthStrategy::LeafWise;
    restricted_params.num_leaves = Some(3_usize);
    restricted_params.max_features = Some(1_usize);
    let restricted = match GradientBoostingConfig::new(restricted_params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let unrestricted = match make_leaf_wise_config(3_usize, 3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let model_restricted = match train_binary(&x_train, &y_train, None, &restricted, &feature_names)
    {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let model_unrestricted =
        match train_binary(&x_train, &y_train, None, &unrestricted, &feature_names) {
            Ok(m) => m,
            Err(e) => return Err(e),
        };

    let preds_restricted = match model_restricted.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let preds_unrestricted = match model_unrestricted.predict_raw(&x_train) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(
        preds_restricted != preds_unrestricted,
        "leaf-wise max_features=1 produced the same predictions as all-features"
    );
    Ok(())
}
