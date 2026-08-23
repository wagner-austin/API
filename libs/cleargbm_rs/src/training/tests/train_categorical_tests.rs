//! End-to-end tests for native categorical splits.
//!
//! The fixture is a non-ordinal category pattern — codes `[0, 1, 2, 3]`
//! labelled `[1, 0, 1, 0]` — which NO threshold over the code order can
//! separate in one split, but the many-vs-many subset `{0, 2}` vs `{1, 3}`
//! separates perfectly. A depth-1 stump therefore discriminates the
//! categorical treatment from the numeric one, and the alternating layout
//! keeps deeper numeric trees visibly behind as well.

use crate::error::ClearGbmError;
use crate::training::{GradientBoostingConfig, GradientBoostingModel, GrowthStrategy};

use super::serde_helpers::{from_json, to_json};
use super::train_helpers::{default_params, train_binary};

/// Twelve rows on two features: feature 0 holds three copies of each code
/// 0..4, feature 1 is constant. Codes 0 and 2 are the positive class.
fn make_categorical_dataset() -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<u8> = Vec::new();
    for (code, label) in [
        (0.0_f64, 1_u8),
        (1.0_f64, 0_u8),
        (2.0_f64, 1_u8),
        (3.0_f64, 0_u8),
    ] {
        for _ in 0_usize..3_usize {
            rows.push(vec![code, 0.0_f64]);
            y_train.push(label);
        }
    }
    (rows, y_train)
}

/// Trains on the categorical fixture.
fn train_on_fixture(
    categorical: bool,
    n_estimators: usize,
    max_depth: usize,
) -> Result<GradientBoostingModel, ClearGbmError> {
    let (rows, y_train) = make_categorical_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["category".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.n_estimators = n_estimators;
    params.max_depth = max_depth;
    if categorical {
        params.categorical_features = Some(vec![0_usize]);
    }
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    train_binary(&x_train, &y_train, None, &config, &feature_names)
}

/// Positive-class probabilities for the four category codes.
fn code_probas(model: &GradientBoostingModel) -> Result<Vec<f64>, ClearGbmError> {
    let queries: Vec<Vec<f64>> = (0_usize..4_usize)
        .map(|i| {
            let mut code = 0.0_f64;
            for _ in 0_usize..i {
                code += 1.0_f64;
            }
            vec![code, 0.0_f64]
        })
        .collect();
    let refs: Vec<&[f64]> = queries.iter().map(Vec::as_slice).collect();
    let probas = match model.predict_proba(&refs) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    Ok(probas.iter().map(|&(_, p1)| p1).collect())
}

#[test]
fn test_categorical_stump_separates_the_non_ordinal_pattern() -> Result<(), ClearGbmError> {
    // One split, perfect partition: {0,2} classify positive, {1,3} negative.
    let model = match train_on_fixture(true, 1_usize, 1_usize) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let probas = match code_probas(&model) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(probas[0] > 0.5_f64, "code 0 should be positive: {probas:?}");
    assert!(probas[1] < 0.5_f64, "code 1 should be negative: {probas:?}");
    assert!(probas[2] > 0.5_f64, "code 2 should be positive: {probas:?}");
    assert!(probas[3] < 0.5_f64, "code 3 should be negative: {probas:?}");

    // Structural check: the stump's root is a set-split carrying exactly
    // the positive codes, and no threshold.
    let root = propagate!(propagate!(model.trees().first().ok_or_else(|| {
        ClearGbmError::TreeConstructionFailed {
            reason: "model has no trees".to_string(),
        }
    }))
    .root());
    let categories = match root.categories_goes_left() {
        Some(c) => c,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "root is not a categorical split".to_string(),
            })
        }
    };
    let left_is_positive = categories == [0.0_f64, 2.0_f64];
    let left_is_negative = categories == [1.0_f64, 3.0_f64];
    assert!(
        left_is_positive || left_is_negative,
        "expected the {{0,2}} vs {{1,3}} partition, got {categories:?}"
    );
    assert!(root.threshold().is_none());
    Ok(())
}

#[test]
fn test_numeric_stump_cannot_express_the_partition() -> Result<(), ClearGbmError> {
    // Same data, codes treated numerically: one threshold cannot separate
    // the alternating labels, so at least one code lands on the wrong side
    // of 0.5 — and the two treatments produce different predictions.
    let numeric = match train_on_fixture(false, 1_usize, 1_usize) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let categorical = match train_on_fixture(true, 1_usize, 1_usize) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let numeric_probas = match code_probas(&numeric) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let categorical_probas = match code_probas(&categorical) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_ne!(numeric_probas, categorical_probas);

    let misclassified = [
        numeric_probas[0] <= 0.5_f64,
        numeric_probas[1] >= 0.5_f64,
        numeric_probas[2] <= 0.5_f64,
        numeric_probas[3] >= 0.5_f64,
    ]
    .iter()
    .filter(|&&wrong| wrong)
    .count();
    assert!(
        misclassified >= 1_usize,
        "a numeric stump separated an inseparable pattern: {numeric_probas:?}"
    );
    Ok(())
}

#[test]
fn test_unseen_category_routes_with_the_non_members() -> Result<(), ClearGbmError> {
    // Code 9 was never seen in training: it is not a member of any left
    // set, so it must land in exactly the leaf the non-member training
    // codes land in.
    let model = match train_on_fixture(true, 1_usize, 1_usize) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let unseen_row = [9.0_f64, 0.0_f64];
    let member_row = [1.0_f64, 0.0_f64];
    let queries: Vec<&[f64]> = vec![&unseen_row, &member_row];
    let probas = match model.predict_proba(&queries) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    // The stump's left set is one of the two label groups; the unseen code
    // shares a leaf with whichever group is NOT the left set. With the
    // learned partition, code 1 (negative group) and code 9 land together
    // when the positive codes went left — and the probabilities are equal
    // bit for bit either way only if they share a leaf.
    let (_, unseen_p) = probas[0];
    let (_, code_one_p) = probas[1];
    let same_leaf = (unseen_p - code_one_p).abs() < 1e-15_f64;
    assert!(
        same_leaf || unseen_p < 0.5_f64,
        "unseen code must route with the non-members: {unseen_p} vs {code_one_p}"
    );
    Ok(())
}

#[test]
fn test_categorical_model_roundtrips_through_json() -> Result<(), ClearGbmError> {
    let model = match train_on_fixture(true, 3_usize, 2_usize) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&model));
    assert!(json.contains(r#""categories_goes_left":["#));
    let decoded: GradientBoostingModel = propagate!(from_json(&json));

    let (rows, _) = make_categorical_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let before = propagate!(model.predict_raw(&x));
    let after = propagate!(decoded.predict_raw(&x));
    assert_eq!(before, after);
    Ok(())
}

#[test]
fn test_categorical_training_is_deterministic() -> Result<(), ClearGbmError> {
    let first = match train_on_fixture(true, 4_usize, 2_usize) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let second = match train_on_fixture(true, 4_usize, 2_usize) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let (rows, _) = make_categorical_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let preds_first = propagate!(first.predict_raw(&x));
    let preds_second = propagate!(second.predict_raw(&x));
    assert_eq!(preds_first, preds_second);
    Ok(())
}

#[test]
fn test_categorical_applies_under_leaf_wise_growth() -> Result<(), ClearGbmError> {
    let (rows, y_train) = make_categorical_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["category".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.n_estimators = 2_usize;
    params.growth_strategy = GrowthStrategy::LeafWise;
    params.num_leaves = Some(3_usize);
    params.categorical_features = Some(vec![0_usize]);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let probas = match code_probas(&model) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(probas[0] > 0.5_f64 && probas[2] > 0.5_f64, "{probas:?}");
    assert!(probas[1] < 0.5_f64 && probas[3] < 0.5_f64, "{probas:?}");
    Ok(())
}

#[test]
fn test_rejects_a_categorical_index_out_of_range() -> Result<(), ClearGbmError> {
    let (rows, y_train) = make_categorical_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["category".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.categorical_features = Some(vec![5_usize]);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an out-of-range categorical index must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "categorical_features");
            assert!(reason.contains("out of range"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_rejects_a_monotonic_constraint_on_a_categorical_feature() -> Result<(), ClearGbmError> {
    use crate::split::MonotonicConstraint;

    let (rows, y_train) = make_categorical_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["category".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.categorical_features = Some(vec![0_usize]);
    params.monotonic_constraints = Some(vec![
        MonotonicConstraint::Increasing,
        MonotonicConstraint::None,
    ]);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a constrained categorical feature must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "categorical_features");
            assert!(reason.contains("monotonic"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_rejects_non_integer_values_in_a_categorical_column() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64, 0.0_f64],
        vec![1.5_f64, 0.0_f64],
        vec![2.0_f64, 0.0_f64],
        vec![3.0_f64, 0.0_f64],
    ];
    let y_train: Vec<u8> = vec![0, 1, 0, 1];
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["category".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.categorical_features = Some(vec![0_usize]);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a non-integer categorical value must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "categorical_features");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_numeric_features_keep_their_constraints_beside_a_categorical_one(
) -> Result<(), ClearGbmError> {
    // Constraints and the categorical axis coexist as long as no
    // categorical feature is itself constrained: the numeric feature's
    // constraint stands, the categorical feature trains unconstrained.
    use crate::split::MonotonicConstraint;

    let (rows, y_train) = make_categorical_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let feature_names = vec!["category".to_string(), "constant".to_string()];
    let mut params = default_params();
    params.n_estimators = 1_usize;
    params.categorical_features = Some(vec![0_usize]);
    params.monotonic_constraints = Some(vec![
        MonotonicConstraint::None,
        MonotonicConstraint::Increasing,
    ]);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let model = match train_binary(&x_train, &y_train, None, &config, &feature_names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let probas = match code_probas(&model) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert!(probas[0] > 0.5_f64 && probas[1] < 0.5_f64, "{probas:?}");
    Ok(())
}
