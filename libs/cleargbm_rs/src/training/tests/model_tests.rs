//! Tests for GradientBoostingModel.

use crate::error::ClearGbmError;
use crate::training::GradientBoostingModel;

use super::train_helpers::{make_config, make_simple_dataset, train_binary};

/// Builds a small trained model for testing.
fn make_test_model() -> Result<GradientBoostingModel, ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let config = match make_config(5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    train_binary(&x_train, &y_train, None, &config, &feature_names)
}

#[test]
fn test_model_accessors() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    assert_eq!(model.n_trees(), 5_usize);
    assert_eq!(model.trees().len(), 5_usize);
    assert!((model.learning_rate() - 0.3_f64).abs() < 1e-15_f64);
    assert_eq!(model.feature_names(), &["f0".to_string(), "f1".to_string()]);
    // base_prediction is the log-odds of the training labels (50% positive → ~0.0)
    let base = model.base_prediction().unwrap_or(f64::NAN);
    assert!(base.is_finite());
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
    let bad_base = valid_model.base_prediction().unwrap_or(f64::NAN);
    let bad_model = GradientBoostingModel::new(
        valid_model.trees().to_vec(),
        crate::training::model::BaseScore::Single(bad_base),
        0.0_f64,
        valid_model.feature_names().to_vec(),
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

#[test]
fn test_from_parts_rejects_a_wrong_length_class_base_vector() -> Result<(), ClearGbmError> {
    use super::train_multiclass_tests::train_on_fixture;

    let model = match train_on_fixture(1_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    // Reassemble with a two-score base vector against n_classes = 3.
    match GradientBoostingModel::from_parts(
        model.trees().to_vec(),
        None,
        Some(vec![0.1_f64, 0.2_f64]),
        model.learning_rate(),
        model.feature_names().to_vec(),
        model.config().clone(),
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a wrong-length class base vector must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "class_base_predictions");
            assert!(reason.contains("n_classes (3)"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_multiclass_predict_rejects_broken_tree_layouts() -> Result<(), ClearGbmError> {
    use super::train_multiclass_tests::{make_multiclass_dataset, train_on_fixture};

    let model = match train_on_fixture(1_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let (rows, _) = make_multiclass_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();

    // Empty feature matrix.
    let empty: Vec<&[f64]> = Vec::new();
    match model.predict_raw_multiclass(&empty) {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "an empty matrix must be rejected".to_string(),
            })
        }
        Err(ClearGbmError::EmptyInput { .. }) => {}
        Err(e) => return Err(e),
    }

    // A tree count that is not a whole number of rounds.
    let broken = match GradientBoostingModel::from_parts(
        model
            .trees()
            .get(0_usize..2_usize)
            .map(<[_]>::to_vec)
            .unwrap_or_default(),
        None,
        model.class_base_predictions().map(<[f64]>::to_vec),
        model.learning_rate(),
        model.feature_names().to_vec(),
        model.config().clone(),
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    match broken.predict_raw_multiclass(&x) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a partial round must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "trees");
            assert!(reason.contains("whole rounds of 3 trees"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_multiclass_predict_propagates_a_tree_walk_failure() -> Result<(), ClearGbmError> {
    use super::train_multiclass_tests::{make_multiclass_dataset, train_on_fixture};
    use crate::types::{TreeNode, TreeNodeConfig};

    let model = match train_on_fixture(1_usize, None) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    // Three malformed single-node "internal" trees whose children do not
    // exist: the walk fails, and both predict_raw_multiclass and
    // predict_class must surface it.
    let bad_tree = crate::tree::Tree::new(
        vec![TreeNode::new_internal(TreeNodeConfig {
            node_id: 0_usize,
            feature_index: 0_usize,
            threshold: 0.5_f64,
            value: 0.0_f64,
            n_samples: 1_usize,
            left_child: 7_usize,
            right_child: 8_usize,
            nan_goes_left: true,
        })],
        1_usize,
        0_usize,
    );
    let broken = match GradientBoostingModel::from_parts(
        vec![bad_tree.clone(), bad_tree.clone(), bad_tree],
        None,
        model.class_base_predictions().map(<[f64]>::to_vec),
        model.learning_rate(),
        model.feature_names().to_vec(),
        model.config().clone(),
    ) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let (rows, _) = make_multiclass_dataset();
    let x: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    assert!(broken.predict_raw_multiclass(&x).is_err());
    assert!(broken.predict_class(&x).is_err());
    assert!(broken.predict_proba_multiclass(&x).is_err());
    Ok(())
}
