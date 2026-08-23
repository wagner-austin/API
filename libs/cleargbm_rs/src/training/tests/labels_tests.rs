//! Direct tests for objective resolution.
//!
//! `resolve_objective` is total over its inputs even though
//! `GradientBoostingConfig::new` already enforces the objective/weight
//! pairing — these tests drive the arms a validated config can never reach,
//! plus label-content validation through the resolver.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::training::labels::{resolve_objective, ResolvedObjective, ResolvedTraining};
use crate::training::{
    train_gradient_boosting, GrowthStrategy, Objective, TrainingLabels, ValidationData,
};
use crate::training::{Parallelism, TrainingRuntime};

use super::train_helpers::{make_config, make_simple_dataset, train_binary};

/// Unwraps the single-score half of a resolution, failing on multiclass.
fn expect_single_score(
    resolved: ResolvedTraining<'_>,
) -> Result<ResolvedObjective<'_>, ClearGbmError> {
    match resolved {
        ResolvedTraining::SingleScore(r) => Ok(r),
        ResolvedTraining::Multiclass(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected a single-score resolution".to_string(),
        }),
    }
}

#[test]
fn test_resolve_binary_ok() -> Result<(), ClearGbmError> {
    let y = [0_u8, 1_u8];
    let resolved = propagate!(expect_single_score(propagate!(resolve_objective(
        Objective::BinaryLogLoss,
        Some(2.0_f64),
        None,
        TrainingLabels::Binary(&y),
        None,
        None,
    ))));
    match resolved {
        ResolvedObjective::Binary {
            y_train,
            weights,
            val,
            scale_pos_weight,
        } => {
            assert!(weights.is_none());
            assert_eq!(y_train, &y);
            assert!(val.is_none());
            assert!((scale_pos_weight - 2.0_f64).abs() < 1e-15_f64);
            Ok(())
        }
        ResolvedObjective::SquaredError { .. } => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected the Binary variant".to_string(),
        }),
    }
}

#[test]
fn test_resolve_squared_error_ok_with_val() -> Result<(), ClearGbmError> {
    let y = [1.5_f64, -0.5_f64];
    let x_val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64]];
    let x_val: Vec<&[f64]> = x_val_rows.iter().map(Vec::as_slice).collect();
    let y_val = [0.25_f64];
    let resolved = propagate!(expect_single_score(propagate!(resolve_objective(
        Objective::SquaredError,
        None,
        None,
        TrainingLabels::Continuous(&y),
        None,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Continuous(&y_val),
            weight: None,
        }),
    ))));
    assert!(resolved_has_val_features(&resolved));
    match &resolved {
        ResolvedObjective::SquaredError {
            y_train,
            weights,
            val,
        } => {
            assert_eq!(*y_train, &y);
            assert!(weights.is_none());
            let v = match val {
                Some(v) => v,
                None => {
                    return Err(ClearGbmError::TreeConstructionFailed {
                        reason: "expected validation data".to_string(),
                    })
                }
            };
            assert_eq!(v.x.len(), 1_usize);
            assert_eq!(v.y, &y_val);
            assert!(v.weight.is_none());
            Ok(())
        }
        ResolvedObjective::Binary { .. } => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected the SquaredError variant".to_string(),
        }),
    }
}

/// Helper so the borrow of `resolved` above stays simple.
fn resolved_has_val_features(resolved: &ResolvedObjective<'_>) -> bool {
    resolved.val_features().is_some()
}

#[test]
fn test_resolve_binary_missing_weight_is_error() -> Result<(), ClearGbmError> {
    // The config constructor forbids this pairing, but the resolver is total
    // rather than trusting its caller.
    let y = [0_u8, 1_u8];
    let result = resolve_objective(
        Objective::BinaryLogLoss,
        None,
        None,
        TrainingLabels::Binary(&y),
        None,
        None,
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "binary without weight must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "scale_pos_weight");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_resolve_squared_error_with_weight_is_error() -> Result<(), ClearGbmError> {
    let y = [1.5_f64, -0.5_f64];
    let result = resolve_objective(
        Objective::SquaredError,
        Some(3.0_f64),
        None,
        TrainingLabels::Continuous(&y),
        None,
        None,
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "squared_error with a weight must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "scale_pos_weight");
            assert!(
                reason.contains('3'),
                "should quote the weight, got: {reason}"
            );
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_resolve_validates_binary_train_label_content() -> Result<(), ClearGbmError> {
    let y = [0_u8, 2_u8];
    let result = resolve_objective(
        Objective::BinaryLogLoss,
        Some(1.0_f64),
        None,
        TrainingLabels::Binary(&y),
        None,
        None,
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a label of 2 must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidLabel { value, index }) => {
            assert_eq!(value, 2_u8);
            assert_eq!(index, 1_usize);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_resolve_validates_binary_val_label_content() -> Result<(), ClearGbmError> {
    let y = [0_u8, 1_u8];
    let x_val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64]];
    let x_val: Vec<&[f64]> = x_val_rows.iter().map(Vec::as_slice).collect();
    let y_val = [7_u8];
    let result = resolve_objective(
        Objective::BinaryLogLoss,
        Some(1.0_f64),
        None,
        TrainingLabels::Binary(&y),
        None,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Binary(&y_val),
            weight: None,
        }),
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a validation label of 7 must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidLabel { value, .. }) => {
            assert_eq!(value, 7_u8);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_resolve_validates_continuous_val_label_content() -> Result<(), ClearGbmError> {
    let y = [1.0_f64];
    let x_val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64]];
    let x_val: Vec<&[f64]> = x_val_rows.iter().map(Vec::as_slice).collect();
    let y_val = [f64::INFINITY];
    let result = resolve_objective(
        Objective::SquaredError,
        None,
        None,
        TrainingLabels::Continuous(&y),
        None,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Continuous(&y_val),
            weight: None,
        }),
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an infinite validation target must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "y_val");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_validation_data_debug_and_copy() {
    let x_rows: Vec<Vec<f64>> = vec![vec![0.5_f64]];
    let x: Vec<&[f64]> = x_rows.iter().map(Vec::as_slice).collect();
    let y = [1_u8];
    let val = ValidationData {
        x: &x,
        y: TrainingLabels::Binary(&y),
        weight: None,
    };
    let copy = val;
    let debug = format!("{copy:?}");
    assert!(debug.contains("ValidationData"));
}

// =============================================================================
// Pairing enforced through the training entry
// =============================================================================

#[test]
fn test_binary_objective_rejects_continuous_labels() -> Result<(), ClearGbmError> {
    // The typed entry makes the mismatch an error instead of a silent
    // reinterpretation.
    let (rows, _, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_continuous: Vec<f64> = vec![0.0_f64; rows.len()];

    let config = match make_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let result = train_gradient_boosting(
        &x_train,
        TrainingLabels::Continuous(&y_continuous),
        None,
        None,
        &config,
        &feature_names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "binary objective must reject continuous labels".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "y_train");
            assert!(
                reason.contains("binary_log_loss"),
                "error should name the objective, got: {reason}"
            );
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_binary_objective_rejects_continuous_val_labels() -> Result<(), ClearGbmError> {
    let (rows, y_train, feature_names) = make_simple_dataset();
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64, 0.2_f64]];
    let x_val: Vec<&[f64]> = val_rows.iter().map(Vec::as_slice).collect();
    let y_val: Vec<f64> = vec![0.5_f64];

    let config = match make_config(2_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let result = train_binary(
        &x_train,
        &y_train,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Continuous(&y_val),
            weight: None,
        }),
        &config,
        &feature_names,
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "binary objective must reject continuous validation labels".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "y_val");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_labels_len_and_is_empty() {
    let binary = TrainingLabels::Binary(&[0_u8, 1_u8]);
    assert_eq!(binary.len(), 2_usize);
    assert!(!binary.is_empty());
    let continuous: TrainingLabels<'_> = TrainingLabels::Continuous(&[]);
    assert_eq!(continuous.len(), 0_usize);
    assert!(continuous.is_empty());
    let debug = format!("{binary:?} {continuous:?}");
    assert!(debug.contains("Binary"));
    assert!(debug.contains("Continuous"));
}

#[test]
fn test_growth_strategy_debug_format() {
    // Keeps the derive covered without a dedicated serde path.
    let debug = format!("{:?}", GrowthStrategy::LeafWise);
    assert!(debug.contains("LeafWise"));
}

#[test]
fn test_resolve_multiclass_pairing_arms() -> Result<(), ClearGbmError> {
    // The multiclass objective rejects wrong-kind training labels...
    let y_bin = [0_u8, 1_u8];
    let result = resolve_objective(
        Objective::MulticlassSoftmax,
        None,
        Some(3_usize),
        TrainingLabels::Binary(&y_bin),
        None,
        None,
    );
    match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "binary labels under multiclass must be rejected".to_string(),
            })
        }
        Err(ClearGbmError::InvalidParameter { reason, .. }) => {
            assert!(reason.contains("multiclass (u32) labels"), "{reason}");
            assert!(reason.contains("binary (u8) labels"), "{reason}");
        }
        Err(e) => return Err(e),
    }

    // ...and wrong-kind validation labels.
    let y_mc = [0_u32, 1_u32, 2_u32];
    let x_val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64]];
    let x_val: Vec<&[f64]> = x_val_rows.iter().map(Vec::as_slice).collect();
    let y_val = [0.5_f64];
    let result = resolve_objective(
        Objective::MulticlassSoftmax,
        None,
        Some(3_usize),
        TrainingLabels::Multiclass(&y_mc),
        None,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Continuous(&y_val),
            weight: None,
        }),
    );
    match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "continuous val labels under multiclass must be rejected".to_string(),
            })
        }
        Err(ClearGbmError::InvalidParameter { name, .. }) => assert_eq!(name, "y_val"),
        Err(e) => return Err(e),
    }

    // A class weight is refused even when the config layer was bypassed...
    let result = resolve_objective(
        Objective::MulticlassSoftmax,
        Some(2.0_f64),
        Some(3_usize),
        TrainingLabels::Multiclass(&y_mc),
        None,
        None,
    );
    match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "scale_pos_weight under multiclass must be rejected".to_string(),
            })
        }
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "scale_pos_weight");
        }
        Err(e) => return Err(e),
    }

    // ...and so is a missing class count.
    let result = resolve_objective(
        Objective::MulticlassSoftmax,
        None,
        None,
        TrainingLabels::Multiclass(&y_mc),
        None,
        None,
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "multiclass without n_classes must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_classes");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_resolve_multiclass_ok_and_single_score_rejects_multiclass_labels(
) -> Result<(), ClearGbmError> {
    // A clean multiclass resolution carries everything through.
    let y_mc = [0_u32, 2_u32, 1_u32];
    let resolved = propagate!(resolve_objective(
        Objective::MulticlassSoftmax,
        None,
        Some(3_usize),
        TrainingLabels::Multiclass(&y_mc),
        None,
        None,
    ));
    match resolved {
        ResolvedTraining::Multiclass(mc) => {
            assert_eq!(mc.n_classes, 3_usize);
            assert_eq!(mc.y_train, &y_mc);
            assert!(mc.weights.is_none());
            assert!(mc.val.is_none());
        }
        ResolvedTraining::SingleScore(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected the multiclass resolution".to_string(),
            })
        }
    }

    // Multiclass labels under the single-score objectives are rejected in
    // both training and validation position.
    let result = resolve_objective(
        Objective::BinaryLogLoss,
        Some(1.0_f64),
        None,
        TrainingLabels::Multiclass(&y_mc),
        None,
        None,
    );
    match result {
        Ok(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "multiclass labels under binary must be rejected".to_string(),
            })
        }
        Err(ClearGbmError::InvalidParameter { reason, .. }) => {
            assert!(reason.contains("multiclass (u32) labels"), "{reason}");
        }
        Err(e) => return Err(e),
    }
    let y_cont = [0.5_f64, 1.5_f64];
    let x_val_rows: Vec<Vec<f64>> = vec![vec![0.1_f64]];
    let x_val: Vec<&[f64]> = x_val_rows.iter().map(Vec::as_slice).collect();
    let y_val_mc = [0_u32];
    let result = resolve_objective(
        Objective::SquaredError,
        None,
        None,
        TrainingLabels::Continuous(&y_cont),
        None,
        Some(ValidationData {
            x: &x_val,
            y: TrainingLabels::Multiclass(&y_val_mc),
            weight: None,
        }),
    );
    match result {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "multiclass val labels under squared error must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "y_val");
            Ok(())
        }
        Err(e) => Err(e),
    }
}
