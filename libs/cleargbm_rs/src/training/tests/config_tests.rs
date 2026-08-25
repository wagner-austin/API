//! Tests for GradientBoostingConfig scalar validation and accessors.
//!
//! Cross-field pairings (growth/leaf budget, objective/weight) and the wire
//! enums live in `config_pairing_tests`.

use crate::error::ClearGbmError;
use crate::split::MonotonicConstraint;
use crate::training::GradientBoostingConfig;

use super::config_pairing_tests::default_params;

#[test]
fn test_valid_config() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.n_estimators(), 100_usize);
    assert_eq!(config.max_depth(), 3_usize);
    assert!((config.learning_rate() - 0.1_f64).abs() < 1e-15_f64);
    assert_eq!(config.min_samples_split(), 2_usize);
    assert_eq!(config.min_samples_leaf(), 1_usize);
    assert_eq!(config.max_bins(), 255_usize);
    assert!((config.subsample() - 1.0_f64).abs() < 1e-15_f64);
    assert_eq!(config.random_state(), 42_u64);
    assert!(config.monotonic_constraints().is_none());
    assert!((config.reg_alpha() - 0.0_f64).abs() < 1e-15_f64);
    assert!((config.reg_lambda() - 1.0_f64).abs() < 1e-15_f64);
    assert!(config.early_stopping_rounds().is_none());
    Ok(())
}

#[test]
fn test_config_with_monotonic_constraints() -> Result<(), ClearGbmError> {
    let mc = vec![
        MonotonicConstraint::Increasing,
        MonotonicConstraint::None,
        MonotonicConstraint::Decreasing,
    ];
    let mut params = default_params();
    params.monotonic_constraints = Some(mc);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let c = match config.monotonic_constraints() {
        Some(v) => v,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "expected Some monotonic constraints".to_string(),
            })
        }
    };
    assert_eq!(c.len(), 3_usize);
    assert_eq!(c[0_usize], MonotonicConstraint::Increasing);
    assert_eq!(c[1_usize], MonotonicConstraint::None);
    assert_eq!(c[2_usize], MonotonicConstraint::Decreasing);
    Ok(())
}

#[test]
fn test_config_with_early_stopping() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.early_stopping_rounds = Some(5_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.early_stopping_rounds(), Some(5_usize));
    Ok(())
}

#[test]
fn test_min_data_in_bin_one_is_refused() -> Result<(), ClearGbmError> {
    // Some(1) aliases the unset behavior; two spellings of one behavior
    // would make configs lie about themselves. Some(0) is equally out.
    for floor in [0_usize, 1_usize] {
        let mut params = default_params();
        params.min_data_in_bin = Some(floor);
        match GradientBoostingConfig::new(params) {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: format!("expected error for min_data_in_bin={floor}"),
                })
            }
            Err(ClearGbmError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "min_data_in_bin");
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[test]
fn test_min_data_in_bin_two_is_accepted() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.min_data_in_bin = Some(2_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.min_data_in_bin(), Some(2_usize));
    Ok(())
}

#[test]
fn test_n_estimators_zero() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.n_estimators = 0_usize;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for n_estimators=0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_estimators");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_max_depth_zero() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.max_depth = 0_usize;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for max_depth=0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "max_depth");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_learning_rate_zero() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.learning_rate = 0.0_f64;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for learning_rate=0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "learning_rate");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_learning_rate_negative() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.learning_rate = -0.1_f64;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for negative learning_rate".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "learning_rate");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_learning_rate_above_one() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.learning_rate = 1.1_f64;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for learning_rate>1".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "learning_rate");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_min_samples_split_one() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.min_samples_split = 1_usize;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for min_samples_split=1".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "min_samples_split");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_min_samples_leaf_zero() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.min_samples_leaf = 0_usize;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for min_samples_leaf=0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "min_samples_leaf");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_max_bins_one() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.max_bins = 1_usize;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for max_bins=1".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "max_bins");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_subsample_zero() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.subsample = 0.0_f64;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for subsample=0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "subsample");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_subsample_above_one() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.subsample = 1.5_f64;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for subsample>1".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "subsample");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_subsample_negative() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.subsample = -0.5_f64;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for negative subsample".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "subsample");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_reg_alpha_negative() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.reg_alpha = -0.1_f64;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for negative reg_alpha".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "reg_alpha");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_reg_lambda_negative() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.reg_lambda = -0.1_f64;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for negative reg_lambda".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "reg_lambda");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_early_stopping_rounds_zero() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.early_stopping_rounds = Some(0_usize);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for early_stopping_rounds=0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "early_stopping_rounds");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_clone_and_eq() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cloned = config.clone();
    assert_eq!(config, cloned);
    Ok(())
}

#[test]
fn test_config_debug() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("GradientBoostingConfig"));
    Ok(())
}

#[test]
fn test_learning_rate_exactly_one() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.learning_rate = 1.0_f64;
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert!((config.learning_rate() - 1.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_subsample_exactly_one() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert!((config.subsample() - 1.0_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_config_rejects_max_bins_above_u8_range() -> Result<(), ClearGbmError> {
    // Bin indices are packed into u8 for cache density, so 255 is the ceiling.
    // Rejecting here is what lets the binning layer treat the invariant as
    // established rather than re-checking per sample.
    let mut params = default_params();
    params.max_bins = 256_usize;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::InvalidParameter {
            name: "max_bins".to_string(),
            reason: "256 must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "max_bins");
            assert!(
                reason.contains("255"),
                "rejection should name the u8 ceiling, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_accepts_max_bins_at_the_u8_ceiling() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.max_bins = 255_usize;
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.max_bins(), 255_usize);
    Ok(())
}

#[test]
fn test_config_rejects_zero_max_features() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.max_features = Some(0_usize);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for zero max_features".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "max_features");
            assert!(reason.contains(">= 1"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_max_features_getter() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.max_features = Some(3_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.max_features(), Some(3_usize));
    Ok(())
}

#[test]
fn test_config_rejects_colsample_bytree_of_one() -> Result<(), ClearGbmError> {
    // Some(1.0) would be a second spelling of "all features"; the null case
    // owns that meaning, so 1.0 is rejected rather than silently equivalent.
    let mut params = default_params();
    params.colsample_bytree = Some(1.0_f64);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for colsample_bytree = 1.0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "colsample_bytree");
            assert!(reason.contains("(0.0, 1.0) exclusive"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_rejects_zero_colsample_bytree() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.colsample_bytree = Some(0.0_f64);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for colsample_bytree = 0.0".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "colsample_bytree");
            assert!(reason.contains("got 0"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_rejects_negative_colsample_bytree() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.colsample_bytree = Some(-0.5_f64);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for negative colsample_bytree".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "colsample_bytree");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_rejects_nan_colsample_bytree() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.colsample_bytree = Some(f64::NAN);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for NaN colsample_bytree".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "colsample_bytree");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_colsample_bytree_getter() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.colsample_bytree = Some(0.5_f64);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let fraction = match config.colsample_bytree() {
        Some(f) => f,
        None => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "colsample_bytree getter dropped the value".to_string(),
            })
        }
    };
    assert!((fraction - 0.5_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_config_rejects_empty_categorical_features() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.categorical_features = Some(Vec::new());
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for empty categorical_features".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "categorical_features");
            assert!(reason.contains("non-empty"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_rejects_unsorted_categorical_features() -> Result<(), ClearGbmError> {
    // Strictly ascending is the one canonical spelling of a set; a
    // duplicate or out-of-order index is rejected, not silently normalized.
    for indices in [vec![2_usize, 1_usize], vec![1_usize, 1_usize]] {
        let mut params = default_params();
        params.categorical_features = Some(indices);
        match GradientBoostingConfig::new(params) {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error for unsorted categorical_features".to_string(),
                })
            }
            Err(ClearGbmError::InvalidParameter { name, reason }) => {
                assert_eq!(name, "categorical_features");
                assert!(reason.contains("strictly ascending"));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[test]
fn test_config_categorical_features_getter() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.categorical_features = Some(vec![1_usize, 4_usize]);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.categorical_features(), Some(&[1_usize, 4_usize][..]));
    Ok(())
}
