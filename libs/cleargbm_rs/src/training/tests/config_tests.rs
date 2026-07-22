//! Tests for GradientBoostingConfig validation and accessors.

use crate::error::ClearGbmError;
use crate::split::MonotonicConstraint;
use crate::training::{GradientBoostingConfig, GradientBoostingConfigParams};

/// Creates default valid params for reuse in tests.
fn default_params() -> GradientBoostingConfigParams {
    GradientBoostingConfigParams {
        n_estimators: 100_usize,
        max_depth: 3_usize,
        learning_rate: 0.1_f64,
        min_samples_split: 2_usize,
        min_samples_leaf: 1_usize,
        max_bins: 255_usize,
        subsample: 1.0_f64,
        random_state: 42_u64,
        monotonic_constraints: None,
        reg_alpha: 0.0_f64,
        reg_lambda: 1.0_f64,
        early_stopping_rounds: None,
    }
}

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
