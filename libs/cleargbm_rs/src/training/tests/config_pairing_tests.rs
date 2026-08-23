//! Tests for GradientBoostingConfig cross-field pairings and wire enums.
//!
//! Two pairings are enforced at construction: `growth_strategy` with
//! `num_leaves`, and `objective` with `scale_pos_weight`. Both exist for the
//! same reason — a config must never state a knob training does not honour.

use crate::error::ClearGbmError;
use crate::training::{
    GradientBoostingConfig, GradientBoostingConfigParams, GrowthStrategy, Objective,
};

/// Creates default valid binary-classification params for reuse in tests.
pub(super) fn default_params() -> GradientBoostingConfigParams {
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
        growth_strategy: GrowthStrategy::DepthWise,
        num_leaves: None,
        objective: Objective::BinaryLogLoss,
        scale_pos_weight: Some(1.0_f64),
        max_features: None,
        colsample_bytree: None,
        categorical_features: None,
        n_classes: None,
        lambdarank_truncation_level: None,
        goss_top_rate: None,
        goss_other_rate: None,
    }
}

/// Creates default valid squared-error regression params for reuse in tests.
pub(super) fn default_regression_params() -> GradientBoostingConfigParams {
    let mut params = default_params();
    params.objective = Objective::SquaredError;
    params.scale_pos_weight = None;
    params
}

// =============================================================================
// Growth strategy
// =============================================================================

#[test]
fn test_growth_strategy_wire_spellings_round_trip() -> Result<(), ClearGbmError> {
    assert_eq!(GrowthStrategy::DepthWise.as_str(), "depth_wise");
    assert_eq!(GrowthStrategy::LeafWise.as_str(), "leaf_wise");
    assert_eq!(
        propagate!(GrowthStrategy::from_wire("depth_wise")),
        GrowthStrategy::DepthWise
    );
    assert_eq!(
        propagate!(GrowthStrategy::from_wire("leaf_wise")),
        GrowthStrategy::LeafWise
    );
    Ok(())
}

#[test]
fn test_growth_strategy_rejects_unknown_spelling() -> Result<(), ClearGbmError> {
    // `lossguide` is XGBoost's name for the same policy. Naming the offending
    // value back to the caller is what turns a silent default into a fixable
    // typo.
    match GrowthStrategy::from_wire("lossguide") {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for unknown growth_strategy".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "growth_strategy");
            assert!(
                reason.contains("lossguide"),
                "rejection should quote the offending value, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_defaults_to_depth_wise_growth() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.growth_strategy(), GrowthStrategy::DepthWise);
    Ok(())
}

#[test]
fn test_config_accepts_leaf_wise_with_a_budget() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.growth_strategy = GrowthStrategy::LeafWise;
    params.num_leaves = Some(31_usize);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.growth_strategy(), GrowthStrategy::LeafWise);
    assert_eq!(config.num_leaves(), Some(31_usize));
    Ok(())
}

#[test]
fn test_config_rejects_leaf_wise_without_a_budget() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.growth_strategy = GrowthStrategy::LeafWise;
    params.num_leaves = None;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "leaf_wise without num_leaves must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "num_leaves");
            assert!(
                reason.contains("must be set"),
                "rejection should say the budget is required, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_rejects_a_leaf_budget_below_two() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.growth_strategy = GrowthStrategy::LeafWise;
    params.num_leaves = Some(1_usize);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a budget of 1 must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "num_leaves");
            assert!(
                reason.contains(">= 2"),
                "rejection should name the minimum, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_rejects_a_budget_under_depth_wise() -> Result<(), ClearGbmError> {
    // Accepting and ignoring it is the defect this rejects: the run would
    // report a leaf budget it never honoured.
    let mut params = default_params();
    params.growth_strategy = GrowthStrategy::DepthWise;
    params.num_leaves = Some(31_usize);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "num_leaves under depth_wise must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "num_leaves");
            assert!(
                reason.contains("would ignore it"),
                "rejection should say why the pairing is wrong, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_depth_wise_carries_no_budget() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.num_leaves(), None);
    Ok(())
}

// =============================================================================
// Objective wire spellings
// =============================================================================

#[test]
fn test_objective_wire_spellings_round_trip() -> Result<(), ClearGbmError> {
    assert_eq!(Objective::BinaryLogLoss.as_str(), "binary_log_loss");
    assert_eq!(Objective::SquaredError.as_str(), "squared_error");
    assert_eq!(
        propagate!(Objective::from_wire("binary_log_loss")),
        Objective::BinaryLogLoss
    );
    assert_eq!(
        propagate!(Objective::from_wire("squared_error")),
        Objective::SquaredError
    );
    assert_eq!(Objective::LambdaRank.as_str(), "lambdarank");
    assert_eq!(
        propagate!(Objective::from_wire("lambdarank")),
        Objective::LambdaRank
    );
    Ok(())
}

#[test]
fn test_objective_rejects_unknown_spelling() -> Result<(), ClearGbmError> {
    // `reg:squarederror` is XGBoost's name for the same loss. Quoting the
    // offending value back is what turns a silent default into a fixable typo.
    match Objective::from_wire("reg:squarederror") {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for unknown objective".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "objective");
            assert!(
                reason.contains("reg:squarederror"),
                "rejection should quote the offending value, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

// =============================================================================
// Objective / scale_pos_weight pairing
// =============================================================================

#[test]
fn test_config_binary_objective_getter() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.objective(), Objective::BinaryLogLoss);
    assert_eq!(config.scale_pos_weight(), Some(1.0_f64));
    Ok(())
}

#[test]
fn test_config_accepts_squared_error_without_weight() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_regression_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.objective(), Objective::SquaredError);
    assert_eq!(config.scale_pos_weight(), None);
    Ok(())
}

#[test]
fn test_config_rejects_binary_without_weight() -> Result<(), ClearGbmError> {
    // Unweighted binary training states 1.0 explicitly; an absent weight is
    // ambiguous, not a default.
    let mut params = default_params();
    params.scale_pos_weight = None;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "binary_log_loss without scale_pos_weight must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "scale_pos_weight");
            assert!(
                reason.contains("must be set"),
                "rejection should say the weight is required, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_rejects_weight_under_squared_error() -> Result<(), ClearGbmError> {
    // Accepting and ignoring it is the defect this rejects: the run would
    // report a class weight it never honoured.
    let mut params = default_regression_params();
    params.scale_pos_weight = Some(3.0_f64);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "scale_pos_weight under squared_error must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "scale_pos_weight");
            assert!(
                reason.contains("no positive class"),
                "rejection should say why the pairing is wrong, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_rejects_zero_scale_pos_weight() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.scale_pos_weight = Some(0.0_f64);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for zero scale_pos_weight".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "scale_pos_weight");
            assert!(reason.contains("finite positive"));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_rejects_nan_scale_pos_weight() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.scale_pos_weight = Some(f64::NAN);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected error for NaN scale_pos_weight".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "scale_pos_weight");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_config_scale_pos_weight_getter() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.scale_pos_weight = Some(2.5_f64);
    let config = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    assert_eq!(config.scale_pos_weight(), Some(2.5_f64));
    Ok(())
}
// =============================================================================
// The lambdarank_truncation_level pairing
// =============================================================================

/// Default valid lambdarank params for the pairing tests.
fn ranking_params() -> GradientBoostingConfigParams {
    let mut params = default_params();
    params.objective = Objective::LambdaRank;
    params.scale_pos_weight = None;
    params.lambdarank_truncation_level = Some(10_usize);
    params
}

#[test]
fn test_config_lambdarank_pairing_accepts_and_reads_back() -> Result<(), ClearGbmError> {
    let config = propagate!(GradientBoostingConfig::new(ranking_params()));
    assert_eq!(config.objective(), Objective::LambdaRank);
    assert_eq!(config.lambdarank_truncation_level(), Some(10_usize));
    Ok(())
}

#[test]
fn test_config_rejects_lambdarank_without_a_truncation_level() -> Result<(), ClearGbmError> {
    let mut params = ranking_params();
    params.lambdarank_truncation_level = None;
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected the missing-truncation refusal".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "lambdarank_truncation_level");
            assert!(reason.contains("must be set"), "{reason}");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_rejects_a_zero_truncation_level() -> Result<(), ClearGbmError> {
    let mut params = ranking_params();
    params.lambdarank_truncation_level = Some(0_usize);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected the zero-truncation refusal".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "lambdarank_truncation_level");
            assert!(reason.contains(">= 1"), "{reason}");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_rejects_a_truncation_level_under_other_objectives() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.lambdarank_truncation_level = Some(10_usize);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected the decorative-truncation refusal".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "lambdarank_truncation_level");
            assert!(reason.contains("binary_log_loss"), "{reason}");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_rejects_a_class_weight_under_lambdarank() -> Result<(), ClearGbmError> {
    let mut params = ranking_params();
    params.scale_pos_weight = Some(2.0_f64);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected the weight refusal".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "scale_pos_weight");
            assert!(reason.contains("sample_weight"), "{reason}");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_config_rejects_a_class_count_under_lambdarank() -> Result<(), ClearGbmError> {
    let mut params = ranking_params();
    params.n_classes = Some(3_usize);
    match GradientBoostingConfig::new(params) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected the class-count refusal".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "n_classes");
            assert!(reason.contains("lambdarank"), "{reason}");
            Ok(())
        }
        Err(other) => Err(other),
    }
}
