//! Serde tests for GradientBoostingConfig and the wire enums.
//!
//! Model serde tests live in `serde_model_tests`; shared helpers in
//! `serde_helpers`.

use crate::error::ClearGbmError;
use crate::split::MonotonicConstraint;
use crate::training::{GradientBoostingConfig, GrowthStrategy, Objective};

use super::serde_helpers::{as_json_object, from_json, reference_config, to_json};
use super::train_helpers::{default_params, default_regression_params};

/// The fields `GradientBoostingConfig` serializes, in declaration order.
const CONFIG_FIELDS: &[&str] = &[
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_samples_split",
    "min_samples_leaf",
    "max_bins",
    "subsample",
    "random_state",
    "monotonic_constraints",
    "reg_alpha",
    "reg_lambda",
    "early_stopping_rounds",
    "growth_strategy",
    "num_leaves",
    "objective",
    "scale_pos_weight",
    "max_features",
    "colsample_bytree",
    "categorical_features",
    "n_classes",
    "lambdarank_truncation_level",
    "goss_top_rate",
    "goss_other_rate",
    "quantized_gradient_bins",
];

// =============================================================================
// Serialize
// =============================================================================

#[test]
fn test_config_serialize_contains_all_field_names() -> Result<(), ClearGbmError> {
    let config = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&config));
    for field in CONFIG_FIELDS {
        assert!(json.contains(field), "missing field {field} in {json}");
    }
    Ok(())
}

#[test]
fn test_config_serialize_preserves_values() -> Result<(), ClearGbmError> {
    let config = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&config));
    assert!(json.contains(r#""n_estimators":5"#));
    assert!(json.contains(r#""max_depth":2"#));
    assert!(json.contains(r#""min_samples_split":2"#));
    assert!(json.contains(r#""random_state":42"#));
    assert!(json.contains(r#""monotonic_constraints":null"#));
    assert!(json.contains(r#""early_stopping_rounds":null"#));
    assert!(json.contains(r#""objective":"binary_log_loss""#));
    assert!(json.contains(r#""scale_pos_weight":1.0"#));
    Ok(())
}

#[test]
fn test_regression_config_serializes_a_null_weight() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_regression_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&config));
    assert!(json.contains(r#""objective":"squared_error""#));
    assert!(json.contains(r#""scale_pos_weight":null"#));
    Ok(())
}

#[test]
fn test_config_serialize_with_monotonic_constraints() -> Result<(), ClearGbmError> {
    let mut p = default_params();
    p.monotonic_constraints = Some(vec![
        MonotonicConstraint::Increasing,
        MonotonicConstraint::None,
        MonotonicConstraint::Decreasing,
    ]);
    let config = match GradientBoostingConfig::new(p) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&config));
    assert!(json.contains(r#""monotonic_constraints":["Increasing","None","Decreasing"]"#));
    Ok(())
}

// =============================================================================
// Wire enums
// =============================================================================

#[test]
fn test_growth_strategy_serializes_to_its_wire_spelling() -> Result<(), ClearGbmError> {
    assert_eq!(
        propagate!(to_json(&GrowthStrategy::DepthWise)),
        "\"depth_wise\""
    );
    assert_eq!(
        propagate!(to_json(&GrowthStrategy::LeafWise)),
        "\"leaf_wise\""
    );
    Ok(())
}

#[test]
fn test_growth_strategy_deserializes_from_its_wire_spelling() -> Result<(), ClearGbmError> {
    let depth: GrowthStrategy = propagate!(from_json("\"depth_wise\""));
    let leaf: GrowthStrategy = propagate!(from_json("\"leaf_wise\""));
    assert_eq!(depth, GrowthStrategy::DepthWise);
    assert_eq!(leaf, GrowthStrategy::LeafWise);
    Ok(())
}

#[test]
fn test_growth_strategy_deserialize_rejects_unknown_spelling() -> Result<(), ClearGbmError> {
    let err = match serde_json::from_str::<GrowthStrategy>("\"lossguide\"") {
        Ok(v) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: format!("expected rejection of unknown spelling, got {v:?}"),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("lossguide"),
        "deserialize error should quote the offending value, got: {err}"
    );
    Ok(())
}

#[test]
fn test_objective_serializes_to_its_wire_spelling() -> Result<(), ClearGbmError> {
    assert_eq!(
        propagate!(to_json(&Objective::BinaryLogLoss)),
        "\"binary_log_loss\""
    );
    assert_eq!(
        propagate!(to_json(&Objective::SquaredError)),
        "\"squared_error\""
    );
    Ok(())
}

#[test]
fn test_objective_deserializes_from_its_wire_spelling() -> Result<(), ClearGbmError> {
    let binary: Objective = propagate!(from_json("\"binary_log_loss\""));
    let squared: Objective = propagate!(from_json("\"squared_error\""));
    assert_eq!(binary, Objective::BinaryLogLoss);
    assert_eq!(squared, Objective::SquaredError);
    Ok(())
}

#[test]
fn test_objective_deserialize_rejects_unknown_spelling() -> Result<(), ClearGbmError> {
    let err = match serde_json::from_str::<Objective>("\"regression\"") {
        Ok(v) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: format!("expected rejection of unknown spelling, got {v:?}"),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("regression"),
        "deserialize error should quote the offending value, got: {err}"
    );
    Ok(())
}

// =============================================================================
// Deserialize (happy paths)
// =============================================================================

#[test]
fn test_config_deserialize_rejects_a_policy_budget_mismatch() -> Result<(), ClearGbmError> {
    // The deserializer routes through `GradientBoostingConfig::new`, so the
    // policy/budget pairing is enforced on persisted payloads too. A stored
    // model claiming leaf_wise with no budget must not load as anything.
    let config = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&config));
    let leaf_json = json.replace(
        r#""growth_strategy":"depth_wise""#,
        r#""growth_strategy":"leaf_wise""#,
    );
    assert_ne!(leaf_json, json, "the payload rewrite must have applied");
    let err = match serde_json::from_str::<GradientBoostingConfig>(&leaf_json) {
        Ok(c) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: format!("expected rejection of a budgetless leaf_wise payload, got {c:?}"),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("num_leaves"),
        "rejection should name the missing budget, got: {err}"
    );
    Ok(())
}

#[test]
fn test_config_deserialize_rejects_an_objective_weight_mismatch() -> Result<(), ClearGbmError> {
    // The same rule for the objective axis: a payload claiming squared_error
    // while carrying a class weight must not load.
    let config = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&config));
    let squared_json = json.replace(
        r#""objective":"binary_log_loss""#,
        r#""objective":"squared_error""#,
    );
    assert_ne!(squared_json, json, "the payload rewrite must have applied");
    let err = match serde_json::from_str::<GradientBoostingConfig>(&squared_json) {
        Ok(c) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: format!(
                    "expected rejection of a weighted squared_error payload, got {c:?}"
                ),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("scale_pos_weight"),
        "rejection should name the mispaired weight, got: {err}"
    );
    Ok(())
}

#[test]
fn test_config_roundtrips_a_leaf_wise_payload() -> Result<(), ClearGbmError> {
    let mut params = default_params();
    params.growth_strategy = GrowthStrategy::LeafWise;
    params.num_leaves = Some(31_usize);
    let original = match GradientBoostingConfig::new(params) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    assert!(json.contains(r#""growth_strategy":"leaf_wise""#));
    assert!(json.contains(r#""num_leaves":31"#));
    let decoded: GradientBoostingConfig = propagate!(from_json(&json));
    assert_eq!(decoded, original);
    Ok(())
}

#[test]
fn test_config_roundtrips_a_regression_payload() -> Result<(), ClearGbmError> {
    let original = match GradientBoostingConfig::new(default_regression_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    let decoded: GradientBoostingConfig = propagate!(from_json(&json));
    assert_eq!(decoded, original);
    assert_eq!(decoded.objective(), Objective::SquaredError);
    assert_eq!(decoded.scale_pos_weight(), None);
    Ok(())
}

#[test]
fn test_config_deserialize_roundtrip_default() -> Result<(), ClearGbmError> {
    let original = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    let decoded: GradientBoostingConfig = propagate!(from_json(&json));
    assert_eq!(decoded, original);
    Ok(())
}

#[test]
fn test_config_deserialize_roundtrip_with_monotonic() -> Result<(), ClearGbmError> {
    let mut p = default_params();
    p.monotonic_constraints = Some(vec![
        MonotonicConstraint::Increasing,
        MonotonicConstraint::Decreasing,
    ]);
    let original = match GradientBoostingConfig::new(p) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    let decoded: GradientBoostingConfig = propagate!(from_json(&json));
    assert_eq!(decoded, original);
    Ok(())
}

#[test]
fn test_config_deserialize_roundtrip_with_early_stopping() -> Result<(), ClearGbmError> {
    let mut p = default_params();
    p.early_stopping_rounds = Some(7_usize);
    let original = match GradientBoostingConfig::new(p) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    let decoded: GradientBoostingConfig = propagate!(from_json(&json));
    assert_eq!(decoded, original);
    Ok(())
}

// =============================================================================
// Deserialize (error paths)
// =============================================================================

#[test]
fn test_config_deserialize_missing_field_errors() -> Result<(), ClearGbmError> {
    let json = r#"{"n_estimators":10}"#;
    let result: Result<GradientBoostingConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_config_deserialize_unknown_field_errors() -> Result<(), ClearGbmError> {
    let cfg = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&cfg));
    let bad = json.replace(r#""n_estimators""#, r#""bogus_field""#);
    let result: Result<GradientBoostingConfig, _> = serde_json::from_str(&bad);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_config_deserialize_invalid_learning_rate_errors() -> Result<(), ClearGbmError> {
    let cfg = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&cfg));
    let bad = json.replace(r#""learning_rate":0.3"#, r#""learning_rate":-0.5"#);
    assert_ne!(bad, json, "the payload rewrite must have applied");
    let result: Result<GradientBoostingConfig, _> = serde_json::from_str(&bad);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_config_deserialize_wrong_type_errors() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<GradientBoostingConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// -----------------------------------------------------------------------------
// Serializer failures
// -----------------------------------------------------------------------------

#[test]
fn test_config_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    for (fail_at, field) in CONFIG_FIELDS.iter().enumerate() {
        let mut ser = FailingSerializer::fail_after(fail_at);
        assert!(
            config.serialize(&mut ser).is_err(),
            "serializer failure at field {fail_at} ({field}) was swallowed"
        );
    }
    Ok(())
}

#[test]
fn test_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_struct();
    assert!(config.serialize(&mut ser).is_err());
    Ok(())
}

#[test]
fn test_config_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_end();
    assert!(config.serialize(&mut ser).is_err());
    Ok(())
}

// -----------------------------------------------------------------------------
// Deserializer failures: one defect per field
// -----------------------------------------------------------------------------

#[test]
fn test_config_deserialize_rejects_each_missing_field() -> Result<(), ClearGbmError> {
    let config = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let full = match as_json_object(&config) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    for field in CONFIG_FIELDS {
        let mut partial = full.clone();
        assert!(
            partial.remove(*field).is_some(),
            "serialized config does not contain '{field}'"
        );
        let text = serde_json::Value::Object(partial).to_string();
        match serde_json::from_str::<GradientBoostingConfig>(&text) {
            Ok(_) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: format!("config missing '{field}' was accepted"),
                })
            }
            Err(e) => assert!(
                e.to_string().contains(field),
                "error for missing '{field}' does not name it: {e}"
            ),
        }
    }
    Ok(())
}

#[test]
fn test_config_deserialize_rejects_each_wrong_value_type() -> Result<(), ClearGbmError> {
    let config = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let full = match as_json_object(&config) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    // A bare "wrong type" string is a valid JSON value but wrong for every
    // config field: the numeric fields reject the type, and the two wire
    // enums reject the unknown spelling.
    for field in CONFIG_FIELDS {
        let mut broken = full.clone();
        let previous = broken.insert(
            (*field).to_string(),
            serde_json::Value::String("wrong type".to_string()),
        );
        assert!(
            previous.is_some(),
            "serialized config does not contain '{field}'"
        );
        let text = serde_json::Value::Object(broken).to_string();
        if serde_json::from_str::<GradientBoostingConfig>(&text).is_ok() {
            return Err(ClearGbmError::DeserializationFailed {
                reason: format!("config with a string in '{field}' was accepted"),
            });
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Field-identifier visitors
// -----------------------------------------------------------------------------

#[test]
fn test_growth_strategy_visitor_expecting_names_both_spellings() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_write_success;
    use crate::training::serde_impl::GrowthStrategyVisitor;

    // The message names both accepted spellings and is 30 bytes; 50 is ample.
    match test_expecting_write_success(&GrowthStrategyVisitor, 50_usize) {
        Ok(()) => Ok(()),
        Err(_) => Err(ClearGbmError::SerializationFailed {
            reason: "expecting() failed with a sufficient buffer".to_string(),
        }),
    }
}

#[test]
fn test_objective_visitor_expecting_names_both_spellings() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_write_success;
    use crate::training::serde_impl::ObjectiveVisitor;

    // The message names all four accepted spellings (~75 bytes); 120 is
    // ample.
    match test_expecting_write_success(&ObjectiveVisitor, 120_usize) {
        Ok(()) => Ok(()),
        Err(_) => Err(ClearGbmError::SerializationFailed {
            reason: "expecting() failed with a sufficient buffer".to_string(),
        }),
    }
}

#[test]
fn test_config_field_visitor_expecting_describes_a_field_identifier() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_write_success;
    use crate::training::serde_impl::GradientBoostingConfigFieldVisitor;

    // "field identifier" is 16 bytes, so 50 is ample.
    match test_expecting_write_success(&GradientBoostingConfigFieldVisitor, 50_usize) {
        Ok(()) => Ok(()),
        Err(_) => Err(ClearGbmError::SerializationFailed {
            reason: "expecting() failed with a sufficient buffer".to_string(),
        }),
    }
}

#[test]
fn test_config_roundtrips_a_colsample_payload() -> Result<(), ClearGbmError> {
    // The per-tree fraction must survive the wire in both spellings: a real
    // value round-trips exactly, and the reference config (colsample unset)
    // serializes an explicit null rather than omitting the key.
    let mut p = default_params();
    p.colsample_bytree = Some(0.5_f64);
    let original = match GradientBoostingConfig::new(p) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    assert!(json.contains(r#""colsample_bytree":0.5"#));
    let decoded: GradientBoostingConfig = propagate!(from_json(&json));
    assert_eq!(decoded, original);

    let unset = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let unset_json = propagate!(to_json(&unset));
    assert!(unset_json.contains(r#""colsample_bytree":null"#));
    Ok(())
}

#[test]
fn test_config_roundtrips_a_categorical_payload() -> Result<(), ClearGbmError> {
    // The categorical axis must survive the wire in both spellings: a real
    // index list round-trips exactly, and the reference config (axis unset)
    // serializes an explicit null rather than omitting the key.
    let mut p = default_params();
    p.categorical_features = Some(vec![0_usize, 3_usize]);
    let original = match GradientBoostingConfig::new(p) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    assert!(json.contains(r#""categorical_features":[0,3]"#));
    let decoded: GradientBoostingConfig = propagate!(from_json(&json));
    assert_eq!(decoded, original);

    let unset = match reference_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let unset_json = propagate!(to_json(&unset));
    assert!(unset_json.contains(r#""categorical_features":null"#));
    Ok(())
}
