//! Serde tests for GradientBoostingConfig + GradientBoostingModel.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::split::MonotonicConstraint;
use crate::training::{
    train_gradient_boosting, GradientBoostingConfig, GradientBoostingConfigParams,
    GradientBoostingModel, GrowthStrategy,
};
use crate::training::{Parallelism, TrainingRuntime};

// =============================================================================
// Helpers
// =============================================================================

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
        growth_strategy: GrowthStrategy::DepthWise,
        num_leaves: None,
    }
}

fn make_test_model() -> Result<GradientBoostingModel, ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![
        vec![0.0_f64, 0.1_f64],
        vec![0.1_f64, 0.0_f64],
        vec![0.2_f64, 0.2_f64],
        vec![0.3_f64, 0.1_f64],
        vec![0.8_f64, 0.9_f64],
        vec![0.9_f64, 0.8_f64],
        vec![1.0_f64, 1.0_f64],
        vec![0.7_f64, 0.9_f64],
    ];
    let x_train: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let y_train: Vec<u8> = vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8, 1_u8, 1_u8, 1_u8];
    let feature_names: Vec<String> = vec!["f0".to_string(), "f1".to_string()];

    let config = match GradientBoostingConfig::new(GradientBoostingConfigParams {
        n_estimators: 5_usize,
        max_depth: 2_usize,
        learning_rate: 0.3_f64,
        min_samples_split: 2_usize,
        min_samples_leaf: 1_usize,
        max_bins: 4_usize,
        subsample: 1.0_f64,
        random_state: 42_u64,
        monotonic_constraints: None,
        reg_alpha: 0.0_f64,
        reg_lambda: 1.0_f64,
        early_stopping_rounds: None,
        growth_strategy: GrowthStrategy::DepthWise,
        num_leaves: None,
    }) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    train_gradient_boosting(
        &x_train,
        &y_train,
        None,
        None,
        &config,
        &feature_names,
        &TrainingRuntime {
            parallelism: Parallelism::Single,
            hooks: &Hooks::default(),
        },
    )
}

/// Test helper: serialize via `serde_json`, mapping errors into
/// `ClearGbmError::SerializationFailed` so a caller can `propagate!` on the
/// result. Tests that need to observe a serialize error use `serde_json`
/// directly and match on `Result`.
fn to_json<T: serde::Serialize>(value: &T) -> Result<String, ClearGbmError> {
    match serde_json::to_string(value) {
        Ok(s) => Ok(s),
        Err(e) => Err(ClearGbmError::SerializationFailed {
            reason: e.to_string(),
        }),
    }
}

/// Test helper: deserialize via `serde_json`, mapping errors into
/// `ClearGbmError::DeserializationFailed`. Tests that need to observe a
/// deserialize error use `serde_json` directly and match on `Result`.
fn from_json<'a, T: serde::Deserialize<'a>>(json: &'a str) -> Result<T, ClearGbmError> {
    match serde_json::from_str::<T>(json) {
        Ok(v) => Ok(v),
        Err(e) => Err(ClearGbmError::DeserializationFailed {
            reason: e.to_string(),
        }),
    }
}

// =============================================================================
// GradientBoostingConfig — serialize
// =============================================================================

#[test]
fn test_config_serialize_contains_all_field_names() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&config));
    for field in [
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
    ] {
        assert!(json.contains(field), "missing field {field} in {json}");
    }
    Ok(())
}

#[test]
fn test_config_serialize_preserves_values() -> Result<(), ClearGbmError> {
    let config = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&config));
    assert!(json.contains(r#""n_estimators":100"#));
    assert!(json.contains(r#""max_depth":3"#));
    assert!(json.contains(r#""min_samples_split":2"#));
    assert!(json.contains(r#""random_state":42"#));
    assert!(json.contains(r#""monotonic_constraints":null"#));
    assert!(json.contains(r#""early_stopping_rounds":null"#));
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
// GradientBoostingConfig — deserialize (happy paths)
// =============================================================================

#[test]
fn test_growth_strategy_serializes_to_its_wire_spelling() -> Result<(), ClearGbmError> {
    // Both variants serialize, including the one no config can currently
    // hold — the enum's wire contract is independent of whether the builder
    // implements the policy.
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
fn test_config_deserialize_rejects_a_policy_budget_mismatch() -> Result<(), ClearGbmError> {
    // The deserializer routes through `GradientBoostingConfig::new`, so the
    // policy/budget pairing is enforced on persisted payloads too. A stored
    // model claiming leaf_wise with no budget must not load as anything.
    let config = match GradientBoostingConfig::new(default_params()) {
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
fn test_config_deserialize_roundtrip_default() -> Result<(), ClearGbmError> {
    let original = match GradientBoostingConfig::new(default_params()) {
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
// GradientBoostingConfig — deserialize (error paths)
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
    let cfg = match GradientBoostingConfig::new(default_params()) {
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
    let cfg = match GradientBoostingConfig::new(default_params()) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&cfg));
    let bad = json.replace(r#""learning_rate":0.1"#, r#""learning_rate":-0.5"#);
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

// =============================================================================
// GradientBoostingModel — serialize + roundtrip
// =============================================================================

#[test]
fn test_model_serialize_contains_all_field_names() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&model));
    for field in [
        "trees",
        "base_prediction",
        "learning_rate",
        "feature_names",
        "n_classes",
        "config",
    ] {
        assert!(json.contains(field), "missing field {field} in {json}");
    }
    Ok(())
}

#[test]
fn test_model_roundtrip_preserves_accessors() -> Result<(), ClearGbmError> {
    let original = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    let decoded: GradientBoostingModel = propagate!(from_json(&json));
    // Tree leaf values may differ by up to 1 ULP after JSON roundtrip (float text
    // representation is shortest-roundtrip, not bit-exact). Structural accessors
    // must match exactly; prediction preservation is proved in the sibling test
    // `test_model_roundtrip_predictions_identical` at 1e-15 tolerance.
    assert_eq!(decoded.n_trees(), original.n_trees());
    assert_eq!(decoded.n_classes(), original.n_classes());
    assert!((decoded.base_prediction() - original.base_prediction()).abs() < 1e-15_f64);
    assert!((decoded.learning_rate() - original.learning_rate()).abs() < 1e-15_f64);
    assert_eq!(decoded.feature_names(), original.feature_names());
    assert_eq!(decoded.config(), original.config());
    Ok(())
}

#[test]
fn test_model_roundtrip_predictions_identical() -> Result<(), ClearGbmError> {
    let original = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    let decoded: GradientBoostingModel = propagate!(from_json(&json));

    let test_rows: Vec<Vec<f64>> = vec![
        vec![0.05_f64, 0.05_f64],
        vec![0.5_f64, 0.5_f64],
        vec![0.95_f64, 0.95_f64],
    ];
    let x_test: Vec<&[f64]> = test_rows.iter().map(Vec::as_slice).collect();

    let original_raw = match original.predict_raw(&x_test) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let decoded_raw = match decoded.predict_raw(&x_test) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(original_raw.len(), decoded_raw.len());
    for (a, b) in original_raw.iter().zip(decoded_raw.iter()) {
        assert!(
            (a - b).abs() < 1e-15_f64,
            "predict_raw mismatch: {a} vs {b}"
        );
    }
    Ok(())
}

#[test]
fn test_model_deserialize_missing_field_errors() -> Result<(), ClearGbmError> {
    let json = r#"{"trees":[]}"#;
    let result: Result<GradientBoostingModel, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_model_deserialize_unknown_field_errors() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&model));
    let bad = json.replace(r#""trees""#, r#""forest""#);
    let result: Result<GradientBoostingModel, _> = serde_json::from_str(&bad);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_model_deserialize_wrong_type_errors() -> Result<(), ClearGbmError> {
    let json = r#""just a string""#;
    let result: Result<GradientBoostingModel, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// =============================================================================
// Error paths
// =============================================================================
//
// The manual serde impls in `training::serde_impl` propagate every serializer
// and deserializer failure by hand rather than with `?`. Each of those arms is
// a place a future edit could silently swallow an error, so they are driven
// individually: field-by-field serializer failures via `testkit`, and
// field-by-field payload defects via round-tripped JSON.

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
];

/// The fields `GradientBoostingModel` serializes, in declaration order.
const MODEL_FIELDS: &[&str] = &[
    "trees",
    "base_prediction",
    "learning_rate",
    "feature_names",
    "n_classes",
    "config",
];

/// Serializes `value` and reparses it as a mutable JSON object.
///
/// Building the reference payload from the real `Serialize` impl (rather than
/// a hand-written literal) keeps these tests honest if a field is renamed: the
/// per-field assertions below fail loudly instead of quietly testing nothing.
///
/// # Args
///
/// * `value` - The value to serialize.
///
/// # Returns
///
/// The serialized form as a JSON object map.
///
/// # Errors
///
/// Returns [`ClearGbmError::SerializationFailed`] if serialization fails, or
/// [`ClearGbmError::DeserializationFailed`] if the result is not a JSON object.
fn as_json_object<T: serde::Serialize>(
    value: &T,
) -> Result<serde_json::Map<String, serde_json::Value>, ClearGbmError> {
    let text = match to_json(value) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };
    let parsed: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    match parsed {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(ClearGbmError::DeserializationFailed {
            reason: "expected a JSON object".to_string(),
        }),
    }
}

/// Builds the reference config used by the error-path tests.
fn reference_config() -> Result<GradientBoostingConfig, ClearGbmError> {
    GradientBoostingConfig::new(default_params())
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

#[test]
fn test_model_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    for (fail_at, field) in MODEL_FIELDS.iter().enumerate() {
        let mut ser = FailingSerializer::fail_after(fail_at);
        assert!(
            model.serialize(&mut ser).is_err(),
            "serializer failure at field {fail_at} ({field}) was swallowed"
        );
    }
    Ok(())
}

#[test]
fn test_model_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_struct();
    assert!(model.serialize(&mut ser).is_err());
    Ok(())
}

#[test]
fn test_model_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_end();
    assert!(model.serialize(&mut ser).is_err());
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
    // A bare string is a valid JSON value but wrong for every config field:
    // the numeric fields, the optional constraint list and the optional
    // early-stopping count all reject it.
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

#[test]
fn test_model_deserialize_rejects_each_missing_field() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let full = match as_json_object(&model) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    for field in MODEL_FIELDS {
        let mut partial = full.clone();
        assert!(
            partial.remove(*field).is_some(),
            "serialized model does not contain '{field}'"
        );
        let text = serde_json::Value::Object(partial).to_string();
        match serde_json::from_str::<GradientBoostingModel>(&text) {
            Ok(_) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: format!("model missing '{field}' was accepted"),
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
fn test_model_deserialize_rejects_each_wrong_value_type() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let full = match as_json_object(&model) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    for field in MODEL_FIELDS {
        let mut broken = full.clone();
        let previous = broken.insert(
            (*field).to_string(),
            serde_json::Value::String("wrong type".to_string()),
        );
        assert!(
            previous.is_some(),
            "serialized model does not contain '{field}'"
        );
        let text = serde_json::Value::Object(broken).to_string();
        if serde_json::from_str::<GradientBoostingModel>(&text).is_ok() {
            return Err(ClearGbmError::DeserializationFailed {
                reason: format!("model with a string in '{field}' was accepted"),
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
fn test_model_field_visitor_expecting_describes_a_field_identifier() -> Result<(), ClearGbmError> {
    use crate::testkit::test_expecting_write_success;
    use crate::training::serde_impl::GradientBoostingModelFieldVisitor;

    match test_expecting_write_success(&GradientBoostingModelFieldVisitor, 50_usize) {
        Ok(()) => Ok(()),
        Err(_) => Err(ClearGbmError::SerializationFailed {
            reason: "expecting() failed with a sufficient buffer".to_string(),
        }),
    }
}
