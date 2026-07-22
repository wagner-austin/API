//! Serde tests for GradientBoostingConfig + GradientBoostingModel.

use crate::error::ClearGbmError;
use crate::split::MonotonicConstraint;
use crate::training::{
    train_gradient_boosting, GradientBoostingConfig, GradientBoostingConfigParams,
    GradientBoostingModel,
};

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
    }) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    train_gradient_boosting(&x_train, &y_train, None, None, &config, &feature_names)
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
fn test_config_deserialize_invalid_learning_rate_propagates_validation_error(
) -> Result<(), ClearGbmError> {
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
        assert!((a - b).abs() < 1e-15_f64, "predict_raw mismatch: {a} vs {b}");
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
