//! Serde tests for GradientBoostingModel.
//!
//! Config serde tests live in `serde_config_tests`; shared helpers in
//! `serde_helpers`.

use crate::error::ClearGbmError;
use crate::training::{GradientBoostingModel, Objective};

use super::serde_helpers::{
    as_json_object, from_json, make_test_model, make_test_regression_model, to_json,
};

/// The fields `GradientBoostingModel` serializes, in declaration order.
const MODEL_FIELDS: &[&str] = &[
    "trees",
    "base_prediction",
    "learning_rate",
    "feature_names",
    "config",
];

#[test]
fn test_model_serialize_contains_all_field_names() -> Result<(), ClearGbmError> {
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&model));
    for field in MODEL_FIELDS {
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
fn test_regression_model_roundtrip_preserves_objective_and_predictions() -> Result<(), ClearGbmError>
{
    // The objective tag must survive persistence — a reloaded regression
    // model that forgot its objective would happily serve sigmoid
    // probabilities of its raw values.
    let original = match make_test_regression_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let json = propagate!(to_json(&original));
    assert!(json.contains(r#""objective":"squared_error""#));
    let decoded: GradientBoostingModel = propagate!(from_json(&json));
    assert_eq!(decoded.config().objective(), Objective::SquaredError);

    let test_rows: Vec<Vec<f64>> = vec![vec![0.2_f64, 0.8_f64], vec![0.9_f64, 0.1_f64]];
    let x_test: Vec<&[f64]> = test_rows.iter().map(Vec::as_slice).collect();
    let original_raw = match original.predict_raw(&x_test) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let decoded_raw = match decoded.predict_raw(&x_test) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    for (a, b) in original_raw.iter().zip(decoded_raw.iter()) {
        assert!(
            (a - b).abs() < 1e-15_f64,
            "predict_raw mismatch: {a} vs {b}"
        );
    }
    // And the reloaded model still refuses probabilities.
    match decoded.predict_proba(&x_test) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a reloaded squared_error model must reject predict_proba".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "objective");
            Ok(())
        }
        Err(e) => Err(e),
    }
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
fn test_model_deserialize_rejects_the_retired_n_classes_field() -> Result<(), ClearGbmError> {
    // `n_classes` was removed 2026-08-22: it was constant 2 for binary,
    // derivable from the objective, and meaningless for regression. An old
    // artifact carrying it must fail loudly, per the no-silent-migration
    // policy.
    let model = match make_test_model() {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let mut full = match as_json_object(&model) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    full.insert(
        "n_classes".to_string(),
        serde_json::Value::Number(serde_json::Number::from(2_i32)),
    );
    let text = serde_json::Value::Object(full).to_string();
    let err = match serde_json::from_str::<GradientBoostingModel>(&text) {
        Ok(_) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: "a payload with the retired n_classes field was accepted".to_string(),
            })
        }
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("n_classes"),
        "rejection should name the unknown field, got: {err}"
    );
    Ok(())
}

#[test]
fn test_model_deserialize_wrong_type_errors() -> Result<(), ClearGbmError> {
    let json = r#""just a string""#;
    let result: Result<GradientBoostingModel, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// -----------------------------------------------------------------------------
// Serializer failures
// -----------------------------------------------------------------------------

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
// Field-identifier visitor
// -----------------------------------------------------------------------------

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
