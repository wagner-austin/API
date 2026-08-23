//! Serde tests for the split enums: `NanDirection` and
//! `MonotonicConstraint`. The `SplitResult` serde battery lives in
//! [`super::serde_result_tests`].

use crate::error::ClearGbmError;
use crate::split::{MonotonicConstraint, NanDirection};

// =========================================================================
// Serde error path tests - NanDirection
// =========================================================================

#[test]
fn test_nan_direction_deserialize_invalid_value() -> Result<(), ClearGbmError> {
    // Invalid string value
    let json = r#""Invalid""#;
    let result: Result<NanDirection, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    // Number instead of string
    let json = r#"123"#;
    let result: Result<NanDirection, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_left() -> Result<(), ClearGbmError> {
    let json = r#""Left""#;
    let dir: NanDirection = match serde_json::from_str(json) {
        Ok(d) => d,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(dir, NanDirection::Left));
    Ok(())
}

#[test]
fn test_nan_direction_deserialize_right() -> Result<(), ClearGbmError> {
    let json = r#""Right""#;
    let dir: NanDirection = match serde_json::from_str(json) {
        Ok(d) => d,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(dir, NanDirection::Right));
    Ok(())
}

// =========================================================================
// Direct NanDirection deserialize via testkit deserializer
// =========================================================================

#[test]
fn test_nan_direction_via_minimal_value_deserializer() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::MinimalValueDeserializer;
    use serde::Deserialize;
    // MinimalValueDeserializer::deserialize_str returns "Right"
    let result = NanDirection::deserialize(MinimalValueDeserializer);
    assert!(result.is_ok());
    let dir = match result {
        Ok(d) => d,
        Err(_) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: "failed".to_string(),
            })
        }
    };
    assert!(matches!(dir, NanDirection::Right));
    Ok(())
}

#[test]
fn test_nan_direction_via_minimal_struct_deserializer() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::MinimalStructDeserializer;
    use serde::Deserialize;
    // MinimalStructDeserializer::deserialize_str returns "Left"
    let result = NanDirection::deserialize(MinimalStructDeserializer);
    assert!(result.is_ok());
    let dir = match result {
        Ok(d) => d,
        Err(_) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: "failed".to_string(),
            })
        }
    };
    assert!(matches!(dir, NanDirection::Left));
    Ok(())
}

// =========================================================================
// Serde tests - MonotonicConstraint
// =========================================================================

#[test]
fn test_monotonic_constraint_serialize_none() -> Result<(), ClearGbmError> {
    let json = match serde_json::to_string(&MonotonicConstraint::None) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(json, r#""None""#);
    Ok(())
}

#[test]
fn test_monotonic_constraint_serialize_increasing() -> Result<(), ClearGbmError> {
    let json = match serde_json::to_string(&MonotonicConstraint::Increasing) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(json, r#""Increasing""#);
    Ok(())
}

#[test]
fn test_monotonic_constraint_serialize_decreasing() -> Result<(), ClearGbmError> {
    let json = match serde_json::to_string(&MonotonicConstraint::Decreasing) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(json, r#""Decreasing""#);
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_none() -> Result<(), ClearGbmError> {
    let json = r#""None""#;
    let c: MonotonicConstraint = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(c, MonotonicConstraint::None));
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_increasing() -> Result<(), ClearGbmError> {
    let json = r#""Increasing""#;
    let c: MonotonicConstraint = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(c, MonotonicConstraint::Increasing));
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_decreasing() -> Result<(), ClearGbmError> {
    let json = r#""Decreasing""#;
    let c: MonotonicConstraint = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(c, MonotonicConstraint::Decreasing));
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_invalid_string() -> Result<(), ClearGbmError> {
    let json = r#""Bogus""#;
    let result: Result<MonotonicConstraint, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_monotonic_constraint_deserialize_wrong_type() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<MonotonicConstraint, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_monotonic_constraint_roundtrip_all_variants() -> Result<(), ClearGbmError> {
    for original in [
        MonotonicConstraint::None,
        MonotonicConstraint::Increasing,
        MonotonicConstraint::Decreasing,
    ] {
        let json = match serde_json::to_string(&original) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let decoded: MonotonicConstraint = match serde_json::from_str(&json) {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(decoded, original);
    }
    Ok(())
}
