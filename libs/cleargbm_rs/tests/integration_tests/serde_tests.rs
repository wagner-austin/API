//! Serde round-trip integration tests.
//!
//! Verifies serialization and deserialization of public types through
//! the `serde_json` interface to catch any regressions in serde instantiations.

use cleargbm_rs::{ClearGbmError, NanDirection};

/// Test NanDirection serde round-trip for both variants
#[test]
fn test_nan_direction_serde_roundtrip() -> std::result::Result<(), ClearGbmError> {
    // Test Left variant
    let left = NanDirection::Left;
    let json_left = match serde_json::to_string(&left) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let parsed_left: NanDirection = match serde_json::from_str(&json_left) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(parsed_left, NanDirection::Left));

    // Test Right variant
    let right = NanDirection::Right;
    let json_right = match serde_json::to_string(&right) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let parsed_right: NanDirection = match serde_json::from_str(&json_right) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(matches!(parsed_right, NanDirection::Right));

    Ok(())
}
