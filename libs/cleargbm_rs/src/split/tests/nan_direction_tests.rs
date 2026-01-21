//! Tests for NanDirection type.

use crate::error::ClearGbmError;
use crate::split::NanDirection;

#[test]
fn test_nan_direction_left() -> Result<(), ClearGbmError> {
    let dir = NanDirection::Left;
    assert!(dir.goes_left());
    assert!(!dir.goes_right());
    Ok(())
}

#[test]
fn test_nan_direction_right() -> Result<(), ClearGbmError> {
    let dir = NanDirection::Right;
    assert!(!dir.goes_left());
    assert!(dir.goes_right());
    Ok(())
}

#[test]
fn test_nan_direction_clone() -> Result<(), ClearGbmError> {
    let dir = NanDirection::Left;
    let cloned = dir;
    assert_eq!(dir, cloned);
    Ok(())
}

#[test]
fn test_nan_direction_serialize_deserialize() -> Result<(), ClearGbmError> {
    let dir = NanDirection::Left;
    let json_str = match serde_json::to_string(&dir) {
        Ok(s) => s,
        Err(e) => {
            return Err(ClearGbmError::SerializationFailed {
                reason: e.to_string(),
            })
        }
    };
    let parsed: NanDirection = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert_eq!(parsed, dir);
    Ok(())
}
