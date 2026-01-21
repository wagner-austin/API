//! Tests for SerError and DeError types.

use crate::error::ClearGbmError;
use crate::testkit::deserializer::DeError;
use crate::testkit::serializer::SerError;
use serde::de;
use serde::ser;

#[test]
fn test_ser_error_display() -> Result<(), ClearGbmError> {
    let err = SerError {
        message: "test error".to_string(),
    };
    assert_eq!(format!("{}", err), "test error");
    Ok(())
}

#[test]
fn test_ser_error_debug() -> Result<(), ClearGbmError> {
    let err = SerError {
        message: "test error".to_string(),
    };
    let debug = format!("{:?}", err);
    assert!(debug.contains("SerError"));
    assert!(debug.contains("test error"));
    Ok(())
}

#[test]
fn test_ser_error_custom() -> Result<(), ClearGbmError> {
    let err = <SerError as ser::Error>::custom("custom msg");
    assert_eq!(err.message, "custom msg");
    Ok(())
}

#[test]
fn test_ser_error_is_error() -> Result<(), ClearGbmError> {
    let err = SerError {
        message: "test".to_string(),
    };
    let _: &dyn std::error::Error = &err;
    Ok(())
}

#[test]
fn test_de_error_display() -> Result<(), ClearGbmError> {
    let err = DeError {
        message: "test error".to_string(),
    };
    assert_eq!(format!("{}", err), "test error");
    Ok(())
}

#[test]
fn test_de_error_debug() -> Result<(), ClearGbmError> {
    let err = DeError {
        message: "test error".to_string(),
    };
    let debug = format!("{:?}", err);
    assert!(debug.contains("DeError"));
    assert!(debug.contains("test error"));
    Ok(())
}

#[test]
fn test_de_error_custom() -> Result<(), ClearGbmError> {
    let err = <DeError as de::Error>::custom("custom msg");
    assert_eq!(err.message, "custom msg");
    Ok(())
}

#[test]
fn test_de_error_is_error() -> Result<(), ClearGbmError> {
    let err = DeError {
        message: "test".to_string(),
    };
    let _: &dyn std::error::Error = &err;
    Ok(())
}
