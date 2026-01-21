//! Tests for forward_to_deserialize_any! generated methods on FailingDeserializer
//! and FieldNameDeserializer.

use crate::error::ClearGbmError;
use crate::testkit::deserializer::{FailingDeserializer, FieldNameDeserializer};
use core::fmt;
use serde::de::Visitor;
use serde::Deserializer;

// =============================================================================
// Visitor implementations for testing
// =============================================================================

/// Visitor that accepts i64 (for FailingDeserializer).
struct I64Visitor;
impl<'de> Visitor<'de> for I64Visitor {
    type Value = i64;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i64")
    }
    fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
        Ok(v)
    }
}

/// Visitor that accepts str (for FieldNameDeserializer).
struct StrVisitor;
impl<'de> Visitor<'de> for StrVisitor {
    type Value = String;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "str")
    }
    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
        Ok(v.to_string())
    }
}

// =============================================================================
// FailingDeserializer forward tests
// =============================================================================

#[test]
fn test_failing_deserializer_forward_bool() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_bool(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_i8() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_i8(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_i16() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_i16(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_i32() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_i32(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_i64() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_i64(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_u8() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_u8(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_u16() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_u16(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_u32() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_u32(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_u64() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_u64(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_f32() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_f32(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_f64() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_f64(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_char() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_char(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_str() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_str(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_string() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_string(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_bytes() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_bytes(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_byte_buf() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_byte_buf(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_option() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_option(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_unit() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_unit(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_unit_struct() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_unit_struct("Test", I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_newtype_struct() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_newtype_struct("Test", I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_seq() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_seq(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_tuple() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_tuple(1, I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_tuple_struct() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_tuple_struct("Test", 1, I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_map() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_map(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_enum() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_enum("Test", &["A"], I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_forward_ignored_any() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    let result = de.deserialize_ignored_any(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

// =============================================================================
// FieldNameDeserializer forward tests
// =============================================================================

#[test]
fn test_field_name_deserializer_forward_bool() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_bool(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_i8() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_i8(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_i16() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_i16(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_i32() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_i32(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_i64() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_i64(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_u8() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_u8(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_u16() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_u16(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_u32() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_u32(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_u64() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_u64(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_f32() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_f32(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_f64() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_f64(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_char() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_char(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_str() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_str(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_string() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_string(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_bytes() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_bytes(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_byte_buf() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_byte_buf(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_option() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_option(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_unit() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_unit(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_unit_struct() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_unit_struct("Test", StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_newtype_struct() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_newtype_struct("Test", StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_seq() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_seq(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_tuple() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_tuple(1, StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_tuple_struct() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_tuple_struct("Test", 1, StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_map() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_map(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_struct() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_struct("Test", &["a"], StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_enum() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_enum("Test", &["A"], StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_ignored_any() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_ignored_any(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}
