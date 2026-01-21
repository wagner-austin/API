//! Tests for forward_to_deserialize_any! generated methods on MinimalStructDeserializer
//! and MinimalValueDeserializer.

use crate::error::ClearGbmError;
use crate::testkit::deserializer::{MinimalStructDeserializer, MinimalValueDeserializer};
use core::fmt;
use serde::de::Visitor;
use serde::Deserializer;

// =============================================================================
// Visitor implementations for testing
// =============================================================================

/// Visitor that accepts i64 (for MinimalValueDeserializer).
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

/// Visitor that accepts map (for MinimalStructDeserializer).
struct EmptyMapVisitor;
impl<'de> Visitor<'de> for EmptyMapVisitor {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "map")
    }
    fn visit_map<A>(self, _map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        Ok(())
    }
}

/// Visitor that accepts strings (for deserialize_str tests).
struct StrVisitor;
impl<'de> Visitor<'de> for StrVisitor {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a string")
    }
    fn visit_str<E>(self, _v: &str) -> Result<Self::Value, E> {
        Ok(())
    }
}

// =============================================================================
// MinimalStructDeserializer forward tests
// =============================================================================

#[test]
fn test_minimal_struct_deserializer_forward_bool() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_bool(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_i8() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_i8(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_i16() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_i16(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_i32() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_i32(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_i64() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_i64(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_u8() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_u8(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_u16() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_u16(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_u32() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_u32(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_u64() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_u64(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_f32() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_f32(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_f64() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_f64(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_char() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_char(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_str() -> Result<(), ClearGbmError> {
    // MinimalStructDeserializer::deserialize_str returns "Left" via visit_str
    let de = MinimalStructDeserializer;
    let result = de.deserialize_str(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_string() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_string(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_bytes() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_bytes(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_byte_buf() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_byte_buf(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_option() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_option(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_unit() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_unit(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_unit_struct() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_unit_struct("Test", EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_newtype_struct() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_newtype_struct("Test", EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_tuple() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_tuple(1, EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_tuple_struct() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_tuple_struct("Test", 1, EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_map() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_map(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_enum() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_enum("Test", &["A"], EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_identifier() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_identifier(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_forward_ignored_any() -> Result<(), ClearGbmError> {
    let de = MinimalStructDeserializer;
    let result = de.deserialize_ignored_any(EmptyMapVisitor);
    assert!(result.is_ok());
    Ok(())
}

// =============================================================================
// MinimalValueDeserializer forward tests
// =============================================================================

#[test]
fn test_minimal_value_deserializer_forward_bool() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_bool(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_i8() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_i8(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_i16() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_i16(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_i32() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_i32(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_i64() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_i64(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_u8() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_u8(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_u16() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_u16(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_u32() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_u32(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_u64() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_u64(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_f32() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_f32(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_f64() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_f64(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_char() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_char(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_str() -> Result<(), ClearGbmError> {
    use serde::de::Visitor;
    struct StrVisitor;
    impl<'de> Visitor<'de> for StrVisitor {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "a string")
        }
        fn visit_str<E>(self, _v: &str) -> Result<Self::Value, E> {
            Ok(())
        }
    }
    let de = MinimalValueDeserializer;
    let result = de.deserialize_str(StrVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_string() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_string(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_bytes() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_bytes(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_byte_buf() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_byte_buf(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_option() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_option(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_unit() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_unit(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_unit_struct() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_unit_struct("Test", I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_newtype_struct() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_newtype_struct("Test", I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_tuple() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_tuple(1, I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_tuple_struct() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_tuple_struct("Test", 1, I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_map() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_map(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_enum() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_enum("Test", &["A"], I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_identifier() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_identifier(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_forward_ignored_any() -> Result<(), ClearGbmError> {
    let de = MinimalValueDeserializer;
    let result = de.deserialize_ignored_any(I64Visitor);
    assert!(result.is_ok());
    Ok(())
}

// =============================================================================
// FieldNameDeserializer forward tests
// =============================================================================

/// Visitor that accepts str and returns String (for FieldNameDeserializer).
struct StrToStringVisitor;
impl<'de> Visitor<'de> for StrToStringVisitor {
    type Value = String;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "str")
    }
    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
        Ok(v.to_string())
    }
}

#[test]
fn test_field_name_deserializer_forward_bool() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_bool(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_i8() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_i8(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_i16() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_i16(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_i32() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_i32(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_i64() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_i64(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_u8() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_u8(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_u16() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_u16(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_u32() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_u32(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_u64() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_u64(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_f32() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_f32(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_f64() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_f64(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_char() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_char(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_str() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_str(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_string() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_string(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_bytes() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_bytes(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_byte_buf() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_byte_buf(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_option() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_option(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_unit() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_unit(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_unit_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_unit_struct("Test", StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_newtype_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_newtype_struct("Test", StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_seq() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_seq(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_tuple() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_tuple(1, StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_tuple_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_tuple_struct("Test", 1, StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_map() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_map(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_struct("Test", &["a"], StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_enum() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_enum("Test", &["A"], StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_field_name_deserializer_forward_ignored_any() -> Result<(), ClearGbmError> {
    use crate::testkit::FieldNameDeserializer;
    let de = FieldNameDeserializer { field: "test" };
    let result = de.deserialize_ignored_any(StrToStringVisitor);
    assert!(result.is_ok());
    Ok(())
}
