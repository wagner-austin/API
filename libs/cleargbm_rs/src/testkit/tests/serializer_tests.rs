//! Tests for FailingSerializer.

use crate::error::ClearGbmError;
use crate::testkit::serializer::FailingSerializer;
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

struct TestStruct {
    a: i32,
    b: i32,
    c: i32,
}

impl Serialize for TestStruct {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("TestStruct", 3) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("a", &self.a) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("b", &self.b) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("c", &self.c) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

#[test]
fn test_failing_serializer_fail_after_0() -> Result<(), ClearGbmError> {
    let mut ser = FailingSerializer::fail_after(0);
    let s = TestStruct { a: 1, b: 2, c: 3 };
    let result = s.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_failing_serializer_fail_after_1() -> Result<(), ClearGbmError> {
    let mut ser = FailingSerializer::fail_after(1);
    let s = TestStruct { a: 1, b: 2, c: 3 };
    let result = s.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_failing_serializer_fail_after_2() -> Result<(), ClearGbmError> {
    let mut ser = FailingSerializer::fail_after(2);
    let s = TestStruct { a: 1, b: 2, c: 3 };
    let result = s.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_failing_serializer_success_when_enough_fields() -> Result<(), ClearGbmError> {
    let mut ser = FailingSerializer::fail_after(10);
    let s = TestStruct { a: 1, b: 2, c: 3 };
    let result = s.serialize(&mut ser);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_serializer_fail_on_struct() -> Result<(), ClearGbmError> {
    let mut ser = FailingSerializer::fail_on_struct();
    let s = TestStruct { a: 1, b: 2, c: 3 };
    let result = s.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_failing_serializer_primitives() -> Result<(), ClearGbmError> {
    let mut ser = FailingSerializer::fail_after(100);
    assert!(ser.serialize_bool(true).is_ok());
    assert!(ser.serialize_i8(1_i8).is_ok());
    assert!(ser.serialize_i16(1_i16).is_ok());
    assert!(ser.serialize_i32(1_i32).is_ok());
    assert!(ser.serialize_i64(1_i64).is_ok());
    assert!(ser.serialize_u8(1_u8).is_ok());
    assert!(ser.serialize_u16(1_u16).is_ok());
    assert!(ser.serialize_u32(1_u32).is_ok());
    assert!(ser.serialize_u64(1_u64).is_ok());
    assert!(ser.serialize_f32(1.0_f32).is_ok());
    assert!(ser.serialize_f64(1.0_f64).is_ok());
    assert!(ser.serialize_char('a').is_ok());
    assert!(ser.serialize_str("test").is_ok());
    assert!(ser.serialize_bytes(&[1_u8, 2_u8]).is_ok());
    assert!(ser.serialize_none().is_ok());
    assert!(ser.serialize_some(&1_i32).is_ok());
    assert!(ser.serialize_unit().is_ok());
    assert!(ser.serialize_unit_struct("Unit").is_ok());
    assert!(ser.serialize_unit_variant("E", 0, "V").is_ok());
    assert!(ser.serialize_newtype_struct("N", &1_i32).is_ok());
    assert!(ser.serialize_newtype_variant("E", 0, "V", &1_i32).is_ok());
    Ok(())
}

#[test]
fn test_failing_serializer_unsupported() -> Result<(), ClearGbmError> {
    let mut ser = FailingSerializer::fail_after(100);
    assert!(ser.serialize_seq(Some(1)).is_err());
    assert!(ser.serialize_tuple(1).is_err());
    assert!(ser.serialize_tuple_struct("T", 1).is_err());
    assert!(ser.serialize_tuple_variant("E", 0, "V", 1).is_err());
    assert!(ser.serialize_map(Some(1)).is_err());
    assert!(ser.serialize_struct_variant("E", 0, "V", 1).is_err());
    Ok(())
}

#[test]
fn test_failing_serializer_struct_end() -> Result<(), ClearGbmError> {
    let mut ser = FailingSerializer::fail_after(100);
    let struct_ser = match (&mut ser).serialize_struct("Test", 0) {
        Ok(s) => s,
        Err(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "failed".to_string(),
            })
        }
    };
    assert!(struct_ser.end().is_ok());
    Ok(())
}

#[test]
fn test_failing_serializer_struct_field_ok_then_fail() -> Result<(), ClearGbmError> {
    let mut ser = FailingSerializer::fail_after(1);
    let mut struct_ser = match (&mut ser).serialize_struct("Test", 2) {
        Ok(s) => s,
        Err(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "failed".to_string(),
            })
        }
    };
    assert!(struct_ser.serialize_field("f1", &1_u32).is_ok());
    assert!(struct_ser.serialize_field("f2", &2_u32).is_err());
    Ok(())
}
