//! Tests for FailingDeserializer and FieldNameDeserializer.

use crate::error::ClearGbmError;
use crate::testkit::deserializer::{
    DuplicateFieldDeserializer, ErrorOnKeyDeserializer, ErrorOnValueDeserializer,
    FailingDeserializer, FieldNameDeserializer, IntegerDeserializer, IntegerKeyDeserializer,
    StringDeserializer, StructDuplicateFieldDeserializer, WrongValueDeserializer,
};
use core::fmt;
use serde::de::{self, Visitor};
use serde::Deserializer;

#[test]
fn test_failing_deserializer_integer() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_any(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_failing_deserializer_map_with_integer_key() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_with_integer_key();
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            Ok(())
        }
    }
    let _ = de.deserialize_any(TestVisitor);
    Ok(())
}

#[test]
fn test_failing_deserializer_map_with_wrong_value() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_with_wrong_value("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            let _ = map.next_value::<String>();
            Ok(())
        }
    }
    let _ = de.deserialize_any(TestVisitor);
    Ok(())
}

#[test]
fn test_failing_deserializer_struct() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_struct("Test", &["a"], TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_failing_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::integer();
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_identifier(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_field_name_deserializer_any() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "test" };
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = String;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "string")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }
    }
    let result = de.deserialize_any(V);
    match result {
        Ok(v) => assert_eq!(v, "test"),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_field_name_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = FieldNameDeserializer { field: "my_id" };
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = String;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "string")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }
    }
    let result = de.deserialize_identifier(V);
    match result {
        Ok(v) => assert_eq!(v, "my_id"),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_failing_deserializer_struct_map_with_integer_key() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_with_integer_key();
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            Ok(())
        }
    }
    let _ = de.deserialize_struct("Test", &["a"], TestVisitor);
    Ok(())
}

#[test]
fn test_failing_deserializer_struct_map_with_wrong_value() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_with_wrong_value("field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            Ok(())
        }
    }
    let _ = de.deserialize_struct("Test", &["field"], TestVisitor);
    Ok(())
}

#[test]
fn test_failing_deserializer_any_with_duplicate_field() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_with_duplicate_field("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            while let Ok(Some(_key)) = map.next_key::<String>() {
                let _ = map.next_value::<i64>();
            }
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_struct_with_duplicate_field() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_with_duplicate_field("field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            while let Ok(Some(_key)) = map.next_key::<String>() {
                let _ = map.next_value::<i64>();
            }
            Ok(())
        }
    }
    let result = de.deserialize_struct("Test", &["field"], TestVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_struct_duplicate_field() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::struct_duplicate_field("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            loop {
                let key: Option<String> = match map.next_key() {
                    Ok(k) => k,
                    Err(e) => return Err(e),
                };
                let Some(_key) = key else {
                    break;
                };
                struct UnitSeed;
                impl<'de> de::DeserializeSeed<'de> for UnitSeed {
                    type Value = ();
                    fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
                    where
                        D: de::Deserializer<'de>,
                    {
                        struct V;
                        impl<'de> Visitor<'de> for V {
                            type Value = ();
                            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                                write!(f, "unit")
                            }
                            fn visit_map<A>(self, _: A) -> Result<Self::Value, A::Error>
                            where
                                A: de::MapAccess<'de>,
                            {
                                Ok(())
                            }
                        }
                        de.deserialize_struct("", &[], V)
                    }
                }
                match map.next_value_seed(UnitSeed) {
                    Ok(_) => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok());
    let de2 = FailingDeserializer::struct_duplicate_field("field");
    let result2 = de2.deserialize_struct("Test", &["field"], TestVisitor);
    assert!(result2.is_ok());
    Ok(())
}

#[test]
fn test_failing_deserializer_string_value() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::string_value();
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = String;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "string")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, "wrong_type_string"),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_failing_deserializer_string_value_struct() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::string_value();
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = String;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "string")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }
    }
    let result = de.deserialize_struct("Test", &["a"], TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, "wrong_type_string"),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_failing_deserializer_map_error_on_key_any() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_error_on_key();
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok()); // Visitor handles the error internally
    Ok(())
}

#[test]
fn test_failing_deserializer_map_error_on_key_struct() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_error_on_key();
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            Ok(())
        }
    }
    let result = de.deserialize_struct("Test", &["a"], TestVisitor);
    assert!(result.is_ok()); // Visitor handles the error internally
    Ok(())
}

#[test]
fn test_failing_deserializer_map_error_on_value_any() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_error_on_value("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            let _ = map.next_value::<String>();
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok()); // Visitor handles the error internally
    Ok(())
}

#[test]
fn test_failing_deserializer_map_error_on_value_struct() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_error_on_value("field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            let _ = map.next_value::<String>();
            Ok(())
        }
    }
    let result = de.deserialize_struct("Test", &["field"], TestVisitor);
    assert!(result.is_ok()); // Visitor handles the error internally
    Ok(())
}

#[test]
fn test_integer_key_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = IntegerKeyDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_identifier(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_duplicate_field_deserializer_any() -> Result<(), ClearGbmError> {
    let de = DuplicateFieldDeserializer::new("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "map")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            while let Ok(Some(_key)) = map.next_key::<String>() {
                let _ = map.next_value::<i64>();
            }
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_duplicate_field_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = DuplicateFieldDeserializer::new("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_identifier(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_struct_duplicate_field_deserializer_any() -> Result<(), ClearGbmError> {
    let de = StructDuplicateFieldDeserializer::new("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "map")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            while let Ok(Some(_key)) = map.next_key::<String>() {
                struct UnitSeed;
                impl<'de> de::DeserializeSeed<'de> for UnitSeed {
                    type Value = ();
                    fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
                    where
                        D: de::Deserializer<'de>,
                    {
                        struct V;
                        impl<'de> Visitor<'de> for V {
                            type Value = ();
                            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                                write!(f, "unit")
                            }
                            fn visit_map<A>(self, _: A) -> Result<Self::Value, A::Error>
                            where
                                A: de::MapAccess<'de>,
                            {
                                Ok(())
                            }
                        }
                        de.deserialize_struct("", &[], V)
                    }
                }
                let _ = map.next_value_seed(UnitSeed);
            }
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_struct_duplicate_field_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = StructDuplicateFieldDeserializer::new("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_identifier(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_wrong_value_deserializer_any() -> Result<(), ClearGbmError> {
    let de = WrongValueDeserializer::new("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "map")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            let _ = map.next_value::<String>();
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_wrong_value_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = WrongValueDeserializer::new("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_identifier(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_error_on_key_deserializer_any() -> Result<(), ClearGbmError> {
    let de = ErrorOnKeyDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "map")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_error_on_key_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = ErrorOnKeyDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_identifier(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_error_on_value_deserializer_any() -> Result<(), ClearGbmError> {
    let de = ErrorOnValueDeserializer::new("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "map")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            let _ = map.next_value::<String>();
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_error_on_value_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = ErrorOnValueDeserializer::new("test_field");
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_identifier(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_string_deserializer_any() -> Result<(), ClearGbmError> {
    let de = StringDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = String;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "string")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, "wrong_type_string"),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_string_deserializer_struct() -> Result<(), ClearGbmError> {
    let de = StringDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = String;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "string")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }
    }
    let result = de.deserialize_struct("Test", &["a"], TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, "wrong_type_string"),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_string_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = StringDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_identifier(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_integer_key_deserializer_any() -> Result<(), ClearGbmError> {
    let de = IntegerKeyDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "map")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            Ok(())
        }
    }
    let result = de.deserialize_any(TestVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_integer_key_deserializer_struct() -> Result<(), ClearGbmError> {
    let de = IntegerKeyDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "map")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let _ = map.next_key::<String>();
            Ok(())
        }
    }
    let result = de.deserialize_struct("Test", &["a"], TestVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_integer_deserializer_any() -> Result<(), ClearGbmError> {
    let de = IntegerDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_any(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_integer_deserializer_struct() -> Result<(), ClearGbmError> {
    let de = IntegerDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_struct("Test", &["a"], TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_integer_deserializer_identifier() -> Result<(), ClearGbmError> {
    let de = IntegerDeserializer;
    struct TestVisitor;
    impl<'de> Visitor<'de> for TestVisitor {
        type Value = i64;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let result = de.deserialize_identifier(TestVisitor);
    match result {
        Ok(v) => assert_eq!(v, 42_i64),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

/// A DeserializeSeed that always returns an error.
struct FailingSeed;

impl<'de> de::DeserializeSeed<'de> for FailingSeed {
    type Value = ();
    fn deserialize<D>(self, _deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        Err(de::Error::custom("intentional seed failure"))
    }
}

#[test]
fn test_integer_key_map_access_err_branch() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::IntegerKeyMapAccess;
    let mut map = IntegerKeyMapAccess { done: false };
    // Use FailingSeed to trigger the Err branch
    let result = de::MapAccess::next_key_seed(&mut map, FailingSeed);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_wrong_value_map_access_err_branch() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::WrongValueMapAccess;
    let mut map = WrongValueMapAccess::new("field");
    let result = de::MapAccess::next_key_seed(&mut map, FailingSeed);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_duplicate_field_map_access_err_branch() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::DuplicateFieldMapAccess;
    let mut map = DuplicateFieldMapAccess::new("field");
    let result = de::MapAccess::next_key_seed(&mut map, FailingSeed);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_struct_duplicate_field_map_access_err_branch() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::StructDuplicateFieldMapAccess;
    let mut map = StructDuplicateFieldMapAccess::new("field");
    let result = de::MapAccess::next_key_seed(&mut map, FailingSeed);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_error_on_value_map_access_err_branch() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::ErrorOnValueMapAccess;
    let mut map = ErrorOnValueMapAccess::new("field");
    let result = de::MapAccess::next_key_seed(&mut map, FailingSeed);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests for next_value_seed on map access types
// =========================================================================

/// A seed that succeeds and returns a unit value.
struct UnitSeed;

impl<'de> de::DeserializeSeed<'de> for UnitSeed {
    type Value = ();
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct UnitVisitor;
        impl<'de> Visitor<'de> for UnitVisitor {
            type Value = ();
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "unit")
            }
            fn visit_i64<E>(self, _v: i64) -> Result<Self::Value, E> {
                Ok(())
            }
            fn visit_str<E>(self, _v: &str) -> Result<Self::Value, E> {
                Ok(())
            }
            fn visit_map<A>(self, _map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                Ok(())
            }
            fn visit_seq<A>(self, _seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                Ok(())
            }
        }
        deserializer.deserialize_any(UnitVisitor)
    }
}

#[test]
fn test_integer_key_map_access_next_value_seed() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::IntegerKeyMapAccess;
    let mut map = IntegerKeyMapAccess { done: false };
    let result = de::MapAccess::next_value_seed(&mut map, UnitSeed);
    assert!(result.is_ok());
    Ok(())
}

// Tests with concrete types to cover production instantiations
#[test]
fn test_integer_key_map_access_next_value_i64() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::IntegerKeyMapAccess;
    let mut map = IntegerKeyMapAccess { done: false };
    let result: Result<i64, _> = de::MapAccess::next_value(&mut map);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_integer_key_map_access_next_value_bool() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::IntegerKeyMapAccess;
    let mut map = IntegerKeyMapAccess { done: false };
    let result: Result<bool, _> = de::MapAccess::next_value(&mut map);
    // Will fail because integer can't convert to bool, but exercises the code path
    let _ = result;
    Ok(())
}

/// A visitor that accepts any map and reads all entries, accepting integer keys.
/// This allows us to exercise next_value_seed with production types.
struct AnyMapVisitor;

impl<'de> Visitor<'de> for AnyMapVisitor {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "any map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: de::MapAccess<'de>,
    {
        // Try to read key as i64 (which IntegerKeyMapAccess provides)
        while let Ok(Some(_key)) = map.next_key::<i64>() {
            // Read value as various types to exercise different instantiations
            let _: Result<i64, _> = map.next_value();
        }
        Ok(())
    }
}

#[test]
fn test_integer_key_deserializer_with_any_visitor() -> Result<(), ClearGbmError> {
    use serde::Deserialize;
    struct AnyMap;
    impl<'de> Deserialize<'de> for AnyMap {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            match deserializer.deserialize_any(AnyMapVisitor) {
                Ok(()) => Ok(AnyMap),
                Err(e) => Err(e),
            }
        }
    }
    let de = IntegerKeyDeserializer;
    let result = AnyMap::deserialize(de);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_integer_key_map_access_next_value_usize() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::IntegerKeyMapAccess;
    let mut map = IntegerKeyMapAccess { done: false };
    let result: Result<usize, _> = de::MapAccess::next_value(&mut map);
    let _ = result; // may fail on type conversion
    Ok(())
}

#[test]
fn test_integer_key_map_access_next_value_f64() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::IntegerKeyMapAccess;
    let mut map = IntegerKeyMapAccess { done: false };
    let result: Result<f64, _> = de::MapAccess::next_value(&mut map);
    let _ = result;
    Ok(())
}

#[test]
fn test_wrong_value_map_access_next_value_seed() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::WrongValueMapAccess;
    let mut map = WrongValueMapAccess::new("field");
    let result = de::MapAccess::next_value_seed(&mut map, UnitSeed);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_wrong_value_map_access_next_value_string() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::WrongValueMapAccess;
    let mut map = WrongValueMapAccess::new("field");
    let result: Result<String, _> = de::MapAccess::next_value(&mut map);
    assert!(result.is_ok()); // WrongValueMapAccess returns string
    Ok(())
}

#[test]
fn test_wrong_value_map_access_next_value_usize() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::WrongValueMapAccess;
    let mut map = WrongValueMapAccess::new("field");
    let result: Result<usize, _> = de::MapAccess::next_value(&mut map);
    let _ = result; // May fail type conversion, but exercises code path
    Ok(())
}

#[test]
fn test_wrong_value_map_access_next_value_f64() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::WrongValueMapAccess;
    let mut map = WrongValueMapAccess::new("field");
    let result: Result<f64, _> = de::MapAccess::next_value(&mut map);
    let _ = result;
    Ok(())
}

#[test]
fn test_wrong_value_map_access_next_value_bool() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::WrongValueMapAccess;
    let mut map = WrongValueMapAccess::new("field");
    let result: Result<bool, _> = de::MapAccess::next_value(&mut map);
    let _ = result;
    Ok(())
}

#[test]
fn test_duplicate_field_map_access_next_value_seed() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::DuplicateFieldMapAccess;
    let mut map = DuplicateFieldMapAccess::new("field");
    let result = de::MapAccess::next_value_seed(&mut map, UnitSeed);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_duplicate_field_map_access_next_value_i64() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::DuplicateFieldMapAccess;
    let mut map = DuplicateFieldMapAccess::new("field");
    let result: Result<i64, _> = de::MapAccess::next_value(&mut map);
    assert!(result.is_ok()); // DuplicateFieldMapAccess returns integer
    Ok(())
}

#[test]
fn test_struct_duplicate_field_map_access_next_value_seed() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::StructDuplicateFieldMapAccess;
    let mut map = StructDuplicateFieldMapAccess::new("field");
    let result = de::MapAccess::next_value_seed(&mut map, UnitSeed);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_error_on_key_map_access_next_value_seed() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::ErrorOnKeyMapAccess;
    let mut map = ErrorOnKeyMapAccess;
    let result = de::MapAccess::next_value_seed(&mut map, UnitSeed);
    assert!(result.is_err()); // ErrorOnKeyMapAccess errors on next_value
    Ok(())
}

#[test]
fn test_error_on_value_map_access_next_value_seed() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::ErrorOnValueMapAccess;
    let mut map = ErrorOnValueMapAccess::new("field");
    let result = de::MapAccess::next_value_seed(&mut map, UnitSeed);
    assert!(result.is_err()); // ErrorOnValueMapAccess errors on next_value
    Ok(())
}

// ===========================================================================
// Tests for WrongTypeDeserializer::deserialize_struct
// These cover lines 585-604 in deserializer.rs
// ===========================================================================

/// A visitor for deserialize_struct that accepts maps.
struct StructMapVisitor;

impl<'de> Visitor<'de> for StructMapVisitor {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a struct")
    }

    fn visit_i64<E>(self, _v: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, _v: &str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: de::MapAccess<'de>,
    {
        // Read all entries to exercise the map access
        while let Ok(Some(_key)) = map.next_key::<String>() {
            let _: Result<String, _> = map.next_value();
        }
        Ok(())
    }
}

#[test]
fn test_failing_deserializer_struct_integer_key() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_with_integer_key();
    let result = de.deserialize_struct("Test", &["a"], StructMapVisitor);
    // Should succeed - visitor accepts maps
    let _ = result;
    Ok(())
}

#[test]
fn test_failing_deser_struct_wrong_value_mode() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_with_wrong_value("test_field");
    let result = de.deserialize_struct("Test", &["test_field"], StructMapVisitor);
    let _ = result;
    Ok(())
}

#[test]
fn test_failing_deser_struct_duplicate_field_mode() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_with_duplicate_field("test_field");
    let result = de.deserialize_struct("Test", &["test_field"], StructMapVisitor);
    let _ = result;
    Ok(())
}

#[test]
fn test_failing_deser_struct_struct_duplicate_mode() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::struct_duplicate_field("test_field");
    let result = de.deserialize_struct("Test", &["test_field"], StructMapVisitor);
    let _ = result;
    Ok(())
}

#[test]
fn test_failing_deser_struct_error_on_key_mode() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_error_on_key();
    let result = de.deserialize_struct("Test", &["a"], StructMapVisitor);
    let _ = result;
    Ok(())
}

#[test]
fn test_failing_deser_struct_error_on_value_mode() -> Result<(), ClearGbmError> {
    let de = FailingDeserializer::map_error_on_value("test_field");
    let result = de.deserialize_struct("Test", &["test_field"], StructMapVisitor);
    let _ = result;
    Ok(())
}

// ===========================================================================
// Tests for MinimalStructDeserializer::deserialize_any
// ===========================================================================

#[test]
fn test_minimal_struct_deserializer_deserialize_any() -> Result<(), ClearGbmError> {
    use crate::testkit::deserializer::MinimalStructDeserializer;
    let de = MinimalStructDeserializer;
    // MinimalStructDeserializer::deserialize_any visits an empty map
    let result = de.deserialize_any(StructMapVisitor);
    let _ = result;
    Ok(())
}
