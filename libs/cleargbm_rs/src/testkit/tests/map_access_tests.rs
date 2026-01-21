//! Tests for MapAccess implementations and related deserializers.

use crate::error::ClearGbmError;
use crate::testkit::deserializer::{
    AllFieldsMapAccess, DuplicateFieldMapAccess, EmptyMapAccess, EmptySeqAccess,
    ErrorOnKeyMapAccess, ErrorOnValueMapAccess, IntegerKeyMapAccess, MinimalStructDeserializer,
    MinimalValueDeserializer, StructDuplicateFieldMapAccess, WrongValueMapAccess,
};
use core::fmt;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::Deserializer;

// =============================================================================
// IntegerKeyMapAccess tests
// =============================================================================

#[test]
fn test_integer_key_map_access_returns_none_when_done() -> Result<(), ClearGbmError> {
    let mut access = IntegerKeyMapAccess { done: true };
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Ok("test".to_string())
        }
    }
    let result = access.next_key_seed(StringSeed);
    assert!(matches!(result, Ok(None)));
    Ok(())
}

#[test]
fn test_integer_key_map_access_next_value() -> Result<(), ClearGbmError> {
    let mut access = IntegerKeyMapAccess { done: false };
    struct I64Seed;
    impl<'de> de::DeserializeSeed<'de> for I64Seed {
        type Value = i64;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = i64;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "i64")
                }
                fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
                    Ok(v)
                }
            }
            de.deserialize_any(V)
        }
    }
    let result = access.next_value_seed(I64Seed);
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
fn test_integer_key_map_access_returns_integer_key() -> Result<(), ClearGbmError> {
    let mut access = IntegerKeyMapAccess { done: false };
    struct I64Seed;
    impl<'de> de::DeserializeSeed<'de> for I64Seed {
        type Value = i64;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = i64;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "i64")
                }
                fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
                    Ok(v)
                }
            }
            de.deserialize_any(V)
        }
    }
    let result = access.next_key_seed(I64Seed);
    match result {
        Ok(Some(v)) => assert_eq!(v, 42_i64),
        _ => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok(Some)".to_string(),
            })
        }
    }
    Ok(())
}

// =============================================================================
// WrongValueMapAccess tests
// =============================================================================

#[test]
fn test_wrong_value_map_access_returns_field_name() -> Result<(), ClearGbmError> {
    let mut access = WrongValueMapAccess::new("my_field");
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
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
            de.deserialize_identifier(V)
        }
    }
    let result = access.next_key_seed(StringSeed);
    match result {
        Ok(Some(v)) => assert_eq!(v, "my_field"),
        _ => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok(Some)".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_wrong_value_map_access_returns_none_when_done() -> Result<(), ClearGbmError> {
    let mut access = WrongValueMapAccess::new("field");
    access.returned_key = true;
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Ok("test".to_string())
        }
    }
    let result = access.next_key_seed(StringSeed);
    assert!(matches!(result, Ok(None)));
    Ok(())
}

#[test]
fn test_wrong_value_map_access_next_value_returns_string() -> Result<(), ClearGbmError> {
    let mut access = WrongValueMapAccess::new("field");
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
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
            de.deserialize_any(V)
        }
    }
    let result = access.next_value_seed(StringSeed);
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

// =============================================================================
// DuplicateFieldMapAccess tests
// =============================================================================

#[test]
fn test_duplicate_field_map_access_returns_none_after_two_keys() -> Result<(), ClearGbmError> {
    let mut access = DuplicateFieldMapAccess::new("field");
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
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
            de.deserialize_identifier(V)
        }
    }
    // First key
    let result1 = access.next_key_seed(StringSeed);
    assert!(matches!(result1, Ok(Some(_))));
    // Second key
    let result2 = access.next_key_seed(StringSeed);
    assert!(matches!(result2, Ok(Some(_))));
    // Third call should return None
    let result3 = access.next_key_seed(StringSeed);
    assert!(matches!(result3, Ok(None)));
    Ok(())
}

#[test]
fn test_duplicate_field_map_access_next_value() -> Result<(), ClearGbmError> {
    let mut access = DuplicateFieldMapAccess::new("field");
    struct I64Seed;
    impl<'de> de::DeserializeSeed<'de> for I64Seed {
        type Value = i64;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = i64;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "i64")
                }
                fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
                    Ok(v)
                }
            }
            de.deserialize_any(V)
        }
    }
    let result = access.next_value_seed(I64Seed);
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
fn test_duplicate_field_map_access_key_error_path() -> Result<(), ClearGbmError> {
    let mut access = DuplicateFieldMapAccess::new("field");
    struct FailingSeed;
    impl<'de> de::DeserializeSeed<'de> for FailingSeed {
        type Value = ();
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Err(de::Error::custom("intentional failure"))
        }
    }
    let result = access.next_key_seed(FailingSeed);
    assert!(result.is_err());
    Ok(())
}

// =============================================================================
// StructDuplicateFieldMapAccess tests
// =============================================================================

#[test]
fn test_struct_duplicate_field_map_access_returns_two_keys() -> Result<(), ClearGbmError> {
    let mut access = StructDuplicateFieldMapAccess::new("field");
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
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
            de.deserialize_identifier(V)
        }
    }
    // First key
    let result1 = access.next_key_seed(StringSeed);
    assert!(matches!(result1, Ok(Some(_))));
    // Second key
    let result2 = access.next_key_seed(StringSeed);
    assert!(matches!(result2, Ok(Some(_))));
    // Third call should return None
    let result3 = access.next_key_seed(StringSeed);
    assert!(matches!(result3, Ok(None)));
    Ok(())
}

#[test]
fn test_struct_duplicate_field_map_access_key_error() -> Result<(), ClearGbmError> {
    let mut access = StructDuplicateFieldMapAccess::new("field");
    struct FailingSeed;
    impl<'de> de::DeserializeSeed<'de> for FailingSeed {
        type Value = ();
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Err(de::Error::custom("intentional failure"))
        }
    }
    let result = access.next_key_seed(FailingSeed);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_struct_duplicate_field_map_access_value() -> Result<(), ClearGbmError> {
    let mut access = StructDuplicateFieldMapAccess::new("field");
    struct StructSeed;
    impl<'de> de::DeserializeSeed<'de> for StructSeed {
        type Value = ();
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = ();
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "struct")
                }
                fn visit_map<A>(self, _map: A) -> Result<Self::Value, A::Error>
                where
                    A: de::MapAccess<'de>,
                {
                    Ok(())
                }
            }
            de.deserialize_struct("Test", &[], V)
        }
    }
    let result = access.next_value_seed(StructSeed);
    assert!(result.is_ok());
    Ok(())
}

// =============================================================================
// MinimalStructDeserializer tests
// =============================================================================

#[test]
fn test_minimal_struct_deserializer_any() -> Result<(), ClearGbmError> {
    struct MapVisitor;
    impl<'de> Visitor<'de> for MapVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "map")
        }
        fn visit_map<A>(self, _map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            Ok(())
        }
    }
    let de = MinimalStructDeserializer;
    let result = de.deserialize_any(MapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_struct() -> Result<(), ClearGbmError> {
    struct MapVisitor;
    impl<'de> Visitor<'de> for MapVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "map")
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
                let _: i64 = match map.next_value() {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                };
            }
            Ok(())
        }
    }
    let de = MinimalStructDeserializer;
    let result = de.deserialize_struct("Test", &["a", "b"], MapVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_struct_deserializer_seq() -> Result<(), ClearGbmError> {
    struct SeqVisitor;
    impl<'de> Visitor<'de> for SeqVisitor {
        type Value = Vec<i64>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "seq")
        }
        fn visit_seq<A>(self, _seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            Ok(vec![])
        }
    }
    let de = MinimalStructDeserializer;
    let result = de.deserialize_seq(SeqVisitor);
    assert!(result.is_ok());
    Ok(())
}

// =============================================================================
// MinimalValueDeserializer tests
// =============================================================================

#[test]
fn test_minimal_value_deserializer_struct() -> Result<(), ClearGbmError> {
    struct NestedStructVisitor;
    impl<'de> Visitor<'de> for NestedStructVisitor {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "nested struct")
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
                let Some(_k) = key else {
                    break;
                };
                let _: i64 = match map.next_value() {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                };
            }
            Ok(())
        }
    }
    let de = MinimalValueDeserializer;
    let result = de.deserialize_struct("Nested", &["field_a", "field_b"], NestedStructVisitor);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_minimal_value_deserializer_seq() -> Result<(), ClearGbmError> {
    struct SeqVisitor;
    impl<'de> Visitor<'de> for SeqVisitor {
        type Value = Vec<i64>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "sequence")
        }
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut result = Vec::new();
            loop {
                let elem: Option<i64> = match seq.next_element() {
                    Ok(e) => e,
                    Err(e) => return Err(e),
                };
                let Some(v) = elem else {
                    break;
                };
                result.push(v);
            }
            Ok(result)
        }
    }
    let de = MinimalValueDeserializer;
    let result = de.deserialize_seq(SeqVisitor);
    match result {
        Ok(v) => assert!(v.is_empty()),
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok".to_string(),
            })
        }
    }
    Ok(())
}

// =============================================================================
// EmptyMapAccess tests
// =============================================================================

#[test]
fn test_empty_map_access_next_key_seed() -> Result<(), ClearGbmError> {
    let mut access = EmptyMapAccess;
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Ok("unused".to_string())
        }
    }
    let result = access.next_key_seed(StringSeed);
    assert!(matches!(result, Ok(None)));
    Ok(())
}

#[test]
fn test_empty_map_access_next_value_seed() -> Result<(), ClearGbmError> {
    let mut access = EmptyMapAccess;
    struct I64Seed;
    impl<'de> de::DeserializeSeed<'de> for I64Seed {
        type Value = i64;
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Ok(0_i64)
        }
    }
    let result = access.next_value_seed(I64Seed);
    assert!(result.is_err());
    Ok(())
}

// =============================================================================
// EmptySeqAccess tests
// =============================================================================

#[test]
fn test_empty_seq_access_next_element_seed() -> Result<(), ClearGbmError> {
    let mut access = EmptySeqAccess;
    struct I64Seed;
    impl<'de> de::DeserializeSeed<'de> for I64Seed {
        type Value = i64;
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Ok(0_i64)
        }
    }
    let result = access.next_element_seed(I64Seed);
    assert!(matches!(result, Ok(None)));
    Ok(())
}

// =============================================================================
// AllFieldsMapAccess tests
// =============================================================================

#[test]
fn test_all_fields_map_access_key_error_path() -> Result<(), ClearGbmError> {
    let mut access = AllFieldsMapAccess::new(&["field_a"]);
    struct FailingSeed;
    impl<'de> de::DeserializeSeed<'de> for FailingSeed {
        type Value = ();
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Err(de::Error::custom("intentional failure"))
        }
    }
    let result = access.next_key_seed(FailingSeed);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_all_fields_map_access_returns_all_fields() -> Result<(), ClearGbmError> {
    let mut access = AllFieldsMapAccess::new(&["field_a", "field_b"]);
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
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
            de.deserialize_identifier(V)
        }
    }
    // First field
    let result1 = access.next_key_seed(StringSeed);
    match result1 {
        Ok(Some(v)) => assert_eq!(v, "field_a"),
        _ => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok(Some)".to_string(),
            })
        }
    }
    // Second field
    let result2 = access.next_key_seed(StringSeed);
    match result2 {
        Ok(Some(v)) => assert_eq!(v, "field_b"),
        _ => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok(Some)".to_string(),
            })
        }
    }
    // Third call should return None
    let result3 = access.next_key_seed(StringSeed);
    assert!(matches!(result3, Ok(None)));
    Ok(())
}

// =============================================================================
// ErrorOnKeyMapAccess tests
// =============================================================================

#[test]
fn test_error_on_key_map_access_next_key_seed() -> Result<(), ClearGbmError> {
    let mut access = ErrorOnKeyMapAccess;
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Ok("unused".to_string())
        }
    }
    let result = access.next_key_seed(StringSeed);
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => String::new(),
    };
    assert!(err_msg.contains("next_key"));
    Ok(())
}

#[test]
fn test_error_on_key_map_access_next_value_seed() -> Result<(), ClearGbmError> {
    let mut access = ErrorOnKeyMapAccess;
    struct I64Seed;
    impl<'de> de::DeserializeSeed<'de> for I64Seed {
        type Value = i64;
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Ok(0_i64)
        }
    }
    let result = access.next_value_seed(I64Seed);
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => String::new(),
    };
    assert!(err_msg.contains("no value"));
    Ok(())
}

// =============================================================================
// ErrorOnValueMapAccess tests
// =============================================================================

#[test]
fn test_error_on_value_map_access_next_key_seed() -> Result<(), ClearGbmError> {
    let mut access = ErrorOnValueMapAccess::new("test_field");
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
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
            de.deserialize_identifier(V)
        }
    }
    let result = access.next_key_seed(StringSeed);
    match result {
        Ok(Some(v)) => assert_eq!(v, "test_field"),
        _ => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected Ok(Some)".to_string(),
            })
        }
    }
    Ok(())
}

#[test]
fn test_error_on_value_map_access_next_key_seed_returns_none() -> Result<(), ClearGbmError> {
    let mut access = ErrorOnValueMapAccess::new("field");
    struct StringSeed;
    impl<'de> de::DeserializeSeed<'de> for StringSeed {
        type Value = String;
        fn deserialize<D>(self, de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
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
            de.deserialize_identifier(V)
        }
    }
    // First call returns the field
    let _ = access.next_key_seed(StringSeed);
    // Second call should return None
    let result = access.next_key_seed(StringSeed);
    assert!(matches!(result, Ok(None)));
    Ok(())
}

#[test]
fn test_error_on_value_map_access_next_value_seed() -> Result<(), ClearGbmError> {
    let mut access = ErrorOnValueMapAccess::new("field");
    struct I64Seed;
    impl<'de> de::DeserializeSeed<'de> for I64Seed {
        type Value = i64;
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Ok(0_i64)
        }
    }
    let result = access.next_value_seed(I64Seed);
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => String::new(),
    };
    assert!(err_msg.contains("next_value"));
    Ok(())
}

#[test]
fn test_error_on_value_map_access_key_error_path() -> Result<(), ClearGbmError> {
    let mut access = ErrorOnValueMapAccess::new("field");
    struct FailingSeed;
    impl<'de> de::DeserializeSeed<'de> for FailingSeed {
        type Value = ();
        fn deserialize<D>(self, _de: D) -> Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Err(de::Error::custom("intentional failure"))
        }
    }
    let result = access.next_key_seed(FailingSeed);
    assert!(result.is_err());
    Ok(())
}
