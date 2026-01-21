//! Type mismatch tests and failing deserializer tests.

use crate::error::ClearGbmError;
use crate::types::{HistogramBuffer, SplitConfig, TreeNode, TreeNodeConfig};

// =========================================================================
// Type mismatch tests to trigger expecting() methods
// =========================================================================

#[test]
fn test_tree_node_config_deserialize_from_array() -> Result<(), ClearGbmError> {
    let json = r#"[1, 2, 3]"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_from_string() -> Result<(), ClearGbmError> {
    let json = r#""not a struct""#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_deserialize_from_number() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_from_array() -> Result<(), ClearGbmError> {
    let json = r#"[1, 2, 3]"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_from_string() -> Result<(), ClearGbmError> {
    let json = r#""not a struct""#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_from_number() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_from_array() -> Result<(), ClearGbmError> {
    let json = r#"[1, 2, 3]"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_from_string() -> Result<(), ClearGbmError> {
    let json = r#""not a struct""#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_deserialize_from_number() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<SplitConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_from_array() -> Result<(), ClearGbmError> {
    let json = r#"[1, 2, 3]"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_from_string() -> Result<(), ClearGbmError> {
    let json = r#""not a struct""#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_deserialize_from_number() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests using WrongTypeDeserializer to trigger expecting() methods
// =========================================================================

#[test]
fn test_tree_node_config_expecting_via_wrong_type() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerDeserializer;
    use serde::Deserialize;
    let de = IntegerDeserializer;
    let result = TreeNodeConfig::deserialize(de);
    let err = match result {
        Ok(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected error but got success".to_string(),
            })
        }
        Err(e) => e,
    };
    let err_msg = format!("{}", err);
    assert!(err_msg.contains("field identifier") || err_msg.contains("invalid type"));
    Ok(())
}

#[test]
fn test_tree_node_expecting_via_wrong_type() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerDeserializer;
    use serde::Deserialize;
    let de = IntegerDeserializer;
    let result = TreeNode::deserialize(de);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_expecting_via_wrong_type() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerDeserializer;
    use serde::Deserialize;
    let de = IntegerDeserializer;
    let result = SplitConfig::deserialize(de);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_expecting_via_wrong_type() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerDeserializer;
    use serde::Deserialize;
    let de = IntegerDeserializer;
    let result = HistogramBuffer::deserialize(de);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests using map_with_integer_key to trigger field visitor expecting()
// =========================================================================

#[test]
fn test_tree_node_config_field_expecting() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;
    let de = IntegerKeyDeserializer;
    let result = TreeNodeConfig::deserialize(de);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_field_expecting() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;
    let de = IntegerKeyDeserializer;
    let result = TreeNode::deserialize(de);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_split_config_field_expecting() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;
    let de = IntegerKeyDeserializer;
    let result = SplitConfig::deserialize(de);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_field_expecting() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;
    let de = IntegerKeyDeserializer;
    let result = HistogramBuffer::deserialize(de);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests to trigger next_value error branches
// =========================================================================

#[test]
fn test_tree_node_config_next_value_error() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    // Test each field
    for field in &[
        "node_id",
        "feature_index",
        "threshold",
        "value",
        "n_samples",
        "left_child",
        "right_child",
        "nan_goes_left",
    ] {
        let de = WrongValueDeserializer::new(field);
        let result = TreeNodeConfig::deserialize(de);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_tree_node_next_value_error() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    for field in &[
        "node_id",
        "is_leaf",
        "feature_index",
        "threshold",
        "value",
        "n_samples",
        "left_child",
        "right_child",
        "nan_goes_left",
    ] {
        let de = WrongValueDeserializer::new(field);
        let result = TreeNode::deserialize(de);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_split_config_next_value_error() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    for field in &[
        "min_samples_split",
        "min_samples_leaf",
        "max_bins",
        "min_gain",
        "reg_lambda",
    ] {
        let de = WrongValueDeserializer::new(field);
        let result = SplitConfig::deserialize(de);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_histogram_buffer_next_value_error() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    for field in &["n_bins", "gradient_sums", "hessian_sums", "counts"] {
        let de = WrongValueDeserializer::new(field);
        let result = HistogramBuffer::deserialize(de);
        assert!(result.is_err());
    }
    Ok(())
}

// =========================================================================
// Failing deserializer coverage tests
// =========================================================================

#[test]
fn test_failing_deserializer_coverage() -> Result<(), ClearGbmError> {
    use crate::testkit::{DeError, IntegerDeserializer, IntegerKeyDeserializer};
    use serde::de::{Deserializer, Error, Visitor};

    // Test DeError Display
    let err = DeError {
        message: "test error".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("test error"));

    // Test DeError custom
    let custom_err = DeError::custom("custom message");
    assert!(custom_err.message.contains("custom"));

    // Test all deserialize methods - they all delegate to deserialize_any
    // which calls visitor.visit_i64, triggering expecting() on most visitors

    // Create a simple visitor that accepts i64
    struct I64Visitor;
    impl<'de> Visitor<'de> for I64Visitor {
        type Value = i64;
        fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }

    // Test deserialize_any
    let de = IntegerDeserializer;
    let result = de.deserialize_any(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_bool
    let de = IntegerDeserializer;
    let result = de.deserialize_bool(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_i8
    let de = IntegerDeserializer;
    let result = de.deserialize_i8(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_i16
    let de = IntegerDeserializer;
    let result = de.deserialize_i16(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_i32
    let de = IntegerDeserializer;
    let result = de.deserialize_i32(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_i64
    let de = IntegerDeserializer;
    let result = de.deserialize_i64(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_u8
    let de = IntegerDeserializer;
    let result = de.deserialize_u8(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_u16
    let de = IntegerDeserializer;
    let result = de.deserialize_u16(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_u32
    let de = IntegerDeserializer;
    let result = de.deserialize_u32(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_u64
    let de = IntegerDeserializer;
    let result = de.deserialize_u64(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_f32
    let de = IntegerDeserializer;
    let result = de.deserialize_f32(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_f64
    let de = IntegerDeserializer;
    let result = de.deserialize_f64(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_char
    let de = IntegerDeserializer;
    let result = de.deserialize_char(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_str
    let de = IntegerDeserializer;
    let result = de.deserialize_str(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_string
    let de = IntegerDeserializer;
    let result = de.deserialize_string(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_bytes
    let de = IntegerDeserializer;
    let result = de.deserialize_bytes(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_byte_buf
    let de = IntegerDeserializer;
    let result = de.deserialize_byte_buf(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_option
    let de = IntegerDeserializer;
    let result = de.deserialize_option(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_unit
    let de = IntegerDeserializer;
    let result = de.deserialize_unit(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_unit_struct
    let de = IntegerDeserializer;
    let result = de.deserialize_unit_struct("Test", I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_newtype_struct
    let de = IntegerDeserializer;
    let result = de.deserialize_newtype_struct("Test", I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_seq
    let de = IntegerDeserializer;
    let result = de.deserialize_seq(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_tuple
    let de = IntegerDeserializer;
    let result = de.deserialize_tuple(2, I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_tuple_struct
    let de = IntegerDeserializer;
    let result = de.deserialize_tuple_struct("Test", 2, I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_map
    let de = IntegerDeserializer;
    let result = de.deserialize_map(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_struct
    let de = IntegerDeserializer;
    let result = de.deserialize_struct("Test", &["field"], I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_enum
    let de = IntegerDeserializer;
    let result = de.deserialize_enum("Test", &["Variant"], I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_identifier
    let de = IntegerDeserializer;
    let result = de.deserialize_identifier(I64Visitor);
    assert!(result.is_ok());

    // Test deserialize_ignored_any
    let de = IntegerDeserializer;
    let result = de.deserialize_ignored_any(I64Visitor);
    assert!(result.is_ok());

    // Test map_with_integer_key mode
    struct MapVisitor;
    impl<'de> Visitor<'de> for MapVisitor {
        type Value = ();
        fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "map")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            // Try to get key which will fail because it's an integer
            let _key: Result<Option<String>, _> = map.next_key();
            Ok(())
        }
    }
    let de = IntegerKeyDeserializer;
    let result = de.deserialize_any(MapVisitor);
    assert!(result.is_ok());

    // Test IntegerKeyMapAccess done state
    use crate::testkit::IntegerKeyMapAccess;
    use serde::de::MapAccess;
    let mut map_access = IntegerKeyMapAccess { done: true };
    let key_result: Result<Option<String>, _> = map_access.next_key();
    assert!(key_result.is_ok());
    assert!(matches!(key_result, Ok(None)));

    // Test IntegerKeyMapAccess next_value (returns integer successfully)
    let mut map_access2 = IntegerKeyMapAccess { done: false };
    let value_result: Result<i64, _> = map_access2.next_value();
    assert!(value_result.is_ok());

    Ok(())
}

// =========================================================================
// Direct expecting() method tests
// =========================================================================

#[test]
fn test_direct_visitor_error_generation() -> Result<(), ClearGbmError> {
    use serde::de::Visitor;
    // Test that visit_i64 generates an error with expecting() message
    struct StringVisitor;
    impl<'de> Visitor<'de> for StringVisitor {
        type Value = String;
        fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter.write_str("a string")
        }
    }
    // The default visit_i64 impl should call expecting() when generating error
    let result: Result<String, serde_json::Error> = StringVisitor.visit_i64(42_i64);
    assert!(result.is_err());
    let err_msg = match result {
        Ok(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "test".to_string(),
                reason: "expected error".to_string(),
            })
        }
        Err(e) => format!("{}", e),
    };
    assert!(err_msg.contains("string") || err_msg.contains("expected"));
    Ok(())
}
