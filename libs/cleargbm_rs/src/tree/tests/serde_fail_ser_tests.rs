//! Tree serialization error paths: the failing-serializer battery that
//! drives every per-field error arm.

use crate::error::ClearGbmError;
use crate::tree::{Tree, TreeBuildConfig};
use crate::types::{SplitConfig, TreeNode};

// =========================================================================
// Serialization error path tests using failing serializer
// =========================================================================

#[test]
fn test_tree_build_config_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let split_config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = match TreeBuildConfig::new(6_usize, 8_usize, 0.1_f64, 1.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    // TreeBuildConfig has 5 fields
    for fail_at in 0_usize..5_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_tree_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
    // Tree has 3 fields
    for fail_at in 0_usize..3_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = tree.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_tree_build_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let split_config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = match TreeBuildConfig::new(6_usize, 8_usize, 0.0_f64, 1.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_struct();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
    let mut ser = FailingSerializer::fail_on_struct();
    let result = tree.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let split_config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let config = match TreeBuildConfig::new(6_usize, 8_usize, 0.0_f64, 1.0_f64, split_config) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_end();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
    let mut ser = FailingSerializer::fail_on_end();
    let result = tree.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_failing_serializer_coverage() -> Result<(), ClearGbmError> {
    use crate::testkit::{FailingSerializer, SerError};
    use serde::ser::{Error, SerializeStruct, Serializer};

    // Test SerError Display
    let err = SerError {
        message: "test".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("test"));

    // Test SerError custom
    let custom_err = SerError::custom("custom error");
    assert!(custom_err.message.contains("custom"));

    // Test all serializer primitive methods
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_bool(true).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_i8(1_i8).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_i16(1_i16).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_i32(1_i32).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_i64(1_i64).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_u8(1_u8).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_u16(1_u16).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_u32(1_u32).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_u64(1_u64).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_f64(1.0_f64).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_f64(1.0_f64).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_char('a').is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_str("test").is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_bytes(&[1_u8, 2_u8]).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_none().is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_some(&1_u32).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_unit().is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_unit_struct("Unit").is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_unit_variant("E", 0, "V").is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_newtype_struct("N", &1_u32).is_ok());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser)
        .serialize_newtype_variant("E", 0, "V", &1_u32)
        .is_ok());

    // Test error methods
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_seq(Some(1)).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_tuple(1).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_tuple_struct("T", 1).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_tuple_variant("E", 0, "V", 1).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_map(Some(1)).is_err());
    let mut ser = FailingSerializer::fail_after(100);
    assert!((&mut ser).serialize_struct_variant("E", 0, "V", 1).is_err());

    // Test serialize_struct
    let mut ser = FailingSerializer::fail_after(100);
    let struct_ser = (&mut ser).serialize_struct("S", 1);
    assert!(struct_ser.is_ok());

    // Test struct end
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

    // Test struct serialize_field Ok then Err
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

// =========================================================================
// Failing deserializer tests
// =========================================================================

#[test]
fn test_tree_build_config_field_expecting() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(IntegerKeyDeserializer);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_field_expecting() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(IntegerKeyDeserializer);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Wrong value tests for TreeBuildConfig
// =========================================================================

#[test]
fn test_tree_build_config_wrong_value_max_depth() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("max_depth"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_wrong_value_max_leaves() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("max_leaves"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_wrong_value_reg_alpha() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("reg_alpha"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_wrong_value_reg_lambda() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("reg_lambda"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_wrong_value_split_config() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = TreeBuildConfig::deserialize(WrongValueDeserializer::new("split_config"));
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Wrong value tests for Tree
// =========================================================================

#[test]
fn test_tree_wrong_value_n_features() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(WrongValueDeserializer::new("n_features"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_wrong_value_nodes() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;
    let result = Tree::deserialize(WrongValueDeserializer::new("nodes"));
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_failing_deserializer_coverage() -> Result<(), ClearGbmError> {
    use crate::testkit::{DeError, IntegerDeserializer, IntegerKeyMapAccess};
    use serde::de::{Deserializer, Error, MapAccess};

    // Test DeError Display
    let err = DeError {
        message: "test".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("test"));

    // Test DeError custom
    let custom_err = DeError::custom("custom");
    assert!(custom_err.message.contains("custom"));

    // Test IntegerDeserializer
    struct I64Visitor;
    impl<'de> serde::de::Visitor<'de> for I64Visitor {
        type Value = i64;
        fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "i64")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }
    let de = IntegerDeserializer;
    let result = de.deserialize_any(I64Visitor);
    assert!(result.is_ok());

    // Test IntegerKeyMapAccess done state
    let mut map_access = IntegerKeyMapAccess { done: true };
    let key_result: Result<Option<String>, _> = map_access.next_key();
    assert!(matches!(key_result, Ok(None)));

    // Test IntegerKeyMapAccess next_value (returns integer successfully)
    let mut map_access2 = IntegerKeyMapAccess { done: false };
    let value_result: Result<i64, _> = map_access2.next_value();
    assert!(value_result.is_ok());

    Ok(())
}
