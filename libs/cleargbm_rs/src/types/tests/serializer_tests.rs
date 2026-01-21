//! Serialization error path tests using failing serializer.

use crate::error::ClearGbmError;
use crate::types::{HistogramBuffer, SplitConfig, TreeNode, TreeNodeConfig};

// =========================================================================
// TreeNodeConfig serialization error tests
// =========================================================================

#[test]
fn test_tree_node_config_serialize_fail_field_1() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.1_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_after(0);
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_field_2() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.1_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_after(1);
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_field_3() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.1_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_after(2);
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_field_4() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.1_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_after(3);
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_field_5() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.1_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_after(4);
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_field_6() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.1_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_after(5);
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_field_7() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.1_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_after(6);
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_field_8() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.1_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_after(7);
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = TreeNodeConfig {
        node_id: 0_usize,
        feature_index: 1_usize,
        threshold: 0.5_f64,
        value: 0.1_f64,
        n_samples: 10_usize,
        left_child: 1_usize,
        right_child: 2_usize,
        nan_goes_left: true,
    };
    let mut ser = FailingSerializer::fail_on_struct();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// TreeNode serialization error tests
// =========================================================================

#[test]
fn test_tree_node_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let node = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
    let mut ser = FailingSerializer::fail_on_struct();
    let result = node.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let node = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
    // TreeNode has 9 fields
    for fail_at in 0_usize..9_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = node.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

// =========================================================================
// SplitConfig serialization error tests
// =========================================================================

#[test]
fn test_split_config_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    // SplitConfig has 5 fields
    for fail_at in 0_usize..5_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_split_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let mut ser = FailingSerializer::fail_on_struct();
    let result = config.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// HistogramBuffer serialization error tests
// =========================================================================

#[test]
fn test_histogram_buffer_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let hist = HistogramBuffer::new(3_usize);
    // HistogramBuffer has 4 fields
    for fail_at in 0_usize..4_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = hist.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_histogram_buffer_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;
    let hist = HistogramBuffer::new(3_usize);
    let mut ser = FailingSerializer::fail_on_struct();
    let result = hist.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// FailingSerializer coverage tests
// =========================================================================

#[test]
fn test_failing_serializer_coverage() -> Result<(), ClearGbmError> {
    use crate::testkit::{FailingSerializer, SerError};
    use serde::ser::{Error, Serializer};

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
    assert!((&mut ser).serialize_f32(1.0_f32).is_ok());

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

    // Test serialize_struct returns Ok and can be used
    let mut ser = FailingSerializer::fail_after(100);
    let struct_ser = (&mut ser).serialize_struct("S", 1);
    assert!(struct_ser.is_ok());

    Ok(())
}

#[test]
fn test_failing_serializer_struct_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::ser::{SerializeStruct, Serializer};

    let mut ser = FailingSerializer::fail_after(100);
    let struct_ser = match (&mut ser).serialize_struct("Test", 0) {
        Ok(s) => s,
        Err(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "failed to create struct serializer".to_string(),
            })
        }
    };
    // Test end() method
    let result = struct_ser.end();
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_failing_serializer_struct_field_ok_then_fail() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::ser::{SerializeStruct, Serializer};

    // Test that serialize_field returns Ok for first field, then Err
    let mut ser = FailingSerializer::fail_after(1);
    let mut struct_ser = match (&mut ser).serialize_struct("Test", 2) {
        Ok(s) => s,
        Err(_) => {
            return Err(ClearGbmError::TreeConstructionFailed {
                reason: "failed to create struct serializer".to_string(),
            })
        }
    };

    // First field should succeed
    let result1 = struct_ser.serialize_field("field1", &1_u32);
    assert!(result1.is_ok());

    // Second field should fail
    let result2 = struct_ser.serialize_field("field2", &2_u32);
    assert!(result2.is_err());

    Ok(())
}
