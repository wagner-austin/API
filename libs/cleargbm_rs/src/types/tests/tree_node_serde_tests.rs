//! Serde error path tests for TreeNode.

use crate::error::ClearGbmError;
use crate::types::TreeNode;

#[test]
fn test_tree_node_deserialize_missing_is_leaf() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_missing_node_id() -> Result<(), ClearGbmError> {
    let json = r#"{"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_missing_feature_index() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_missing_threshold() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_missing_value() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_missing_n_samples() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_missing_left_child() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_missing_right_child() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_missing_nan_goes_left() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_unknown_field() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"bogus":123}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_leaf_all_fields() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let node: TreeNode = match serde_json::from_str(json) {
        Ok(n) => n,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(node.is_leaf());
    assert_eq!(node.node_id(), 0_usize);
    assert!((node.value() - 0.5_f64).abs() < 1e-10_f64);
    assert_eq!(node.n_samples(), 100_usize);
    Ok(())
}

#[test]
fn test_tree_node_deserialize_internal_all_fields() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":false,"feature_index":1,"threshold":0.5,"value":0.0,"n_samples":100,"left_child":1,"right_child":2,"nan_goes_left":true,"categories_goes_left":null}"#;
    let node: TreeNode = match serde_json::from_str(json) {
        Ok(n) => n,
        Err(e) => {
            return Err(ClearGbmError::DeserializationFailed {
                reason: e.to_string(),
            })
        }
    };
    assert!(!node.is_leaf());
    assert_eq!(node.feature_index(), Some(1_usize));
    let threshold = match node.threshold() {
        Some(t) => t,
        None => {
            return Err(ClearGbmError::EmptyInput {
                context: "threshold missing".to_string(),
            })
        }
    };
    assert!((threshold - 0.5_f64).abs() < 1e-10_f64);
    assert_eq!(node.left_child(), Some(1_usize));
    assert_eq!(node.right_child(), Some(2_usize));
    assert!(node.nan_goes_left());
    Ok(())
}

// =========================================================================
// serde_json invalid type tests (covers visit_map error paths)
// =========================================================================

#[test]
fn test_tree_node_serde_json_invalid_node_id_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":"not_a_number","is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_serde_json_invalid_is_leaf_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":"not_a_bool","feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_serde_json_invalid_value_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":"not_a_number","n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_serde_json_invalid_n_samples_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":"not_a_number","left_child":null,"right_child":null,"nan_goes_left":true,"categories_goes_left":null}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_serde_json_invalid_nan_goes_left_type() -> Result<(), ClearGbmError> {
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":"not_a_bool"}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// FailingDeserializer tests for TreeNode
// =========================================================================

#[test]
fn test_tree_node_deserialize_with_integer_key() -> Result<(), ClearGbmError> {
    use crate::testkit::IntegerKeyDeserializer;
    use serde::Deserialize;

    let deser = IntegerKeyDeserializer;
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_value_node_id() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("node_id");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_value_is_leaf() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("is_leaf");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_value_feature_index() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("feature_index");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_value_threshold() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("threshold");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_value_value() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("value");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_value_n_samples() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("n_samples");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_value_left_child() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("left_child");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_value_right_child() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("right_child");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_value_nan_goes_left() -> Result<(), ClearGbmError> {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("nan_goes_left");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_key() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnKeyDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnKeyDeserializer;
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_value_node_id() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("node_id");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_value_is_leaf() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("is_leaf");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_value_feature_index() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("feature_index");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_value_threshold() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("threshold");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_value_value() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("value");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_value_n_samples() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("n_samples");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_value_left_child() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("left_child");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_value_right_child() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("right_child");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_error_on_value_nan_goes_left() -> Result<(), ClearGbmError> {
    use crate::testkit::ErrorOnValueDeserializer;
    use serde::Deserialize;

    let deser = ErrorOnValueDeserializer::new("nan_goes_left");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// Tests to exercise DuplicateFieldMapAccess and StructDuplicateFieldMapAccess
// These types don't check for duplicates, but calling these exercises the code paths.
// =========================================================================

#[test]
fn test_tree_node_deserialize_duplicate_field_map_access() -> Result<(), ClearGbmError> {
    use crate::testkit::DuplicateFieldDeserializer;
    use serde::Deserialize;

    // DuplicateFieldMapAccess returns the same field twice with integer values.
    // TreeNode doesn't check for duplicates, but this fails because only one field is provided.
    let deser = DuplicateFieldDeserializer::new("node_id");
    let result = TreeNode::deserialize(deser);
    // Should fail due to missing other required fields
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_struct_duplicate_field_map_access() -> Result<(), ClearGbmError> {
    use crate::testkit::StructDuplicateFieldDeserializer;
    use serde::Deserialize;

    // StructDuplicateFieldMapAccess returns the same field twice with struct/seq values.
    // TreeNode expects specific field types, so this will fail on type mismatch.
    let deser = StructDuplicateFieldDeserializer::new("node_id");
    let result = TreeNode::deserialize(deser);
    // Should fail due to type mismatch or missing fields
    assert!(result.is_err());
    Ok(())
}

// =========================================================================
// FailingSerializer tests for TreeNode
// =========================================================================

#[test]
fn test_tree_node_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    let mut ser = FailingSerializer::fail_on_struct();
    let result = node.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_serialize_fail_each_field() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    // TreeNode has 9 fields
    for fail_at in 0_usize..9_usize {
        let mut ser = FailingSerializer::fail_after(fail_at);
        let result = node.serialize(&mut ser);
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn test_tree_node_serialize_fail_on_end() -> Result<(), ClearGbmError> {
    use crate::testkit::FailingSerializer;
    use serde::Serialize;

    let node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
    let mut ser = FailingSerializer::fail_on_end();
    let result = node.serialize(&mut ser);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_node_deserialize_wrong_categories_type() {
    use crate::testkit::WrongValueDeserializer;
    use serde::Deserialize;

    let deser = WrongValueDeserializer::new("categories_goes_left");
    let result = TreeNode::deserialize(deser);
    assert!(result.is_err());
}

#[test]
fn test_tree_node_deserialize_missing_categories_field() {
    // Nine complete fields but no categories_goes_left: pre-categorical
    // node payloads are rejected, not silently read as numeric.
    let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true}"#;
    let result: Result<TreeNode, _> = serde_json::from_str(json);
    let err_text = match result {
        Ok(_) => String::new(),
        Err(e) => e.to_string(),
    };
    assert!(
        err_text.contains("categories_goes_left"),
        "a payload without categories_goes_left must be rejected, got: {err_text:?}"
    );
}
