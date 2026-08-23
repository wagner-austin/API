//! Serde error-path tests for tree types: type mismatches that drive
//! every expecting() method, and failing-serializer per-field batteries.

use crate::error::ClearGbmError;
use crate::tree::{Tree, TreeBuildConfig};

// =========================================================================
// Type mismatch tests to trigger expecting() methods
// =========================================================================

#[test]
fn test_tree_build_config_deserialize_from_array() -> Result<(), ClearGbmError> {
    let json = r#"[1, 2, 3]"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_from_string() -> Result<(), ClearGbmError> {
    let json = r#""not a struct""#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_build_config_deserialize_from_number() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_deserialize_from_array() -> Result<(), ClearGbmError> {
    let json = r#"[1, 2, 3]"#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_deserialize_from_string() -> Result<(), ClearGbmError> {
    let json = r#""not a struct""#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_tree_deserialize_from_number() -> Result<(), ClearGbmError> {
    let json = r#"42"#;
    let result: Result<Tree, _> = serde_json::from_str(json);
    assert!(result.is_err());
    Ok(())
}
