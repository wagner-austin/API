//! Manual serde implementations for tree types.
//!
//! These implementations avoid the `?` operator per project rules.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::types::{SplitConfig, TreeNode};

use super::{Tree, TreeBuildConfig};

// =============================================================================
// TreeBuildConfig Serialization
// =============================================================================

impl Serialize for TreeBuildConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("TreeBuildConfig", 5) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("max_depth", &self.max_depth) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("max_leaves", &self.max_leaves) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("reg_alpha", &self.reg_alpha) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("reg_lambda", &self.reg_lambda) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("split_config", &self.split_config) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `TreeBuildConfig` deserialization.
enum TreeBuildConfigField {
    /// The max depth field.
    MaxDepth,
    /// The max leaves field.
    MaxLeaves,
    /// The regularization alpha field.
    RegAlpha,
    /// The regularization lambda field.
    RegLambda,
    /// The split configuration field.
    SplitConfig,
}

/// Visitor for deserializing `TreeBuildConfigField` from string.
struct TreeBuildConfigFieldVisitor;

impl<'de> Visitor<'de> for TreeBuildConfigFieldVisitor {
    type Value = TreeBuildConfigField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "max_depth" => Ok(TreeBuildConfigField::MaxDepth),
            "max_leaves" => Ok(TreeBuildConfigField::MaxLeaves),
            "reg_alpha" => Ok(TreeBuildConfigField::RegAlpha),
            "reg_lambda" => Ok(TreeBuildConfigField::RegLambda),
            "split_config" => Ok(TreeBuildConfigField::SplitConfig),
            _ => Err(de::Error::unknown_field(value, TREE_BUILD_CONFIG_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for TreeBuildConfigField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(TreeBuildConfigFieldVisitor)
    }
}

/// Field names for TreeBuildConfig serialization.
const TREE_BUILD_CONFIG_FIELDS: &[&str] = &[
    "max_depth",
    "max_leaves",
    "reg_alpha",
    "reg_lambda",
    "split_config",
];

impl<'de> Deserialize<'de> for TreeBuildConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TreeBuildConfigVisitor;

        impl<'de> Visitor<'de> for TreeBuildConfigVisitor {
            type Value = TreeBuildConfig;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct TreeBuildConfig")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut max_depth: Option<usize> = None;
                let mut max_leaves: Option<usize> = None;
                let mut reg_alpha: Option<f64> = None;
                let mut reg_lambda: Option<f64> = None;
                let mut split_config: Option<SplitConfig> = None;

                loop {
                    let key: Option<TreeBuildConfigField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let Some(key) = key else {
                        break;
                    };

                    match key {
                        TreeBuildConfigField::MaxDepth => {
                            if max_depth.is_some() {
                                return Err(de::Error::duplicate_field("max_depth"));
                            }
                            max_depth = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeBuildConfigField::MaxLeaves => {
                            if max_leaves.is_some() {
                                return Err(de::Error::duplicate_field("max_leaves"));
                            }
                            max_leaves = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeBuildConfigField::RegAlpha => {
                            if reg_alpha.is_some() {
                                return Err(de::Error::duplicate_field("reg_alpha"));
                            }
                            reg_alpha = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeBuildConfigField::RegLambda => {
                            if reg_lambda.is_some() {
                                return Err(de::Error::duplicate_field("reg_lambda"));
                            }
                            reg_lambda = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeBuildConfigField::SplitConfig => {
                            if split_config.is_some() {
                                return Err(de::Error::duplicate_field("split_config"));
                            }
                            split_config = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

                let max_depth = match max_depth {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("max_depth")),
                };
                let max_leaves = match max_leaves {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("max_leaves")),
                };
                let reg_alpha = match reg_alpha {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("reg_alpha")),
                };
                let reg_lambda = match reg_lambda {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("reg_lambda")),
                };
                let split_config = match split_config {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("split_config")),
                };

                Ok(TreeBuildConfig {
                    max_depth,
                    max_leaves,
                    reg_alpha,
                    reg_lambda,
                    split_config,
                })
            }
        }

        deserializer.deserialize_struct(
            "TreeBuildConfig",
            TREE_BUILD_CONFIG_FIELDS,
            TreeBuildConfigVisitor,
        )
    }
}

// =============================================================================
// Tree Serialization
// =============================================================================

impl Serialize for Tree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("Tree", 3) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("nodes", &self.nodes) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("max_depth", &self.max_depth) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("n_leaves", &self.n_leaves) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `Tree` deserialization.
enum TreeField {
    /// The nodes field.
    Nodes,
    /// The max depth field.
    MaxDepth,
    /// The number of leaves field.
    NLeaves,
}

/// Visitor for deserializing `TreeField` from string.
struct TreeFieldVisitor;

impl<'de> Visitor<'de> for TreeFieldVisitor {
    type Value = TreeField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "nodes" => Ok(TreeField::Nodes),
            "max_depth" => Ok(TreeField::MaxDepth),
            "n_leaves" => Ok(TreeField::NLeaves),
            _ => Err(de::Error::unknown_field(value, TREE_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for TreeField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(TreeFieldVisitor)
    }
}

/// Field names for Tree serialization.
const TREE_FIELDS: &[&str] = &["nodes", "max_depth", "n_leaves"];

impl<'de> Deserialize<'de> for Tree {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TreeVisitor;

        impl<'de> Visitor<'de> for TreeVisitor {
            type Value = Tree;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct Tree")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut nodes: Option<Vec<TreeNode>> = None;
                let mut max_depth: Option<usize> = None;
                let mut n_leaves: Option<usize> = None;

                loop {
                    let key: Option<TreeField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let Some(key) = key else {
                        break;
                    };

                    match key {
                        TreeField::Nodes => {
                            if nodes.is_some() {
                                return Err(de::Error::duplicate_field("nodes"));
                            }
                            nodes = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeField::MaxDepth => {
                            if max_depth.is_some() {
                                return Err(de::Error::duplicate_field("max_depth"));
                            }
                            max_depth = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeField::NLeaves => {
                            if n_leaves.is_some() {
                                return Err(de::Error::duplicate_field("n_leaves"));
                            }
                            n_leaves = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

                let nodes = match nodes {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("nodes")),
                };
                let max_depth = match max_depth {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("max_depth")),
                };
                let n_leaves = match n_leaves {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("n_leaves")),
                };

                Ok(Tree {
                    nodes,
                    max_depth,
                    n_leaves,
                })
            }
        }

        deserializer.deserialize_struct("Tree", TREE_FIELDS, TreeVisitor)
    }
}
