//! Manual serde implementation for `TreeNode`.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::types::TreeNode;

impl Serialize for TreeNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("TreeNode", 10) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("node_id", &self.node_id()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("is_leaf", &self.is_leaf()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("feature_index", &self.feature_index()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("threshold", &self.threshold()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("value", &self.value()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("n_samples", &self.n_samples()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("left_child", &self.left_child()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("right_child", &self.right_child()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("nan_goes_left", &self.nan_goes_left()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("categories_goes_left", &self.categories_goes_left()) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `TreeNode` deserialization.
enum TreeNodeField {
    /// The node ID field.
    NodeId,
    /// The is_leaf field.
    IsLeaf,
    /// The feature index field.
    FeatureIndex,
    /// The threshold field.
    Threshold,
    /// The value field.
    Value,
    /// The n_samples field.
    NSamples,
    /// The left child field.
    LeftChild,
    /// The right child field.
    RightChild,
    /// The nan_goes_left field.
    NanGoesLeft,
    /// The categories_goes_left field.
    CategoriesGoesLeft,
}

/// Visitor for deserializing `TreeNodeField` from string.
struct TreeNodeFieldVisitor;

impl<'de> Visitor<'de> for TreeNodeFieldVisitor {
    type Value = TreeNodeField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "node_id" => Ok(TreeNodeField::NodeId),
            "is_leaf" => Ok(TreeNodeField::IsLeaf),
            "feature_index" => Ok(TreeNodeField::FeatureIndex),
            "threshold" => Ok(TreeNodeField::Threshold),
            "value" => Ok(TreeNodeField::Value),
            "n_samples" => Ok(TreeNodeField::NSamples),
            "left_child" => Ok(TreeNodeField::LeftChild),
            "right_child" => Ok(TreeNodeField::RightChild),
            "nan_goes_left" => Ok(TreeNodeField::NanGoesLeft),
            "categories_goes_left" => Ok(TreeNodeField::CategoriesGoesLeft),
            _ => Err(E::unknown_field(value, TREE_NODE_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for TreeNodeField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(TreeNodeFieldVisitor)
    }
}

/// Field names for `TreeNode` serialization.
const TREE_NODE_FIELDS: &[&str] = &[
    "node_id",
    "is_leaf",
    "feature_index",
    "threshold",
    "value",
    "n_samples",
    "left_child",
    "right_child",
    "nan_goes_left",
    "categories_goes_left",
];

impl<'de> Deserialize<'de> for TreeNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TreeNodeVisitor;

        impl<'de> Visitor<'de> for TreeNodeVisitor {
            type Value = TreeNode;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct TreeNode")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TreeNode, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut node_id = None;
                let mut is_leaf = None;
                let mut feature_index = None;
                let mut threshold = None;
                let mut value = None;
                let mut n_samples = None;
                let mut left_child = None;
                let mut right_child = None;
                let mut nan_goes_left = None;
                let mut categories_goes_left: Option<Option<Vec<f64>>> = None;

                loop {
                    let key: Option<TreeNodeField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        TreeNodeField::NodeId => {
                            node_id = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeField::IsLeaf => {
                            is_leaf = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeField::FeatureIndex => {
                            feature_index = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeField::Threshold => {
                            threshold = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeField::Value => {
                            value = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeField::NSamples => {
                            n_samples = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeField::LeftChild => {
                            left_child = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeField::RightChild => {
                            right_child = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeField::NanGoesLeft => {
                            nan_goes_left = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeField::CategoriesGoesLeft => {
                            categories_goes_left = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

                let node_id = match node_id {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("node_id")),
                };
                let is_leaf = match is_leaf {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("is_leaf")),
                };
                let feature_index = match feature_index {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("feature_index")),
                };
                let threshold = match threshold {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("threshold")),
                };
                let value = match value {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("value")),
                };
                let n_samples = match n_samples {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("n_samples")),
                };
                let left_child = match left_child {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("left_child")),
                };
                let right_child = match right_child {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("right_child")),
                };
                let nan_goes_left = match nan_goes_left {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("nan_goes_left")),
                };
                let categories_goes_left = match categories_goes_left {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("categories_goes_left")),
                };

                Ok(TreeNode {
                    node_id,
                    is_leaf,
                    feature_index,
                    threshold,
                    value,
                    n_samples,
                    left_child,
                    right_child,
                    nan_goes_left,
                    categories_goes_left,
                })
            }
        }

        deserializer.deserialize_struct("TreeNode", TREE_NODE_FIELDS, TreeNodeVisitor)
    }
}
