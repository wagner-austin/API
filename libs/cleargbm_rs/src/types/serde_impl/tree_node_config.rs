//! Manual serde implementation for `TreeNodeConfig`.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::types::TreeNodeConfig;

impl Serialize for TreeNodeConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("TreeNodeConfig", 8) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("node_id", &self.node_id) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("feature_index", &self.feature_index) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("threshold", &self.threshold) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("value", &self.value) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("n_samples", &self.n_samples) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("left_child", &self.left_child) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("right_child", &self.right_child) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("nan_goes_left", &self.nan_goes_left) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `TreeNodeConfig` deserialization.
enum TreeNodeConfigField {
    /// The node ID field.
    NodeId,
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
}

/// Visitor for deserializing `TreeNodeConfigField` from string.
struct TreeNodeConfigFieldVisitor;

impl<'de> Visitor<'de> for TreeNodeConfigFieldVisitor {
    type Value = TreeNodeConfigField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "node_id" => Ok(TreeNodeConfigField::NodeId),
            "feature_index" => Ok(TreeNodeConfigField::FeatureIndex),
            "threshold" => Ok(TreeNodeConfigField::Threshold),
            "value" => Ok(TreeNodeConfigField::Value),
            "n_samples" => Ok(TreeNodeConfigField::NSamples),
            "left_child" => Ok(TreeNodeConfigField::LeftChild),
            "right_child" => Ok(TreeNodeConfigField::RightChild),
            "nan_goes_left" => Ok(TreeNodeConfigField::NanGoesLeft),
            _ => Err(E::unknown_field(value, TREE_NODE_CONFIG_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for TreeNodeConfigField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(TreeNodeConfigFieldVisitor)
    }
}

/// Field names for `TreeNodeConfig` serialization.
const TREE_NODE_CONFIG_FIELDS: &[&str] = &[
    "node_id",
    "feature_index",
    "threshold",
    "value",
    "n_samples",
    "left_child",
    "right_child",
    "nan_goes_left",
];

impl<'de> Deserialize<'de> for TreeNodeConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TreeNodeConfigVisitor;

        impl<'de> Visitor<'de> for TreeNodeConfigVisitor {
            type Value = TreeNodeConfig;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct TreeNodeConfig")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TreeNodeConfig, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut node_id = None;
                let mut feature_index = None;
                let mut threshold = None;
                let mut value = None;
                let mut n_samples = None;
                let mut left_child = None;
                let mut right_child = None;
                let mut nan_goes_left = None;

                loop {
                    let key: Option<TreeNodeConfigField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        TreeNodeConfigField::NodeId => {
                            node_id = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeConfigField::FeatureIndex => {
                            feature_index = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeConfigField::Threshold => {
                            threshold = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeConfigField::Value => {
                            value = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeConfigField::NSamples => {
                            n_samples = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeConfigField::LeftChild => {
                            left_child = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeConfigField::RightChild => {
                            right_child = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeNodeConfigField::NanGoesLeft => {
                            nan_goes_left = Some(match map.next_value() {
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

                Ok(TreeNodeConfig {
                    node_id,
                    feature_index,
                    threshold,
                    value,
                    n_samples,
                    left_child,
                    right_child,
                    nan_goes_left,
                })
            }
        }

        deserializer.deserialize_struct(
            "TreeNodeConfig",
            TREE_NODE_CONFIG_FIELDS,
            TreeNodeConfigVisitor,
        )
    }
}
