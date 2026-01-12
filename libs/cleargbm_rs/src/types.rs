//! Core data structures for `ClearGBM`.
//!
//! All types are immutable after construction. Use builder patterns
//! or constructor functions to create instances.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::ClearGbmError;

/// Configuration for creating an internal (split) tree node.
///
/// Used to avoid having too many function arguments while maintaining
/// explicit, named parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeNodeConfig {
    /// Unique identifier for this node within the tree.
    pub node_id: usize,
    /// Index of feature used for split.
    pub feature_index: usize,
    /// Split threshold value.
    pub threshold: f64,
    /// Node value (sum of gradients / sum of hessians).
    pub value: f64,
    /// Number of samples at this node.
    pub n_samples: usize,
    /// ID of left child node.
    pub left_child: usize,
    /// ID of right child node.
    pub right_child: usize,
    /// Whether NaN values go to left child.
    pub nan_goes_left: bool,
}

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

/// A node in a gradient boosted decision tree.
///
/// Nodes are either internal (with split information) or leaf (with prediction value).
/// This is equivalent to the Python `TreeNode` `TypedDict`.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    /// Unique identifier for this node within the tree.
    node_id: usize,

    /// Whether this is a leaf node (no children).
    is_leaf: bool,

    /// Feature index used for splitting (None for leaf nodes).
    feature_index: Option<usize>,

    /// Threshold value for the split (None for leaf nodes).
    threshold: Option<f64>,

    /// Prediction value (leaf value or intermediate sum).
    value: f64,

    /// Number of training samples that reached this node.
    n_samples: usize,

    /// Left child node ID (None for leaf nodes).
    left_child: Option<usize>,

    /// Right child node ID (None for leaf nodes).
    right_child: Option<usize>,

    /// Direction for NaN values: true = left, false = right.
    nan_goes_left: bool,
}

impl Serialize for TreeNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("TreeNode", 9) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("node_id", &self.node_id) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("is_leaf", &self.is_leaf) {
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
                })
            }
        }

        deserializer.deserialize_struct("TreeNode", TREE_NODE_FIELDS, TreeNodeVisitor)
    }
}

impl TreeNode {
    /// Creates a new leaf node.
    ///
    /// # Args
    ///
    /// * `node_id` - Unique identifier for this node.
    /// * `value` - Prediction value for this leaf.
    /// * `n_samples` - Number of samples that reached this leaf.
    ///
    /// # Returns
    ///
    /// A new leaf `TreeNode`.
    #[must_use]
    pub const fn new_leaf(node_id: usize, value: f64, n_samples: usize) -> Self {
        Self {
            node_id,
            is_leaf: true,
            feature_index: None,
            threshold: None,
            value,
            n_samples,
            left_child: None,
            right_child: None,
            nan_goes_left: true,
        }
    }

    /// Creates a new internal (split) node from configuration.
    ///
    /// # Args
    ///
    /// * `config` - Configuration containing all node parameters.
    ///
    /// # Returns
    ///
    /// A new internal `TreeNode`.
    #[must_use]
    pub const fn new_internal(config: TreeNodeConfig) -> Self {
        Self {
            node_id: config.node_id,
            is_leaf: false,
            feature_index: Some(config.feature_index),
            threshold: Some(config.threshold),
            value: config.value,
            n_samples: config.n_samples,
            left_child: Some(config.left_child),
            right_child: Some(config.right_child),
            nan_goes_left: config.nan_goes_left,
        }
    }

    /// Returns the node ID.
    #[must_use]
    pub const fn node_id(&self) -> usize {
        self.node_id
    }

    /// Returns whether this is a leaf node.
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    /// Returns the feature index (None for leaves).
    #[must_use]
    pub const fn feature_index(&self) -> Option<usize> {
        self.feature_index
    }

    /// Returns the threshold (None for leaves).
    #[must_use]
    pub const fn threshold(&self) -> Option<f64> {
        self.threshold
    }

    /// Returns the node value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Returns the sample count.
    #[must_use]
    pub const fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Returns the left child ID (None for leaves).
    #[must_use]
    pub const fn left_child(&self) -> Option<usize> {
        self.left_child
    }

    /// Returns the right child ID (None for leaves).
    #[must_use]
    pub const fn right_child(&self) -> Option<usize> {
        self.right_child
    }

    /// Returns whether NaN values go left.
    #[must_use]
    pub const fn nan_goes_left(&self) -> bool {
        self.nan_goes_left
    }
}

/// Histogram buffer for gradient/hessian accumulation.
///
/// Used during split finding to accumulate statistics per bin.
/// Equivalent to Python `HistogramBuffer` but with explicit sizing.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBuffer {
    /// Sum of gradients per bin.
    gradient_sums: Vec<f64>,

    /// Sum of hessians per bin.
    hessian_sums: Vec<f64>,

    /// Count of samples per bin.
    counts: Vec<usize>,

    /// Number of bins (fixed at construction).
    n_bins: usize,
}

impl HistogramBuffer {
    /// Creates a new zeroed histogram buffer.
    ///
    /// # Args
    ///
    /// * `n_bins` - Number of bins (including NaN bin).
    ///
    /// # Returns
    ///
    /// A new zeroed `HistogramBuffer`.
    #[must_use]
    pub fn new(n_bins: usize) -> Self {
        Self {
            gradient_sums: vec![0.0_f64; n_bins],
            hessian_sums: vec![0.0_f64; n_bins],
            counts: vec![0_usize; n_bins],
            n_bins,
        }
    }

    /// Returns the number of bins.
    #[must_use]
    pub const fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Accumulates a sample into the appropriate bin.
    ///
    /// # Args
    ///
    /// * `bin` - Bin index for this sample.
    /// * `gradient` - Gradient value.
    /// * `hessian` - Hessian value.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::BinIndexOutOfBounds` if `bin` >= `n_bins`.
    pub fn accumulate(
        &mut self,
        bin: usize,
        gradient: f64,
        hessian: f64,
    ) -> Result<(), ClearGbmError> {
        if bin >= self.n_bins {
            return Err(ClearGbmError::BinIndexOutOfBounds {
                bin,
                n_bins: self.n_bins,
            });
        }
        self.gradient_sums[bin] += gradient;
        self.hessian_sums[bin] += hessian;
        self.counts[bin] += 1;
        Ok(())
    }

    /// Returns the gradient sum for a bin.
    ///
    /// # Args
    ///
    /// * `bin` - Bin index.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::BinIndexOutOfBounds` if `bin` >= `n_bins`.
    pub fn gradient_sum(&self, bin: usize) -> Result<f64, ClearGbmError> {
        self.gradient_sums
            .get(bin)
            .copied()
            .ok_or(ClearGbmError::BinIndexOutOfBounds {
                bin,
                n_bins: self.n_bins,
            })
    }

    /// Returns the hessian sum for a bin.
    ///
    /// # Args
    ///
    /// * `bin` - Bin index.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::BinIndexOutOfBounds` if `bin` >= `n_bins`.
    pub fn hessian_sum(&self, bin: usize) -> Result<f64, ClearGbmError> {
        self.hessian_sums
            .get(bin)
            .copied()
            .ok_or(ClearGbmError::BinIndexOutOfBounds {
                bin,
                n_bins: self.n_bins,
            })
    }

    /// Returns the count for a bin.
    ///
    /// # Args
    ///
    /// * `bin` - Bin index.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::BinIndexOutOfBounds` if `bin` >= `n_bins`.
    pub fn count(&self, bin: usize) -> Result<usize, ClearGbmError> {
        self.counts
            .get(bin)
            .copied()
            .ok_or(ClearGbmError::BinIndexOutOfBounds {
                bin,
                n_bins: self.n_bins,
            })
    }

    /// Returns a slice of all gradient sums.
    #[must_use]
    pub fn gradient_sums(&self) -> &[f64] {
        &self.gradient_sums
    }

    /// Returns a slice of all hessian sums.
    #[must_use]
    pub fn hessian_sums(&self) -> &[f64] {
        &self.hessian_sums
    }

    /// Returns a slice of all counts.
    #[must_use]
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// Resets all bins to zero (for reuse).
    pub fn reset(&mut self) {
        self.gradient_sums.fill(0.0_f64);
        self.hessian_sums.fill(0.0_f64);
        self.counts.fill(0_usize);
    }

    /// Computes sibling histogram by subtraction: self = parent - child.
    ///
    /// This is the "histogram trick" for 2x speedup: instead of building
    /// both child histograms, build one and subtract from parent.
    ///
    /// # Args
    ///
    /// * `parent` - Parent node histogram.
    /// * `child` - One child's histogram.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::ShapeMismatch` if bin counts don't match.
    pub fn subtract_into(&mut self, parent: &Self, child: &Self) -> Result<(), ClearGbmError> {
        if parent.n_bins != self.n_bins || child.n_bins != self.n_bins {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!(
                    "self.n_bins={}, parent.n_bins={}, child.n_bins={}",
                    self.n_bins, parent.n_bins, child.n_bins
                ),
                got: "bin counts must all match".to_string(),
            });
        }

        for i in 0_usize..self.n_bins {
            self.gradient_sums[i] = parent.gradient_sums[i] - child.gradient_sums[i];
            self.hessian_sums[i] = parent.hessian_sums[i] - child.hessian_sums[i];
            self.counts[i] = parent.counts[i].saturating_sub(child.counts[i]);
        }

        Ok(())
    }

    /// Copies contents from another histogram buffer.
    ///
    /// # Args
    ///
    /// * `other` - Source histogram buffer.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::ShapeMismatch` if bin counts don't match.
    pub fn copy_from(&mut self, other: &Self) -> Result<(), ClearGbmError> {
        if other.n_bins != self.n_bins {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("n_bins={}", self.n_bins),
                got: format!("n_bins={}", other.n_bins),
            });
        }

        self.gradient_sums.copy_from_slice(&other.gradient_sums);
        self.hessian_sums.copy_from_slice(&other.hessian_sums);
        self.counts.copy_from_slice(&other.counts);

        Ok(())
    }
}

impl Serialize for HistogramBuffer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("HistogramBuffer", 4) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("n_bins", &self.n_bins) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("gradient_sums", &self.gradient_sums) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("hessian_sums", &self.hessian_sums) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("counts", &self.counts) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `HistogramBuffer` deserialization.
enum HistogramBufferField {
    /// The n_bins field.
    NBins,
    /// The gradient_sums field.
    GradientSums,
    /// The hessian_sums field.
    HessianSums,
    /// The counts field.
    Counts,
}

/// Visitor for deserializing `HistogramBufferField` from string.
struct HistogramBufferFieldVisitor;

impl<'de> Visitor<'de> for HistogramBufferFieldVisitor {
    type Value = HistogramBufferField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "n_bins" => Ok(HistogramBufferField::NBins),
            "gradient_sums" => Ok(HistogramBufferField::GradientSums),
            "hessian_sums" => Ok(HistogramBufferField::HessianSums),
            "counts" => Ok(HistogramBufferField::Counts),
            _ => Err(E::unknown_field(value, HISTOGRAM_BUFFER_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for HistogramBufferField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(HistogramBufferFieldVisitor)
    }
}

/// Field names for `HistogramBuffer` serialization.
const HISTOGRAM_BUFFER_FIELDS: &[&str] = &["n_bins", "gradient_sums", "hessian_sums", "counts"];

impl<'de> Deserialize<'de> for HistogramBuffer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HistogramBufferVisitor;

        impl<'de> Visitor<'de> for HistogramBufferVisitor {
            type Value = HistogramBuffer;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct HistogramBuffer")
            }

            fn visit_map<V>(self, mut map: V) -> Result<HistogramBuffer, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut n_bins = None;
                let mut gradient_sums = None;
                let mut hessian_sums = None;
                let mut counts = None;

                loop {
                    let key: Option<HistogramBufferField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        HistogramBufferField::NBins => {
                            n_bins = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        HistogramBufferField::GradientSums => {
                            gradient_sums = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        HistogramBufferField::HessianSums => {
                            hessian_sums = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        HistogramBufferField::Counts => {
                            counts = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

                let n_bins = match n_bins {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("n_bins")),
                };
                let gradient_sums = match gradient_sums {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("gradient_sums")),
                };
                let hessian_sums = match hessian_sums {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("hessian_sums")),
                };
                let counts = match counts {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("counts")),
                };

                Ok(HistogramBuffer {
                    n_bins,
                    gradient_sums,
                    hessian_sums,
                    counts,
                })
            }
        }

        deserializer.deserialize_struct(
            "HistogramBuffer",
            HISTOGRAM_BUFFER_FIELDS,
            HistogramBufferVisitor,
        )
    }
}

/// Configuration for histogram-based split finding.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitConfig {
    /// Minimum samples required to split a node.
    min_samples_split: usize,

    /// Minimum samples required in a leaf.
    min_samples_leaf: usize,

    /// Maximum number of bins for histogram.
    max_bins: usize,

    /// L2 regularization parameter.
    reg_lambda: f64,

    /// Minimum gain required to make a split.
    min_gain: f64,
}

impl Serialize for SplitConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = match serializer.serialize_struct("SplitConfig", 5) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        match state.serialize_field("min_samples_split", &self.min_samples_split) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("min_samples_leaf", &self.min_samples_leaf) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("max_bins", &self.max_bins) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("reg_lambda", &self.reg_lambda) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match state.serialize_field("min_gain", &self.min_gain) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        state.end()
    }
}

/// Field identifiers for `SplitConfig` deserialization.
enum SplitConfigField {
    /// The min_samples_split field.
    MinSamplesSplit,
    /// The min_samples_leaf field.
    MinSamplesLeaf,
    /// The max_bins field.
    MaxBins,
    /// The reg_lambda field.
    RegLambda,
    /// The min_gain field.
    MinGain,
}

/// Visitor for deserializing `SplitConfigField` from string.
struct SplitConfigFieldVisitor;

impl<'de> Visitor<'de> for SplitConfigFieldVisitor {
    type Value = SplitConfigField;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("field identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "min_samples_split" => Ok(SplitConfigField::MinSamplesSplit),
            "min_samples_leaf" => Ok(SplitConfigField::MinSamplesLeaf),
            "max_bins" => Ok(SplitConfigField::MaxBins),
            "reg_lambda" => Ok(SplitConfigField::RegLambda),
            "min_gain" => Ok(SplitConfigField::MinGain),
            _ => Err(E::unknown_field(value, SPLIT_CONFIG_FIELDS)),
        }
    }
}

impl<'de> Deserialize<'de> for SplitConfigField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(SplitConfigFieldVisitor)
    }
}

/// Field names for `SplitConfig` serialization.
const SPLIT_CONFIG_FIELDS: &[&str] = &[
    "min_samples_split",
    "min_samples_leaf",
    "max_bins",
    "reg_lambda",
    "min_gain",
];

impl<'de> Deserialize<'de> for SplitConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SplitConfigVisitor;

        impl<'de> Visitor<'de> for SplitConfigVisitor {
            type Value = SplitConfig;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("struct SplitConfig")
            }

            fn visit_map<V>(self, mut map: V) -> Result<SplitConfig, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut min_samples_split = None;
                let mut min_samples_leaf = None;
                let mut max_bins = None;
                let mut reg_lambda = None;
                let mut min_gain = None;

                loop {
                    let key: Option<SplitConfigField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        SplitConfigField::MinSamplesSplit => {
                            min_samples_split = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitConfigField::MinSamplesLeaf => {
                            min_samples_leaf = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitConfigField::MaxBins => {
                            max_bins = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitConfigField::RegLambda => {
                            reg_lambda = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        SplitConfigField::MinGain => {
                            min_gain = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                    }
                }

                let min_samples_split = match min_samples_split {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("min_samples_split")),
                };
                let min_samples_leaf = match min_samples_leaf {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("min_samples_leaf")),
                };
                let max_bins = match max_bins {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("max_bins")),
                };
                let reg_lambda = match reg_lambda {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("reg_lambda")),
                };
                let min_gain = match min_gain {
                    Some(v) => v,
                    None => return Err(de::Error::missing_field("min_gain")),
                };

                Ok(SplitConfig {
                    min_samples_split,
                    min_samples_leaf,
                    max_bins,
                    reg_lambda,
                    min_gain,
                })
            }
        }

        deserializer.deserialize_struct("SplitConfig", SPLIT_CONFIG_FIELDS, SplitConfigVisitor)
    }
}

impl SplitConfig {
    /// Creates a new split configuration.
    ///
    /// # Args
    ///
    /// * `min_samples_split` - Minimum samples to split.
    /// * `min_samples_leaf` - Minimum samples in leaf.
    /// * `max_bins` - Maximum histogram bins.
    /// * `reg_lambda` - L2 regularization.
    /// * `min_gain` - Minimum split gain.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if parameters are invalid.
    pub fn new(
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_bins: usize,
        reg_lambda: f64,
        min_gain: f64,
    ) -> Result<Self, ClearGbmError> {
        if min_samples_split < 2_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "min_samples_split".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if min_samples_leaf < 1_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "min_samples_leaf".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if max_bins < 2_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "max_bins".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if reg_lambda < 0.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "reg_lambda".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        if min_gain < 0.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "min_gain".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }

        Ok(Self {
            min_samples_split,
            min_samples_leaf,
            max_bins,
            reg_lambda,
            min_gain,
        })
    }

    /// Returns minimum samples to split.
    #[must_use]
    pub const fn min_samples_split(&self) -> usize {
        self.min_samples_split
    }

    /// Returns minimum samples in leaf.
    #[must_use]
    pub const fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Returns maximum bins.
    #[must_use]
    pub const fn max_bins(&self) -> usize {
        self.max_bins
    }

    /// Returns L2 regularization.
    #[must_use]
    pub const fn reg_lambda(&self) -> f64 {
        self.reg_lambda
    }

    /// Returns minimum gain.
    #[must_use]
    pub const fn min_gain(&self) -> f64 {
        self.min_gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Failing serializer for testing error propagation paths.
    mod failing_serializer {
        use core::fmt::{self, Display};
        use serde::ser::{self, Serialize};

        /// Error type for failing serializer.
        #[derive(Debug)]
        pub struct FailError {
            /// Error message.
            pub message: String,
        }

        impl Display for FailError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.message)
            }
        }

        impl std::error::Error for FailError {}

        impl ser::Error for FailError {
            fn custom<T: Display>(msg: T) -> Self {
                FailError {
                    message: msg.to_string(),
                }
            }
        }

        /// Serializer that fails after N fields.
        pub struct FailAfterN {
            /// Fields serialized so far.
            count: usize,
            /// Fail after this many fields.
            fail_after: usize,
            /// Whether to fail on serialize_struct call.
            fail_on_struct: bool,
        }

        impl FailAfterN {
            /// Create serializer that fails after n fields.
            pub fn new(fail_after: usize) -> Self {
                FailAfterN {
                    count: 0,
                    fail_after,
                    fail_on_struct: false,
                }
            }

            /// Create serializer that fails immediately on serialize_struct.
            pub fn fail_on_struct() -> Self {
                FailAfterN {
                    count: 0,
                    fail_after: usize::MAX,
                    fail_on_struct: true,
                }
            }
        }

        /// Struct serializer state.
        pub struct FailAfterNStruct<'a> {
            ser: &'a mut FailAfterN,
        }

        impl<'a> ser::SerializeStruct for FailAfterNStruct<'a> {
            type Ok = ();
            type Error = FailError;

            fn serialize_field<T>(
                &mut self,
                _key: &'static str,
                _value: &T,
            ) -> Result<(), Self::Error>
            where
                T: ?Sized + Serialize,
            {
                self.ser.count += 1;
                if self.ser.count > self.ser.fail_after {
                    Err(FailError {
                        message: "intentional failure".to_string(),
                    })
                } else {
                    Ok(())
                }
            }

            fn end(self) -> Result<Self::Ok, Self::Error> {
                Ok(())
            }
        }

        impl<'a> ser::Serializer for &'a mut FailAfterN {
            type Ok = ();
            type Error = FailError;
            type SerializeSeq = ser::Impossible<(), FailError>;
            type SerializeTuple = ser::Impossible<(), FailError>;
            type SerializeTupleStruct = ser::Impossible<(), FailError>;
            type SerializeTupleVariant = ser::Impossible<(), FailError>;
            type SerializeMap = ser::Impossible<(), FailError>;
            type SerializeStruct = FailAfterNStruct<'a>;
            type SerializeStructVariant = ser::Impossible<(), FailError>;

            fn serialize_bool(self, _v: bool) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_i8(self, _v: i8) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_i16(self, _v: i16) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_i32(self, _v: i32) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_i64(self, _v: i64) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_u8(self, _v: u8) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_u16(self, _v: u16) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_u32(self, _v: u32) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_u64(self, _v: u64) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_f32(self, _v: f32) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_f64(self, _v: f64) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_char(self, _v: char) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_str(self, _v: &str) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_bytes(self, _v: &[u8]) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_none(self) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_some<T>(self, _value: &T) -> Result<(), FailError>
            where
                T: ?Sized + Serialize,
            {
                Ok(())
            }
            fn serialize_unit(self) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_unit_struct(self, _name: &'static str) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_unit_variant(
                self,
                _name: &'static str,
                _idx: u32,
                _variant: &'static str,
            ) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_newtype_struct<T>(
                self,
                _name: &'static str,
                _value: &T,
            ) -> Result<(), FailError>
            where
                T: ?Sized + Serialize,
            {
                Ok(())
            }
            fn serialize_newtype_variant<T>(
                self,
                _name: &'static str,
                _idx: u32,
                _variant: &'static str,
                _value: &T,
            ) -> Result<(), FailError>
            where
                T: ?Sized + Serialize,
            {
                Ok(())
            }
            fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, FailError> {
                Err(FailError {
                    message: "seq not supported".to_string(),
                })
            }
            fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, FailError> {
                Err(FailError {
                    message: "tuple not supported".to_string(),
                })
            }
            fn serialize_tuple_struct(
                self,
                _name: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeTupleStruct, FailError> {
                Err(FailError {
                    message: "tuple_struct not supported".to_string(),
                })
            }
            fn serialize_tuple_variant(
                self,
                _name: &'static str,
                _idx: u32,
                _variant: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeTupleVariant, FailError> {
                Err(FailError {
                    message: "tuple_variant not supported".to_string(),
                })
            }
            fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, FailError> {
                Err(FailError {
                    message: "map not supported".to_string(),
                })
            }
            fn serialize_struct(
                self,
                _name: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeStruct, FailError> {
                if self.fail_on_struct {
                    Err(FailError {
                        message: "intentional failure on serialize_struct".to_string(),
                    })
                } else {
                    Ok(FailAfterNStruct { ser: self })
                }
            }
            fn serialize_struct_variant(
                self,
                _name: &'static str,
                _idx: u32,
                _variant: &'static str,
                _len: usize,
            ) -> Result<Self::SerializeStructVariant, FailError> {
                Err(FailError {
                    message: "struct_variant not supported".to_string(),
                })
            }
        }
    }

    /// Failing deserializer for testing visitor expecting() methods.
    mod failing_deserializer {
        use core::fmt::{self, Display};
        use serde::de::{self, Visitor};

        /// Error type for failing deserializer.
        #[derive(Debug)]
        pub struct DeError {
            /// Error message.
            pub message: String,
        }

        impl Display for DeError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.message)
            }
        }

        impl std::error::Error for DeError {}

        impl de::Error for DeError {
            fn custom<T: Display>(msg: T) -> Self {
                DeError {
                    message: msg.to_string(),
                }
            }
        }

        /// Deserializer that provides wrong types to trigger expecting().
        pub struct WrongTypeDeserializer {
            mode: WrongTypeMode,
        }

        /// What wrong type to provide.
        pub enum WrongTypeMode {
            /// Provide an i64 when something else is expected.
            Integer,
            /// Provide a map with an integer key to trigger field expecting().
            MapWithIntegerKey,
            /// Provide a map with valid key but wrong value type.
            MapWithWrongValue(&'static str),
        }

        impl WrongTypeDeserializer {
            /// Create a deserializer that provides an integer.
            pub fn integer() -> Self {
                WrongTypeDeserializer {
                    mode: WrongTypeMode::Integer,
                }
            }

            /// Create a deserializer that provides a map with an integer key.
            pub fn map_with_integer_key() -> Self {
                WrongTypeDeserializer {
                    mode: WrongTypeMode::MapWithIntegerKey,
                }
            }

            /// Create a deserializer that provides a map with wrong value type.
            pub fn map_with_wrong_value(field_name: &'static str) -> Self {
                WrongTypeDeserializer {
                    mode: WrongTypeMode::MapWithWrongValue(field_name),
                }
            }
        }

        /// MapAccess that provides an integer as the first key.
        pub struct IntegerKeyMapAccess {
            /// Whether we've returned a key yet.
            pub done: bool,
        }

        impl<'de> de::MapAccess<'de> for IntegerKeyMapAccess {
            type Error = DeError;

            fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
            where
                K: de::DeserializeSeed<'de>,
            {
                if self.done {
                    return Ok(None);
                }
                self.done = true;
                // Deserialize the key with an integer deserializer
                // This will call expecting() on the key visitor
                seed.deserialize(WrongTypeDeserializer::integer()).map(Some)
            }

            fn next_value_seed<V>(&mut self, _seed: V) -> Result<V::Value, Self::Error>
            where
                V: de::DeserializeSeed<'de>,
            {
                Err(DeError {
                    message: "should not reach value".to_string(),
                })
            }
        }

        /// Deserializer that provides a string value.
        pub struct StringDeserializer {
            /// The string to provide.
            pub value: &'static str,
        }

        impl<'de> de::Deserializer<'de> for StringDeserializer {
            type Error = DeError;

            fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                visitor.visit_str(self.value)
            }

            serde::forward_to_deserialize_any! {
                bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string
                bytes byte_buf option unit unit_struct newtype_struct seq
                tuple tuple_struct map struct enum identifier ignored_any
            }
        }

        /// MapAccess that provides a valid field name but fails on value.
        pub struct FailOnValueMapAccess {
            /// State: 0 = return key, 1 = fail on value, 2 = done.
            pub state: usize,
            /// Field name to return.
            pub field_name: &'static str,
        }

        impl<'de> de::MapAccess<'de> for FailOnValueMapAccess {
            type Error = DeError;

            fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
            where
                K: de::DeserializeSeed<'de>,
            {
                if self.state >= 2 {
                    return Ok(None);
                }
                if self.state == 1 {
                    self.state = 2;
                    return Ok(None);
                }
                self.state = 1;
                seed.deserialize(StringDeserializer {
                    value: self.field_name,
                })
                .map(Some)
            }

            fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
            where
                V: de::DeserializeSeed<'de>,
            {
                // Deserialize with wrong type to trigger error
                seed.deserialize(WrongTypeDeserializer::integer())
            }
        }

        impl<'de> de::Deserializer<'de> for WrongTypeDeserializer {
            type Error = DeError;

            fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                match self.mode {
                    WrongTypeMode::Integer => visitor.visit_i64(42_i64),
                    WrongTypeMode::MapWithIntegerKey => {
                        visitor.visit_map(IntegerKeyMapAccess { done: false })
                    }
                    WrongTypeMode::MapWithWrongValue(field_name) => {
                        visitor.visit_map(FailOnValueMapAccess {
                            state: 0,
                            field_name,
                        })
                    }
                }
            }

            fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_char<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_unit_struct<V>(
                self,
                _name: &'static str,
                visitor: V,
            ) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_newtype_struct<V>(
                self,
                _name: &'static str,
                visitor: V,
            ) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_tuple_struct<V>(
                self,
                _name: &'static str,
                _len: usize,
                visitor: V,
            ) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_struct<V>(
                self,
                _name: &'static str,
                _fields: &'static [&'static str],
                visitor: V,
            ) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_enum<V>(
                self,
                _name: &'static str,
                _variants: &'static [&'static str],
                visitor: V,
            ) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }

            fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                self.deserialize_any(visitor)
            }
        }
    }

    // TreeNode tests

    #[test]
    fn test_tree_node_new_leaf() -> Result<(), ClearGbmError> {
        let node = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        assert_eq!(node.node_id(), 0_usize);
        assert!(node.is_leaf());
        assert_eq!(node.feature_index(), None);
        assert_eq!(node.threshold(), None);
        assert!((node.value() - 0.5_f64).abs() < f64::EPSILON);
        assert_eq!(node.n_samples(), 100_usize);
        assert_eq!(node.left_child(), None);
        assert_eq!(node.right_child(), None);
        assert!(node.nan_goes_left());
        Ok(())
    }

    #[test]
    fn test_tree_node_new_internal() -> Result<(), ClearGbmError> {
        let config = TreeNodeConfig {
            node_id: 1_usize,
            feature_index: 3_usize,
            threshold: 0.25_f64,
            value: 0.1_f64,
            n_samples: 50_usize,
            left_child: 2_usize,
            right_child: 3_usize,
            nan_goes_left: false,
        };
        let node = TreeNode::new_internal(config);
        assert_eq!(node.node_id(), 1_usize);
        assert!(!node.is_leaf());
        assert_eq!(node.feature_index(), Some(3_usize));
        assert_eq!(node.threshold(), Some(0.25_f64));
        assert!((node.value() - 0.1_f64).abs() < f64::EPSILON);
        assert_eq!(node.n_samples(), 50_usize);
        assert_eq!(node.left_child(), Some(2_usize));
        assert_eq!(node.right_child(), Some(3_usize));
        assert!(!node.nan_goes_left());
        Ok(())
    }

    #[test]
    fn test_tree_node_clone() -> Result<(), ClearGbmError> {
        let node = TreeNode::new_leaf(5_usize, 1.0_f64, 10_usize);
        let cloned = node.clone();
        assert_eq!(node, cloned);
        Ok(())
    }

    #[test]
    fn test_tree_node_debug() -> Result<(), ClearGbmError> {
        let node = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        let debug_str = format!("{node:?}");
        assert!(debug_str.contains("TreeNode"));
        assert!(debug_str.contains("node_id: 0"));
        Ok(())
    }

    #[test]
    fn test_tree_node_serialize_deserialize() -> Result<(), ClearGbmError> {
        let config = TreeNodeConfig {
            node_id: 1_usize,
            feature_index: 2_usize,
            threshold: 0.5_f64,
            value: 0.3_f64,
            n_samples: 200_usize,
            left_child: 3_usize,
            right_child: 4_usize,
            nan_goes_left: true,
        };
        let node = TreeNode::new_internal(config);
        let json_str = match serde_json::to_string(&node) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let parsed: TreeNode = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(parsed, node);
        Ok(())
    }

    // HistogramBuffer tests

    #[test]
    fn test_histogram_buffer_new() -> Result<(), ClearGbmError> {
        let hist = HistogramBuffer::new(5_usize);
        assert_eq!(hist.n_bins(), 5_usize);
        for i in 0_usize..5_usize {
            let grad = match hist.gradient_sum(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert!(grad.abs() < f64::EPSILON);
            let hess = match hist.hessian_sum(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert!(hess.abs() < f64::EPSILON);
            let count_val = match hist.count(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert_eq!(count_val, 0_usize);
        }
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_accumulate() -> Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        match hist.accumulate(1_usize, 0.5_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        let grad = match hist.gradient_sum(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((grad - 0.5_f64).abs() < f64::EPSILON);
        let hess = match hist.hessian_sum(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((hess - 1.0_f64).abs() < f64::EPSILON);
        let count_val = match hist.count(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_val, 1_usize);
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_accumulate_multiple() -> Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        match hist.accumulate(0_usize, 0.1_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist.accumulate(0_usize, 0.2_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist.accumulate(0_usize, 0.3_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        let grad = match hist.gradient_sum(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((grad - 0.6_f64).abs() < 1e-10_f64);
        let hess = match hist.hessian_sum(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((hess - 3.0_f64).abs() < f64::EPSILON);
        let count_val = match hist.count(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_val, 3_usize);
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_accumulate_out_of_bounds() -> Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        let result = hist.accumulate(5_usize, 0.5_f64, 1.0_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::BinIndexOutOfBounds {
                bin: 5_usize,
                n_bins: 3_usize
            })
        ));
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_gradient_sum_out_of_bounds() -> Result<(), ClearGbmError> {
        let hist = HistogramBuffer::new(3_usize);
        let result = hist.gradient_sum(10_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_hessian_sum_out_of_bounds() -> Result<(), ClearGbmError> {
        let hist = HistogramBuffer::new(3_usize);
        let result = hist.hessian_sum(10_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_count_out_of_bounds() -> Result<(), ClearGbmError> {
        let hist = HistogramBuffer::new(3_usize);
        let result = hist.count(10_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_slices() -> Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        match hist.accumulate(0_usize, 0.1_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist.accumulate(1_usize, 0.2_f64, 2.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist.accumulate(2_usize, 0.3_f64, 3.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }

        assert_eq!(hist.gradient_sums().len(), 3_usize);
        assert_eq!(hist.hessian_sums().len(), 3_usize);
        assert_eq!(hist.counts().len(), 3_usize);

        assert!((hist.gradient_sums()[0_usize] - 0.1_f64).abs() < f64::EPSILON);
        assert!((hist.hessian_sums()[1_usize] - 2.0_f64).abs() < f64::EPSILON);
        assert_eq!(hist.counts()[2_usize], 1_usize);
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_reset() -> Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        match hist.accumulate(0_usize, 0.5_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist.accumulate(1_usize, 0.3_f64, 1.5_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        hist.reset();
        for i in 0_usize..3_usize {
            let grad = match hist.gradient_sum(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert!(grad.abs() < f64::EPSILON);
            let hess = match hist.hessian_sum(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert!(hess.abs() < f64::EPSILON);
            let count_val = match hist.count(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert_eq!(count_val, 0_usize);
        }
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_subtract_into() -> Result<(), ClearGbmError> {
        let mut parent = HistogramBuffer::new(3_usize);
        match parent.accumulate(0_usize, 0.5_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match parent.accumulate(0_usize, 0.3_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match parent.accumulate(1_usize, 0.2_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }

        let mut child = HistogramBuffer::new(3_usize);
        match child.accumulate(0_usize, 0.5_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }

        let mut sibling = HistogramBuffer::new(3_usize);
        match sibling.subtract_into(&parent, &child) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }

        // Bin 0: parent (0.8, 2.0, 2), child (0.5, 1.0, 1), sibling should be (0.3, 1.0, 1)
        let grad = match sibling.gradient_sum(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((grad - 0.3_f64).abs() < 1e-10_f64);
        let hess = match sibling.hessian_sum(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((hess - 1.0_f64).abs() < f64::EPSILON);
        let count_val = match sibling.count(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_val, 1_usize);

        // Bin 1: parent (0.2, 1.0, 1), child (0, 0, 0), sibling should be (0.2, 1.0, 1)
        let count_val_1 = match sibling.count(1_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_val_1, 1_usize);
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_subtract_into_shape_mismatch() -> Result<(), ClearGbmError> {
        let parent = HistogramBuffer::new(3_usize);
        let child = HistogramBuffer::new(5_usize);
        let mut sibling = HistogramBuffer::new(3_usize);

        let result = sibling.subtract_into(&parent, &child);
        assert!(result.is_err());
        assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_copy_from() -> Result<(), ClearGbmError> {
        let mut source = HistogramBuffer::new(3_usize);
        match source.accumulate(0_usize, 0.5_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match source.accumulate(1_usize, 0.3_f64, 2.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }

        let mut dest = HistogramBuffer::new(3_usize);
        match dest.copy_from(&source) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }

        assert_eq!(dest.gradient_sums(), source.gradient_sums());
        assert_eq!(dest.hessian_sums(), source.hessian_sums());
        assert_eq!(dest.counts(), source.counts());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_copy_from_shape_mismatch() -> Result<(), ClearGbmError> {
        let source = HistogramBuffer::new(5_usize);
        let mut dest = HistogramBuffer::new(3_usize);
        let result = dest.copy_from(&source);
        assert!(result.is_err());
        assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_clone() -> Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        match hist.accumulate(0_usize, 0.5_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        let cloned = hist.clone();
        assert_eq!(hist, cloned);
        Ok(())
    }

    // SplitConfig tests

    #[test]
    fn test_split_config_new_valid() -> Result<(), ClearGbmError> {
        let c = match SplitConfig::new(2_usize, 1_usize, 64_usize, 1.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        assert_eq!(c.min_samples_split(), 2_usize);
        assert_eq!(c.min_samples_leaf(), 1_usize);
        assert_eq!(c.max_bins(), 64_usize);
        assert!((c.reg_lambda() - 1.0_f64).abs() < f64::EPSILON);
        assert!(c.min_gain().abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn test_split_config_min_samples_split_too_small() -> Result<(), ClearGbmError> {
        let result = SplitConfig::new(1_usize, 1_usize, 64_usize, 1.0_f64, 0.0_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "min_samples_split"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_min_samples_leaf_zero() -> Result<(), ClearGbmError> {
        let result = SplitConfig::new(2_usize, 0_usize, 64_usize, 1.0_f64, 0.0_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "min_samples_leaf"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_max_bins_too_small() -> Result<(), ClearGbmError> {
        let result = SplitConfig::new(2_usize, 1_usize, 1_usize, 1.0_f64, 0.0_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "max_bins"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_negative_reg_lambda() -> Result<(), ClearGbmError> {
        let result = SplitConfig::new(2_usize, 1_usize, 64_usize, -1.0_f64, 0.0_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "reg_lambda"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_negative_min_gain() -> Result<(), ClearGbmError> {
        let result = SplitConfig::new(2_usize, 1_usize, 64_usize, 1.0_f64, -0.1_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "min_gain"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_clone() -> Result<(), ClearGbmError> {
        let c = match SplitConfig::new(2_usize, 1_usize, 64_usize, 1.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cloned = c.clone();
        assert_eq!(c, cloned);
        Ok(())
    }

    #[test]
    fn test_split_config_serialize_deserialize() -> Result<(), ClearGbmError> {
        let c = match SplitConfig::new(10_usize, 5_usize, 128_usize, 0.5_f64, 0.01_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let json_str = match serde_json::to_string(&c) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let parsed: SplitConfig = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(parsed, c);
        Ok(())
    }

    // =========================================================================
    // Serde error path tests - TreeNodeConfig
    // =========================================================================

    #[test]
    fn test_tree_node_config_deserialize_missing_field() -> Result<(), ClearGbmError> {
        // Missing nan_goes_left field
        let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3}"#;
        let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_deserialize_unknown_field() -> Result<(), ClearGbmError> {
        let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true,"unknown_field":42}"#;
        let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_deserialize_wrong_type() -> Result<(), ClearGbmError> {
        // node_id should be usize, not string
        let json = r#"{"node_id":"not_a_number","feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
        let result: Result<TreeNodeConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_deserialize_all_fields() -> Result<(), ClearGbmError> {
        let json = r#"{"node_id":1,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
        let config: TreeNodeConfig = match serde_json::from_str(json) {
            Ok(c) => c,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(config.node_id, 1_usize);
        assert_eq!(config.feature_index, 2_usize);
        assert!((config.threshold - 0.5_f64).abs() < 1e-10_f64);
        assert!((config.value - 1.0_f64).abs() < 1e-10_f64);
        assert_eq!(config.n_samples, 100_usize);
        assert_eq!(config.left_child, 2_usize);
        assert_eq!(config.right_child, 3_usize);
        assert!(config.nan_goes_left);
        Ok(())
    }

    #[test]
    fn test_tree_node_config_deserialize_duplicate_field() -> Result<(), ClearGbmError> {
        // Duplicate node_id field - serde_json uses last value
        let json = r#"{"node_id":1,"node_id":99,"feature_index":2,"threshold":0.5,"value":1.0,"n_samples":100,"left_child":2,"right_child":3,"nan_goes_left":true}"#;
        let config: TreeNodeConfig = match serde_json::from_str(json) {
            Ok(c) => c,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        // serde_json takes the last value for duplicate keys
        assert_eq!(config.node_id, 99_usize);
        Ok(())
    }

    #[test]
    fn test_tree_node_config_serialize_roundtrip() -> Result<(), ClearGbmError> {
        let config = TreeNodeConfig {
            node_id: 5_usize,
            feature_index: 2_usize,
            threshold: 0.75_f64,
            value: 0.123_f64,
            n_samples: 500_usize,
            left_child: 10_usize,
            right_child: 11_usize,
            nan_goes_left: false,
        };
        let json_str = match serde_json::to_string(&config) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let parsed: TreeNodeConfig = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(parsed.node_id, config.node_id);
        assert_eq!(parsed.feature_index, config.feature_index);
        assert!((parsed.threshold - config.threshold).abs() < 1e-10_f64);
        assert!((parsed.value - config.value).abs() < 1e-10_f64);
        assert_eq!(parsed.n_samples, config.n_samples);
        assert_eq!(parsed.left_child, config.left_child);
        assert_eq!(parsed.right_child, config.right_child);
        assert_eq!(parsed.nan_goes_left, config.nan_goes_left);
        Ok(())
    }

    // =========================================================================
    // Serde error path tests - TreeNode
    // =========================================================================

    #[test]
    fn test_tree_node_deserialize_missing_field() -> Result<(), ClearGbmError> {
        // Missing is_leaf field
        let json = r#"{"node_id":0,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":null}"#;
        let result: Result<TreeNode, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_deserialize_unknown_field() -> Result<(), ClearGbmError> {
        let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":null,"bogus":123}"#;
        let result: Result<TreeNode, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_deserialize_leaf_all_fields() -> Result<(), ClearGbmError> {
        let json = r#"{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":100,"left_child":null,"right_child":null,"nan_goes_left":true}"#;
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
        let json = r#"{"node_id":0,"is_leaf":false,"feature_index":1,"threshold":0.5,"value":0.0,"n_samples":100,"left_child":1,"right_child":2,"nan_goes_left":true}"#;
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
    // Serde error path tests - SplitConfig
    // =========================================================================

    #[test]
    fn test_split_config_deserialize_missing_field() -> Result<(), ClearGbmError> {
        // Missing min_gain field
        let json = r#"{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0}"#;
        let result: Result<SplitConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_config_deserialize_unknown_field() -> Result<(), ClearGbmError> {
        let json = r#"{"min_samples_split":2,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0,"extra":999}"#;
        let result: Result<SplitConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_config_deserialize_wrong_type() -> Result<(), ClearGbmError> {
        // min_samples_split should be usize, not bool
        let json = r#"{"min_samples_split":true,"min_samples_leaf":1,"max_bins":64,"reg_lambda":0.0,"min_gain":0.0}"#;
        let result: Result<SplitConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_config_deserialize_all_fields() -> Result<(), ClearGbmError> {
        let json = r#"{"min_samples_split":10,"min_samples_leaf":5,"max_bins":128,"reg_lambda":0.5,"min_gain":0.01}"#;
        let config: SplitConfig = match serde_json::from_str(json) {
            Ok(c) => c,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(config.min_samples_split(), 10_usize);
        assert_eq!(config.min_samples_leaf(), 5_usize);
        assert_eq!(config.max_bins(), 128_usize);
        assert!((config.reg_lambda() - 0.5_f64).abs() < 1e-10_f64);
        assert!((config.min_gain() - 0.01_f64).abs() < 1e-10_f64);
        Ok(())
    }

    // =========================================================================
    // Serde error path tests - HistogramBuffer
    // =========================================================================

    #[test]
    fn test_histogram_buffer_deserialize_wrong_type() -> Result<(), ClearGbmError> {
        // gradient_sums should be an array, not a number
        let json =
            r#"{"n_bins":3,"gradient_sums":123,"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0]}"#;
        let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_deserialize_missing_field() -> Result<(), ClearGbmError> {
        // Missing counts field
        let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0]}"#;
        let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_deserialize_unknown_field() -> Result<(), ClearGbmError> {
        let json = r#"{"n_bins":3,"gradient_sums":[0.0,0.0,0.0],"hessian_sums":[0.0,0.0,0.0],"counts":[0,0,0],"unknown":true}"#;
        let result: Result<HistogramBuffer, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_deserialize_all_fields() -> Result<(), ClearGbmError> {
        let json = r#"{"n_bins":3,"gradient_sums":[1.0,2.0,3.0],"hessian_sums":[0.5,1.0,1.5],"counts":[10,20,30]}"#;
        let hist: HistogramBuffer = match serde_json::from_str(json) {
            Ok(h) => h,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(hist.n_bins(), 3_usize);
        let g0 = match hist.gradient_sum(0_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!((g0 - 1.0_f64).abs() < 1e-10_f64);
        let c2 = match hist.count(2_usize) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(c2, 30_usize);
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_serialize_roundtrip() -> Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(4_usize);
        match hist.accumulate(0_usize, 1.5_f64, 2.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist.accumulate(1_usize, 3.0_f64, 4.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist.accumulate(2_usize, 0.0_f64, 1.0_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        match hist.accumulate(3_usize, -1.0_f64, 0.5_f64) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }

        let json_str = match serde_json::to_string(&hist) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };

        let parsed: HistogramBuffer = match serde_json::from_str(&json_str) {
            Ok(h) => h,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };

        assert_eq!(parsed.n_bins(), hist.n_bins());
        for i in 0_usize..4_usize {
            let orig_g = match hist.gradient_sum(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let parsed_g = match parsed.gradient_sum(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert!((orig_g - parsed_g).abs() < 1e-10_f64);

            let orig_h = match hist.hessian_sum(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let parsed_h = match parsed.hessian_sum(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert!((orig_h - parsed_h).abs() < 1e-10_f64);

            let orig_c = match hist.count(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let parsed_c = match parsed.count(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            assert_eq!(orig_c, parsed_c);
        }
        Ok(())
    }

    // Type mismatch tests to trigger expecting() methods

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

    // Tests using WrongTypeDeserializer to trigger expecting() methods

    #[test]
    fn test_tree_node_config_expecting_via_wrong_type() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        let de = WrongTypeDeserializer::integer();
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
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        let de = WrongTypeDeserializer::integer();
        let result = TreeNode::deserialize(de);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_config_expecting_via_wrong_type() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        let de = WrongTypeDeserializer::integer();
        let result = SplitConfig::deserialize(de);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_expecting_via_wrong_type() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        let de = WrongTypeDeserializer::integer();
        let result = HistogramBuffer::deserialize(de);
        assert!(result.is_err());
        Ok(())
    }

    // Tests using map_with_integer_key to trigger field visitor expecting()
    #[test]
    fn test_tree_node_config_field_expecting() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        let de = WrongTypeDeserializer::map_with_integer_key();
        let result = TreeNodeConfig::deserialize(de);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_field_expecting() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        let de = WrongTypeDeserializer::map_with_integer_key();
        let result = TreeNode::deserialize(de);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_config_field_expecting() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        let de = WrongTypeDeserializer::map_with_integer_key();
        let result = SplitConfig::deserialize(de);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_field_expecting() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        let de = WrongTypeDeserializer::map_with_integer_key();
        let result = HistogramBuffer::deserialize(de);
        assert!(result.is_err());
        Ok(())
    }

    // Tests to trigger next_value error branches
    #[test]
    fn test_tree_node_config_next_value_error() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
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
            let de = WrongTypeDeserializer::map_with_wrong_value(field);
            let result = TreeNodeConfig::deserialize(de);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_tree_node_next_value_error() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
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
            let de = WrongTypeDeserializer::map_with_wrong_value(field);
            let result = TreeNode::deserialize(de);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_split_config_next_value_error() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        for field in &[
            "min_samples_split",
            "min_samples_leaf",
            "max_bins",
            "min_gain",
            "reg_lambda",
        ] {
            let de = WrongTypeDeserializer::map_with_wrong_value(field);
            let result = SplitConfig::deserialize(de);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_next_value_error() -> Result<(), ClearGbmError> {
        use failing_deserializer::WrongTypeDeserializer;
        use serde::Deserialize;
        for field in &["n_bins", "gradient_sums", "hessian_sums", "counts"] {
            let de = WrongTypeDeserializer::map_with_wrong_value(field);
            let result = HistogramBuffer::deserialize(de);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_failing_deserializer_coverage() -> Result<(), ClearGbmError> {
        use failing_deserializer::{DeError, WrongTypeDeserializer};
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
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_any(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_bool
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_bool(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_i8
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_i8(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_i16
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_i16(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_i32
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_i32(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_i64
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_i64(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_u8
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_u8(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_u16
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_u16(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_u32
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_u32(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_u64
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_u64(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_f32
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_f32(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_f64
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_f64(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_char
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_char(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_str
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_str(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_string
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_string(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_bytes
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_bytes(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_byte_buf
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_byte_buf(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_option
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_option(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_unit
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_unit(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_unit_struct
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_unit_struct("Test", I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_newtype_struct
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_newtype_struct("Test", I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_seq
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_seq(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_tuple
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_tuple(2, I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_tuple_struct
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_tuple_struct("Test", 2, I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_map
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_map(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_struct
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_struct("Test", &["field"], I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_enum
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_enum("Test", &["Variant"], I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_identifier
        let de = WrongTypeDeserializer::integer();
        let result = de.deserialize_identifier(I64Visitor);
        assert!(result.is_ok());

        // Test deserialize_ignored_any
        let de = WrongTypeDeserializer::integer();
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
        let de = WrongTypeDeserializer::map_with_integer_key();
        let result = de.deserialize_any(MapVisitor);
        assert!(result.is_ok());

        // Test IntegerKeyMapAccess done state
        use failing_deserializer::IntegerKeyMapAccess;
        use serde::de::MapAccess;
        let mut map_access = IntegerKeyMapAccess { done: true };
        let key_result: Result<Option<String>, _> = map_access.next_key();
        assert!(key_result.is_ok());
        assert!(matches!(key_result, Ok(None)));

        // Test IntegerKeyMapAccess next_value_seed
        let mut map_access2 = IntegerKeyMapAccess { done: false };
        let value_result: Result<i64, _> = map_access2.next_value();
        assert!(value_result.is_err());

        Ok(())
    }

    // Serialization error path tests using failing serializer

    #[test]
    fn test_tree_node_config_serialize_fail_field_1() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
        let mut ser = FailAfterN::new(0);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_serialize_fail_field_2() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
        let mut ser = FailAfterN::new(1);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_serialize_fail_field_3() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
        let mut ser = FailAfterN::new(2);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_serialize_fail_field_4() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
        let mut ser = FailAfterN::new(3);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_serialize_fail_field_5() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
        let mut ser = FailAfterN::new(4);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_serialize_fail_field_6() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
        let mut ser = FailAfterN::new(5);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_serialize_fail_field_7() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
        let mut ser = FailAfterN::new(6);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_serialize_fail_field_8() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
        let mut ser = FailAfterN::new(7);
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
        let mut ser = FailAfterN::fail_on_struct();
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let node = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        let mut ser = FailAfterN::fail_on_struct();
        let result = node.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_node_serialize_fail_each_field() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let node = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        // TreeNode has 9 fields
        for fail_at in 0_usize..9_usize {
            let mut ser = FailAfterN::new(fail_at);
            let result = node.serialize(&mut ser);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_split_config_serialize_fail_each_field() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        // SplitConfig has 5 fields
        for fail_at in 0_usize..5_usize {
            let mut ser = FailAfterN::new(fail_at);
            let result = config.serialize(&mut ser);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_serialize_fail_each_field() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let hist = HistogramBuffer::new(3_usize);
        // HistogramBuffer has 4 fields
        for fail_at in 0_usize..4_usize {
            let mut ser = FailAfterN::new(fail_at);
            let result = hist.serialize(&mut ser);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let hist = HistogramBuffer::new(3_usize);
        let mut ser = FailAfterN::fail_on_struct();
        let result = hist.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_split_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let mut ser = FailAfterN::fail_on_struct();
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    // Tests to cover all failing serializer methods
    #[test]
    fn test_failing_serializer_coverage() -> Result<(), ClearGbmError> {
        use failing_serializer::{FailAfterN, FailError};
        use serde::ser::{Error, Serializer};

        // Test FailError Display
        let err = FailError {
            message: "test".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("test"));

        // Test FailError custom
        let custom_err = FailError::custom("custom error");
        assert!(custom_err.message.contains("custom"));

        // Test all serializer primitive methods
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_bool(true).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_i8(1_i8).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_i16(1_i16).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_i32(1_i32).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_i64(1_i64).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_u8(1_u8).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_u16(1_u16).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_u32(1_u32).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_u64(1_u64).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_f32(1.0_f32).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_f64(1.0_f64).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_char('a').is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_str("test").is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_bytes(&[1_u8, 2_u8]).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_none().is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_some(&1_u32).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_unit().is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_unit_struct("Unit").is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_unit_variant("E", 0, "V").is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_newtype_struct("N", &1_u32).is_ok());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser)
            .serialize_newtype_variant("E", 0, "V", &1_u32)
            .is_ok());

        // Test error methods
        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_seq(Some(1)).is_err());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_tuple(1).is_err());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_tuple_struct("T", 1).is_err());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_tuple_variant("E", 0, "V", 1).is_err());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_map(Some(1)).is_err());

        let mut ser = FailAfterN::new(100);
        assert!((&mut ser).serialize_struct_variant("E", 0, "V", 1).is_err());

        // Test serialize_struct returns Ok and can be used
        let mut ser = FailAfterN::new(100);
        let struct_ser = (&mut ser).serialize_struct("S", 1);
        assert!(struct_ser.is_ok());

        Ok(())
    }

    #[test]
    fn test_failing_serializer_struct_end() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::ser::{SerializeStruct, Serializer};

        let mut ser = FailAfterN::new(100);
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
        use failing_serializer::FailAfterN;
        use serde::ser::{SerializeStruct, Serializer};

        // Test that serialize_field returns Ok for first field, then Err
        let mut ser = FailAfterN::new(1);
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
}
