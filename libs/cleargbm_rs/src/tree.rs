//! Tree construction for gradient boosting.
//!
//! Implements depth-first tree building with histogram-based split finding.
//! Uses the sibling histogram subtraction trick for 2x speedup on histogram building.
//!
//! # Algorithm
//!
//! 1. Start with all samples at root node
//! 2. For each node (depth-first via stack):
//!    a. Check stopping criteria (max_depth, min_samples, etc.)
//!    b. Build histograms for each feature
//!    c. Find best split using O(K) histogram scan
//!    d. If valid split found, create internal node and push children to stack
//!    e. Otherwise create leaf node
//! 3. Return completed tree with all nodes

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::ClearGbmError;
use crate::histogram::subtract_histogram;
use crate::hooks::Hooks;
use crate::split::{find_best_split_from_histogram, MonotonicConstraint, SplitResult};
use crate::types::{HistogramBuffer, SplitConfig, TreeNode, TreeNodeConfig};

/// Epsilon for floating-point comparisons.
const EPSILON: f64 = 1e-10_f64;

/// Configuration for tree building.
///
/// Controls tree growth constraints like maximum depth and leaf limits.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeBuildConfig {
    /// Maximum depth of tree (0 = unlimited).
    max_depth: usize,

    /// Maximum number of leaves (0 = unlimited).
    max_leaves: usize,

    /// L1 regularization term for leaf values.
    reg_alpha: f64,

    /// L2 regularization term for leaf values.
    reg_lambda: f64,

    /// Split configuration.
    split_config: SplitConfig,
}

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
            _ => Err(E::unknown_field(value, TREE_BUILD_CONFIG_FIELDS)),
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

/// Field names for `TreeBuildConfig` serialization.
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

            fn visit_map<V>(self, mut map: V) -> Result<TreeBuildConfig, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut max_depth = None;
                let mut max_leaves = None;
                let mut reg_alpha = None;
                let mut reg_lambda = None;
                let mut split_config = None;

                loop {
                    let key: Option<TreeBuildConfigField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        TreeBuildConfigField::MaxDepth => {
                            max_depth = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeBuildConfigField::MaxLeaves => {
                            max_leaves = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeBuildConfigField::RegAlpha => {
                            reg_alpha = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeBuildConfigField::RegLambda => {
                            reg_lambda = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeBuildConfigField::SplitConfig => {
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

impl TreeBuildConfig {
    /// Creates a new tree build configuration.
    ///
    /// # Args
    ///
    /// * `max_depth` - Maximum tree depth (0 = unlimited).
    /// * `max_leaves` - Maximum number of leaves (0 = unlimited).
    /// * `reg_alpha` - L1 regularization term.
    /// * `reg_lambda` - L2 regularization term.
    /// * `split_config` - Configuration for split finding.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if:
    /// - `reg_alpha` < 0.0
    /// - `reg_lambda` < 0.0
    pub fn new(
        max_depth: usize,
        max_leaves: usize,
        reg_alpha: f64,
        reg_lambda: f64,
        split_config: SplitConfig,
    ) -> Result<Self, ClearGbmError> {
        if reg_alpha < 0.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "reg_alpha".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        if reg_lambda < 0.0_f64 {
            return Err(ClearGbmError::InvalidParameter {
                name: "reg_lambda".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }

        Ok(Self {
            max_depth,
            max_leaves,
            reg_alpha,
            reg_lambda,
            split_config,
        })
    }

    /// Returns maximum depth.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns maximum leaves.
    #[must_use]
    pub const fn max_leaves(&self) -> usize {
        self.max_leaves
    }

    /// Returns L1 regularization.
    #[must_use]
    pub const fn reg_alpha(&self) -> f64 {
        self.reg_alpha
    }

    /// Returns L2 regularization.
    #[must_use]
    pub const fn reg_lambda(&self) -> f64 {
        self.reg_lambda
    }

    /// Returns split configuration.
    #[must_use]
    pub const fn split_config(&self) -> &SplitConfig {
        &self.split_config
    }
}

/// A complete decision tree.
///
/// Contains all nodes and metadata about the tree structure.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    /// All nodes in the tree (index = node_id).
    nodes: Vec<TreeNode>,

    /// Actual maximum depth achieved.
    max_depth: usize,

    /// Number of leaf nodes.
    n_leaves: usize,
}

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
    /// The n_leaves field.
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
            _ => Err(E::unknown_field(value, TREE_FIELDS)),
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

/// Field names for `Tree` serialization.
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

            fn visit_map<V>(self, mut map: V) -> Result<Tree, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut nodes = None;
                let mut max_depth = None;
                let mut n_leaves = None;

                loop {
                    let key: Option<TreeField> = match map.next_key() {
                        Ok(k) => k,
                        Err(e) => return Err(e),
                    };
                    let key = match key {
                        Some(k) => k,
                        None => break,
                    };
                    match key {
                        TreeField::Nodes => {
                            nodes = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeField::MaxDepth => {
                            max_depth = Some(match map.next_value() {
                                Ok(v) => v,
                                Err(e) => return Err(e),
                            });
                        }
                        TreeField::NLeaves => {
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

impl Tree {
    /// Creates a new tree from nodes.
    ///
    /// # Args
    ///
    /// * `nodes` - Vector of tree nodes.
    /// * `max_depth` - Maximum depth of tree.
    /// * `n_leaves` - Number of leaf nodes.
    #[must_use]
    pub fn new(nodes: Vec<TreeNode>, max_depth: usize, n_leaves: usize) -> Self {
        Self {
            nodes,
            max_depth,
            n_leaves,
        }
    }

    /// Returns the nodes slice.
    #[must_use]
    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    /// Returns a specific node by ID.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::NodeNotFound` if node_id is out of bounds.
    pub fn node(&self, node_id: usize) -> Result<&TreeNode, ClearGbmError> {
        self.nodes
            .get(node_id)
            .ok_or(ClearGbmError::NodeNotFound { node_id })
    }

    /// Returns the root node.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::NodeNotFound` if tree is empty.
    pub fn root(&self) -> Result<&TreeNode, ClearGbmError> {
        self.node(0_usize)
    }

    /// Returns maximum depth.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns number of leaves.
    #[must_use]
    pub const fn n_leaves(&self) -> usize {
        self.n_leaves
    }

    /// Returns total number of nodes.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Computes the optimal leaf value from gradient and hessian sums.
///
/// The optimal value minimizes the regularized loss:
/// - Without L1: leaf = -G / (H + λ)
/// - With L1 (alpha): leaf = -sign(G) * max(|G| - alpha, 0) / (H + λ)
///
/// # Args
///
/// * `gradient_sum` - Sum of gradients in this leaf.
/// * `hessian_sum` - Sum of hessians in this leaf.
/// * `reg_alpha` - L1 regularization term.
/// * `reg_lambda` - L2 regularization term.
///
/// # Returns
///
/// Optimal leaf prediction value.
#[must_use]
pub fn compute_leaf_value(
    gradient_sum: f64,
    hessian_sum: f64,
    reg_alpha: f64,
    reg_lambda: f64,
) -> f64 {
    // L2 regularization: add lambda to hessian sum
    let h_reg = hessian_sum + reg_lambda;

    // Avoid division by zero
    if h_reg.abs() < EPSILON {
        return 0.0_f64;
    }

    // L1 regularization: soft threshold on gradient
    if reg_alpha > 0.0_f64 {
        let abs_g = gradient_sum.abs();
        if abs_g <= reg_alpha {
            return 0.0_f64;
        }
        let sign_g = if gradient_sum > 0.0_f64 {
            1.0_f64
        } else {
            -1.0_f64
        };
        return -sign_g * (abs_g - reg_alpha) / h_reg;
    }

    // Standard case (no L1): -G / (H + lambda)
    -gradient_sum / h_reg
}

/// Internal struct for tracking pending nodes during tree building.
#[derive(Debug)]
struct PendingNode {
    /// Sample indices at this node.
    sample_indices: Vec<usize>,

    /// Current depth.
    depth: usize,

    /// Parent node ID (None for root).
    parent_id: Option<usize>,

    /// Whether this is the left child of parent.
    is_left_child: bool,

    /// Cached histograms from parent's sibling subtraction (for 2x speedup).
    cached_histograms: Option<Vec<HistogramBuffer>>,
}

/// Internal struct for building tree nodes before finalization.
#[derive(Debug)]
struct BuildNode {
    /// Node ID.
    node_id: usize,

    /// Whether this is a leaf.
    is_leaf: bool,

    /// Feature index for split (None for leaf).
    feature_index: Option<usize>,

    /// Split bin index (None for leaf).
    split_bin: Option<usize>,

    /// Node value (leaf value or intermediate).
    value: f64,

    /// Number of samples.
    n_samples: usize,

    /// NaN direction.
    nan_goes_left: bool,
}

/// Configuration for `build_tree` to avoid too many arguments.
#[derive(Debug, Clone)]
pub struct BuildTreeInput<'a> {
    /// Sample indices to include in tree.
    pub sample_indices: &'a [usize],

    /// Gradients for all samples.
    pub gradients: &'a [f64],

    /// Hessians for all samples.
    pub hessians: &'a [f64],

    /// Bin assignments per sample per feature (n_samples x n_features).
    pub bins: &'a [Vec<usize>],

    /// Number of regular bins (excluding NaN bin).
    pub n_regular_bins: usize,

    /// Bin thresholds per feature for converting bin index to threshold.
    pub bin_thresholds: &'a [Vec<f64>],

    /// Tree build configuration.
    pub config: &'a TreeBuildConfig,

    /// Optional monotonic constraints per feature.
    pub monotonic_constraints: Option<&'a [MonotonicConstraint]>,
}

/// Builds a decision tree using histogram-based split finding.
///
/// Uses depth-first traversal with the sibling histogram subtraction trick
/// for 2x speedup on histogram building.
///
/// # Args
///
/// * `input` - Build tree input configuration.
/// * `hooks` - Dependency injection hooks for histogram building.
///
/// # Returns
///
/// Built decision tree.
///
/// # Errors
///
/// Returns error if:
/// - Input validation fails
/// - Histogram building fails
/// - Split finding fails
pub fn build_tree(input: &BuildTreeInput<'_>, hooks: &Hooks) -> Result<Tree, ClearGbmError> {
    let n_samples = input.sample_indices.len();

    // Handle empty input
    if n_samples == 0_usize {
        return Err(ClearGbmError::EmptyInput {
            context: "sample_indices".to_string(),
        });
    }

    // Validate input lengths
    if input.gradients.len() < n_samples {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("gradients.len() >= {n_samples}"),
            got: format!("gradients.len() = {}", input.gradients.len()),
        });
    }
    if input.hessians.len() < n_samples {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("hessians.len() >= {n_samples}"),
            got: format!("hessians.len() = {}", input.hessians.len()),
        });
    }

    let n_features = if input.bins.is_empty() {
        0_usize
    } else {
        input.bins.first().map_or(0_usize, Vec::len)
    };

    if n_features == 0_usize {
        return Err(ClearGbmError::EmptyInput {
            context: "bins (no features)".to_string(),
        });
    }

    // Number of bins in histogram (regular + 1 for NaN)
    let n_bins = input.n_regular_bins + 1_usize;

    let config = input.config;
    let split_config = config.split_config();

    // Build tree using depth-first stack
    let mut nodes: Vec<BuildNode> = Vec::new();
    let mut next_node_id = 0_usize;
    let mut n_leaves = 0_usize;
    let mut max_depth_found = 0_usize;

    // Child pointer tracking: node_id -> (left_child, right_child)
    let mut child_pointers: Vec<(Option<usize>, Option<usize>)> = Vec::new();

    // Stack of pending nodes
    let mut stack: Vec<PendingNode> = Vec::new();
    stack.push(PendingNode {
        sample_indices: input.sample_indices.to_vec(),
        depth: 0_usize,
        parent_id: None,
        is_left_child: false,
        cached_histograms: None,
    });

    while let Some(pending) = stack.pop() {
        let node_id = next_node_id;
        next_node_id += 1_usize;

        // Ensure child_pointers has space
        while child_pointers.len() <= node_id {
            child_pointers.push((None, None));
        }

        // Update parent's child pointer
        if let Some(parent_id) = pending.parent_id {
            if pending.is_left_child {
                child_pointers[parent_id].0 = Some(node_id);
            } else {
                child_pointers[parent_id].1 = Some(node_id);
            }
        }

        let current_n_samples = pending.sample_indices.len();
        let depth = pending.depth;

        if depth > max_depth_found {
            max_depth_found = depth;
        }

        // Check stopping criteria
        let should_be_leaf = should_stop(
            depth,
            current_n_samples,
            n_leaves,
            config.max_depth(),
            config.max_leaves(),
            split_config.min_samples_split(),
            split_config.min_samples_leaf(),
        );

        if should_be_leaf {
            // Create leaf node
            let (g_sum, h_sum) =
                compute_sums(&pending.sample_indices, input.gradients, input.hessians);
            let leaf_value =
                compute_leaf_value(g_sum, h_sum, config.reg_alpha(), config.reg_lambda());

            nodes.push(BuildNode {
                node_id,
                is_leaf: true,
                feature_index: None,
                split_bin: None,
                value: leaf_value,
                n_samples: current_n_samples,
                nan_goes_left: true,
            });
            n_leaves += 1_usize;
            continue;
        }

        // Build histograms for all features (uses cache from sibling subtraction if available)
        let hist_config = BuildHistogramConfig {
            sample_indices: &pending.sample_indices,
            gradients: input.gradients,
            hessians: input.hessians,
            bins: input.bins,
            n_features,
            n_bins,
            hooks,
            cached_histograms: pending.cached_histograms.as_deref(),
        };
        let histograms = match build_feature_histograms(&hist_config) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        // Find best split across all features
        let best_split = match find_best_split_across_features_internal(
            &histograms,
            split_config,
            input.n_regular_bins,
            input.monotonic_constraints,
        ) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };

        // If no valid split, create leaf
        let Some(split) = best_split else {
            let (g_sum, h_sum) =
                compute_sums(&pending.sample_indices, input.gradients, input.hessians);
            let leaf_value =
                compute_leaf_value(g_sum, h_sum, config.reg_alpha(), config.reg_lambda());

            nodes.push(BuildNode {
                node_id,
                is_leaf: true,
                feature_index: None,
                split_bin: None,
                value: leaf_value,
                n_samples: current_n_samples,
                nan_goes_left: true,
            });
            n_leaves += 1_usize;
            continue;
        };

        // Create internal node
        let (g_sum, h_sum) = compute_sums(&pending.sample_indices, input.gradients, input.hessians);
        let node_value = compute_leaf_value(g_sum, h_sum, config.reg_alpha(), config.reg_lambda());

        nodes.push(BuildNode {
            node_id,
            is_leaf: false,
            feature_index: Some(split.feature_index()),
            split_bin: Some(split.split_bin()),
            value: node_value,
            n_samples: current_n_samples,
            nan_goes_left: split.nan_goes_left(),
        });

        // Split samples into left and right
        let (left_indices, right_indices) = split_samples(
            &pending.sample_indices,
            input.bins,
            split.feature_index(),
            split.split_bin(),
            split.nan_goes_left(),
            input.n_regular_bins,
        );

        // Compute child histograms using sibling subtraction trick
        let child_hist_config = ChildHistogramConfig {
            left_indices: &left_indices,
            right_indices: &right_indices,
            gradients: input.gradients,
            hessians: input.hessians,
            bins: input.bins,
            n_features,
            n_bins,
            parent_histograms: &histograms,
            hooks,
        };
        let (left_histograms, right_histograms) = match compute_child_histograms(&child_hist_config)
        {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        // Push children to stack (right first so left is processed first)
        stack.push(PendingNode {
            sample_indices: right_indices,
            depth: depth + 1_usize,
            parent_id: Some(node_id),
            is_left_child: false,
            cached_histograms: Some(right_histograms),
        });
        stack.push(PendingNode {
            sample_indices: left_indices,
            depth: depth + 1_usize,
            parent_id: Some(node_id),
            is_left_child: true,
            cached_histograms: Some(left_histograms),
        });
    }

    // Finalize nodes with child pointers and convert to TreeNode
    let final_nodes = match finalize_nodes(&nodes, &child_pointers, input.bin_thresholds) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };

    Ok(Tree::new(final_nodes, max_depth_found, n_leaves))
}

/// Checks if node should be a leaf based on stopping criteria.
fn should_stop(
    depth: usize,
    n_samples: usize,
    n_leaves: usize,
    max_depth: usize,
    max_leaves: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
) -> bool {
    // Check max depth (0 = unlimited)
    if max_depth > 0_usize && depth >= max_depth {
        return true;
    }

    // Check max leaves (0 = unlimited)
    // Note: splitting adds 1 net leaf (removes 1, adds 2)
    if max_leaves > 0_usize && n_leaves + 1_usize >= max_leaves {
        return true;
    }

    // Check min samples to split
    if n_samples < min_samples_split {
        return true;
    }

    // Check if we can satisfy min_samples_leaf for both children
    if n_samples < 2_usize * min_samples_leaf {
        return true;
    }

    false
}

/// Computes gradient and hessian sums for a set of sample indices.
fn compute_sums(sample_indices: &[usize], gradients: &[f64], hessians: &[f64]) -> (f64, f64) {
    let mut g_sum = 0.0_f64;
    let mut h_sum = 0.0_f64;

    for &idx in sample_indices {
        if idx < gradients.len() {
            g_sum += gradients[idx];
        }
        if idx < hessians.len() {
            h_sum += hessians[idx];
        }
    }

    (g_sum, h_sum)
}

/// Configuration for building feature histograms.
struct BuildHistogramConfig<'a> {
    /// Sample indices.
    sample_indices: &'a [usize],
    /// Gradients for all samples.
    gradients: &'a [f64],
    /// Hessians for all samples.
    hessians: &'a [f64],
    /// Bin assignments per sample per feature.
    bins: &'a [Vec<usize>],
    /// Number of features.
    n_features: usize,
    /// Number of bins.
    n_bins: usize,
    /// Dependency injection hooks.
    hooks: &'a Hooks,
    /// Cached histograms from parent's sibling subtraction (for 2x speedup).
    cached_histograms: Option<&'a [HistogramBuffer]>,
}

/// Builds histograms for all features.
///
/// Uses cached histograms from parent's sibling subtraction when available,
/// otherwise builds histograms from scratch.
fn build_feature_histograms(
    config: &BuildHistogramConfig<'_>,
) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    // Use cached histograms if available (from parent's sibling subtraction)
    if let Some(cached) = config.cached_histograms {
        if cached.len() == config.n_features {
            return Ok(cached.to_vec());
        }
    }

    // Build histograms from scratch
    let mut histograms = Vec::with_capacity(config.n_features);

    for feat_idx in 0_usize..config.n_features {
        // Extract bins for this feature
        let feat_bins: Vec<usize> = config
            .sample_indices
            .iter()
            .filter_map(|&idx| config.bins.get(idx).and_then(|b| b.get(feat_idx).copied()))
            .collect();

        // Build histogram
        let sample_idx_vec: Vec<usize> = (0_usize..feat_bins.len()).collect();
        let feat_gradients: Vec<f64> = config
            .sample_indices
            .iter()
            .filter_map(|&idx| config.gradients.get(idx).copied())
            .collect();
        let feat_hessians: Vec<f64> = config
            .sample_indices
            .iter()
            .filter_map(|&idx| config.hessians.get(idx).copied())
            .collect();

        let hist = match (config.hooks.build_histogram)(
            &sample_idx_vec,
            &feat_gradients,
            &feat_hessians,
            &feat_bins,
            config.n_bins,
        ) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };
        histograms.push(hist);
    }

    Ok(histograms)
}

/// Finds best split across all features.
fn find_best_split_across_features_internal(
    histograms: &[HistogramBuffer],
    config: &SplitConfig,
    n_regular_bins: usize,
    monotonic_constraints: Option<&[MonotonicConstraint]>,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let mut best_split: Option<SplitResult> = None;

    for (feature_idx, histogram) in histograms.iter().enumerate() {
        let constraint = monotonic_constraints
            .and_then(|constraints| constraints.get(feature_idx).copied())
            .unwrap_or(MonotonicConstraint::None);

        let maybe_split = match find_best_split_from_histogram(
            histogram,
            feature_idx,
            config,
            n_regular_bins,
            constraint,
        ) {
            Ok(s) => s,
            Err(e) => return Err(e),
        };

        if let Some(split) = maybe_split {
            let is_better = best_split
                .as_ref()
                .is_none_or(|current| split.gain() > current.gain());

            if is_better {
                best_split = Some(split);
            }
        }
    }

    Ok(best_split)
}

/// Splits samples into left and right based on split result.
fn split_samples(
    sample_indices: &[usize],
    bins: &[Vec<usize>],
    feature_index: usize,
    split_bin: usize,
    nan_goes_left: bool,
    n_regular_bins: usize,
) -> (Vec<usize>, Vec<usize>) {
    let nan_bin = n_regular_bins;
    let mut left = Vec::new();
    let mut right = Vec::new();

    for &idx in sample_indices {
        let bin = bins
            .get(idx)
            .and_then(|b| b.get(feature_index).copied())
            .unwrap_or(nan_bin);

        // NaN bin handling
        if bin == nan_bin {
            if nan_goes_left {
                left.push(idx);
            } else {
                right.push(idx);
            }
        } else if bin <= split_bin {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    (left, right)
}

/// Configuration for computing child histograms.
struct ChildHistogramConfig<'a> {
    /// Left child sample indices.
    left_indices: &'a [usize],
    /// Right child sample indices.
    right_indices: &'a [usize],
    /// Gradients for all samples.
    gradients: &'a [f64],
    /// Hessians for all samples.
    hessians: &'a [f64],
    /// Bin assignments per sample per feature.
    bins: &'a [Vec<usize>],
    /// Number of features.
    n_features: usize,
    /// Number of bins.
    n_bins: usize,
    /// Parent histograms.
    parent_histograms: &'a [HistogramBuffer],
    /// Dependency injection hooks.
    hooks: &'a Hooks,
}

/// Computes child histograms using sibling subtraction trick.
///
/// Builds histogram for smaller child, derives larger child via subtraction.
fn compute_child_histograms(
    config: &ChildHistogramConfig<'_>,
) -> Result<(Vec<HistogramBuffer>, Vec<HistogramBuffer>), ClearGbmError> {
    let n_left = config.left_indices.len();
    let n_right = config.right_indices.len();
    let left_is_smaller = n_left <= n_right;

    let smaller_indices = if left_is_smaller {
        config.left_indices
    } else {
        config.right_indices
    };

    let mut left_histograms = Vec::with_capacity(config.n_features);
    let mut right_histograms = Vec::with_capacity(config.n_features);

    for feat_idx in 0_usize..config.n_features {
        // Build histogram for smaller child
        let feat_bins: Vec<usize> = smaller_indices
            .iter()
            .filter_map(|&idx| config.bins.get(idx).and_then(|b| b.get(feat_idx).copied()))
            .collect();

        let sample_idx_vec: Vec<usize> = (0_usize..feat_bins.len()).collect();
        let feat_gradients: Vec<f64> = smaller_indices
            .iter()
            .filter_map(|&idx| config.gradients.get(idx).copied())
            .collect();
        let feat_hessians: Vec<f64> = smaller_indices
            .iter()
            .filter_map(|&idx| config.hessians.get(idx).copied())
            .collect();

        let smaller_hist = match (config.hooks.build_histogram)(
            &sample_idx_vec,
            &feat_gradients,
            &feat_hessians,
            &feat_bins,
            config.n_bins,
        ) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        // Derive larger child via subtraction
        let parent_hist = match config.parent_histograms.get(feat_idx) {
            Some(h) => h,
            None => {
                return Err(ClearGbmError::FeatureIndexOutOfBounds {
                    index: feat_idx,
                    n_features: config.n_features,
                })
            }
        };

        let larger_hist = match subtract_histogram(parent_hist, &smaller_hist) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };

        if left_is_smaller {
            left_histograms.push(smaller_hist);
            right_histograms.push(larger_hist);
        } else {
            left_histograms.push(larger_hist);
            right_histograms.push(smaller_hist);
        }
    }

    Ok((left_histograms, right_histograms))
}

/// Finalizes build nodes into TreeNode with proper child pointers and thresholds.
fn finalize_nodes(
    build_nodes: &[BuildNode],
    child_pointers: &[(Option<usize>, Option<usize>)],
    bin_thresholds: &[Vec<f64>],
) -> Result<Vec<TreeNode>, ClearGbmError> {
    let mut final_nodes = Vec::with_capacity(build_nodes.len());

    for node in build_nodes {
        let (left_child, right_child) = child_pointers
            .get(node.node_id)
            .copied()
            .unwrap_or((None, None));

        if node.is_leaf {
            final_nodes.push(TreeNode::new_leaf(node.node_id, node.value, node.n_samples));
        } else {
            // Convert split_bin to threshold
            let feature_index = match node.feature_index {
                Some(f) => f,
                None => {
                    return Err(ClearGbmError::TreeConstructionFailed {
                        reason: "internal node missing feature_index".to_string(),
                    })
                }
            };
            let split_bin = match node.split_bin {
                Some(s) => s,
                None => {
                    return Err(ClearGbmError::TreeConstructionFailed {
                        reason: "internal node missing split_bin".to_string(),
                    })
                }
            };

            // Get threshold from bin_thresholds
            // Threshold is the upper bound of the split_bin
            let threshold = bin_thresholds
                .get(feature_index)
                .and_then(|thresholds| thresholds.get(split_bin).copied())
                .unwrap_or(0.0_f64);

            let left_id = match left_child {
                Some(l) => l,
                None => {
                    return Err(ClearGbmError::TreeConstructionFailed {
                        reason: format!("internal node {} missing left_child", node.node_id),
                    })
                }
            };
            let right_id = match right_child {
                Some(r) => r,
                None => {
                    return Err(ClearGbmError::TreeConstructionFailed {
                        reason: format!("internal node {} missing right_child", node.node_id),
                    })
                }
            };

            final_nodes.push(TreeNode::new_internal(TreeNodeConfig {
                node_id: node.node_id,
                feature_index,
                threshold,
                value: node.value,
                n_samples: node.n_samples,
                left_child: left_id,
                right_child: right_id,
                nan_goes_left: node.nan_goes_left,
            }));
        }
    }

    Ok(final_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;

    /// Failing serializer for testing error propagation paths.
    mod failing_serializer {
        use core::fmt::{self, Display};
        use serde::ser::{self, Serialize};

        #[derive(Debug)]
        pub struct FailError {
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

        pub struct FailAfterN {
            count: usize,
            fail_after: usize,
            fail_on_struct: bool,
        }

        impl FailAfterN {
            pub fn new(fail_after: usize) -> Self {
                FailAfterN {
                    count: 0,
                    fail_after,
                    fail_on_struct: false,
                }
            }

            pub fn fail_on_struct() -> Self {
                FailAfterN {
                    count: 0,
                    fail_after: usize::MAX,
                    fail_on_struct: true,
                }
            }
        }

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
            fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<(), FailError> {
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
            fn serialize_newtype_struct<T: ?Sized + Serialize>(
                self,
                _name: &'static str,
                _value: &T,
            ) -> Result<(), FailError> {
                Ok(())
            }
            fn serialize_newtype_variant<T: ?Sized + Serialize>(
                self,
                _name: &'static str,
                _idx: u32,
                _variant: &'static str,
                _value: &T,
            ) -> Result<(), FailError> {
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

    /// Failing deserializer for testing error propagation paths.
    mod failing_deserializer {
        use core::fmt::{self, Display};
        use serde::de::{self, Visitor};

        #[derive(Debug)]
        pub struct DeError {
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

        pub struct IntegerDeserializer;

        impl<'de> de::Deserializer<'de> for IntegerDeserializer {
            type Error = DeError;

            fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                visitor.visit_i64(42_i64)
            }

            serde::forward_to_deserialize_any! {
                bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string
                bytes byte_buf option unit unit_struct newtype_struct seq
                tuple tuple_struct map struct enum identifier ignored_any
            }
        }

        pub struct MapWithIntegerKeyDeserializer;

        pub struct IntegerKeyMapAccess {
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
                seed.deserialize(IntegerDeserializer).map(Some)
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

        impl<'de> de::Deserializer<'de> for MapWithIntegerKeyDeserializer {
            type Error = DeError;

            fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
            where
                V: Visitor<'de>,
            {
                visitor.visit_map(IntegerKeyMapAccess { done: false })
            }

            serde::forward_to_deserialize_any! {
                bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string
                bytes byte_buf option unit unit_struct newtype_struct seq
                tuple tuple_struct map struct enum identifier ignored_any
            }
        }
    }

    // =========================================================================
    // Property-based tests with proptest
    // =========================================================================

    #[test]
    fn prop_compute_leaf_value_zero_hessian_returns_zero() -> Result<(), ClearGbmError> {
        let config = proptest::test_runner::Config::with_cases(100);
        let mut runner = proptest::test_runner::TestRunner::new(config);
        runner
            .run(
                &(
                    -1000.0_f64..1000.0_f64,
                    0.0_f64..10.0_f64,
                    0.0_f64..10.0_f64,
                ),
                |(gradient, reg_alpha, reg_lambda)| {
                    // When hessian + lambda is near zero, should return 0
                    let hessian = 0.0_f64;
                    if reg_lambda < EPSILON {
                        let value = compute_leaf_value(gradient, hessian, reg_alpha, reg_lambda);
                        prop_assert!(value.abs() < EPSILON, "Expected 0, got {}", value);
                    }
                    Ok(())
                },
            )
            .map_err(|e| ClearGbmError::InvalidParameter {
                name: "proptest".to_string(),
                reason: format!("{}", e),
            })
    }

    #[test]
    fn prop_compute_leaf_value_l1_soft_threshold() -> Result<(), ClearGbmError> {
        let config = proptest::test_runner::Config::with_cases(100);
        let mut runner = proptest::test_runner::TestRunner::new(config);
        runner
            .run(
                &(
                    -100.0_f64..100.0_f64,
                    1.0_f64..100.0_f64,
                    0.0_f64..50.0_f64,
                    0.0_f64..10.0_f64,
                ),
                |(gradient, hessian, reg_alpha, reg_lambda)| {
                    let value = compute_leaf_value(gradient, hessian, reg_alpha, reg_lambda);

                    // L1 soft threshold: if |G| <= alpha, value should be 0
                    if gradient.abs() <= reg_alpha {
                        prop_assert!(
                            value.abs() < EPSILON,
                            "Expected 0 when |G| <= alpha, got {}",
                            value
                        );
                    }

                    // Value should be finite
                    prop_assert!(value.is_finite(), "Value should be finite, got {}", value);
                    Ok(())
                },
            )
            .map_err(|e| ClearGbmError::InvalidParameter {
                name: "proptest".to_string(),
                reason: format!("{}", e),
            })
    }

    #[test]
    fn prop_compute_leaf_value_sign_correct() -> Result<(), ClearGbmError> {
        let config = proptest::test_runner::Config::with_cases(100);
        let mut runner = proptest::test_runner::TestRunner::new(config);
        runner
            .run(
                &(-100.0_f64..100.0_f64, 1.0_f64..100.0_f64),
                |(gradient, hessian)| {
                    // Without regularization: -G/H
                    let value = compute_leaf_value(gradient, hessian, 0.0_f64, 0.0_f64);

                    // Sign should be opposite of gradient (when hessian > 0)
                    if gradient.abs() > EPSILON {
                        let expected_sign = if gradient > 0.0_f64 {
                            -1.0_f64
                        } else {
                            1.0_f64
                        };
                        let actual_sign = if value > 0.0_f64 { 1.0_f64 } else { -1.0_f64 };
                        prop_assert_eq!(
                            expected_sign,
                            actual_sign,
                            "Sign mismatch: G={}, value={}",
                            gradient,
                            value
                        );
                    }
                    Ok(())
                },
            )
            .map_err(|e| ClearGbmError::InvalidParameter {
                name: "proptest".to_string(),
                reason: format!("{}", e),
            })
    }

    #[test]
    fn prop_should_stop_respects_constraints() -> Result<(), ClearGbmError> {
        let config = proptest::test_runner::Config::with_cases(100);
        let mut runner = proptest::test_runner::TestRunner::new(config);
        runner
            .run(
                &(
                    0_usize..20_usize,
                    1_usize..1000_usize,
                    0_usize..100_usize,
                    0_usize..15_usize,
                    0_usize..50_usize,
                    2_usize..50_usize,
                    1_usize..25_usize,
                ),
                |(
                    depth,
                    n_samples,
                    n_leaves,
                    max_depth,
                    max_leaves,
                    min_samples_split,
                    min_samples_leaf,
                )| {
                    let result = should_stop(
                        depth,
                        n_samples,
                        n_leaves,
                        max_depth,
                        max_leaves,
                        min_samples_split,
                        min_samples_leaf,
                    );

                    // If max_depth > 0 and depth >= max_depth, must stop
                    if max_depth > 0_usize && depth >= max_depth {
                        prop_assert!(result, "Should stop when depth >= max_depth");
                    }

                    // If max_leaves > 0 and n_leaves + 1 >= max_leaves, must stop
                    if max_leaves > 0_usize && n_leaves + 1_usize >= max_leaves {
                        prop_assert!(result, "Should stop when approaching max_leaves");
                    }

                    // If n_samples < min_samples_split, must stop
                    if n_samples < min_samples_split {
                        prop_assert!(result, "Should stop when n_samples < min_samples_split");
                    }

                    // If n_samples < 2 * min_samples_leaf, must stop
                    if n_samples < 2_usize * min_samples_leaf {
                        prop_assert!(result, "Should stop when n_samples < 2 * min_samples_leaf");
                    }
                    Ok(())
                },
            )
            .map_err(|e| ClearGbmError::InvalidParameter {
                name: "proptest".to_string(),
                reason: format!("{}", e),
            })
    }

    #[test]
    fn prop_split_samples_preserves_count() -> Result<(), ClearGbmError> {
        let config = proptest::test_runner::Config::with_cases(100);
        let mut runner = proptest::test_runner::TestRunner::new(config);
        runner
            .run(
                &(2_usize..20_usize, 0_usize..5_usize, proptest::bool::ANY),
                |(n_samples, split_bin, nan_goes_left)| {
                    let n_regular_bins = 6_usize;
                    let sample_indices: Vec<usize> = (0_usize..n_samples).collect();

                    // Create bins that distribute samples across bins
                    let bins: Vec<Vec<usize>> = (0_usize..n_samples)
                        .map(|i| vec![i % n_regular_bins])
                        .collect();

                    let (left, right) = split_samples(
                        &sample_indices,
                        &bins,
                        0_usize,
                        split_bin,
                        nan_goes_left,
                        n_regular_bins,
                    );

                    // Total samples should be preserved
                    prop_assert_eq!(
                        left.len() + right.len(),
                        n_samples,
                        "Sample count not preserved: left={}, right={}, total={}",
                        left.len(),
                        right.len(),
                        n_samples
                    );

                    // No duplicates
                    let mut all: Vec<usize> = left.iter().chain(right.iter()).copied().collect();
                    all.sort();
                    all.dedup();
                    prop_assert_eq!(all.len(), n_samples, "Duplicate samples found");
                    Ok(())
                },
            )
            .map_err(|e| ClearGbmError::InvalidParameter {
                name: "proptest".to_string(),
                reason: format!("{}", e),
            })
    }

    // =========================================================================
    // TreeBuildConfig tests
    // =========================================================================

    #[test]
    fn test_tree_build_config_new_valid() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let c = match TreeBuildConfig::new(5_usize, 10_usize, 0.0_f64, 1.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        assert_eq!(c.max_depth(), 5_usize);
        assert_eq!(c.max_leaves(), 10_usize);
        assert!(c.reg_alpha().abs() < EPSILON);
        assert!((c.reg_lambda() - 1.0_f64).abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_tree_build_config_negative_reg_alpha() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let config = TreeBuildConfig::new(5_usize, 10_usize, -0.1_f64, 1.0_f64, sc);

        assert!(config.is_err());
        assert!(matches!(
            config.err(),
            Some(ClearGbmError::InvalidParameter { name, .. }) if name == "reg_alpha"
        ));
        Ok(())
    }

    #[test]
    fn test_tree_build_config_negative_reg_lambda() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let config = TreeBuildConfig::new(5_usize, 10_usize, 0.0_f64, -1.0_f64, sc);

        assert!(config.is_err());
        assert!(matches!(
            config.err(),
            Some(ClearGbmError::InvalidParameter { name, .. }) if name == "reg_lambda"
        ));
        Ok(())
    }

    // =========================================================================
    // Tree tests
    // =========================================================================

    #[test]
    fn test_tree_new() -> Result<(), ClearGbmError> {
        let leaf = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        let tree = Tree::new(vec![leaf], 0_usize, 1_usize);

        assert_eq!(tree.n_nodes(), 1_usize);
        assert_eq!(tree.n_leaves(), 1_usize);
        assert_eq!(tree.max_depth(), 0_usize);
        Ok(())
    }

    #[test]
    fn test_tree_nodes_accessor() -> Result<(), ClearGbmError> {
        let leaf1 = TreeNode::new_leaf(0_usize, 0.5_f64, 50_usize);
        let leaf2 = TreeNode::new_leaf(1_usize, -0.5_f64, 50_usize);
        let tree = Tree::new(vec![leaf1, leaf2], 0_usize, 2_usize);

        let nodes = tree.nodes();
        assert_eq!(nodes.len(), 2_usize);
        assert_eq!(nodes[0_usize].node_id(), 0_usize);
        assert_eq!(nodes[1_usize].node_id(), 1_usize);
        Ok(())
    }

    #[test]
    fn test_tree_node_access() -> Result<(), ClearGbmError> {
        let leaf = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        let tree = Tree::new(vec![leaf.clone()], 0_usize, 1_usize);

        let node = match tree.root() {
            Ok(n) => n,
            Err(e) => return Err(e),
        };
        assert_eq!(node.node_id(), 0_usize);
        assert!(node.is_leaf());

        let node0 = match tree.node(0_usize) {
            Ok(n) => n,
            Err(e) => return Err(e),
        };
        assert_eq!(node0.node_id(), 0_usize);

        let missing = tree.node(99_usize);
        assert!(missing.is_err());
        assert!(matches!(
            missing.err(),
            Some(ClearGbmError::NodeNotFound { node_id: 99_usize })
        ));
        Ok(())
    }

    #[test]
    fn test_tree_empty_root_error() -> Result<(), ClearGbmError> {
        let tree = Tree::new(vec![], 0_usize, 0_usize);
        let root = tree.root();
        assert!(root.is_err());
        assert!(matches!(
            root.err(),
            Some(ClearGbmError::NodeNotFound { node_id: 0_usize })
        ));
        Ok(())
    }

    #[test]
    fn test_tree_serialize_deserialize() -> Result<(), ClearGbmError> {
        let leaf = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        let tree = Tree::new(vec![leaf], 1_usize, 1_usize);

        let json_str = match serde_json::to_string(&tree) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let p: Tree = match serde_json::from_str(&json_str) {
            Ok(t) => t,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };

        assert_eq!(p.n_nodes(), 1_usize);
        assert_eq!(p.max_depth(), 1_usize);
        assert_eq!(p.n_leaves(), 1_usize);
        Ok(())
    }

    #[test]
    fn test_tree_build_config_getters() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.5_f64, 0.01_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let c = match TreeBuildConfig::new(5_usize, 10_usize, 0.1_f64, 0.5_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        assert_eq!(c.max_depth(), 5_usize);
        assert_eq!(c.max_leaves(), 10_usize);
        assert!((c.reg_alpha() - 0.1_f64).abs() < EPSILON);
        assert!((c.reg_lambda() - 0.5_f64).abs() < EPSILON);
        assert_eq!(c.split_config().min_samples_split(), 2_usize);
        Ok(())
    }

    #[test]
    fn test_tree_build_config_serialize_deserialize() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let c = match TreeBuildConfig::new(5_usize, 10_usize, 0.1_f64, 0.5_f64, sc) {
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
        let p: TreeBuildConfig = match serde_json::from_str(&json_str) {
            Ok(c) => c,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };

        assert_eq!(p.max_depth(), 5_usize);
        assert_eq!(p.max_leaves(), 10_usize);
        Ok(())
    }

    // =========================================================================
    // compute_leaf_value tests
    // =========================================================================

    #[test]
    fn test_compute_leaf_value_basic() -> Result<(), ClearGbmError> {
        // Simple case: -G/H = -2.0/10.0 = -0.2
        let value = compute_leaf_value(2.0_f64, 10.0_f64, 0.0_f64, 0.0_f64);
        assert!((value - (-0.2_f64)).abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_compute_leaf_value_with_l2() -> Result<(), ClearGbmError> {
        // With L2: -G/(H + lambda) = -2.0/(10.0 + 1.0) = -2.0/11.0
        let value = compute_leaf_value(2.0_f64, 10.0_f64, 0.0_f64, 1.0_f64);
        let expected = -2.0_f64 / 11.0_f64;
        assert!((value - expected).abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_compute_leaf_value_with_l1() -> Result<(), ClearGbmError> {
        // With L1: soft threshold
        // G = 2.0, alpha = 0.5
        // sign(G) = 1, |G| = 2.0 > alpha
        // value = -1 * (2.0 - 0.5) / (10.0 + 0.0) = -1.5 / 10.0 = -0.15
        let value = compute_leaf_value(2.0_f64, 10.0_f64, 0.5_f64, 0.0_f64);
        let expected = -1.5_f64 / 10.0_f64;
        assert!((value - expected).abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_compute_leaf_value_l1_below_threshold() -> Result<(), ClearGbmError> {
        // With L1: |G| <= alpha, value = 0
        let value = compute_leaf_value(0.3_f64, 10.0_f64, 0.5_f64, 0.0_f64);
        assert!(value.abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_compute_leaf_value_zero_hessian() -> Result<(), ClearGbmError> {
        // Zero hessian should return 0
        let value = compute_leaf_value(2.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        assert!(value.abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_compute_leaf_value_negative_gradient() -> Result<(), ClearGbmError> {
        // Negative gradient: -(-2.0)/10.0 = 0.2
        let value = compute_leaf_value(-2.0_f64, 10.0_f64, 0.0_f64, 0.0_f64);
        assert!((value - 0.2_f64).abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_compute_leaf_value_negative_gradient_with_l1() -> Result<(), ClearGbmError> {
        // Negative gradient with L1: soft threshold
        // G = -2.0, alpha = 0.5
        // sign(G) = -1, |G| = 2.0 > alpha
        // value = -(-1) * (2.0 - 0.5) / (10.0 + 0.0) = 1.5 / 10.0 = 0.15
        let value = compute_leaf_value(-2.0_f64, 10.0_f64, 0.5_f64, 0.0_f64);
        let expected = 1.5_f64 / 10.0_f64;
        assert!((value - expected).abs() < EPSILON);
        Ok(())
    }

    // =========================================================================
    // should_stop tests
    // =========================================================================

    #[test]
    fn test_should_stop_max_depth() -> Result<(), ClearGbmError> {
        assert!(should_stop(
            5_usize, 100_usize, 0_usize, 5_usize, 0_usize, 2_usize, 1_usize
        ));
        assert!(!should_stop(
            4_usize, 100_usize, 0_usize, 5_usize, 0_usize, 2_usize, 1_usize
        ));
        Ok(())
    }

    #[test]
    fn test_should_stop_unlimited_depth() -> Result<(), ClearGbmError> {
        // max_depth = 0 means unlimited
        assert!(!should_stop(
            100_usize, 100_usize, 0_usize, 0_usize, 0_usize, 2_usize, 1_usize
        ));
        Ok(())
    }

    #[test]
    fn test_should_stop_max_leaves() -> Result<(), ClearGbmError> {
        // max_leaves = 10, n_leaves = 9, would add 1 more -> stop
        assert!(should_stop(
            2_usize, 100_usize, 9_usize, 0_usize, 10_usize, 2_usize, 1_usize
        ));
        assert!(!should_stop(
            2_usize, 100_usize, 8_usize, 0_usize, 10_usize, 2_usize, 1_usize
        ));
        Ok(())
    }

    #[test]
    fn test_should_stop_min_samples_split() -> Result<(), ClearGbmError> {
        assert!(should_stop(
            2_usize, 5_usize, 0_usize, 0_usize, 0_usize, 10_usize, 1_usize
        ));
        assert!(!should_stop(
            2_usize, 15_usize, 0_usize, 0_usize, 0_usize, 10_usize, 1_usize
        ));
        Ok(())
    }

    #[test]
    fn test_should_stop_min_samples_leaf() -> Result<(), ClearGbmError> {
        // n_samples = 5, min_samples_leaf = 3, need 6 samples minimum
        assert!(should_stop(
            2_usize, 5_usize, 0_usize, 0_usize, 0_usize, 2_usize, 3_usize
        ));
        assert!(!should_stop(
            2_usize, 10_usize, 0_usize, 0_usize, 0_usize, 2_usize, 3_usize
        ));
        Ok(())
    }

    // =========================================================================
    // split_samples tests
    // =========================================================================

    #[test]
    fn test_split_samples_basic() -> Result<(), ClearGbmError> {
        // 5 samples, bins for feature 0
        let bins = vec![
            vec![0_usize], // sample 0, feature 0 -> bin 0
            vec![1_usize], // sample 1, feature 0 -> bin 1
            vec![2_usize], // sample 2, feature 0 -> bin 2
            vec![0_usize], // sample 3, feature 0 -> bin 0
            vec![1_usize], // sample 4, feature 0 -> bin 1
        ];
        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize, 4_usize];

        // Split at bin 0 (samples in bin <= 0 go left)
        let (left, right) = split_samples(&sample_indices, &bins, 0_usize, 0_usize, true, 3_usize);

        // Left: bins 0 (samples 0, 3)
        assert_eq!(left.len(), 2_usize);
        assert!(left.contains(&0_usize));
        assert!(left.contains(&3_usize));

        // Right: bins 1, 2 (samples 1, 2, 4)
        assert_eq!(right.len(), 3_usize);
        assert!(right.contains(&1_usize));
        assert!(right.contains(&2_usize));
        assert!(right.contains(&4_usize));
        Ok(())
    }

    #[test]
    fn test_split_samples_nan_handling() -> Result<(), ClearGbmError> {
        // Sample with NaN bin (= n_regular_bins)
        let bins = vec![
            vec![0_usize], // sample 0 -> bin 0
            vec![3_usize], // sample 1 -> NaN bin (n_regular_bins = 3)
        ];
        let sample_indices = vec![0_usize, 1_usize];

        // NaN goes left
        let (left, right) = split_samples(&sample_indices, &bins, 0_usize, 0_usize, true, 3_usize);
        assert!(left.contains(&0_usize)); // bin 0
        assert!(left.contains(&1_usize)); // NaN goes left
        assert!(right.is_empty());

        // NaN goes right
        let (left2, right2) =
            split_samples(&sample_indices, &bins, 0_usize, 0_usize, false, 3_usize);
        assert!(left2.contains(&0_usize)); // bin 0
        assert!(right2.contains(&1_usize)); // NaN goes right
        Ok(())
    }

    // =========================================================================
    // build_tree tests
    // =========================================================================

    #[test]
    fn test_build_tree_single_leaf() -> Result<(), ClearGbmError> {
        // Create simple data that results in a single leaf
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(1_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize];
        let gradients = vec![0.1_f64, 0.1_f64, 0.1_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![0_usize], vec![0_usize]];
        let bin_thresholds = vec![vec![0.5_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 1_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let tree = match build_tree(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };

        // Should be a single leaf (max_depth = 1, samples in same bin)
        assert_eq!(tree.n_leaves(), 1_usize);
        let _ = match tree.root() {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        Ok(())
    }

    #[test]
    fn test_build_tree_with_split() -> Result<(), ClearGbmError> {
        // Create data with clear split
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
        // Left samples (bins 0,1) have positive gradients
        // Right samples (bins 2,3) have negative gradients
        let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let tree = match build_tree(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };

        // Should have split
        assert!(tree.n_nodes() >= 3_usize);
        assert!(tree.n_leaves() >= 2_usize);

        // Root should not be a leaf
        let root = match tree.root() {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        assert!(!root.is_leaf());
        Ok(())
    }

    #[test]
    fn test_build_tree_empty_input() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices: Vec<usize> = vec![];
        let gradients: Vec<f64> = vec![];
        let hessians: Vec<f64> = vec![];
        let bins: Vec<Vec<usize>> = vec![];
        let bin_thresholds: Vec<Vec<f64>> = vec![];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let result = build_tree(&input, &Hooks::default());
        assert!(result.is_err());
        assert!(matches!(
            result.err(),
            Some(ClearGbmError::EmptyInput { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_build_tree_max_depth_constraint() -> Result<(), ClearGbmError> {
        // max_depth = 1 should create root + 2 leaves max
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(1_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
        let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let tree = match build_tree(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };

        // Max depth = 1, so max 3 nodes (root + 2 leaves)
        assert!(tree.n_nodes() <= 3_usize);
        assert!(tree.max_depth() <= 1_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_max_leaves_constraint() -> Result<(), ClearGbmError> {
        // max_leaves = 2
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(10_usize, 2_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
        let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let tree = match build_tree(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(tree.n_leaves() <= 2_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_gradients_too_short() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize];
        let gradients = vec![1.0_f64]; // Too short!
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize]];
        let bin_thresholds = vec![vec![0.5_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 3_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let result = build_tree(&input, &Hooks::default());
        assert!(result.is_err());
        assert!(matches!(
            result.err(),
            Some(ClearGbmError::ShapeMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_build_tree_hessians_too_short() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize];
        let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let hessians = vec![1.0_f64]; // Too short!
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize]];
        let bin_thresholds = vec![vec![0.5_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 3_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let result = build_tree(&input, &Hooks::default());
        assert!(result.is_err());
        assert!(matches!(
            result.err(),
            Some(ClearGbmError::ShapeMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_build_tree_no_features() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize];
        let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
        // No features - empty inner vecs
        let bins: Vec<Vec<usize>> = vec![vec![], vec![], vec![]];
        let bin_thresholds: Vec<Vec<f64>> = vec![];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 3_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let result = build_tree(&input, &Hooks::default());
        assert!(result.is_err());
        assert!(matches!(
            result.err(),
            Some(ClearGbmError::EmptyInput { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_build_tree_empty_bins_vec() -> Result<(), ClearGbmError> {
        // Test where bins vec itself is empty (different from empty inner vecs)
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize];
        let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
        // Completely empty bins vec
        let bins: Vec<Vec<usize>> = vec![];
        let bin_thresholds: Vec<Vec<f64>> = vec![];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 3_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let result = build_tree(&input, &Hooks::default());
        assert!(result.is_err());
        assert!(matches!(
            result.err(),
            Some(ClearGbmError::EmptyInput { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_build_tree_with_monotonic_constraints() -> Result<(), ClearGbmError> {
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
        let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];
        let constraints = vec![MonotonicConstraint::Increasing];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: Some(&constraints),
        };

        // Should succeed (constraint may or may not affect the split)
        let _ = match build_tree(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        Ok(())
    }

    #[test]
    fn test_build_tree_with_l1_regularization() -> Result<(), ClearGbmError> {
        // Use L1 regularization
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.5_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
        let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let _ = match build_tree(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        Ok(())
    }

    #[test]
    fn test_build_tree_left_larger_than_right() -> Result<(), ClearGbmError> {
        // Test where left child has more samples than right
        // This exercises the else branch in compute_child_histograms
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        // 6 samples: 4 in low bins (left), 2 in high bins (right)
        // This makes left child larger than right child
        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize, 4_usize, 5_usize];
        let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, -2.0_f64, -2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        // First 4 samples in bins 0-1 (left), last 2 in bins 2-3 (right)
        let bins = vec![
            vec![0_usize],
            vec![0_usize],
            vec![1_usize],
            vec![1_usize],
            vec![2_usize],
            vec![3_usize],
        ];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let tree = match build_tree(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };

        // Should have split into left (4 samples) and right (2 samples)
        assert!(tree.n_nodes() >= 3_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_deep_tree() -> Result<(), ClearGbmError> {
        // Test building a deeper tree to exercise more code paths
        // Allow deep tree
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(10_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        // 8 samples with varying gradients
        let sample_indices = vec![
            0_usize, 1_usize, 2_usize, 3_usize, 4_usize, 5_usize, 6_usize, 7_usize,
        ];
        let gradients = vec![
            4.0_f64, 3.0_f64, 2.0_f64, 1.0_f64, -1.0_f64, -2.0_f64, -3.0_f64, -4.0_f64,
        ];
        let hessians = vec![
            1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64,
        ];
        let bins = vec![
            vec![0_usize],
            vec![1_usize],
            vec![2_usize],
            vec![3_usize],
            vec![4_usize],
            vec![5_usize],
            vec![6_usize],
            vec![7_usize],
        ];
        let bin_thresholds = vec![vec![
            0.125_f64, 0.25_f64, 0.375_f64, 0.5_f64, 0.625_f64, 0.75_f64, 0.875_f64, 1.0_f64,
        ]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 8_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let tree = match build_tree(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };

        // Should have multiple nodes
        assert!(tree.n_nodes() > 1_usize);
        // Should have at least 2 leaves
        assert!(tree.n_leaves() >= 2_usize);
        // Test nodes accessor
        let nodes = tree.nodes();
        assert_eq!(nodes.len(), tree.n_nodes());
        Ok(())
    }

    // =========================================================================
    // finalize_nodes error path tests
    // =========================================================================

    #[test]
    fn test_finalize_nodes_internal_node_missing_feature_index() -> Result<(), ClearGbmError> {
        // Create an internal node (is_leaf=false) without feature_index
        let build_nodes = vec![BuildNode {
            node_id: 0_usize,
            is_leaf: false, // internal node
            value: 0.0_f64,
            n_samples: 10_usize,
            feature_index: None, // missing!
            split_bin: Some(1_usize),
            nan_goes_left: true,
        }];
        let child_pointers = vec![(Some(1_usize), Some(2_usize))];
        let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

        let result = finalize_nodes(&build_nodes, &child_pointers, &bin_thresholds);
        assert!(matches!(
            result,
            Err(ClearGbmError::TreeConstructionFailed { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_finalize_nodes_internal_node_missing_split_bin() -> Result<(), ClearGbmError> {
        // Create an internal node without split_bin
        let build_nodes = vec![BuildNode {
            node_id: 0_usize,
            is_leaf: false,
            value: 0.0_f64,
            n_samples: 10_usize,
            feature_index: Some(0_usize),
            split_bin: None, // missing!
            nan_goes_left: true,
        }];
        let child_pointers = vec![(Some(1_usize), Some(2_usize))];
        let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

        let result = finalize_nodes(&build_nodes, &child_pointers, &bin_thresholds);
        assert!(matches!(
            result,
            Err(ClearGbmError::TreeConstructionFailed { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_finalize_nodes_internal_node_missing_left_child() -> Result<(), ClearGbmError> {
        // Create an internal node with missing left child in child_pointers
        let build_nodes = vec![BuildNode {
            node_id: 0_usize,
            is_leaf: false,
            value: 0.0_f64,
            n_samples: 10_usize,
            feature_index: Some(0_usize),
            split_bin: Some(0_usize),
            nan_goes_left: true,
        }];
        let child_pointers = vec![(None, Some(2_usize))]; // left is None!
        let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

        let result = finalize_nodes(&build_nodes, &child_pointers, &bin_thresholds);
        assert!(matches!(
            result,
            Err(ClearGbmError::TreeConstructionFailed { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_finalize_nodes_internal_node_missing_right_child() -> Result<(), ClearGbmError> {
        // Create an internal node with missing right child in child_pointers
        let build_nodes = vec![BuildNode {
            node_id: 0_usize,
            is_leaf: false,
            value: 0.0_f64,
            n_samples: 10_usize,
            feature_index: Some(0_usize),
            split_bin: Some(0_usize),
            nan_goes_left: true,
        }];
        let child_pointers = vec![(Some(1_usize), None)]; // right is None!
        let bin_thresholds = vec![vec![0.5_f64, 1.0_f64]];

        let result = finalize_nodes(&build_nodes, &child_pointers, &bin_thresholds);
        assert!(matches!(
            result,
            Err(ClearGbmError::TreeConstructionFailed { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_finalize_nodes_leaf_node_success() -> Result<(), ClearGbmError> {
        // Leaf nodes should finalize without needing feature_index, split_bin, or children
        let build_nodes = vec![BuildNode {
            node_id: 0_usize,
            is_leaf: true,
            value: 1.5_f64,
            n_samples: 10_usize,
            feature_index: None,
            split_bin: None,
            nan_goes_left: false,
        }];
        let child_pointers = vec![(None, None)];
        let bin_thresholds: Vec<Vec<f64>> = vec![];

        let nodes = match finalize_nodes(&build_nodes, &child_pointers, &bin_thresholds) {
            Ok(n) => n,
            Err(e) => return Err(e),
        };
        assert_eq!(nodes.len(), 1_usize);
        assert!(nodes[0_usize].is_leaf());
        assert!((nodes[0_usize].value() - 1.5_f64).abs() < 1e-10_f64);
        Ok(())
    }

    // =========================================================================
    // Internal function error path tests
    // =========================================================================

    #[test]
    fn test_build_feature_histograms_empty_features() -> Result<(), ClearGbmError> {
        // Test with n_features = 0
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins: Vec<Vec<usize>> = vec![vec![], vec![]]; // No features

        let hooks = Hooks::default();
        let config = BuildHistogramConfig {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_features: 0_usize,
            n_bins: 3_usize,
            hooks: &hooks,
            cached_histograms: None,
        };
        let result = build_feature_histograms(&config);

        // Should return Ok with empty vec (no error, but no histograms)
        assert!(matches!(result, Ok(ref h) if h.is_empty()));
        Ok(())
    }

    #[test]
    fn test_find_best_split_across_features_internal_error() -> Result<(), ClearGbmError> {
        // Create a histogram and config that will trigger an error
        // n_regular_bins > n_bins should cause an error
        let histogram = HistogramBuffer::new(3_usize);
        let histograms = vec![histogram];
        let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let result = find_best_split_across_features_internal(
            &histograms,
            &config,
            10_usize, // n_regular_bins > n_bins (3)
            None,
        );

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_find_best_split_across_features_internal_multiple_features() -> Result<(), ClearGbmError>
    {
        // Test with 2 features that both have valid splits to cover the comparison closure
        let mut hist0 = HistogramBuffer::new(4_usize);
        for _ in 0_usize..10_usize {
            match hist0.accumulate(0_usize, 0.5_f64, 1.0_f64) {
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match hist0.accumulate(1_usize, -0.5_f64, 1.0_f64) {
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }

        let mut hist1 = HistogramBuffer::new(4_usize);
        for _ in 0_usize..10_usize {
            match hist1.accumulate(0_usize, 0.3_f64, 1.0_f64) {
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }
        for _ in 0_usize..10_usize {
            match hist1.accumulate(1_usize, -0.3_f64, 1.0_f64) {
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }

        let histograms = vec![hist0, hist1];
        let config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let result =
            match find_best_split_across_features_internal(&histograms, &config, 3_usize, None) {
                Ok(r) => r,
                Err(e) => return Err(e),
            };

        // Should find the best split (feature 0 has higher gain due to larger gradient magnitude)
        assert!(matches!(result, Some(ref s) if s.feature_index() == 0_usize));
        Ok(())
    }

    #[test]
    fn test_compute_child_histograms_parent_histograms_too_short() -> Result<(), ClearGbmError> {
        // Test error when parent_histograms has fewer entries than n_features
        let left_indices = vec![0_usize, 1_usize];
        let right_indices = vec![2_usize, 3_usize];
        let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![
            vec![0_usize, 0_usize], // 2 features
            vec![1_usize, 1_usize],
            vec![0_usize, 0_usize],
            vec![1_usize, 1_usize],
        ];

        // Only 1 parent histogram, but n_features = 2
        let parent_histograms = vec![HistogramBuffer::new(3_usize)];

        let hooks = Hooks::default();
        let config = ChildHistogramConfig {
            left_indices: &left_indices,
            right_indices: &right_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_features: 2_usize, // 2 features, but only 1 parent histogram
            n_bins: 3_usize,
            parent_histograms: &parent_histograms,
            hooks: &hooks,
        };

        let result = compute_child_histograms(&config);
        assert!(matches!(
            result,
            Err(ClearGbmError::FeatureIndexOutOfBounds { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_compute_child_histograms_success() -> Result<(), ClearGbmError> {
        // Test successful child histogram computation
        let left_indices = vec![0_usize, 1_usize];
        let right_indices = vec![2_usize, 3_usize];
        let gradients = vec![1.0_f64, 2.0_f64, -1.0_f64, -2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![0_usize], vec![1_usize], vec![1_usize]];

        // Create parent histogram with proper values
        let mut parent_hist = HistogramBuffer::new(3_usize);
        match parent_hist.accumulate(0_usize, 3.0_f64, 2.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
        match parent_hist.accumulate(1_usize, -3.0_f64, 2.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }

        let parent_histograms = vec![parent_hist];

        let hooks = Hooks::default();
        let config = ChildHistogramConfig {
            left_indices: &left_indices,
            right_indices: &right_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_features: 1_usize,
            n_bins: 3_usize,
            parent_histograms: &parent_histograms,
            hooks: &hooks,
        };

        let (left_hists, right_hists) = match compute_child_histograms(&config) {
            Ok(h) => h,
            Err(e) => return Err(e),
        };
        assert_eq!(left_hists.len(), 1_usize);
        assert_eq!(right_hists.len(), 1_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_with_large_n_regular_bins() -> Result<(), ClearGbmError> {
        // Test with n_regular_bins much larger than actual bins used
        // This succeeds because histogram is built with n_bins = n_regular_bins + 1
        let sc = match SplitConfig::new(2_usize, 1_usize, 4_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
        let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 100_usize, // Large but histogram will have n_bins = 101
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        // This succeeds because histogram.n_bins() = n_regular_bins + 1 = 101 > 100
        let tree = match build_tree(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        assert!(tree.n_nodes() >= 1_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_bins_out_of_bounds() -> Result<(), ClearGbmError> {
        // Test with bin values that exceed n_bins to trigger error in build_feature_histograms
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
        let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        // Bin value 100 exceeds n_bins (n_regular_bins + 1 = 5)
        let bins = vec![vec![0_usize], vec![1_usize], vec![100_usize], vec![3_usize]];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        let result = build_tree(&input, &Hooks::default());
        // This should error due to bin out of bounds
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_is_increasing_is_decreasing_methods() -> Result<(), ClearGbmError> {
        // Test the is_increasing and is_decreasing methods
        let inc = MonotonicConstraint::Increasing;
        let dec = MonotonicConstraint::Decreasing;
        let none = MonotonicConstraint::None;

        assert!(inc.is_increasing());
        assert!(!inc.is_decreasing());
        assert!(!inc.is_none());

        assert!(!dec.is_increasing());
        assert!(dec.is_decreasing());
        assert!(!dec.is_none());

        assert!(!none.is_increasing());
        assert!(!none.is_decreasing());
        assert!(none.is_none());

        Ok(())
    }

    // =========================================================================
    // Hook-based error injection tests
    // =========================================================================

    /// Histogram builder that always returns an error (for testing error propagation)
    fn error_histogram(
        _: &[usize],
        _: &[f64],
        _: &[f64],
        _: &[usize],
        _: usize,
    ) -> Result<HistogramBuffer, ClearGbmError> {
        Err(ClearGbmError::EmptyInput {
            context: "injected error from hook".to_string(),
        })
    }

    #[test]
    fn test_build_tree_hooks_error_in_histogram_building() -> Result<(), ClearGbmError> {
        // Use hooks to inject error during histogram building (exercises line 474's `?`)
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
        let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize,
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        // Inject error via hook
        let error_hooks = Hooks::with_histogram_builder(error_histogram);
        let result = build_tree(&input, &error_hooks);

        assert!(result.is_err());
        assert!(matches!(
            result.err(),
            Some(ClearGbmError::EmptyInput { context }) if context.contains("injected")
        ));
        Ok(())
    }

    #[test]
    fn test_build_feature_histograms_with_cached_histograms() -> Result<(), ClearGbmError> {
        // Test that cached histograms are used when available
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize]];

        // Create cached histograms
        let mut cached = HistogramBuffer::new(3_usize);
        match cached.accumulate(0_usize, 1.0_f64, 1.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
        match cached.accumulate(1_usize, 2.0_f64, 1.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
        let cached_histograms = vec![cached];

        let hooks = Hooks::default();
        let config = BuildHistogramConfig {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_features: 1_usize,
            n_bins: 3_usize,
            hooks: &hooks,
            cached_histograms: Some(&cached_histograms),
        };

        let result = match build_feature_histograms(&config) {
            Ok(r) => r,
            Err(e) => return Err(e),
        };

        // Should return the cached histograms
        assert_eq!(result.len(), 1_usize);
        Ok(())
    }

    #[test]
    fn test_build_feature_histograms_with_wrong_size_cache() -> Result<(), ClearGbmError> {
        // Test that wrong-size cache is ignored and histograms are built fresh
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize, 1_usize], vec![1_usize, 0_usize]]; // 2 features

        // Create cache with wrong size (1 instead of 2)
        let cached = HistogramBuffer::new(3_usize);
        let cached_histograms = vec![cached];

        let hooks = Hooks::default();
        let config = BuildHistogramConfig {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_features: 2_usize, // 2 features
            n_bins: 3_usize,
            hooks: &hooks,
            cached_histograms: Some(&cached_histograms), // but only 1 cached
        };

        // Should build from scratch since cache size doesn't match
        let result = match build_feature_histograms(&config) {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        assert_eq!(result.len(), 2_usize);
        Ok(())
    }

    #[test]
    fn test_compute_child_histograms_hooks_error() -> Result<(), ClearGbmError> {
        // Test error propagation from hooks in compute_child_histograms
        let left_indices = vec![0_usize, 1_usize];
        let right_indices = vec![2_usize, 3_usize];
        let gradients = vec![1.0_f64, 2.0_f64, -1.0_f64, -2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![0_usize], vec![1_usize], vec![1_usize]];

        let mut parent_hist = HistogramBuffer::new(3_usize);
        match parent_hist.accumulate(0_usize, 3.0_f64, 2.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
        match parent_hist.accumulate(1_usize, -3.0_f64, 2.0_f64) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
        let parent_histograms = vec![parent_hist];

        // Inject error via hook
        let error_hooks = Hooks::with_histogram_builder(error_histogram);
        let config = ChildHistogramConfig {
            left_indices: &left_indices,
            right_indices: &right_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_features: 1_usize,
            n_bins: 3_usize,
            parent_histograms: &parent_histograms,
            hooks: &error_hooks,
        };

        let result = compute_child_histograms(&config);
        assert!(result.is_err());
        assert!(matches!(
            result.err(),
            Some(ClearGbmError::EmptyInput { context }) if context.contains("injected")
        ));
        Ok(())
    }

    /// Histogram builder that returns undersized histogram (for testing error propagation)
    fn undersized_histogram(
        _: &[usize],
        _: &[f64],
        _: &[f64],
        _: &[usize],
        _: usize,
    ) -> Result<HistogramBuffer, ClearGbmError> {
        // Return a histogram with only 2 bins, regardless of requested size
        // This will cause find_best_split_from_histogram to fail when n_regular_bins > 2
        Ok(HistogramBuffer::new(2_usize))
    }

    #[test]
    fn test_build_tree_hooks_error_in_split_finding() -> Result<(), ClearGbmError> {
        // Use hooks to inject undersized histogram, causing split finding to fail
        // This exercises the `?` at line 482 in build_tree
        let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let sample_indices = vec![0_usize, 1_usize, 2_usize, 3_usize];
        let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize], vec![2_usize], vec![3_usize]];
        let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

        let input = BuildTreeInput {
            sample_indices: &sample_indices,
            gradients: &gradients,
            hessians: &hessians,
            bins: &bins,
            n_regular_bins: 4_usize, // 4 regular bins, but hook returns 2-bin histogram
            bin_thresholds: &bin_thresholds,
            config: &cfg,
            monotonic_constraints: None,
        };

        // Inject undersized histogram via hook - this causes split finding error
        let undersized_hooks = Hooks::with_histogram_builder(undersized_histogram);
        let result = build_tree(&input, &undersized_hooks);

        // Should fail because n_regular_bins (4) > histogram.n_bins() (2)
        assert!(result.is_err());
        Ok(())
    }

    // ==================== SERDE ERROR PATH TESTS ====================

    // TreeBuildConfig serde tests

    #[test]
    fn test_tree_build_config_deserialize_missing_max_depth() -> Result<(), ClearGbmError> {
        let json = r#"{"max_leaves":8,"reg_alpha":0.0,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
        let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("max_depth"));
        Ok(())
    }

    #[test]
    fn test_tree_build_config_deserialize_missing_max_leaves() -> Result<(), ClearGbmError> {
        let json = r#"{"max_depth":6,"reg_alpha":0.0,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
        let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("max_leaves"));
        Ok(())
    }

    #[test]
    fn test_tree_build_config_deserialize_missing_reg_alpha() -> Result<(), ClearGbmError> {
        let json = r#"{"max_depth":6,"max_leaves":8,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
        let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("reg_alpha"));
        Ok(())
    }

    #[test]
    fn test_tree_build_config_deserialize_missing_reg_lambda() -> Result<(), ClearGbmError> {
        let json = r#"{"max_depth":6,"max_leaves":8,"reg_alpha":0.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
        let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("reg_lambda"));
        Ok(())
    }

    #[test]
    fn test_tree_build_config_deserialize_missing_split_config() -> Result<(), ClearGbmError> {
        let json = r#"{"max_depth":6,"max_leaves":8,"reg_alpha":0.0,"reg_lambda":1.0}"#;
        let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("split_config"));
        Ok(())
    }

    #[test]
    fn test_tree_build_config_deserialize_unknown_field() -> Result<(), ClearGbmError> {
        let json = r#"{"max_depth":6,"max_leaves":8,"reg_alpha":0.0,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0},"unknown_field":"value"}"#;
        let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("unknown field"));
        Ok(())
    }

    #[test]
    fn test_tree_build_config_deserialize_wrong_type() -> Result<(), ClearGbmError> {
        let json = r#"{"max_depth":"six","max_leaves":8,"reg_alpha":0.0,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
        let result: Result<TreeBuildConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_build_config_deserialize_all_fields() -> Result<(), ClearGbmError> {
        let json = r#"{"max_depth":6,"max_leaves":8,"reg_alpha":0.1,"reg_lambda":1.0,"split_config":{"min_samples_split":2,"min_samples_leaf":1,"max_bins":256,"reg_lambda":0.0,"min_gain":0.0}}"#;
        let config: TreeBuildConfig = match serde_json::from_str(json) {
            Ok(c) => c,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(config.max_depth(), 6_usize);
        assert_eq!(config.max_leaves(), 8_usize);
        assert!((config.reg_alpha() - 0.1_f64).abs() < 1e-10_f64);
        assert!((config.reg_lambda() - 1.0_f64).abs() < 1e-10_f64);
        Ok(())
    }

    #[test]
    fn test_tree_build_config_serialize_roundtrip() -> Result<(), ClearGbmError> {
        let split_config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let config = match TreeBuildConfig::new(6_usize, 8_usize, 0.1_f64, 1.0_f64, split_config) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let json = match serde_json::to_string(&config) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let deserialized: TreeBuildConfig = match serde_json::from_str(&json) {
            Ok(c) => c,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(config, deserialized);
        Ok(())
    }

    // Tree serde tests

    #[test]
    fn test_tree_deserialize_missing_nodes() -> Result<(), ClearGbmError> {
        let json = r#"{"max_depth":3,"n_leaves":4}"#;
        let result: Result<Tree, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("nodes"));
        Ok(())
    }

    #[test]
    fn test_tree_deserialize_missing_max_depth() -> Result<(), ClearGbmError> {
        let json = r#"{"nodes":[],"n_leaves":0}"#;
        let result: Result<Tree, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("max_depth"));
        Ok(())
    }

    #[test]
    fn test_tree_deserialize_missing_n_leaves() -> Result<(), ClearGbmError> {
        let json = r#"{"nodes":[],"max_depth":0}"#;
        let result: Result<Tree, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("n_leaves"));
        Ok(())
    }

    #[test]
    fn test_tree_deserialize_unknown_field() -> Result<(), ClearGbmError> {
        let json = r#"{"nodes":[],"max_depth":0,"n_leaves":0,"unknown":"value"}"#;
        let result: Result<Tree, _> = serde_json::from_str(json);
        let err_msg = match result {
            Ok(_) => {
                return Err(ClearGbmError::TreeConstructionFailed {
                    reason: "expected error".to_string(),
                })
            }
            Err(e) => e.to_string(),
        };
        assert!(err_msg.contains("unknown field"));
        Ok(())
    }

    #[test]
    fn test_tree_deserialize_wrong_type() -> Result<(), ClearGbmError> {
        let json = r#"{"nodes":"not_an_array","max_depth":0,"n_leaves":0}"#;
        let result: Result<Tree, _> = serde_json::from_str(json);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_deserialize_all_fields_empty() -> Result<(), ClearGbmError> {
        let json = r#"{"nodes":[],"max_depth":0,"n_leaves":0}"#;
        let tree: Tree = match serde_json::from_str(json) {
            Ok(t) => t,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(tree.n_nodes(), 0_usize);
        assert_eq!(tree.max_depth(), 0_usize);
        assert_eq!(tree.n_leaves(), 0_usize);
        Ok(())
    }

    #[test]
    fn test_tree_serialize_roundtrip() -> Result<(), ClearGbmError> {
        let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
        let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
        let json = match serde_json::to_string(&tree) {
            Ok(s) => s,
            Err(e) => {
                return Err(ClearGbmError::SerializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        let deserialized: Tree = match serde_json::from_str(&json) {
            Ok(t) => t,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(tree.n_nodes(), deserialized.n_nodes());
        assert_eq!(tree.max_depth(), deserialized.max_depth());
        assert_eq!(tree.n_leaves(), deserialized.n_leaves());
        Ok(())
    }

    #[test]
    fn test_tree_deserialize_with_nodes() -> Result<(), ClearGbmError> {
        // Create a proper JSON with a leaf node
        let json = r#"{"nodes":[{"node_id":0,"is_leaf":true,"feature_index":null,"threshold":null,"value":0.5,"n_samples":10,"left_child":null,"right_child":null,"nan_goes_left":true}],"max_depth":0,"n_leaves":1}"#;
        let tree: Tree = match serde_json::from_str(json) {
            Ok(t) => t,
            Err(e) => {
                return Err(ClearGbmError::DeserializationFailed {
                    reason: e.to_string(),
                })
            }
        };
        assert_eq!(tree.n_nodes(), 1_usize);
        assert_eq!(tree.max_depth(), 0_usize);
        assert_eq!(tree.n_leaves(), 1_usize);
        Ok(())
    }

    // Type mismatch tests to trigger expecting() methods

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

    // Serialization error path tests using failing serializer

    #[test]
    fn test_tree_build_config_serialize_fail_each_field() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
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
            let mut ser = FailAfterN::new(fail_at);
            let result = config.serialize(&mut ser);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_tree_serialize_fail_each_field() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
        let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
        // Tree has 3 fields
        for fail_at in 0_usize..3_usize {
            let mut ser = FailAfterN::new(fail_at);
            let result = tree.serialize(&mut ser);
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_tree_build_config_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let split_config = match SplitConfig::new(2_usize, 1_usize, 256_usize, 0.0_f64, 0.0_f64) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let config = match TreeBuildConfig::new(6_usize, 8_usize, 0.0_f64, 1.0_f64, split_config) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let mut ser = FailAfterN::fail_on_struct();
        let result = config.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_serialize_fail_on_struct() -> Result<(), ClearGbmError> {
        use failing_serializer::FailAfterN;
        use serde::Serialize;
        let leaf_node = TreeNode::new_leaf(0_usize, 0.5_f64, 10_usize);
        let tree = Tree::new(vec![leaf_node], 0_usize, 1_usize);
        let mut ser = FailAfterN::fail_on_struct();
        let result = tree.serialize(&mut ser);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_failing_serializer_coverage() -> Result<(), ClearGbmError> {
        use failing_serializer::{FailAfterN, FailError};
        use serde::ser::{Error, SerializeStruct, Serializer};

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

        // Test serialize_struct
        let mut ser = FailAfterN::new(100);
        let struct_ser = (&mut ser).serialize_struct("S", 1);
        assert!(struct_ser.is_ok());

        // Test struct end
        let mut ser = FailAfterN::new(100);
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
        let mut ser = FailAfterN::new(1);
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
        use failing_deserializer::MapWithIntegerKeyDeserializer;
        use serde::Deserialize;
        let result = TreeBuildConfig::deserialize(MapWithIntegerKeyDeserializer);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_tree_field_expecting() -> Result<(), ClearGbmError> {
        use failing_deserializer::MapWithIntegerKeyDeserializer;
        use serde::Deserialize;
        let result = Tree::deserialize(MapWithIntegerKeyDeserializer);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_failing_deserializer_coverage() -> Result<(), ClearGbmError> {
        use failing_deserializer::{DeError, IntegerDeserializer, IntegerKeyMapAccess};
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

        // Test IntegerKeyMapAccess next_value
        let mut map_access2 = IntegerKeyMapAccess { done: false };
        let value_result: Result<i64, _> = map_access2.next_value();
        assert!(value_result.is_err());

        Ok(())
    }
}
