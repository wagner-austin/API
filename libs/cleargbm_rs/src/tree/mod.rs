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

mod builder;
mod categorical;
mod feature_subsample;
mod histograms;
mod leafwise;
mod nodes;
mod serde_impl;

pub use builder::{
    build_tree, build_tree_with_leaf_assignment, compute_leaf_value, BuildTreeInput,
    QuantizedTreeData,
};
pub use categorical::CategoricalLayout;
pub use feature_subsample::{select_tree_features, tree_column_budget, FeatureSubsample};
pub use leafwise::{build_tree_leaf_wise, build_tree_leaf_wise_with_leaf_assignment};

use crate::error::ClearGbmError;
use crate::types::{SplitConfig, TreeNode};

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

impl TreeBuildConfig {
    /// Creates a new tree build configuration.
    ///
    /// # Args
    ///
    /// * `max_depth` - Maximum depth of tree (0 = unlimited).
    /// * `max_leaves` - Maximum number of leaves (0 = unlimited).
    /// * `reg_alpha` - L1 regularization term.
    /// * `reg_lambda` - L2 regularization term.
    /// * `split_config` - Split configuration.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if regularization terms are negative.
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

    /// Returns L1 regularization term.
    #[must_use]
    pub const fn reg_alpha(&self) -> f64 {
        self.reg_alpha
    }

    /// Returns L2 regularization term.
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

/// A decision tree built from gradient boosting.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    /// Tree nodes in breadth-first order.
    pub(crate) nodes: Vec<TreeNode>,

    /// Maximum depth reached.
    pub(crate) max_depth: usize,

    /// Number of leaf nodes.
    pub(crate) n_leaves: usize,
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

#[cfg(test)]
mod tests;
