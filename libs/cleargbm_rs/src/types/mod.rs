//! Core data structures for `ClearGBM`.
//!
//! All types are immutable after construction. Use builder patterns
//! or constructor functions to create instances.

/// Serde serialization and deserialization implementations.
///
/// Made `pub(crate)` to allow testing of visitor `expecting()` error paths.
/// The histogram accumulator types.
mod histogram;
pub(crate) mod serde_impl;

pub(crate) use histogram::BinAccumulator;
pub use histogram::HistogramBuffer;

use crate::error::ClearGbmError;

#[cfg(test)]
mod tests;

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

/// A node in a gradient boosted decision tree.
///
/// Nodes are either internal (with split information) or leaf (with prediction value).
/// This is equivalent to the Python `TreeNode` `TypedDict`.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    /// Unique identifier for this node within the tree.
    pub(crate) node_id: usize,

    /// Whether this is a leaf node (no children).
    pub(crate) is_leaf: bool,

    /// Feature index used for splitting (None for leaf nodes).
    pub(crate) feature_index: Option<usize>,

    /// Threshold value for the split (None for leaf nodes).
    pub(crate) threshold: Option<f64>,

    /// Prediction value (leaf value or intermediate sum).
    pub(crate) value: f64,

    /// Number of training samples that reached this node.
    pub(crate) n_samples: usize,

    /// Left child node ID (None for leaf nodes).
    pub(crate) left_child: Option<usize>,

    /// Right child node ID (None for leaf nodes).
    pub(crate) right_child: Option<usize>,

    /// Direction for NaN values: true = left, false = right.
    pub(crate) nan_goes_left: bool,

    /// Category codes routed to the left child (categorical internal nodes
    /// only; None for threshold splits and leaves). Sorted ascending; a
    /// prediction value goes left exactly when it equals one of these codes,
    /// right otherwise, with NaN following `nan_goes_left`.
    pub(crate) categories_goes_left: Option<Vec<f64>>,
}

/// Configuration for creating a categorical internal (set-split) node.
///
/// The categorical sibling of [`TreeNodeConfig`]: it carries the left-routed
/// category codes where the threshold config carries a boundary value —
/// neither shape can express the other's split, so neither carries a
/// placeholder.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalNodeConfig {
    /// Unique identifier for this node within the tree.
    pub node_id: usize,
    /// Index of feature used for split.
    pub feature_index: usize,
    /// Category codes routed to the left child, ascending.
    pub categories_goes_left: Vec<f64>,
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
            categories_goes_left: None,
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
            categories_goes_left: None,
        }
    }

    /// Creates a new categorical internal (set-split) node from configuration.
    ///
    /// # Args
    ///
    /// * `config` - Configuration containing all node parameters, with the
    ///   left-routed category codes in place of a threshold.
    ///
    /// # Returns
    ///
    /// A new categorical internal `TreeNode` (its `threshold` is `None`).
    #[must_use]
    pub fn new_categorical_internal(config: CategoricalNodeConfig) -> Self {
        Self {
            node_id: config.node_id,
            is_leaf: false,
            feature_index: Some(config.feature_index),
            threshold: None,
            value: config.value,
            n_samples: config.n_samples,
            left_child: Some(config.left_child),
            right_child: Some(config.right_child),
            nan_goes_left: config.nan_goes_left,
            categories_goes_left: Some(config.categories_goes_left),
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

    /// Returns the category codes routed left (None for threshold splits
    /// and leaves).
    #[must_use]
    pub fn categories_goes_left(&self) -> Option<&[f64]> {
        self.categories_goes_left.as_deref()
    }
}

/// Configuration for histogram-based split finding.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitConfig {
    /// Minimum samples required to split a node.
    pub(crate) min_samples_split: usize,

    /// Minimum samples required in a leaf.
    pub(crate) min_samples_leaf: usize,

    /// Maximum number of bins for histogram.
    pub(crate) max_bins: usize,

    /// L2 regularization parameter.
    pub(crate) reg_lambda: f64,

    /// Minimum gain required to make a split.
    pub(crate) min_gain: f64,
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
