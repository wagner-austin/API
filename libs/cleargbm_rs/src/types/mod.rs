//! Core data structures for `ClearGBM`.
//!
//! All types are immutable after construction. Use builder patterns
//! or constructor functions to create instances.

/// Serde serialization and deserialization implementations.
///
/// Made `pub(crate)` to allow testing of visitor `expecting()` error paths.
pub(crate) mod serde_impl;

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
    /// Per-bin accumulators, stored interleaved so one sample's update
    /// touches one contiguous 24-byte record instead of three parallel
    /// arrays. See [`BinAccumulator`] for why.
    pub(crate) bins: Vec<BinAccumulator>,

    /// Number of bins (fixed at construction).
    pub(crate) n_bins: usize,
}

/// One bin's accumulators, interleaved.
///
/// The histogram hot loop performs a read-modify-write on all three values
/// for one bin per sample. With three parallel arrays that update touches
/// three cache lines and pays three bounds checks; interleaved it touches one
/// 24-byte record (one line, occasionally two when the record straddles a
/// boundary) and pays a single bounds check. This is LightGBM's `hist_t`
/// grad/hess interleaving, extended to carry the sample count this codebase
/// also accumulates. Splitting the count into a parallel `u32` array to
/// shrink the record to 16 bytes was measured 2026-08-22 at +20% fit time —
/// the second memory touch per update costs far more than the straddle it
/// removes — so the count stays fused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BinAccumulator {
    /// Sum of gradients in this bin.
    pub(crate) gradient_sum: f64,

    /// Sum of hessians in this bin.
    pub(crate) hessian_sum: f64,

    /// Count of samples in this bin.
    pub(crate) count: usize,
}

impl BinAccumulator {
    /// The all-zero accumulator a fresh histogram starts from.
    pub(crate) const ZERO: Self = Self {
        gradient_sum: 0.0_f64,
        hessian_sum: 0.0_f64,
        count: 0_usize,
    };
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
            bins: vec![BinAccumulator::ZERO; n_bins],
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
        let n_bins = self.n_bins;
        let Some(acc) = self.bins.get_mut(bin) else {
            return Err(ClearGbmError::BinIndexOutOfBounds { bin, n_bins });
        };
        acc.gradient_sum += gradient;
        acc.hessian_sum += hessian;
        acc.count += 1;
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
        self.bins
            .get(bin)
            .map(|acc| acc.gradient_sum)
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
        self.bins
            .get(bin)
            .map(|acc| acc.hessian_sum)
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
        self.bins
            .get(bin)
            .map(|acc| acc.count)
            .ok_or(ClearGbmError::BinIndexOutOfBounds {
                bin,
                n_bins: self.n_bins,
            })
    }

    /// Returns all gradient sums, materialized in bin order.
    ///
    /// Materializes from the interleaved storage; intended for
    /// serialization and inspection, not for hot-path use.
    #[must_use]
    pub fn gradient_sums(&self) -> Vec<f64> {
        self.bins.iter().map(|acc| acc.gradient_sum).collect()
    }

    /// Returns all hessian sums, materialized in bin order.
    ///
    /// Materializes from the interleaved storage; intended for
    /// serialization and inspection, not for hot-path use.
    #[must_use]
    pub fn hessian_sums(&self) -> Vec<f64> {
        self.bins.iter().map(|acc| acc.hessian_sum).collect()
    }

    /// Returns all counts, materialized in bin order.
    ///
    /// Materializes from the interleaved storage; intended for
    /// serialization and inspection, not for hot-path use.
    #[must_use]
    pub fn counts(&self) -> Vec<usize> {
        self.bins.iter().map(|acc| acc.count).collect()
    }

    /// Resets all bins to zero (for reuse).
    pub fn reset(&mut self) {
        self.bins.fill(BinAccumulator::ZERO);
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
            self.bins[i] = BinAccumulator {
                gradient_sum: parent.bins[i].gradient_sum - child.bins[i].gradient_sum,
                hessian_sum: parent.bins[i].hessian_sum - child.bins[i].hessian_sum,
                count: parent.bins[i].count.saturating_sub(child.bins[i].count),
            };
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

        self.bins.copy_from_slice(&other.bins);

        Ok(())
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
