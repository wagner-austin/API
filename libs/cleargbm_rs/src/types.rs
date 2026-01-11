//! Core data structures for `ClearGBM`.
//!
//! All types are immutable after construction. Use builder patterns
//! or constructor functions to create instances.

use serde::{Deserialize, Serialize};

use crate::error::ClearGbmError;

/// Configuration for creating an internal (split) tree node.
///
/// Used to avoid having too many function arguments while maintaining
/// explicit, named parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    ) -> std::result::Result<(), ClearGbmError> {
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
    pub fn gradient_sum(&self, bin: usize) -> std::result::Result<f64, ClearGbmError> {
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
    pub fn hessian_sum(&self, bin: usize) -> std::result::Result<f64, ClearGbmError> {
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
    pub fn count(&self, bin: usize) -> std::result::Result<usize, ClearGbmError> {
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
    pub fn subtract_into(
        &mut self,
        parent: &Self,
        child: &Self,
    ) -> std::result::Result<(), ClearGbmError> {
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
    pub fn copy_from(&mut self, other: &Self) -> std::result::Result<(), ClearGbmError> {
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

/// Configuration for histogram-based split finding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    ) -> std::result::Result<Self, ClearGbmError> {
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

    /// Boxed error type for tests with multiple error sources.
    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    // TreeNode tests

    #[test]
    fn test_tree_node_new_leaf() -> TestResult {
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
    fn test_tree_node_new_internal() -> TestResult {
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
    fn test_tree_node_clone() -> TestResult {
        let node = TreeNode::new_leaf(5_usize, 1.0_f64, 10_usize);
        let cloned = node.clone();
        assert_eq!(node, cloned);
        Ok(())
    }

    #[test]
    fn test_tree_node_debug() -> TestResult {
        let node = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        let debug_str = format!("{node:?}");
        assert!(debug_str.contains("TreeNode"));
        assert!(debug_str.contains("node_id: 0"));
        Ok(())
    }

    #[test]
    fn test_tree_node_serialize_deserialize() -> TestResult {
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
        let json_str = serde_json::to_string(&node)?;
        let parsed: TreeNode = serde_json::from_str(&json_str)?;
        assert_eq!(parsed, node);
        Ok(())
    }

    // HistogramBuffer tests

    #[test]
    fn test_histogram_buffer_new() -> std::result::Result<(), ClearGbmError> {
        let hist = HistogramBuffer::new(5_usize);
        assert_eq!(hist.n_bins(), 5_usize);
        for i in 0_usize..5_usize {
            let grad = hist.gradient_sum(i)?;
            assert!(grad.abs() < f64::EPSILON);
            let hess = hist.hessian_sum(i)?;
            assert!(hess.abs() < f64::EPSILON);
            assert_eq!(hist.count(i)?, 0_usize);
        }
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_accumulate() -> std::result::Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        hist.accumulate(1_usize, 0.5_f64, 1.0_f64)?;
        let grad = hist.gradient_sum(1_usize)?;
        assert!((grad - 0.5_f64).abs() < f64::EPSILON);
        let hess = hist.hessian_sum(1_usize)?;
        assert!((hess - 1.0_f64).abs() < f64::EPSILON);
        assert_eq!(hist.count(1_usize)?, 1_usize);
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_accumulate_multiple() -> std::result::Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        hist.accumulate(0_usize, 0.1_f64, 1.0_f64)?;
        hist.accumulate(0_usize, 0.2_f64, 1.0_f64)?;
        hist.accumulate(0_usize, 0.3_f64, 1.0_f64)?;
        let grad = hist.gradient_sum(0_usize)?;
        assert!((grad - 0.6_f64).abs() < 1e-10_f64);
        let hess = hist.hessian_sum(0_usize)?;
        assert!((hess - 3.0_f64).abs() < f64::EPSILON);
        assert_eq!(hist.count(0_usize)?, 3_usize);
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_accumulate_out_of_bounds() -> TestResult {
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
    fn test_histogram_buffer_gradient_sum_out_of_bounds() -> TestResult {
        let hist = HistogramBuffer::new(3_usize);
        let result = hist.gradient_sum(10_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_hessian_sum_out_of_bounds() -> TestResult {
        let hist = HistogramBuffer::new(3_usize);
        let result = hist.hessian_sum(10_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_count_out_of_bounds() -> TestResult {
        let hist = HistogramBuffer::new(3_usize);
        let result = hist.count(10_usize);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_slices() -> std::result::Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        hist.accumulate(0_usize, 0.1_f64, 1.0_f64)?;
        hist.accumulate(1_usize, 0.2_f64, 2.0_f64)?;
        hist.accumulate(2_usize, 0.3_f64, 3.0_f64)?;

        assert_eq!(hist.gradient_sums().len(), 3_usize);
        assert_eq!(hist.hessian_sums().len(), 3_usize);
        assert_eq!(hist.counts().len(), 3_usize);

        assert!((hist.gradient_sums()[0_usize] - 0.1_f64).abs() < f64::EPSILON);
        assert!((hist.hessian_sums()[1_usize] - 2.0_f64).abs() < f64::EPSILON);
        assert_eq!(hist.counts()[2_usize], 1_usize);
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_reset() -> std::result::Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        hist.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
        hist.accumulate(1_usize, 0.3_f64, 1.5_f64)?;
        hist.reset();
        for i in 0_usize..3_usize {
            let grad = hist.gradient_sum(i)?;
            assert!(grad.abs() < f64::EPSILON);
            let hess = hist.hessian_sum(i)?;
            assert!(hess.abs() < f64::EPSILON);
            assert_eq!(hist.count(i)?, 0_usize);
        }
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_subtract_into() -> std::result::Result<(), ClearGbmError> {
        let mut parent = HistogramBuffer::new(3_usize);
        parent.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
        parent.accumulate(0_usize, 0.3_f64, 1.0_f64)?;
        parent.accumulate(1_usize, 0.2_f64, 1.0_f64)?;

        let mut child = HistogramBuffer::new(3_usize);
        child.accumulate(0_usize, 0.5_f64, 1.0_f64)?;

        let mut sibling = HistogramBuffer::new(3_usize);
        sibling.subtract_into(&parent, &child)?;

        // Bin 0: parent (0.8, 2.0, 2), child (0.5, 1.0, 1), sibling should be (0.3, 1.0, 1)
        let grad = sibling.gradient_sum(0_usize)?;
        assert!((grad - 0.3_f64).abs() < 1e-10_f64);
        let hess = sibling.hessian_sum(0_usize)?;
        assert!((hess - 1.0_f64).abs() < f64::EPSILON);
        assert_eq!(sibling.count(0_usize)?, 1_usize);

        // Bin 1: parent (0.2, 1.0, 1), child (0, 0, 0), sibling should be (0.2, 1.0, 1)
        assert_eq!(sibling.count(1_usize)?, 1_usize);
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_subtract_into_shape_mismatch() -> TestResult {
        let parent = HistogramBuffer::new(3_usize);
        let child = HistogramBuffer::new(5_usize);
        let mut sibling = HistogramBuffer::new(3_usize);

        let result = sibling.subtract_into(&parent, &child);
        assert!(result.is_err());
        assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_copy_from() -> std::result::Result<(), ClearGbmError> {
        let mut source = HistogramBuffer::new(3_usize);
        source.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
        source.accumulate(1_usize, 0.3_f64, 2.0_f64)?;

        let mut dest = HistogramBuffer::new(3_usize);
        dest.copy_from(&source)?;

        assert_eq!(dest.gradient_sums(), source.gradient_sums());
        assert_eq!(dest.hessian_sums(), source.hessian_sums());
        assert_eq!(dest.counts(), source.counts());
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_copy_from_shape_mismatch(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let source = HistogramBuffer::new(5_usize);
        let mut dest = HistogramBuffer::new(3_usize);
        let result = dest.copy_from(&source);
        assert!(result.is_err());
        assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn test_histogram_buffer_clone() -> std::result::Result<(), ClearGbmError> {
        let mut hist = HistogramBuffer::new(3_usize);
        hist.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
        let cloned = hist.clone();
        assert_eq!(hist, cloned);
        Ok(())
    }

    // SplitConfig tests

    #[test]
    fn test_split_config_new_valid() -> std::result::Result<(), ClearGbmError> {
        let c = SplitConfig::new(2_usize, 1_usize, 64_usize, 1.0_f64, 0.0_f64)?;
        assert_eq!(c.min_samples_split(), 2_usize);
        assert_eq!(c.min_samples_leaf(), 1_usize);
        assert_eq!(c.max_bins(), 64_usize);
        assert!((c.reg_lambda() - 1.0_f64).abs() < f64::EPSILON);
        assert!(c.min_gain().abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn test_split_config_min_samples_split_too_small() -> TestResult {
        let result = SplitConfig::new(1_usize, 1_usize, 64_usize, 1.0_f64, 0.0_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "min_samples_split"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_min_samples_leaf_zero() -> TestResult {
        let result = SplitConfig::new(2_usize, 0_usize, 64_usize, 1.0_f64, 0.0_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "min_samples_leaf"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_max_bins_too_small() -> TestResult {
        let result = SplitConfig::new(2_usize, 1_usize, 1_usize, 1.0_f64, 0.0_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "max_bins"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_negative_reg_lambda() -> TestResult {
        let result = SplitConfig::new(2_usize, 1_usize, 64_usize, -1.0_f64, 0.0_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "reg_lambda"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_negative_min_gain() -> TestResult {
        let result = SplitConfig::new(2_usize, 1_usize, 64_usize, 1.0_f64, -0.1_f64);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ClearGbmError::InvalidParameter { ref name, .. }) if name == "min_gain"
        ));
        Ok(())
    }

    #[test]
    fn test_split_config_clone() -> std::result::Result<(), ClearGbmError> {
        let c = SplitConfig::new(2_usize, 1_usize, 64_usize, 1.0_f64, 0.0_f64)?;
        let cloned = c.clone();
        assert_eq!(c, cloned);
        Ok(())
    }

    #[test]
    fn test_split_config_serialize_deserialize() -> TestResult {
        let c = SplitConfig::new(10_usize, 5_usize, 128_usize, 0.5_f64, 0.01_f64)?;
        let json_str = serde_json::to_string(&c)?;
        let parsed: SplitConfig = serde_json::from_str(&json_str)?;
        assert_eq!(parsed, c);
        Ok(())
    }
}
