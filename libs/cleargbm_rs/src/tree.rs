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

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    ) -> std::result::Result<Self, ClearGbmError> {
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tree {
    /// All nodes in the tree (index = node_id).
    nodes: Vec<TreeNode>,

    /// Actual maximum depth achieved.
    max_depth: usize,

    /// Number of leaf nodes.
    n_leaves: usize,
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
    pub fn node(&self, node_id: usize) -> std::result::Result<&TreeNode, ClearGbmError> {
        self.nodes
            .get(node_id)
            .ok_or(ClearGbmError::NodeNotFound { node_id })
    }

    /// Returns the root node.
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::NodeNotFound` if tree is empty.
    pub fn root(&self) -> std::result::Result<&TreeNode, ClearGbmError> {
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
pub fn build_tree(
    input: &BuildTreeInput<'_>,
    hooks: &Hooks,
) -> std::result::Result<Tree, ClearGbmError> {
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
        let histograms = build_feature_histograms(&hist_config)?;

        // Find best split across all features
        let best_split = find_best_split_across_features_internal(
            &histograms,
            split_config,
            input.n_regular_bins,
            input.monotonic_constraints,
        )?;

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
        let (left_histograms, right_histograms) = compute_child_histograms(&child_hist_config)?;

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
    let final_nodes = finalize_nodes(&nodes, &child_pointers, input.bin_thresholds)?;

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
) -> std::result::Result<Vec<HistogramBuffer>, ClearGbmError> {
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

        let hist = (config.hooks.build_histogram)(
            &sample_idx_vec,
            &feat_gradients,
            &feat_hessians,
            &feat_bins,
            config.n_bins,
        )?;
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
) -> std::result::Result<Option<SplitResult>, ClearGbmError> {
    let mut best_split: Option<SplitResult> = None;

    for (feature_idx, histogram) in histograms.iter().enumerate() {
        let constraint = monotonic_constraints
            .and_then(|constraints| constraints.get(feature_idx).copied())
            .unwrap_or(MonotonicConstraint::None);

        if let Some(split) = find_best_split_from_histogram(
            histogram,
            feature_idx,
            config,
            n_regular_bins,
            constraint,
        )? {
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
) -> std::result::Result<(Vec<HistogramBuffer>, Vec<HistogramBuffer>), ClearGbmError> {
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

        let smaller_hist = (config.hooks.build_histogram)(
            &sample_idx_vec,
            &feat_gradients,
            &feat_hessians,
            &feat_bins,
            config.n_bins,
        )?;

        // Derive larger child via subtraction
        let parent_hist = config.parent_histograms.get(feat_idx).ok_or(
            ClearGbmError::FeatureIndexOutOfBounds {
                index: feat_idx,
                n_features: config.n_features,
            },
        )?;

        let larger_hist = subtract_histogram(parent_hist, &smaller_hist)?;

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
) -> std::result::Result<Vec<TreeNode>, ClearGbmError> {
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
            let feature_index =
                node.feature_index
                    .ok_or_else(|| ClearGbmError::TreeConstructionFailed {
                        reason: "internal node missing feature_index".to_string(),
                    })?;
            let split_bin =
                node.split_bin
                    .ok_or_else(|| ClearGbmError::TreeConstructionFailed {
                        reason: "internal node missing split_bin".to_string(),
                    })?;

            // Get threshold from bin_thresholds
            // Threshold is the upper bound of the split_bin
            let threshold = bin_thresholds
                .get(feature_index)
                .and_then(|thresholds| thresholds.get(split_bin).copied())
                .unwrap_or(0.0_f64);

            let left_id = left_child.ok_or_else(|| ClearGbmError::TreeConstructionFailed {
                reason: format!("internal node {} missing left_child", node.node_id),
            })?;
            let right_id = right_child.ok_or_else(|| ClearGbmError::TreeConstructionFailed {
                reason: format!("internal node {} missing right_child", node.node_id),
            })?;

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
    use proptest::prelude::*;

    // =========================================================================
    // Property-based tests with proptest
    // =========================================================================

    proptest! {
        #[test]
        fn prop_compute_leaf_value_zero_hessian_returns_zero(
            gradient in -1000.0_f64..1000.0_f64,
            reg_alpha in 0.0_f64..10.0_f64,
            reg_lambda in 0.0_f64..10.0_f64,
        ) {
            // When hessian + lambda is near zero, should return 0
            let hessian = 0.0_f64;
            if reg_lambda < EPSILON {
                let value = compute_leaf_value(gradient, hessian, reg_alpha, reg_lambda);
                prop_assert!(value.abs() < EPSILON, "Expected 0, got {}", value);
            }
        }

        #[test]
        fn prop_compute_leaf_value_l1_soft_threshold(
            gradient in -100.0_f64..100.0_f64,
            hessian in 1.0_f64..100.0_f64,
            reg_alpha in 0.0_f64..50.0_f64,
            reg_lambda in 0.0_f64..10.0_f64,
        ) {
            let value = compute_leaf_value(gradient, hessian, reg_alpha, reg_lambda);

            // L1 soft threshold: if |G| <= alpha, value should be 0
            if gradient.abs() <= reg_alpha {
                prop_assert!(value.abs() < EPSILON, "Expected 0 when |G| <= alpha, got {}", value);
            }

            // Value should be finite
            prop_assert!(value.is_finite(), "Value should be finite, got {}", value);
        }

        #[test]
        fn prop_compute_leaf_value_sign_correct(
            gradient in -100.0_f64..100.0_f64,
            hessian in 1.0_f64..100.0_f64,
        ) {
            // Without regularization: -G/H
            let value = compute_leaf_value(gradient, hessian, 0.0_f64, 0.0_f64);

            // Sign should be opposite of gradient (when hessian > 0)
            if gradient.abs() > EPSILON {
                let expected_sign = if gradient > 0.0_f64 { -1.0_f64 } else { 1.0_f64 };
                let actual_sign = if value > 0.0_f64 { 1.0_f64 } else { -1.0_f64 };
                prop_assert_eq!(expected_sign, actual_sign, "Sign mismatch: G={}, value={}", gradient, value);
            }
        }

        #[test]
        fn prop_should_stop_respects_constraints(
            depth in 0_usize..20_usize,
            n_samples in 1_usize..1000_usize,
            n_leaves in 0_usize..100_usize,
            max_depth in 0_usize..15_usize,
            max_leaves in 0_usize..50_usize,
            min_samples_split in 2_usize..50_usize,
            min_samples_leaf in 1_usize..25_usize,
        ) {
            let result = should_stop(depth, n_samples, n_leaves, max_depth, max_leaves, min_samples_split, min_samples_leaf);

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
        }

        #[test]
        fn prop_split_samples_preserves_count(
            n_samples in 2_usize..20_usize,
            split_bin in 0_usize..5_usize,
            nan_goes_left in proptest::bool::ANY,
        ) {
            let n_regular_bins = 6_usize;
            let sample_indices: Vec<usize> = (0_usize..n_samples).collect();

            // Create bins that distribute samples across bins
            let bins: Vec<Vec<usize>> = (0_usize..n_samples)
                .map(|i| vec![i % n_regular_bins])
                .collect();

            let (left, right) = split_samples(&sample_indices, &bins, 0_usize, split_bin, nan_goes_left, n_regular_bins);

            // Total samples should be preserved
            prop_assert_eq!(
                left.len() + right.len(),
                n_samples,
                "Sample count not preserved: left={}, right={}, total={}",
                left.len(), right.len(), n_samples
            );

            // No duplicates
            let mut all: Vec<usize> = left.iter().chain(right.iter()).copied().collect();
            all.sort();
            all.dedup();
            prop_assert_eq!(all.len(), n_samples, "Duplicate samples found");
        }
    }

    // =========================================================================
    // TreeBuildConfig tests
    // =========================================================================

    #[test]
    fn test_tree_build_config_new_valid() -> std::result::Result<(), ClearGbmError> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let c = TreeBuildConfig::new(5_usize, 10_usize, 0.0_f64, 1.0_f64, sc)?;

        assert_eq!(c.max_depth(), 5_usize);
        assert_eq!(c.max_leaves(), 10_usize);
        assert!(c.reg_alpha().abs() < EPSILON);
        assert!((c.reg_lambda() - 1.0_f64).abs() < EPSILON);
        Ok(())
    }

    #[test]
    fn test_tree_build_config_negative_reg_alpha() -> std::result::Result<(), ClearGbmError> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let config = TreeBuildConfig::new(5_usize, 10_usize, -0.1_f64, 1.0_f64, sc);

        assert!(config.is_err());
        assert!(matches!(
            config.err(),
            Some(ClearGbmError::InvalidParameter { name, .. }) if name == "reg_alpha"
        ));
        Ok(())
    }

    #[test]
    fn test_tree_build_config_negative_reg_lambda() -> std::result::Result<(), ClearGbmError> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
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
    fn test_tree_new() {
        let leaf = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        let tree = Tree::new(vec![leaf], 0_usize, 1_usize);

        assert_eq!(tree.n_nodes(), 1_usize);
        assert_eq!(tree.n_leaves(), 1_usize);
        assert_eq!(tree.max_depth(), 0_usize);
    }

    #[test]
    fn test_tree_nodes_accessor() {
        let leaf1 = TreeNode::new_leaf(0_usize, 0.5_f64, 50_usize);
        let leaf2 = TreeNode::new_leaf(1_usize, -0.5_f64, 50_usize);
        let tree = Tree::new(vec![leaf1, leaf2], 0_usize, 2_usize);

        let nodes = tree.nodes();
        assert_eq!(nodes.len(), 2_usize);
        assert_eq!(nodes[0_usize].node_id(), 0_usize);
        assert_eq!(nodes[1_usize].node_id(), 1_usize);
    }

    #[test]
    fn test_tree_node_access() -> std::result::Result<(), ClearGbmError> {
        let leaf = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        let tree = Tree::new(vec![leaf.clone()], 0_usize, 1_usize);

        let node = tree.root()?;
        assert_eq!(node.node_id(), 0_usize);
        assert!(node.is_leaf());

        let node0 = tree.node(0_usize)?;
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
    fn test_tree_empty_root_error() {
        let tree = Tree::new(vec![], 0_usize, 0_usize);
        let root = tree.root();
        assert!(root.is_err());
        assert!(matches!(
            root.err(),
            Some(ClearGbmError::NodeNotFound { node_id: 0_usize })
        ));
    }

    #[test]
    fn test_tree_serialize_deserialize() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let leaf = TreeNode::new_leaf(0_usize, 0.5_f64, 100_usize);
        let tree = Tree::new(vec![leaf], 1_usize, 1_usize);

        let json_str = serde_json::to_string(&tree)?;
        let p: Tree = serde_json::from_str(&json_str)?;

        assert_eq!(p.n_nodes(), 1_usize);
        assert_eq!(p.max_depth(), 1_usize);
        assert_eq!(p.n_leaves(), 1_usize);
        Ok(())
    }

    #[test]
    fn test_tree_build_config_getters() -> std::result::Result<(), ClearGbmError> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.5_f64, 0.01_f64)?;
        let c = TreeBuildConfig::new(5_usize, 10_usize, 0.1_f64, 0.5_f64, sc)?;

        assert_eq!(c.max_depth(), 5_usize);
        assert_eq!(c.max_leaves(), 10_usize);
        assert!((c.reg_alpha() - 0.1_f64).abs() < EPSILON);
        assert!((c.reg_lambda() - 0.5_f64).abs() < EPSILON);
        assert_eq!(c.split_config().min_samples_split(), 2_usize);
        Ok(())
    }

    #[test]
    fn test_tree_build_config_serialize_deserialize(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let c = TreeBuildConfig::new(5_usize, 10_usize, 0.1_f64, 0.5_f64, sc)?;

        let json_str = serde_json::to_string(&c)?;
        let p: TreeBuildConfig = serde_json::from_str(&json_str)?;

        assert_eq!(p.max_depth(), 5_usize);
        assert_eq!(p.max_leaves(), 10_usize);
        Ok(())
    }

    // =========================================================================
    // compute_leaf_value tests
    // =========================================================================

    #[test]
    fn test_compute_leaf_value_basic() {
        // Simple case: -G/H = -2.0/10.0 = -0.2
        let value = compute_leaf_value(2.0_f64, 10.0_f64, 0.0_f64, 0.0_f64);
        assert!((value - (-0.2_f64)).abs() < EPSILON);
    }

    #[test]
    fn test_compute_leaf_value_with_l2() {
        // With L2: -G/(H + lambda) = -2.0/(10.0 + 1.0) = -2.0/11.0
        let value = compute_leaf_value(2.0_f64, 10.0_f64, 0.0_f64, 1.0_f64);
        let expected = -2.0_f64 / 11.0_f64;
        assert!((value - expected).abs() < EPSILON);
    }

    #[test]
    fn test_compute_leaf_value_with_l1() {
        // With L1: soft threshold
        // G = 2.0, alpha = 0.5
        // sign(G) = 1, |G| = 2.0 > alpha
        // value = -1 * (2.0 - 0.5) / (10.0 + 0.0) = -1.5 / 10.0 = -0.15
        let value = compute_leaf_value(2.0_f64, 10.0_f64, 0.5_f64, 0.0_f64);
        let expected = -1.5_f64 / 10.0_f64;
        assert!((value - expected).abs() < EPSILON);
    }

    #[test]
    fn test_compute_leaf_value_l1_below_threshold() {
        // With L1: |G| <= alpha, value = 0
        let value = compute_leaf_value(0.3_f64, 10.0_f64, 0.5_f64, 0.0_f64);
        assert!(value.abs() < EPSILON);
    }

    #[test]
    fn test_compute_leaf_value_zero_hessian() {
        // Zero hessian should return 0
        let value = compute_leaf_value(2.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        assert!(value.abs() < EPSILON);
    }

    #[test]
    fn test_compute_leaf_value_negative_gradient() {
        // Negative gradient: -(-2.0)/10.0 = 0.2
        let value = compute_leaf_value(-2.0_f64, 10.0_f64, 0.0_f64, 0.0_f64);
        assert!((value - 0.2_f64).abs() < EPSILON);
    }

    #[test]
    fn test_compute_leaf_value_negative_gradient_with_l1() {
        // Negative gradient with L1: soft threshold
        // G = -2.0, alpha = 0.5
        // sign(G) = -1, |G| = 2.0 > alpha
        // value = -(-1) * (2.0 - 0.5) / (10.0 + 0.0) = 1.5 / 10.0 = 0.15
        let value = compute_leaf_value(-2.0_f64, 10.0_f64, 0.5_f64, 0.0_f64);
        let expected = 1.5_f64 / 10.0_f64;
        assert!((value - expected).abs() < EPSILON);
    }

    // =========================================================================
    // should_stop tests
    // =========================================================================

    #[test]
    fn test_should_stop_max_depth() {
        assert!(should_stop(
            5_usize, 100_usize, 0_usize, 5_usize, 0_usize, 2_usize, 1_usize
        ));
        assert!(!should_stop(
            4_usize, 100_usize, 0_usize, 5_usize, 0_usize, 2_usize, 1_usize
        ));
    }

    #[test]
    fn test_should_stop_unlimited_depth() {
        // max_depth = 0 means unlimited
        assert!(!should_stop(
            100_usize, 100_usize, 0_usize, 0_usize, 0_usize, 2_usize, 1_usize
        ));
    }

    #[test]
    fn test_should_stop_max_leaves() {
        // max_leaves = 10, n_leaves = 9, would add 1 more -> stop
        assert!(should_stop(
            2_usize, 100_usize, 9_usize, 0_usize, 10_usize, 2_usize, 1_usize
        ));
        assert!(!should_stop(
            2_usize, 100_usize, 8_usize, 0_usize, 10_usize, 2_usize, 1_usize
        ));
    }

    #[test]
    fn test_should_stop_min_samples_split() {
        assert!(should_stop(
            2_usize, 5_usize, 0_usize, 0_usize, 0_usize, 10_usize, 1_usize
        ));
        assert!(!should_stop(
            2_usize, 15_usize, 0_usize, 0_usize, 0_usize, 10_usize, 1_usize
        ));
    }

    #[test]
    fn test_should_stop_min_samples_leaf() {
        // n_samples = 5, min_samples_leaf = 3, need 6 samples minimum
        assert!(should_stop(
            2_usize, 5_usize, 0_usize, 0_usize, 0_usize, 2_usize, 3_usize
        ));
        assert!(!should_stop(
            2_usize, 10_usize, 0_usize, 0_usize, 0_usize, 2_usize, 3_usize
        ));
    }

    // =========================================================================
    // split_samples tests
    // =========================================================================

    #[test]
    fn test_split_samples_basic() {
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
    }

    #[test]
    fn test_split_samples_nan_handling() {
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
    }

    // =========================================================================
    // build_tree tests
    // =========================================================================

    #[test]
    fn test_build_tree_single_leaf() -> std::result::Result<(), ClearGbmError> {
        // Create simple data that results in a single leaf
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(1_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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

        let tree = build_tree(&input, &Hooks::default())?;

        // Should be a single leaf (max_depth = 1, samples in same bin)
        assert_eq!(tree.n_leaves(), 1_usize);
        let _ = tree.root()?;
        Ok(())
    }

    #[test]
    fn test_build_tree_with_split() -> std::result::Result<(), ClearGbmError> {
        // Create data with clear split
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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

        let tree = build_tree(&input, &Hooks::default())?;

        // Should have split
        assert!(tree.n_nodes() >= 3_usize);
        assert!(tree.n_leaves() >= 2_usize);

        // Root should not be a leaf
        let root = tree.root()?;
        assert!(!root.is_leaf());
        Ok(())
    }

    #[test]
    fn test_build_tree_empty_input() -> std::result::Result<(), ClearGbmError> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
    fn test_build_tree_max_depth_constraint() -> std::result::Result<(), ClearGbmError> {
        // max_depth = 1 should create root + 2 leaves max
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(1_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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

        let tree = build_tree(&input, &Hooks::default())?;

        // Max depth = 1, so max 3 nodes (root + 2 leaves)
        assert!(tree.n_nodes() <= 3_usize);
        assert!(tree.max_depth() <= 1_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_max_leaves_constraint() -> std::result::Result<(), ClearGbmError> {
        // max_leaves = 2
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(10_usize, 2_usize, 0.0_f64, 0.0_f64, sc)?;

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

        let tree = build_tree(&input, &Hooks::default())?;
        assert!(tree.n_leaves() <= 2_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_gradients_too_short() -> std::result::Result<(), ClearGbmError> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
    fn test_build_tree_hessians_too_short() -> std::result::Result<(), ClearGbmError> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
    fn test_build_tree_no_features() -> std::result::Result<(), ClearGbmError> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
    fn test_build_tree_empty_bins_vec() -> std::result::Result<(), ClearGbmError> {
        // Test where bins vec itself is empty (different from empty inner vecs)
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
    fn test_build_tree_with_monotonic_constraints() -> std::result::Result<(), ClearGbmError> {
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
        let _ = build_tree(&input, &Hooks::default())?;
        Ok(())
    }

    #[test]
    fn test_build_tree_with_l1_regularization() -> std::result::Result<(), ClearGbmError> {
        // Use L1 regularization
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.5_f64, 0.0_f64, sc)?;

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

        let _ = build_tree(&input, &Hooks::default())?;
        Ok(())
    }

    #[test]
    fn test_build_tree_left_larger_than_right() -> std::result::Result<(), ClearGbmError> {
        // Test where left child has more samples than right
        // This exercises the else branch in compute_child_histograms
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(2_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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

        let tree = build_tree(&input, &Hooks::default())?;

        // Should have split into left (4 samples) and right (2 samples)
        assert!(tree.n_nodes() >= 3_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_deep_tree() -> std::result::Result<(), ClearGbmError> {
        // Test building a deeper tree to exercise more code paths
        // Allow deep tree
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(10_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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

        let tree = build_tree(&input, &Hooks::default())?;

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
    fn test_finalize_nodes_internal_node_missing_feature_index(
    ) -> std::result::Result<(), ClearGbmError> {
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
    fn test_finalize_nodes_internal_node_missing_split_bin(
    ) -> std::result::Result<(), ClearGbmError> {
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
    fn test_finalize_nodes_internal_node_missing_left_child(
    ) -> std::result::Result<(), ClearGbmError> {
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
    fn test_finalize_nodes_internal_node_missing_right_child(
    ) -> std::result::Result<(), ClearGbmError> {
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
    fn test_finalize_nodes_leaf_node_success() -> std::result::Result<(), ClearGbmError> {
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

        let nodes = finalize_nodes(&build_nodes, &child_pointers, &bin_thresholds)?;
        assert_eq!(nodes.len(), 1_usize);
        assert!(nodes[0_usize].is_leaf());
        assert!((nodes[0_usize].value() - 1.5_f64).abs() < 1e-10_f64);
        Ok(())
    }

    // =========================================================================
    // Internal function error path tests
    // =========================================================================

    #[test]
    fn test_build_feature_histograms_empty_features() -> std::result::Result<(), ClearGbmError> {
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
    fn test_find_best_split_across_features_internal_error(
    ) -> std::result::Result<(), ClearGbmError> {
        // Create a histogram and config that will trigger an error
        // n_regular_bins > n_bins should cause an error
        let histogram = HistogramBuffer::new(3_usize);
        let histograms = vec![histogram];
        let config = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;

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
    fn test_find_best_split_across_features_internal_multiple_features(
    ) -> std::result::Result<(), ClearGbmError> {
        // Test with 2 features that both have valid splits to cover the comparison closure
        let mut hist0 = HistogramBuffer::new(4_usize);
        for _ in 0_usize..10_usize {
            hist0.accumulate(0_usize, 0.5_f64, 1.0_f64)?;
        }
        for _ in 0_usize..10_usize {
            hist0.accumulate(1_usize, -0.5_f64, 1.0_f64)?;
        }

        let mut hist1 = HistogramBuffer::new(4_usize);
        for _ in 0_usize..10_usize {
            hist1.accumulate(0_usize, 0.3_f64, 1.0_f64)?;
        }
        for _ in 0_usize..10_usize {
            hist1.accumulate(1_usize, -0.3_f64, 1.0_f64)?;
        }

        let histograms = vec![hist0, hist1];
        let config = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;

        let result = find_best_split_across_features_internal(&histograms, &config, 3_usize, None)?;

        // Should find the best split (feature 0 has higher gain due to larger gradient magnitude)
        assert!(matches!(result, Some(ref s) if s.feature_index() == 0_usize));
        Ok(())
    }

    #[test]
    fn test_compute_child_histograms_parent_histograms_too_short(
    ) -> std::result::Result<(), ClearGbmError> {
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
    fn test_compute_child_histograms_success() -> std::result::Result<(), ClearGbmError> {
        // Test successful child histogram computation
        let left_indices = vec![0_usize, 1_usize];
        let right_indices = vec![2_usize, 3_usize];
        let gradients = vec![1.0_f64, 2.0_f64, -1.0_f64, -2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![0_usize], vec![1_usize], vec![1_usize]];

        // Create parent histogram with proper values
        let mut parent_hist = HistogramBuffer::new(3_usize);
        parent_hist.accumulate(0_usize, 3.0_f64, 2.0_f64)?;
        parent_hist.accumulate(1_usize, -3.0_f64, 2.0_f64)?;

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

        let (left_hists, right_hists) = compute_child_histograms(&config)?;
        assert_eq!(left_hists.len(), 1_usize);
        assert_eq!(right_hists.len(), 1_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_with_large_n_regular_bins() -> std::result::Result<(), ClearGbmError> {
        // Test with n_regular_bins much larger than actual bins used
        // This succeeds because histogram is built with n_bins = n_regular_bins + 1
        let sc = SplitConfig::new(2_usize, 1_usize, 4_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
        let tree = build_tree(&input, &Hooks::default())?;
        assert!(tree.n_nodes() >= 1_usize);
        Ok(())
    }

    #[test]
    fn test_build_tree_bins_out_of_bounds() -> std::result::Result<(), ClearGbmError> {
        // Test with bin values that exceed n_bins to trigger error in build_feature_histograms
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
    fn test_is_increasing_is_decreasing_methods() -> std::result::Result<(), ClearGbmError> {
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
    ) -> std::result::Result<HistogramBuffer, ClearGbmError> {
        Err(ClearGbmError::EmptyInput {
            context: "injected error from hook".to_string(),
        })
    }

    #[test]
    fn test_build_tree_hooks_error_in_histogram_building() -> std::result::Result<(), ClearGbmError>
    {
        // Use hooks to inject error during histogram building (exercises line 474's `?`)
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
    fn test_build_feature_histograms_with_cached_histograms(
    ) -> std::result::Result<(), ClearGbmError> {
        // Test that cached histograms are used when available
        let sample_indices = vec![0_usize, 1_usize];
        let gradients = vec![1.0_f64, 2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![1_usize]];

        // Create cached histograms
        let mut cached = HistogramBuffer::new(3_usize);
        cached.accumulate(0_usize, 1.0_f64, 1.0_f64)?;
        cached.accumulate(1_usize, 2.0_f64, 1.0_f64)?;
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

        let result = build_feature_histograms(&config)?;

        // Should return the cached histograms
        assert_eq!(result.len(), 1_usize);
        Ok(())
    }

    #[test]
    fn test_build_feature_histograms_with_wrong_size_cache(
    ) -> std::result::Result<(), ClearGbmError> {
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
        let result = build_feature_histograms(&config)?;
        assert_eq!(result.len(), 2_usize);
        Ok(())
    }

    #[test]
    fn test_compute_child_histograms_hooks_error() -> std::result::Result<(), ClearGbmError> {
        // Test error propagation from hooks in compute_child_histograms
        let left_indices = vec![0_usize, 1_usize];
        let right_indices = vec![2_usize, 3_usize];
        let gradients = vec![1.0_f64, 2.0_f64, -1.0_f64, -2.0_f64];
        let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let bins = vec![vec![0_usize], vec![0_usize], vec![1_usize], vec![1_usize]];

        let mut parent_hist = HistogramBuffer::new(3_usize);
        parent_hist.accumulate(0_usize, 3.0_f64, 2.0_f64)?;
        parent_hist.accumulate(1_usize, -3.0_f64, 2.0_f64)?;
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
    ) -> std::result::Result<HistogramBuffer, ClearGbmError> {
        // Return a histogram with only 2 bins, regardless of requested size
        // This will cause find_best_split_from_histogram to fail when n_regular_bins > 2
        Ok(HistogramBuffer::new(2_usize))
    }

    #[test]
    fn test_build_tree_hooks_error_in_split_finding() -> std::result::Result<(), ClearGbmError> {
        // Use hooks to inject undersized histogram, causing split finding to fail
        // This exercises the `?` at line 482 in build_tree
        let sc = SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)?;
        let cfg = TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc)?;

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
}
