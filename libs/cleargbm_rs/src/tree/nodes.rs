//! Tree-node bookkeeping used by [`super::builder`].
//!
//! Holds the internal node representations, the leaf-value formula, the
//! stopping-criterion predicate, and the node-finalization pass that
//! converts intermediate build records into the immutable [`TreeNode`]
//! form. This is the "structural" half of the tree builder; the
//! "histogram" half lives in [`super::histograms`].

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::split::SplitDecision;
use crate::types::{CategoricalNodeConfig, HistogramBuffer, TreeNode, TreeNodeConfig};

use super::categorical::CategoricalLayout;

/// Epsilon for floating-point comparisons.
pub(crate) const EPSILON: f64 = 1e-10_f64;

/// Internal struct for tracking pending nodes during tree building.
#[derive(Debug)]
pub(super) struct PendingNode {
    /// Sample indices at this node (u32 per lightgbm-score-t-float
    /// `data_size_t = int32` pattern; widened via `crate::narrow::index_widen`
    /// at access sites).
    pub(super) sample_indices: Vec<u32>,

    /// Current depth.
    pub(super) depth: usize,

    /// Parent node ID (None for root).
    pub(super) parent_id: Option<usize>,

    /// Whether this is the left child of parent.
    pub(super) is_left_child: bool,

    /// Cached histograms from parent's sibling subtraction (for 2x speedup).
    pub(super) cached_histograms: Option<Vec<HistogramBuffer>>,
}

/// Internal struct for building tree nodes before finalization.
#[derive(Debug)]
pub(super) struct BuildNode {
    /// Node ID.
    pub(super) node_id: usize,

    /// Whether this is a leaf.
    pub(super) is_leaf: bool,

    /// Feature index for split (None for leaf).
    pub(super) feature_index: Option<usize>,

    /// How the split partitions samples (None for leaf).
    pub(super) decision: Option<SplitDecision>,

    /// Node value (leaf value or intermediate).
    pub(super) value: f64,

    /// Number of samples.
    pub(super) n_samples: usize,

    /// NaN direction.
    pub(super) nan_goes_left: bool,
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

/// Checks if node should be a leaf based on stopping criteria.
pub(super) fn should_stop(
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
///
/// Handles out-of-bounds indices defensively by skipping them.
pub(super) fn compute_sums(
    sample_indices: &[u32],
    gradients: &[f64],
    hessians: &[f64],
) -> (f64, f64) {
    let mut g_sum = 0.0_f64;
    let mut h_sum = 0.0_f64;

    for &idx in sample_indices {
        let idx_usize = crate::narrow::index_widen(idx);
        if idx_usize < gradients.len() {
            g_sum += gradients[idx_usize];
        }
        if idx_usize < hessians.len() {
            h_sum += hessians[idx_usize];
        }
    }

    (g_sum, h_sum)
}

/// Splits samples into left and right based on a split result.
///
/// Reads bin values from a flat column-major `u8` slice. Samples whose bin
/// index is out of range for the flat slice are treated as NaN — the
/// pre-refactor behavior (missing per-row Vec → NaN) carried over.
pub(super) fn split_samples(
    sample_indices: &[u32],
    bins_rows: &[u8],
    n_features: usize,
    feature_index: usize,
    decision: SplitDecision,
    nan_goes_left: bool,
    n_regular_bins: usize,
) -> (Vec<u32>, Vec<u32>) {
    let nan_bin = n_regular_bins;
    // The row count is implied by the matrix: its length is validated as
    // n_samples * n_features at the build_tree boundary.
    let n_samples = bins_rows.len() / n_features;
    // Reserve the node's full sample count for both sides up front. Growing
    // from empty reallocates and copies on every doubling — about sixteen
    // times for a root node of ~55k samples, per side, per split — and the
    // copied bytes roughly double the partition's write volume. One side
    // always ends up over-reserved, but peak cost is bounded by the largest
    // single node (the root: two buffers of `n * 4` bytes) because the
    // builder walks the tree depth-first and only one split is live at a time.
    let n_at_node = sample_indices.len();
    let mut left = Vec::with_capacity(n_at_node);
    let mut right = Vec::with_capacity(n_at_node);

    for &idx in sample_indices {
        let idx_usize = crate::narrow::index_widen(idx);
        let bin = if idx_usize < n_samples {
            usize::from(bins_rows[idx_usize * n_features + feature_index])
        } else {
            nan_bin
        };

        // NaN bin handling
        if bin == nan_bin {
            if nan_goes_left {
                left.push(idx);
            } else {
                right.push(idx);
            }
            continue;
        }
        let goes_left = match decision {
            SplitDecision::Threshold { split_bin } => bin <= split_bin,
            SplitDecision::CategorySubset { left_bins } => left_bins.contains(bin),
        };
        if goes_left {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    (left, right)
}

/// Finalizes build nodes into TreeNode with proper child pointers.
///
/// Threshold splits read their boundary from `bin_thresholds`; categorical
/// splits translate their left-routed bins into raw category codes through
/// `categorical` (required whenever such a split exists).
pub(super) fn finalize_nodes(
    build_nodes: &[BuildNode],
    child_pointers: &[(Option<usize>, Option<usize>)],
    bin_thresholds: &[Vec<f64>],
    categorical: Option<&CategoricalLayout>,
    hooks: &Hooks,
) -> Result<Vec<TreeNode>, ClearGbmError> {
    // Check for injected error (for testing error propagation)
    if let Some(ref err) = hooks.finalize_nodes_error {
        return Err(err.clone());
    }

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
            let decision = match node.decision {
                Some(d) => d,
                None => {
                    return Err(ClearGbmError::TreeConstructionFailed {
                        reason: "internal node missing split decision".to_string(),
                    })
                }
            };

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

            match decision {
                SplitDecision::Threshold { split_bin } => {
                    // Get threshold from bin_thresholds
                    // Threshold is the upper bound of the split_bin
                    let threshold = bin_thresholds
                        .get(feature_index)
                        .and_then(|thresholds| thresholds.get(split_bin).copied())
                        .unwrap_or(0.0_f64);

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
                SplitDecision::CategorySubset { left_bins } => {
                    let layout = match categorical {
                        Some(layout) => layout,
                        None => {
                            return Err(ClearGbmError::TreeConstructionFailed {
                                reason: format!(
                                    "node {} split categorically but no categorical                                      layout was provided",
                                    node.node_id
                                ),
                            })
                        }
                    };
                    let categories_goes_left = match layout.left_codes(feature_index, left_bins) {
                        Ok(codes) => codes,
                        Err(e) => return Err(e),
                    };
                    final_nodes.push(TreeNode::new_categorical_internal(CategoricalNodeConfig {
                        node_id: node.node_id,
                        feature_index,
                        categories_goes_left,
                        value: node.value,
                        n_samples: node.n_samples,
                        left_child: left_id,
                        right_child: right_id,
                        nan_goes_left: node.nan_goes_left,
                    }));
                }
            }
        }
    }

    Ok(final_nodes)
}
