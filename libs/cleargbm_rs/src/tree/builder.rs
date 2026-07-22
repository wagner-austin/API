//! Tree building logic for gradient boosting.
//!
//! Contains the core tree building algorithm with histogram-based split finding.

use crate::error::ClearGbmError;
use crate::histogram::subtract_histogram;
use crate::hooks::Hooks;
use crate::split::{find_best_split_from_histogram, MonotonicConstraint, SplitResult};
use crate::types::{HistogramBuffer, SplitConfig, TreeNode, TreeNodeConfig};

use super::{Tree, TreeBuildConfig};

/// Epsilon for floating-point comparisons.
pub(super) const EPSILON: f64 = 1e-10_f64;

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
pub(super) struct BuildNode {
    /// Node ID.
    pub(super) node_id: usize,

    /// Whether this is a leaf.
    pub(super) is_leaf: bool,

    /// Feature index for split (None for leaf).
    pub(super) feature_index: Option<usize>,

    /// Split bin index (None for leaf).
    pub(super) split_bin: Option<usize>,

    /// Node value (leaf value or intermediate).
    pub(super) value: f64,

    /// Number of samples.
    pub(super) n_samples: usize,

    /// NaN direction.
    pub(super) nan_goes_left: bool,
}

/// Configuration for `build_tree` to avoid too many arguments.
///
/// # Bin storage
///
/// `bins` is a flat column-major slice: bin `[feat_idx, sample_idx]` lives at
/// `bins[feat_idx * n_samples + sample_idx]`. Each entry is a `u8` (bin index,
/// `0..=n_regular_bins`). Column-major means a per-feature histogram scan
/// walks `n_samples` contiguous bytes — cache-friendly and SIMD-loadable —
/// instead of striding through fragmented row-major heap allocations.
#[derive(Debug, Clone)]
pub struct BuildTreeInput<'a> {
    /// Sample indices to include in tree.
    pub sample_indices: &'a [usize],

    /// Gradients for all samples.
    pub gradients: &'a [f64],

    /// Hessians for all samples.
    pub hessians: &'a [f64],

    /// Bin assignments in flat column-major u8 storage.
    ///
    /// Length must equal `n_samples * n_features`. Access as
    /// `bins[feat_idx * n_samples + sample_idx]`.
    pub bins: &'a [u8],

    /// Sample count (row count of the original feature matrix).
    pub n_samples: usize,

    /// Feature count (column count of the original feature matrix).
    pub n_features: usize,

    /// Number of regular bins (excluding NaN bin).
    pub n_regular_bins: usize,

    /// Bin thresholds per feature for converting bin index to threshold.
    pub bin_thresholds: &'a [Vec<f64>],

    /// Tree build configuration.
    pub config: &'a TreeBuildConfig,

    /// Optional monotonic constraints per feature.
    pub monotonic_constraints: Option<&'a [MonotonicConstraint]>,
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

    let n_features = input.n_features;

    if n_features == 0_usize {
        return Err(ClearGbmError::EmptyInput {
            context: "bins (no features)".to_string(),
        });
    }

    // Validate that the flat bin slice matches the declared shape.
    let expected_bins_len = input.n_samples * n_features;
    if input.bins.len() != expected_bins_len {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!(
                "bins length {expected_bins_len} (n_samples={} * n_features={})",
                input.n_samples, n_features
            ),
            got: format!("bins length {}", input.bins.len()),
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
            n_samples: input.n_samples,
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
            input.n_samples,
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
            n_samples: input.n_samples,
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
    let final_nodes = match finalize_nodes(&nodes, &child_pointers, input.bin_thresholds, hooks) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };

    Ok(Tree::new(final_nodes, max_depth_found, n_leaves))
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
    sample_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
) -> (f64, f64) {
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
pub(super) struct BuildHistogramConfig<'a> {
    /// Sample indices.
    pub(super) sample_indices: &'a [usize],
    /// Gradients for all samples.
    pub(super) gradients: &'a [f64],
    /// Hessians for all samples.
    pub(super) hessians: &'a [f64],
    /// Bin assignments in flat column-major u8 layout, length `n_samples * n_features`.
    pub(super) bins: &'a [u8],
    /// Sample count (row count of the original feature matrix).
    pub(super) n_samples: usize,
    /// Number of features.
    pub(super) n_features: usize,
    /// Number of bins.
    pub(super) n_bins: usize,
    /// Dependency injection hooks.
    pub(super) hooks: &'a Hooks,
    /// Cached histograms from parent's sibling subtraction (for 2x speedup).
    pub(super) cached_histograms: Option<&'a [HistogramBuffer]>,
}

/// Builds histograms for all features.
///
/// Uses cached histograms from parent's sibling subtraction when available,
/// otherwise builds histograms from scratch by dispatching to the histogram
/// hook with a per-feature contiguous bin slice.
///
/// # Column-major fast path
///
/// The bins slice is column-major, so the per-feature bin column is
/// `bins[feat_idx * n_samples..(feat_idx + 1) * n_samples]` — a contiguous
/// `n_samples`-long byte slice. The full `sample_indices`, `gradients`, and
/// `hessians` are threaded through directly; the histogram builder does the
/// per-index gather. This eliminates the three per-(node, feature) allocations
/// the pre-refactor row-major path required.
pub(super) fn build_feature_histograms(
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
        let feat_col_start = feat_idx * config.n_samples;
        let feat_col_end = feat_col_start + config.n_samples;
        let feat_bins = &config.bins[feat_col_start..feat_col_end];

        let hist = match (config.hooks.build_histogram)(
            config.sample_indices,
            config.gradients,
            config.hessians,
            feat_bins,
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
pub(super) fn find_best_split_across_features_internal(
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
///
/// Reads bin values from a flat column-major `u8` slice. Samples whose bin
/// index is out of range for the flat slice are treated as NaN — the
/// pre-refactor behavior (missing per-row Vec → NaN) carried over.
pub(super) fn split_samples(
    sample_indices: &[usize],
    bins: &[u8],
    n_samples: usize,
    feature_index: usize,
    split_bin: usize,
    nan_goes_left: bool,
    n_regular_bins: usize,
) -> (Vec<usize>, Vec<usize>) {
    let nan_bin = n_regular_bins;
    let feat_col_start = feature_index * n_samples;
    let mut left = Vec::new();
    let mut right = Vec::new();

    for &idx in sample_indices {
        let bin = if idx < n_samples {
            usize::from(bins[feat_col_start + idx])
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
        } else if bin <= split_bin {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    (left, right)
}

/// Configuration for computing child histograms.
pub(super) struct ChildHistogramConfig<'a> {
    /// Left child sample indices.
    pub(super) left_indices: &'a [usize],
    /// Right child sample indices.
    pub(super) right_indices: &'a [usize],
    /// Gradients for all samples.
    pub(super) gradients: &'a [f64],
    /// Hessians for all samples.
    pub(super) hessians: &'a [f64],
    /// Bin assignments in flat column-major u8 layout, length `n_samples * n_features`.
    pub(super) bins: &'a [u8],
    /// Sample count (row count of the original feature matrix).
    pub(super) n_samples: usize,
    /// Number of features.
    pub(super) n_features: usize,
    /// Number of bins.
    pub(super) n_bins: usize,
    /// Parent histograms.
    pub(super) parent_histograms: &'a [HistogramBuffer],
    /// Dependency injection hooks.
    pub(super) hooks: &'a Hooks,
}

/// Computes child histograms using sibling subtraction trick.
///
/// Builds histogram for smaller child, derives larger child via subtraction.
pub(super) fn compute_child_histograms(
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
        let feat_col_start = feat_idx * config.n_samples;
        let feat_col_end = feat_col_start + config.n_samples;
        let feat_bins = &config.bins[feat_col_start..feat_col_end];

        let smaller_hist = match (config.hooks.build_histogram)(
            smaller_indices,
            config.gradients,
            config.hessians,
            feat_bins,
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
pub(super) fn finalize_nodes(
    build_nodes: &[BuildNode],
    child_pointers: &[(Option<usize>, Option<usize>)],
    bin_thresholds: &[Vec<f64>],
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
