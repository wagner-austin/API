//! Tree-building main loop.
//!
//! Drives one depth-first traversal that materializes a single decision
//! tree from a sample set + gradients + hessians + column-major bins. The
//! per-node bookkeeping (node structs, leaf-value formula, stopping
//! criteria, node finalization) lives in [`super::nodes`]; the
//! per-node histogram construction and split search live in
//! [`super::histograms`]. This file is deliberately kept to orchestration
//! only.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::split::MonotonicConstraint;
use crate::types::TreeNode;

use super::histograms::{
    build_feature_histograms, compute_child_histograms, find_best_split_across_features_internal,
    BuildHistogramConfig, ChildHistogramConfig,
};
use super::nodes::{
    compute_sums, finalize_nodes, should_stop, split_samples, BuildNode, PendingNode,
};
use super::{Tree, TreeBuildConfig};

// Re-export the piece external callers touch. `compute_leaf_value` originates
// in `super::nodes`; the re-export keeps the `crate::tree::builder::…` paths
// that tests + the crate root use working.
pub use super::nodes::compute_leaf_value;

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
    /// Sample indices to include in tree (u32 per lightgbm-score-t-float
    /// `data_size_t = int32` pattern; widened at access sites via
    /// `crate::narrow::index_widen`).
    pub sample_indices: &'a [u32],

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

/// Records a leaf's value against every sample that reached it.
///
/// `sample_indices` are `u32` per the `lightgbm-score-t-float` note; they are
/// widened for `Vec<f64>` access via the infallible
/// [`crate::narrow::index_widen`].
///
/// Infallible: [`build_tree_with_leaf_assignment`] rejects an out-of-range
/// sample index before construction begins, so by the time a leaf is finalized
/// every index here addresses a real slot. Indexed directly rather than
/// through `get_mut` for that reason — a `None` arm would be a branch this
/// function's only caller has already made unreachable.
///
/// # Args
///
/// * `sample_indices` - Rows that reached this leaf.
/// * `leaf_value` - The leaf's prediction.
/// * `leaf_value_per_sample` - Per-sample output, length `n_samples`.
pub(super) fn record_leaf_values(
    sample_indices: &[u32],
    leaf_value: f64,
    leaf_value_per_sample: &mut [f64],
) {
    for &sample_idx in sample_indices {
        let sample_idx_usize = crate::narrow::index_widen(sample_idx);
        leaf_value_per_sample[sample_idx_usize] = leaf_value;
    }
}

/// Validates the shapes and index ranges a tree build depends on.
///
/// Shared by both growth policies — the preconditions are properties of the
/// input, not of the order nodes get split in, so a second copy in
/// [`super::leafwise`] could only drift out of step with this one.
///
/// # Args
///
/// * `input` - Build tree input configuration.
///
/// # Errors
///
/// Returns `ClearGbmError::EmptyInput` for an empty sample set or zero
/// features, `ClearGbmError::ShapeMismatch` if the gradient, hessian or bin
/// slices do not match the declared shape, and
/// `ClearGbmError::SampleIndexOutOfBounds` if any sample index falls outside
/// the per-sample output.
pub(super) fn validate_build_input(input: &BuildTreeInput<'_>) -> Result<(), ClearGbmError> {
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
    // Every sample index must address a slot in the per-sample leaf-value
    // output. Checked once here rather than per write: the recording loop runs
    // for every sample in every leaf, so a bound test inside it would be a
    // branch in the hot path that this single up-front pass makes redundant.
    for &sample_idx in input.sample_indices {
        let sample_idx_usize = crate::narrow::index_widen(sample_idx);
        if sample_idx_usize >= input.n_samples {
            return Err(ClearGbmError::SampleIndexOutOfBounds {
                index: sample_idx_usize,
                n_samples: input.n_samples,
            });
        }
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

    Ok(())
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
    match build_tree_with_leaf_assignment(input, hooks) {
        Ok((tree, _leaf_assignment)) => Ok(tree),
        Err(e) => Err(e),
    }
}

/// Builds a tree AND returns a per-sample-index → leaf-value mapping.
///
/// The mapping is populated as leaves are finalized during construction:
/// for every sample that ends up in a leaf, `leaf_value_per_sample[sample_idx]`
/// records that leaf's prediction value. Callers whose sample_indices cover
/// the full training set (i.e. `subsample = 1.0`) can use this to skip
/// `predict_tree` on those samples entirely — direct O(N) lookup + add
/// instead of an O(N × depth) tree walk. Samples NOT in the input
/// `sample_indices` (subsampled-out) will not be updated here and are
/// left at the caller-supplied initial value (typically `f64::NAN` as a
/// sentinel so the caller knows to fall back to tree-walk).
///
/// The caller passes a pre-sized `Vec<f64>` of length ≥ the max
/// sample-index encountered in the tree. It's populated in place as a
/// side effect of tree construction.
///
/// # Errors
///
/// Same as [`build_tree`].
pub fn build_tree_with_leaf_assignment(
    input: &BuildTreeInput<'_>,
    hooks: &Hooks,
) -> Result<(Tree, Vec<f64>), ClearGbmError> {
    match validate_build_input(input) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let n_features = input.n_features;

    // Number of bins in histogram (regular + 1 for NaN)
    let n_bins = input.n_regular_bins + 1_usize;

    let config = input.config;
    let split_config = config.split_config();

    // Per-sample leaf-value output. Size = input.n_samples (max index the
    // tree could touch). NaN sentinel: samples not in sample_indices
    // (subsampled-out) keep NaN so the caller can detect + fall back to
    // predict_tree for them.
    let mut leaf_value_per_sample: Vec<f64> = vec![f64::NAN; input.n_samples];

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

            record_leaf_values(
                &pending.sample_indices,
                leaf_value,
                &mut leaf_value_per_sample,
            );

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

            record_leaf_values(
                &pending.sample_indices,
                leaf_value,
                &mut leaf_value_per_sample,
            );

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
    let final_nodes: Vec<TreeNode> =
        match finalize_nodes(&nodes, &child_pointers, input.bin_thresholds, hooks) {
            Ok(n) => n,
            Err(e) => return Err(e),
        };

    Ok((
        Tree::new(final_nodes, max_depth_found, n_leaves),
        leaf_value_per_sample,
    ))
}
