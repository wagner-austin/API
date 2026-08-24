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
    BuildHistogramConfig, ChildHistogramConfig, OrderedScratch,
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
/// `bins_rows` is a flat row-major slice: bin `[sample_idx, feat_idx]` lives
/// at `bins_rows[sample_idx * n_features + feat_idx]`. Each entry is a `u8`
/// (bin index, `0..=n_regular_bins`). Row-major means the single-pass node
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
    pub bins_rows: &'a [u8],

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

    /// Optional per-split feature subsampling (`max_features`). `None`
    /// considers every feature at every split, bit-identical to the
    /// history before this axis existed.
    pub feature_subsample: Option<super::feature_subsample::FeatureSubsample>,

    /// Optional per-TREE feature mask (`colsample_bytree`), derived once
    /// per boosting round. `None` lets the tree use every feature. When
    /// both axes are set, the per-split draw selects within this mask.
    pub tree_feature_mask: Option<&'a [bool]>,

    /// Optional per-feature category tables. `None` treats every feature
    /// as numeric, bit-identical to the history before the categorical
    /// axis existed. Required whenever a feature is categorical: the split
    /// search partitions its bins by set membership and finalization
    /// translates the winning bins into raw codes.
    pub categorical: Option<&'a super::categorical::CategoricalLayout>,

    /// Optional quantized-training data for this round. `None` runs the
    /// float histogram path, bit-identical to the history before the
    /// quantized axis existed. `Some` switches histogram construction,
    /// sibling subtraction, and the threshold scan to packed integers;
    /// leaf values are computed from the original float gradients either
    /// way, so quantization only ever affects which splits are chosen.
    pub quantized: Option<QuantizedTreeData<'a>>,
}

/// One round's quantized streams and decode scales, as the tree consumes
/// them.
///
/// Produced by the training loop's per-round discretization (see
/// `training::quantize`); the tree never re-derives scales or re-rounds.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedTreeData<'a> {
    /// Interleaved `int8` stream: hessian at `2i`, gradient at `2i + 1`,
    /// length `2 * n_samples`.
    pub packed_int8: &'a [i8],
    /// Integer gradient sums multiply by this to recover gradient space.
    pub grad_scale: f64,
    /// Integer hessian sums multiply by this to recover hessian space.
    pub hess_scale: f64,
    /// The quantization bin count (`quantized_gradient_bins`).
    pub n_quant_bins: usize,
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

    // Quantized invariants, checked before any length math that would
    // require materializing the bin matrix: the 32-bit packed halves
    // hold per-bin sums bounded by `n_samples * n_quant_bins`, so that
    // product must fit the hessian half (u32); and the interleaved
    // stream must cover every row's pair.
    if let Some(quantized) = input.quantized {
        let n_u64 = u64::try_from(input.n_samples).unwrap_or(u64::MAX);
        let bins_u64 = u64::try_from(quantized.n_quant_bins).unwrap_or(u64::MAX);
        let max_stat = n_u64.saturating_mul(bins_u64);
        if max_stat > u64::from(u32::MAX) {
            return Err(ClearGbmError::InvalidParameter {
                name: "quantized_gradient_bins".to_string(),
                reason: format!(
                    "n_samples ({}) x quantized_gradient_bins ({}) must not exceed u32::MAX \
                     (the packed 32-bit histogram half's capacity)",
                    input.n_samples, quantized.n_quant_bins
                ),
            });
        }
        if quantized.packed_int8.len() != 2_usize * input.n_samples {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("packed_int8 length {}", 2_usize * input.n_samples),
                got: format!("packed_int8 length {}", quantized.packed_int8.len()),
            });
        }
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
    if input.bins_rows.len() != expected_bins_len {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!(
                "bins length {expected_bins_len} (n_samples={} * n_features={})",
                input.n_samples, n_features
            ),
            got: format!("bins length {}", input.bins_rows.len()),
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

    // One pair of ordered-stream scratch buffers reused by every node in
    // this tree; see `OrderedScratch` for the allocation-churn rationale.
    let mut scratch = OrderedScratch::new(input.sample_indices.len());

    // Stack of pending nodes
    let mut stack: Vec<PendingNode> = Vec::new();
    stack.push(PendingNode {
        sample_indices: input.sample_indices.to_vec(),
        depth: 0_usize,
        parent_id: None,
        is_left_child: false,
        cached_histograms: None,
    });

    while let Some(mut pending) = stack.pop() {
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
                decision: None,
                value: leaf_value,
                n_samples: current_n_samples,
                nan_goes_left: true,
            });
            n_leaves += 1_usize;
            continue;
        }

        // Histograms for all features: the sibling-subtraction cache is
        // TAKEN from the pending node and used directly — it was cloned
        // here before, 18 buffer allocations and ~27 KB of copying per
        // cached node for values the node already owned.
        let histograms = match pending.cached_histograms.take() {
            Some(cache) => cache,
            None => {
                let hist_config = BuildHistogramConfig {
                    sample_indices: &pending.sample_indices,
                    gradients: input.gradients,
                    hessians: input.hessians,
                    bins_rows: input.bins_rows,
                    n_features,
                    n_bins,
                    quantized: input.quantized,
                    hooks,
                };
                match build_feature_histograms(&hist_config, &mut scratch) {
                    Ok(h) => h,
                    Err(e) => return Err(e),
                }
            }
        };

        // Find best split across the features this node may consider:
        // the per-node draw (within the tree mask when one is active), or
        // the tree mask alone, or everything.
        let feature_mask = match input.feature_subsample {
            Some(fs) => Some(super::feature_subsample::select_split_features(
                fs,
                n_features,
                node_id,
                input.tree_feature_mask,
            )),
            None => input.tree_feature_mask.map(<[bool]>::to_vec),
        };
        let best_split = match find_best_split_across_features_internal(
            &histograms,
            split_config,
            input.n_regular_bins,
            input.monotonic_constraints,
            feature_mask.as_deref(),
            input.categorical,
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
                decision: None,
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
            decision: Some(split.decision()),
            value: node_value,
            n_samples: current_n_samples,
            nan_goes_left: split.nan_goes_left(),
        });

        // Split samples into left and right
        let (left_indices, right_indices) = split_samples(
            &pending.sample_indices,
            input.bins_rows,
            input.n_features,
            split.feature_index(),
            split.decision(),
            split.nan_goes_left(),
            input.n_regular_bins,
        );

        // Compute child histograms using sibling subtraction trick
        let child_hist_config = ChildHistogramConfig {
            left_indices: &left_indices,
            right_indices: &right_indices,
            gradients: input.gradients,
            hessians: input.hessians,
            bins_rows: input.bins_rows,
            n_features,
            n_bins,
            quantized: input.quantized,
            parent_histograms: &histograms,
            hooks,
        };
        let (left_histograms, right_histograms) =
            match compute_child_histograms(&child_hist_config, &mut scratch) {
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
    let final_nodes: Vec<TreeNode> = match finalize_nodes(
        &nodes,
        &child_pointers,
        input.bin_thresholds,
        input.categorical,
        hooks,
    ) {
        Ok(n) => n,
        Err(e) => return Err(e),
    };

    Ok((
        Tree::new(final_nodes, max_depth_found, n_leaves),
        leaf_value_per_sample,
    ))
}
