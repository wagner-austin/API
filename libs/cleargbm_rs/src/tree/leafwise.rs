//! Best-first (leaf-wise) tree construction.
//!
//! The sibling of [`super::builder`], which grows depth-wise. Both consume the
//! same [`BuildTreeInput`] and produce the same [`Tree`]; they differ only in
//! the order nodes are chosen for splitting, and therefore in the shape of the
//! tree that comes out.
//!
//! # Algorithm
//!
//! 1. The root becomes a provisional leaf, and its best split is evaluated.
//! 2. While the leaf budget allows, take the frontier candidate with the
//!    largest split gain, convert it from a leaf into an internal node, and
//!    evaluate both new children for candidacy.
//! 3. Whatever remains on the frontier is already recorded as a leaf.
//!
//! Only the two new children are evaluated per split — every other candidate's
//! best split is unchanged by a split elsewhere in the tree, so it is computed
//! once when the node is created and reused until the node is chosen. This is
//! the property that makes best-first affordable (Shi 2007; the same structure
//! LightGBM's `SerialTreeLearner` uses).
//!
//! # Blocked leaves
//!
//! A node that cannot be split — depth budget reached, too few samples, or no
//! positive-gain split — is simply never placed on the frontier. That is Shi's
//! *removal* handling rather than LightGBM's *gain poisoning*. The two differ
//! only when a blocked leaf could later become splittable, and here none can:
//! depth never decreases, a node's sample count never grows, and its
//! histograms never change once built. The cheaper handling is therefore the
//! equivalent one, not an approximation of it.
//!
//! # Memory
//!
//! Each frontier candidate retains the histograms it was evaluated with, so
//! peak histogram memory scales with the leaf budget rather than with tree
//! depth. That is the standing cost of best-first growth: depth-wise holds one
//! root-to-node path, leaf-wise holds the whole frontier.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::split::SplitResult;
use crate::types::{HistogramBuffer, TreeNode};

use super::builder::{record_leaf_values, validate_build_input, BuildTreeInput};
use super::histograms::{
    build_feature_histograms, compute_child_histograms, find_best_split_across_features_internal,
    BuildHistogramConfig, ChildHistogramConfig, OrderedScratch,
};
use super::nodes::{
    compute_leaf_value, compute_sums, finalize_nodes, should_stop, split_samples, BuildNode,
};
use super::{Tree, TreeBuildConfig};

/// A leaf that is eligible to be split, together with the split it would make.
///
/// Only splittable nodes become candidates; see the module docs on blocked
/// leaves. The `split` is evaluated once, when the node is created, and stays
/// valid until the node is chosen — nothing about this node changes when some
/// other leaf elsewhere in the tree is split.
#[derive(Debug)]
struct Candidate {
    /// Index of this node's record in the builder's node vector.
    node_id: usize,
    /// Rows that reached this node.
    sample_indices: Vec<u32>,
    /// Distance from the root.
    depth: usize,
    /// Histograms this node's split was found from, reused to derive its
    /// children's by sibling subtraction.
    histograms: Vec<HistogramBuffer>,
    /// The best split available at this node.
    split: SplitResult,
}

/// Everything the per-node bookkeeping needs that does not change per node.
struct GrowthContext<'a> {
    /// The tree-building inputs.
    input: &'a BuildTreeInput<'a>,
    /// Bin count including the NaN bin.
    n_bins: usize,
}

/// Returns the index of the frontier candidate with the largest split gain.
///
/// Ties resolve to the earliest candidate, which is the earliest-created node.
/// Scanning in frontier order rather than sorting keeps the choice a pure
/// function of creation order, so a rerun on identical data picks the same
/// node — determinism the benchmark protocol depends on.
///
/// # Args
///
/// * `frontier` - Splittable candidates.
///
/// # Returns
///
/// The winning index, or `None` if the frontier is empty.
fn argmax_by_gain(frontier: &[Candidate]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (index, candidate) in frontier.iter().enumerate() {
        let gain = candidate.split.gain();
        let replace = match best {
            Some((_, best_gain)) => gain > best_gain,
            None => true,
        };
        if replace {
            best = Some((index, gain));
        }
    }
    best.map(|(index, _)| index)
}

/// Appends a provisional leaf record and returns its node id.
///
/// The record is written as a leaf because that is what the node is until it
/// is chosen for splitting. Its value is the node's own optimal leaf value,
/// which is also what an internal node carries, so promoting the record later
/// never has to recompute it.
///
/// Leaf values are recorded per sample eagerly, here. If this node is later
/// split, its two children partition its samples exactly and overwrite every
/// slot this call wrote, so the final write for any sample always comes from
/// the leaf it actually lands in.
///
/// # Args
///
/// * `context` - Shared growth inputs.
/// * `sample_indices` - Rows that reached this node.
/// * `nodes` - Node records, appended to.
/// * `child_pointers` - Child-pointer table, extended in step with `nodes`.
/// * `leaf_value_per_sample` - Per-sample leaf-value output.
///
/// # Returns
///
/// The new node's id, equal to its index in `nodes`.
fn push_leaf_record(
    context: &GrowthContext<'_>,
    sample_indices: &[u32],
    nodes: &mut Vec<BuildNode>,
    child_pointers: &mut Vec<(Option<usize>, Option<usize>)>,
    leaf_value_per_sample: &mut [f64],
) -> usize {
    let config = context.input.config;
    let node_id = nodes.len();
    let (g_sum, h_sum) = compute_sums(
        sample_indices,
        context.input.gradients,
        context.input.hessians,
    );
    let value = compute_leaf_value(g_sum, h_sum, config.reg_alpha(), config.reg_lambda());

    record_leaf_values(sample_indices, value, leaf_value_per_sample);

    nodes.push(BuildNode {
        node_id,
        is_leaf: true,
        feature_index: None,
        split_bin: None,
        value,
        n_samples: sample_indices.len(),
        nan_goes_left: true,
    });
    child_pointers.push((None, None));

    node_id
}

/// Evaluates a node for frontier candidacy, given the histograms it owns.
///
/// # Args
///
/// * `context` - Shared growth inputs.
/// * `node_id` - The node's record index.
/// * `sample_indices` - Rows that reached this node.
/// * `depth` - Distance from the root.
/// * `histograms` - This node's per-feature histograms.
///
/// # Returns
///
/// A candidate if the node may be split and a positive-gain split exists,
/// otherwise `None` — the node stays a leaf.
///
/// # Errors
///
/// Returns any error raised by the split search.
fn evaluate_candidate(
    context: &GrowthContext<'_>,
    node_id: usize,
    sample_indices: Vec<u32>,
    depth: usize,
    histograms: Vec<HistogramBuffer>,
) -> Result<Option<Candidate>, ClearGbmError> {
    let config: &TreeBuildConfig = context.input.config;
    let split_config = config.split_config();

    // `n_leaves`/`max_leaves` are passed as 0 so this asks only "may this node
    // be split at all", not "is there budget left". The budget is global and
    // belongs to the growth loop; folding it in here would retire a candidate
    // permanently on a condition that is really about the tree as a whole.
    let blocked = should_stop(
        depth,
        sample_indices.len(),
        0_usize,
        config.max_depth(),
        0_usize,
        split_config.min_samples_split(),
        split_config.min_samples_leaf(),
    );
    if blocked {
        return Ok(None);
    }

    let feature_mask = match context.input.feature_subsample {
        Some(fs) => Some(super::feature_subsample::select_split_features(
            fs,
            context.input.n_features,
            node_id,
            context.input.tree_feature_mask,
        )),
        None => context.input.tree_feature_mask.map(<[bool]>::to_vec),
    };
    let best_split = match find_best_split_across_features_internal(
        &histograms,
        split_config,
        context.input.n_regular_bins,
        context.input.monotonic_constraints,
        feature_mask.as_deref(),
    ) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    let Some(split) = best_split else {
        return Ok(None);
    };

    Ok(Some(Candidate {
        node_id,
        sample_indices,
        depth,
        histograms,
        split,
    }))
}

/// Builds a decision tree by best-first (leaf-wise) growth.
///
/// Returns the tree and the per-sample leaf-value mapping described on
/// [`super::builder::build_tree_with_leaf_assignment`], with identical
/// semantics: samples outside `sample_indices` keep the `f64::NAN` sentinel.
///
/// # Args
///
/// * `input` - Build tree input configuration. `config.max_leaves()` is the
///   leaf budget and must be at least 2.
/// * `hooks` - Dependency injection hooks for histogram building.
///
/// # Returns
///
/// The built tree and its per-sample leaf values.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` if the leaf budget is below 2,
/// plus every error [`super::builder::build_tree_with_leaf_assignment`] can
/// raise from input validation, histogram building, and split finding.
pub fn build_tree_leaf_wise_with_leaf_assignment(
    input: &BuildTreeInput<'_>,
    hooks: &Hooks,
) -> Result<(Tree, Vec<f64>), ClearGbmError> {
    match validate_build_input(input) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let config = input.config;
    let budget = config.max_leaves();
    if budget < 2_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "max_leaves".to_string(),
            reason: format!("leaf-wise growth needs a budget of at least 2, got {budget}"),
        });
    }

    let context = GrowthContext {
        input,
        n_bins: input.n_regular_bins + 1_usize,
    };

    let mut leaf_value_per_sample: Vec<f64> = vec![f64::NAN; input.n_samples];
    let mut nodes: Vec<BuildNode> = Vec::new();
    let mut child_pointers: Vec<(Option<usize>, Option<usize>)> = Vec::new();
    let mut frontier: Vec<Candidate> = Vec::new();
    let mut max_depth_found = 0_usize;

    // The root: recorded as a leaf, then evaluated for candidacy.
    let root_indices = input.sample_indices.to_vec();
    let root_id = push_leaf_record(
        &context,
        &root_indices,
        &mut nodes,
        &mut child_pointers,
        &mut leaf_value_per_sample,
    );

    // One pair of ordered-stream scratch buffers reused by every node in
    // this tree; see `OrderedScratch` for the allocation-churn rationale.
    let mut scratch = OrderedScratch::new(root_indices.len());

    let root_hist_config = BuildHistogramConfig {
        sample_indices: &root_indices,
        gradients: input.gradients,
        hessians: input.hessians,
        bins_rows: input.bins_rows,
        n_features: input.n_features,
        n_bins: context.n_bins,
        hooks,
    };
    let root_histograms = match build_feature_histograms(&root_hist_config, &mut scratch) {
        Ok(h) => h,
        Err(e) => return Err(e),
    };

    // Nodes awaiting a split search. The root and both children of every
    // split go through this one queue, so evaluation — and its error
    // handling — exists in exactly one place.
    let mut to_evaluate: Vec<(usize, Vec<u32>, usize, Vec<HistogramBuffer>)> =
        vec![(root_id, root_indices, 0_usize, root_histograms)];
    let mut n_leaves = 1_usize;

    loop {
        for (node_id, indices, depth, histograms) in core::mem::take(&mut to_evaluate) {
            match evaluate_candidate(&context, node_id, indices, depth, histograms) {
                Ok(Some(candidate)) => frontier.push(candidate),
                Ok(None) => {}
                Err(e) => return Err(e),
            }
        }

        if n_leaves >= budget {
            break;
        }
        let Some(best_index) = argmax_by_gain(&frontier) else {
            break;
        };
        // `remove` rather than `swap_remove`: the frontier stays in creation
        // order, so the tie-break in `argmax_by_gain` means the same thing on
        // every run.
        let candidate = frontier.remove(best_index);
        let split = &candidate.split;

        // Promote the chosen leaf's record to an internal node. Its value is
        // already this node's optimal value and does not change.
        nodes[candidate.node_id].is_leaf = false;
        nodes[candidate.node_id].feature_index = Some(split.feature_index());
        nodes[candidate.node_id].split_bin = Some(split.split_bin());
        nodes[candidate.node_id].nan_goes_left = split.nan_goes_left();

        let (left_indices, right_indices) = split_samples(
            &candidate.sample_indices,
            input.bins_rows,
            input.n_features,
            split.feature_index(),
            split.split_bin(),
            split.nan_goes_left(),
            input.n_regular_bins,
        );

        let child_hist_config = ChildHistogramConfig {
            left_indices: &left_indices,
            right_indices: &right_indices,
            gradients: input.gradients,
            hessians: input.hessians,
            bins_rows: input.bins_rows,
            n_features: input.n_features,
            n_bins: context.n_bins,
            parent_histograms: &candidate.histograms,
            hooks,
        };
        let (left_histograms, right_histograms) =
            match compute_child_histograms(&child_hist_config, &mut scratch) {
                Ok(h) => h,
                Err(e) => return Err(e),
            };

        let child_depth = candidate.depth + 1_usize;
        if child_depth > max_depth_found {
            max_depth_found = child_depth;
        }

        let left_id = push_leaf_record(
            &context,
            &left_indices,
            &mut nodes,
            &mut child_pointers,
            &mut leaf_value_per_sample,
        );
        let right_id = push_leaf_record(
            &context,
            &right_indices,
            &mut nodes,
            &mut child_pointers,
            &mut leaf_value_per_sample,
        );
        child_pointers[candidate.node_id] = (Some(left_id), Some(right_id));

        // One leaf became an internal node and two leaves appeared: net +1.
        n_leaves += 1_usize;

        to_evaluate.push((left_id, left_indices, child_depth, left_histograms));
        to_evaluate.push((right_id, right_indices, child_depth, right_histograms));
    }

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

/// Builds a decision tree by best-first growth, discarding the leaf mapping.
///
/// # Args
///
/// * `input` - Build tree input configuration.
/// * `hooks` - Dependency injection hooks for histogram building.
///
/// # Returns
///
/// The built tree.
///
/// # Errors
///
/// Same as [`build_tree_leaf_wise_with_leaf_assignment`].
pub fn build_tree_leaf_wise(
    input: &BuildTreeInput<'_>,
    hooks: &Hooks,
) -> Result<Tree, ClearGbmError> {
    match build_tree_leaf_wise_with_leaf_assignment(input, hooks) {
        Ok((tree, _leaf_assignment)) => Ok(tree),
        Err(e) => Err(e),
    }
}
