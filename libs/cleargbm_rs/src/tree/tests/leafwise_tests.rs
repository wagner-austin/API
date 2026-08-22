//! Tests for best-first (leaf-wise) tree construction.
//!
//! The load-bearing test here is
//! `test_leaf_wise_matches_depth_wise_when_the_budget_never_binds`: with an
//! unreachable leaf budget both policies must exhaust exactly the same set of
//! splittable nodes, so their predictions must agree bit for bit. Growth order
//! changes which node is split *first*, never which nodes are splittable, and
//! that test is what pins the distinction.

use crate::error::ClearGbmError;
use crate::histogram::NodeHistogramRequest;
use crate::hooks::Hooks;
use crate::predict::predict_tree;
use crate::tree::{
    build_tree, build_tree_leaf_wise, build_tree_leaf_wise_with_leaf_assignment, BuildTreeInput,
    TreeBuildConfig,
};
use crate::types::{HistogramBuffer, SplitConfig};

/// Sample count of the shared fixture's root node.
const ROOT_SAMPLES: usize = 8_usize;

/// Eight samples over two features, both informative but at different
/// strengths, so the frontier holds several candidates with distinct gains and
/// the growth order is observable rather than forced.
fn fixture_bins() -> Vec<u8> {
    // Column-major: feature 0 then feature 1, 8 samples each.
    vec![
        // feature 0: a clean 4/4 break
        0_u8, 0_u8, 1_u8, 1_u8, 2_u8, 2_u8, 3_u8, 3_u8, //
        // feature 1: a different partition
        0_u8, 1_u8, 2_u8, 3_u8, 0_u8, 1_u8, 2_u8, 3_u8,
    ]
}

/// Row-major features matching [`fixture_bins`], for prediction.
fn fixture_rows() -> Vec<Vec<f64>> {
    vec![
        vec![0.1_f64, 0.1_f64],
        vec![0.1_f64, 0.4_f64],
        vec![0.4_f64, 0.6_f64],
        vec![0.4_f64, 0.9_f64],
        vec![0.6_f64, 0.1_f64],
        vec![0.6_f64, 0.4_f64],
        vec![0.9_f64, 0.6_f64],
        vec![0.9_f64, 0.9_f64],
    ]
}

fn fixture_gradients() -> Vec<f64> {
    vec![
        1.0_f64, 0.9_f64, 0.6_f64, 0.4_f64, -0.4_f64, -0.6_f64, -0.9_f64, -1.0_f64,
    ]
}

fn fixture_hessians() -> Vec<f64> {
    vec![1.0_f64; 8_usize]
}

fn fixture_indices() -> Vec<u32> {
    vec![0_u32, 1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32]
}

fn fixture_thresholds() -> Vec<Vec<f64>> {
    vec![
        vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64],
        vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64],
    ]
}

/// Builds a config with the given depth bound and leaf budget.
fn make_config(max_depth: usize, max_leaves: usize) -> Result<TreeBuildConfig, ClearGbmError> {
    let split_config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    TreeBuildConfig::new(max_depth, max_leaves, 0.0_f64, 0.0_f64, split_config)
}

/// Runs the leaf-wise builder over the shared fixture.
fn grow_leaf_wise(
    config: &TreeBuildConfig,
) -> Result<(crate::tree::Tree, Vec<f64>), ClearGbmError> {
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 8_usize, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 8_usize,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config,
        monotonic_constraints: None,
    };

    build_tree_leaf_wise_with_leaf_assignment(&input, &Hooks::default())
}

/// Runs the depth-wise builder over the same fixture.
fn grow_depth_wise(config: &TreeBuildConfig) -> Result<crate::tree::Tree, ClearGbmError> {
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 8_usize, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 8_usize,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config,
        monotonic_constraints: None,
    };

    build_tree(&input, &Hooks::default())
}

/// Predicts every fixture row through a tree.
fn predict_fixture(tree: &crate::tree::Tree) -> Result<Vec<f64>, ClearGbmError> {
    let rows = fixture_rows();
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    predict_tree(tree, &row_slices)
}

// =========================================================================
// Budget
// =========================================================================

#[test]
fn test_leaf_wise_respects_the_leaf_budget() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let (tree, _assignment) = match grow_leaf_wise(&config) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    assert_eq!(tree.n_leaves(), 3_usize);

    // The recorded leaf count must match the nodes actually built, not just
    // the counter: an off-by-one there would silently mislabel every manifest.
    let leaf_nodes = tree.nodes().iter().filter(|n| n.is_leaf()).count();
    assert_eq!(leaf_nodes, 3_usize);
    Ok(())
}

#[test]
fn test_leaf_wise_budget_below_two_is_rejected() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 1_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    match grow_leaf_wise(&config) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a budget of 1 must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "max_leaves");
            assert!(
                reason.contains("at least 2"),
                "rejection should name the minimum, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_leaf_wise_stops_when_no_candidate_remains() -> Result<(), ClearGbmError> {
    // Depth 1 caps the tree at a single split, so growth runs out of
    // candidates long before the budget of 64 is touched.
    let config = match make_config(1_usize, 64_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let (tree, _assignment) = match grow_leaf_wise(&config) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    assert_eq!(tree.n_leaves(), 2_usize);
    assert_eq!(tree.max_depth(), 1_usize);
    Ok(())
}

// =========================================================================
// Growth order
// =========================================================================

#[test]
fn test_growing_the_budget_only_refines_the_previous_tree() -> Result<(), ClearGbmError> {
    // Best-first growth is monotone in the budget: raising it adds splits and
    // never revises earlier ones, because a node's best split does not depend
    // on how many other leaves the tree is allowed. Every sample's prediction
    // may change as its own leaf is refined, but the partition can only get
    // finer — two samples separated at budget k stay separated at budget k+1.
    let mut previous_partition: Vec<Vec<bool>> = Vec::new();

    for budget in 2_usize..6_usize {
        let config = match make_config(10_usize, budget) {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let (_tree, assignment) = match grow_leaf_wise(&config) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };

        // Same-leaf relation over the eight samples.
        let mut partition: Vec<Vec<bool>> = Vec::new();
        for left in 0_usize..8_usize {
            let mut row: Vec<bool> = Vec::new();
            for right in 0_usize..8_usize {
                row.push((assignment[left] - assignment[right]).abs() < 1e-12_f64);
            }
            partition.push(row);
        }

        if !previous_partition.is_empty() {
            for left in 0_usize..8_usize {
                for right in 0_usize..8_usize {
                    if !previous_partition[left][right] {
                        assert!(
                            !partition[left][right],
                            "samples {left} and {right} were split apart at a smaller \
                             budget and must not be rejoined at budget {budget}"
                        );
                    }
                }
            }
        }
        previous_partition = partition;
    }
    Ok(())
}

// =========================================================================
// Equivalence with depth-wise
// =========================================================================

#[test]
fn test_leaf_wise_matches_depth_wise_when_the_budget_never_binds() -> Result<(), ClearGbmError> {
    // With a budget no tree of this depth can reach, both policies split every
    // splittable node. Growth order decides which node is taken first, never
    // which nodes are takeable, so the two trees must predict identically.
    // This is the property that makes leaf-wise an ordering change rather than
    // a different learner.
    let leaf_wise_config = match make_config(4_usize, 1024_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let depth_wise_config = match make_config(4_usize, 0_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let (leaf_tree, _assignment) = match grow_leaf_wise(&leaf_wise_config) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    let depth_tree = match grow_depth_wise(&depth_wise_config) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    assert_eq!(leaf_tree.n_leaves(), depth_tree.n_leaves());
    assert_eq!(leaf_tree.max_depth(), depth_tree.max_depth());

    let leaf_preds = match predict_fixture(&leaf_tree) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let depth_preds = match predict_fixture(&depth_tree) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(
        leaf_preds, depth_preds,
        "exhausted leaf-wise and depth-wise trees must predict identically"
    );
    Ok(())
}

// =========================================================================
// Structure
// =========================================================================

#[test]
fn test_leaf_wise_node_ids_index_their_own_slots() -> Result<(), ClearGbmError> {
    // The predictor follows `left_child`/`right_child` as indices into the
    // node vector. Best-first assigns ids in split order rather than traversal
    // order, so this invariant is the one most at risk from the reordering.
    let config = match make_config(10_usize, 5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let (tree, _assignment) = match grow_leaf_wise(&config) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    for (index, node) in tree.nodes().iter().enumerate() {
        assert_eq!(node.node_id(), index);
    }
    Ok(())
}

#[test]
fn test_leaf_wise_assignment_matches_walking_the_tree() -> Result<(), ClearGbmError> {
    // The per-sample leaf values are written eagerly as nodes are created and
    // overwritten by each node's children. If that overwrite order were ever
    // wrong, a sample would keep an ancestor's value; walking the finished
    // tree is the independent check.
    let config = match make_config(10_usize, 5_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let (tree, assignment) = match grow_leaf_wise(&config) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    let walked = match predict_fixture(&tree) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    assert_eq!(assignment, walked);
    Ok(())
}

#[test]
fn test_leaf_wise_respects_max_depth() -> Result<(), ClearGbmError> {
    let config = match make_config(2_usize, 64_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let (tree, _assignment) = match grow_leaf_wise(&config) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    assert!(
        tree.max_depth() <= 2_usize,
        "depth bound must hold under best-first growth, got {}",
        tree.max_depth()
    );
    // A depth-2 binary tree cannot exceed four leaves however large the budget.
    assert!(
        tree.n_leaves() <= 4_usize,
        "depth 2 admits at most 4 leaves, got {}",
        tree.n_leaves()
    );
    Ok(())
}

// =========================================================================
// Shared validation + the discarding wrapper
// =========================================================================

#[test]
fn test_leaf_wise_rejects_empty_sample_indices() -> Result<(), ClearGbmError> {
    let config = match make_config(3_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let sample_indices: Vec<u32> = Vec::new();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 8_usize, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 8_usize,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &config,
        monotonic_constraints: None,
    };

    match build_tree_leaf_wise(&input, &Hooks::default()) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "empty sample_indices must be rejected".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert_eq!(context, "sample_indices");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

// =========================================================================
// Error propagation
// =========================================================================

/// Runs the leaf-wise builder over the fixture with injected hooks.
fn grow_with_hooks(
    config: &TreeBuildConfig,
    hooks: &Hooks,
) -> Result<(crate::tree::Tree, Vec<f64>), ClearGbmError> {
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, ROOT_SAMPLES, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: ROOT_SAMPLES,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config,
        monotonic_constraints: None,
    };

    build_tree_leaf_wise_with_leaf_assignment(&input, hooks)
}

/// Histogram builder that always fails.
fn error_histogram(_: NodeHistogramRequest<'_>) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    Err(ClearGbmError::EmptyInput {
        context: "injected error from hook".to_string(),
    })
}

/// Histogram builder that fails only below the root.
///
/// Keyed on the request's sample count rather than a call counter: the tree
/// suite runs its tests concurrently, so a shared counter would make which
/// call fails depend on scheduling. The root is the only node holding all
/// eight fixture samples, so this reliably lets root construction succeed and
/// fails the first child.
fn error_histogram_below_root(
    request: NodeHistogramRequest<'_>,
) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    if request.sample_indices.len() == ROOT_SAMPLES {
        return Ok(crate::histogram::build_node_histograms_single_pass(request));
    }
    Err(ClearGbmError::EmptyInput {
        context: "injected child histogram failure".to_string(),
    })
}

/// Histogram builder that always returns a too-small buffer.
fn undersized_histogram(
    _: NodeHistogramRequest<'_>,
) -> Result<Vec<HistogramBuffer>, ClearGbmError> {
    Ok(vec![HistogramBuffer::new(2_usize)])
}

#[test]
fn test_leaf_wise_propagates_a_root_histogram_failure() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let hooks = Hooks::with_histogram_builder(error_histogram);
    match grow_with_hooks(&config, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a root histogram failure must propagate".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert_eq!(context, "injected error from hook");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_leaf_wise_propagates_a_root_split_search_failure() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let hooks = Hooks::with_histogram_builder(undersized_histogram);
    match grow_with_hooks(&config, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a root split-search failure must propagate".to_string(),
        }),
        Err(_) => Ok(()),
    }
}

#[test]
fn test_leaf_wise_propagates_a_child_histogram_failure() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let hooks = Hooks::with_histogram_builder(error_histogram_below_root);
    match grow_with_hooks(&config, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a child histogram failure must propagate".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { context }) => {
            assert_eq!(context, "injected child histogram failure");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_leaf_wise_propagates_a_finalize_failure() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let hooks = Hooks {
        finalize_nodes_error: Some(ClearGbmError::TreeConstructionFailed {
            reason: "injected finalize failure".to_string(),
        }),
        ..Hooks::default()
    };
    match grow_with_hooks(&config, &hooks) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a finalize failure must propagate".to_string(),
        }),
        Err(ClearGbmError::TreeConstructionFailed { reason }) => {
            assert_eq!(reason, "injected finalize failure");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_leaf_wise_handles_an_unsplittable_root() -> Result<(), ClearGbmError> {
    // Every sample in one bin: no split has positive gain, so the root never
    // reaches the frontier and growth ends before consuming any budget.
    let config = match make_config(10_usize, 4_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins: Vec<u8> = vec![0_u8; 16_usize];
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, ROOT_SAMPLES, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: ROOT_SAMPLES,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &config,
        monotonic_constraints: None,
    };

    let (tree, _assignment) =
        match build_tree_leaf_wise_with_leaf_assignment(&input, &Hooks::default()) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
    assert_eq!(tree.n_leaves(), 1_usize);
    assert_eq!(tree.n_nodes(), 1_usize);
    assert_eq!(tree.max_depth(), 0_usize);
    Ok(())
}

#[test]
fn test_build_tree_leaf_wise_discards_the_assignment() -> Result<(), ClearGbmError> {
    let config = match make_config(10_usize, 3_usize) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 8_usize, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 8_usize,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &config,
        monotonic_constraints: None,
    };

    let tree = match build_tree_leaf_wise(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    assert_eq!(tree.n_leaves(), 3_usize);
    Ok(())
}
