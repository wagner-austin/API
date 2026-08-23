//! Tests for best-first (leaf-wise) tree construction.
//!
//! The load-bearing test here is
//! `test_leaf_wise_matches_depth_wise_when_the_budget_never_binds`: with an
//! unreachable leaf budget both policies must exhaust exactly the same set of
//! splittable nodes, so their predictions must agree bit for bit. Growth order
//! changes which node is split *first*, never which nodes are splittable, and
//! that test is what pins the distinction.

use super::leafwise_helpers::{grow_depth_wise, grow_leaf_wise, make_config, predict_fixture};

use crate::error::ClearGbmError;
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
