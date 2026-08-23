//! Tests for the feature-subsampling mask derivations.
//!
//! Covers the per-tree mask (`select_tree_features`, the `colsample_bytree`
//! axis) and the per-node draw's composition with an active tree mask —
//! the pool restriction and the budget cap that make the two axes nest
//! (the LightGBM semantics).

use super::super::feature_subsample::{select_split_features, FeatureSubsample};
use crate::error::ClearGbmError;
use crate::tree::{select_tree_features, tree_column_budget};

// =============================================================================
// select_tree_features (per-tree mask)
// =============================================================================

#[test]
fn test_tree_mask_enables_exactly_k_features() -> Result<(), ClearGbmError> {
    for k_tree in 1_usize..=10_usize {
        let mask = select_tree_features(42_u64, 0_usize, k_tree, 10_usize);
        assert_eq!(mask.len(), 10_usize);
        let enabled = mask.iter().filter(|&&b| b).count();
        assert_eq!(
            enabled, k_tree,
            "k_tree={k_tree} enabled {enabled} features"
        );
    }
    Ok(())
}

#[test]
fn test_tree_mask_is_a_pure_function_of_seed_and_round() -> Result<(), ClearGbmError> {
    let first = select_tree_features(42_u64, 7_usize, 3_usize, 10_usize);
    let second = select_tree_features(42_u64, 7_usize, 3_usize, 10_usize);
    assert_eq!(first, second);
    Ok(())
}

#[test]
fn test_tree_mask_varies_across_rounds() -> Result<(), ClearGbmError> {
    // Each boosting round draws its own mask; a derivation that ignored the
    // round would give every tree the same columns, silently weakening the
    // regularizer to a one-time projection.
    let masks: Vec<Vec<bool>> = (0_usize..8_usize)
        .map(|round| select_tree_features(42_u64, round, 3_usize, 10_usize))
        .collect();
    let distinct = masks
        .iter()
        .any(|m| masks.first().is_some_and(|head| m != head));
    assert!(distinct, "eight rounds all drew the identical mask");
    Ok(())
}

#[test]
fn test_tree_mask_varies_across_seeds() -> Result<(), ClearGbmError> {
    let a = select_tree_features(42_u64, 0_usize, 4_usize, 12_usize);
    let b = select_tree_features(43_u64, 0_usize, 4_usize, 12_usize);
    assert_ne!(a, b, "different seeds drew the identical mask");
    Ok(())
}

#[test]
fn test_tree_mask_full_budget_enables_everything() -> Result<(), ClearGbmError> {
    // k_tree = n_features is reachable when floor(f * n) == n - 1 rounds up
    // through max(1, ..) on tiny feature counts; the mask must degrade to
    // all-true rather than dropping a column.
    let mask = select_tree_features(42_u64, 3_usize, 5_usize, 5_usize);
    assert!(mask.iter().all(|&b| b));
    Ok(())
}

// =============================================================================
// select_split_features composition with a tree mask
// =============================================================================

#[test]
fn test_split_draw_selects_within_the_tree_mask() -> Result<(), ClearGbmError> {
    // With both axes active, every feature the node draws must come from the
    // tree's sampled set — the composition, not an independent draw.
    let tree_mask = select_tree_features(42_u64, 0_usize, 4_usize, 10_usize);
    let subsample = FeatureSubsample {
        k: 2_usize,
        seed: 42_u64,
    };
    for node_id in 0_usize..16_usize {
        let node_mask = select_split_features(subsample, 10_usize, node_id, Some(&tree_mask));
        assert_eq!(node_mask.iter().filter(|&&b| b).count(), 2_usize);
        for feature in 0_usize..10_usize {
            assert!(
                !node_mask[feature] || tree_mask[feature],
                "node {node_id} drew feature {feature} outside the tree mask"
            );
        }
    }
    Ok(())
}

#[test]
fn test_split_budget_caps_at_the_tree_pool() -> Result<(), ClearGbmError> {
    // max_features larger than the tree's sampled set cannot manufacture
    // columns: the draw keeps the whole pool and nothing more.
    let tree_mask = select_tree_features(42_u64, 0_usize, 3_usize, 10_usize);
    let subsample = FeatureSubsample {
        k: 8_usize,
        seed: 42_u64,
    };
    let node_mask = select_split_features(subsample, 10_usize, 0_usize, Some(&tree_mask));
    assert_eq!(node_mask, tree_mask);
    Ok(())
}

#[test]
fn test_split_draw_without_tree_mask_uses_every_feature_as_pool() -> Result<(), ClearGbmError> {
    // The colsample-off path: the candidate pool is all features, exactly
    // the pre-colsample derivation, so unrestricted training stays
    // bit-identical to the history before the axis existed.
    let subsample = FeatureSubsample {
        k: 3_usize,
        seed: 42_u64,
    };
    let mask = select_split_features(subsample, 10_usize, 5_usize, None);
    assert_eq!(mask.len(), 10_usize);
    assert_eq!(mask.iter().filter(|&&b| b).count(), 3_usize);
    Ok(())
}

// =============================================================================
// tree_column_budget
// =============================================================================

#[test]
fn test_tree_column_budget_floors_the_product() -> Result<(), crate::error::ClearGbmError> {
    let five = match tree_column_budget(0.5_f64, 10_usize) {
        Ok(k) => k,
        Err(e) => return Err(e),
    };
    assert_eq!(five, 5_usize);
    let seven = match tree_column_budget(0.79_f64, 10_usize) {
        Ok(k) => k,
        Err(e) => return Err(e),
    };
    assert_eq!(seven, 7_usize);
    Ok(())
}

#[test]
fn test_tree_column_budget_never_drops_below_one() -> Result<(), crate::error::ClearGbmError> {
    // floor(0.05 * 10) = 0; the budget clamps to one column so a tiny
    // fraction thins the tree rather than making it unbuildable.
    let k = match tree_column_budget(0.05_f64, 10_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(k, 1_usize);
    Ok(())
}

#[test]
fn test_tree_column_budget_rejects_counts_beyond_u32() -> Result<(), crate::error::ClearGbmError> {
    // The same ceiling row subsampling imposes on n_samples: the mask math
    // runs in u32-derived f64 space, so a wider count must error, not wrap.
    match tree_column_budget(0.5_f64, usize::MAX) {
        Ok(_) => Err(crate::error::ClearGbmError::TreeConstructionFailed {
            reason: "a feature count beyond u32::MAX must be rejected".to_string(),
        }),
        Err(crate::error::ClearGbmError::IntegerConversion { context }) => {
            assert!(context.contains("u32::MAX"), "got: {context}");
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_tree_column_budget_propagates_a_bad_product() -> Result<(), crate::error::ClearGbmError> {
    // The config layer validates the fraction, but this function's own
    // contract still propagates the checked-conversion failure a malformed
    // product would produce rather than assuming its caller.
    match tree_column_budget(f64::NAN, 10_usize) {
        Ok(_) => Err(crate::error::ClearGbmError::TreeConstructionFailed {
            reason: "a NaN product must be rejected".to_string(),
        }),
        Err(crate::error::ClearGbmError::IntegerConversion { context }) => {
            assert!(context.contains("tree column budget"), "got: {context}");
            Ok(())
        }
        Err(other) => Err(other),
    }
}
