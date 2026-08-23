//! Feature subsampling: per-tree (`colsample_bytree`) and per-split
//! (`max_features`).
//!
//! The per-tree axis restricts which columns one boosting round may use at
//! all; the per-split axis draws a fresh subset for every node's split
//! search — and when both are set, the per-node draw selects WITHIN the
//! tree's sampled set (the LightGBM composition). Histograms are still
//! built for every feature (sibling subtraction needs complete parent
//! histograms); only the split *search* is restricted, so these knobs
//! change which splits win, not how histograms are constructed.
//!
//! Determinism is stream-free: the tree mask is a pure function of
//! `(seed, round)` and each node's subset of `(seed, node_id)` via
//! dedicated [`SimpleRng`]s, so drawing subsets never advances the run RNG
//! that row subsampling reads. With both knobs unset no subset is ever
//! derived, which keeps unsampled training bit-identical to the history
//! before these axes existed.

use crate::training::rng::SimpleRng;

/// The per-tree inputs feature subsampling needs, carried on
/// [`super::builder::BuildTreeInput`].
#[derive(Debug, Clone, Copy)]
pub struct FeatureSubsample {
    /// Number of features each split may consider (>= 1, <= n_features).
    pub k: usize,

    /// Seed for the per-node subset derivation. The trainer mixes the
    /// boosting round into it, so the same node id in different trees
    /// draws different subsets.
    pub seed: u64,
}

/// Odd multiplier for mixing the node id into the subset seed
/// (splitmix64's golden-ratio constant).
const NODE_MIX: u64 = 0x9E37_79B9_7F4A_7C15_u64;

/// Odd multiplier for mixing the boosting round into the TREE-mask seed
/// (xxhash64's prime-2). Deliberately distinct from [`NODE_MIX`] so the
/// per-tree and per-node derivations can never collide into the same
/// `SimpleRng` stream.
const TREE_MIX: u64 = 0xC2B2_AE3D_27D4_EB4F_u64;

/// Derives the feature subset one node's split search may consider.
///
/// Runs a partial Fisher-Yates shuffle over `0..n_features` with a
/// [`SimpleRng`] seeded from `(subsample.seed, node_id)` and keeps the
/// first `k` positions. The modulo draw carries negligible bias at feature
/// counts and is deterministic, which is the property that matters here.
///
/// `k <= n_features` is the trainer's validated invariant
/// (`train_gradient_boosting` step 9b), not re-checked here.
///
/// # Args
///
/// * `subsample` - The subset size and derivation seed.
/// * `n_features` - Total feature count.
/// * `node_id` - The node whose split search this subset restricts.
///
/// # Returns
///
/// A mask with exactly `k` features enabled.
#[must_use]
pub(super) fn select_split_features(
    subsample: FeatureSubsample,
    n_features: usize,
    node_id: usize,
    tree_mask: Option<&[bool]>,
) -> Vec<bool> {
    // usize -> u64 is lossless on every supported target; the error arm is
    // statically dead (same rationale as `crate::narrow::index_widen`).
    let node_u64 = u64::try_from(node_id).unwrap_or(u64::MAX);
    let mut rng = SimpleRng::new(subsample.seed ^ node_u64.wrapping_mul(NODE_MIX));

    // The per-node draw selects within the tree's sampled set when a tree
    // mask is active — the per-split budget applies to the columns the tree
    // may use, capped at that set's size. Without a tree mask the candidate
    // pool is every feature, exactly the pre-colsample derivation, so the
    // colsample-off path stays bit-identical.
    let mut order: Vec<usize> = match tree_mask {
        Some(mask) => (0_usize..n_features).filter(|&f| mask[f]).collect(),
        None => (0_usize..n_features).collect(),
    };
    let pool = order.len();
    let k = subsample.k.min(pool);
    for slot in 0_usize..k {
        let remaining = pool - slot;
        // remaining <= n_features <= u32::MAX in practice; the modulo is
        // over a usize-sized window derived from the low 32 bits.
        let draw = crate::training::rng::u64_to_usize_via_u32(
            rng.next_u64() % 0x1_0000_0000_u64,
            "feature subsample draw",
        )
        .unwrap_or(0_usize)
            % remaining;
        order.swap(slot, slot + draw);
    }

    let mut mask = vec![false; n_features];
    for &feature_idx in order.iter().take(k) {
        mask[feature_idx] = true;
    }
    mask
}

/// Derives the feature mask one boosting round's tree may use.
///
/// Runs the same partial Fisher-Yates as the per-node draw, seeded from
/// `(random_state, round)` through [`TREE_MIX`], keeping `k_tree` of the
/// `n_features` columns. The trainer computes `k_tree = max(1,
/// floor(colsample_bytree * n_features))` and guarantees `1 <= k_tree <=
/// n_features`.
///
/// # Args
///
/// * `random_state` - The run seed.
/// * `round` - The boosting round (each tree draws a different mask).
/// * `k_tree` - Number of columns this tree may use.
/// * `n_features` - Total feature count.
///
/// # Returns
///
/// A mask with exactly `k_tree` features enabled.
#[must_use]
pub fn select_tree_features(
    random_state: u64,
    round: usize,
    k_tree: usize,
    n_features: usize,
) -> Vec<bool> {
    let round_u64 = u64::try_from(round).unwrap_or(u64::MAX);
    let mut rng = SimpleRng::new(random_state ^ round_u64.wrapping_mul(TREE_MIX));

    let mut order: Vec<usize> = (0_usize..n_features).collect();
    for slot in 0_usize..k_tree {
        let remaining = n_features - slot;
        let draw = crate::training::rng::u64_to_usize_via_u32(
            rng.next_u64() % 0x1_0000_0000_u64,
            "tree feature subsample draw",
        )
        .unwrap_or(0_usize)
            % remaining;
        order.swap(slot, slot + draw);
    }

    let mut mask = vec![false; n_features];
    for &feature_idx in order.iter().take(k_tree) {
        mask[feature_idx] = true;
    }
    mask
}

/// Resolves the per-tree column budget from the `colsample_bytree` fraction:
/// `k_tree = max(1, floor(fraction * n_features))`, the same convention row
/// subsampling uses for its sample count.
///
/// # Args
///
/// * `fraction` - The `colsample_bytree` value, validated by the config
///   layer to lie in `(0.0, 1.0)` exclusive.
/// * `n_features` - Total feature count.
///
/// # Returns
///
/// The number of columns each tree may use, on `[1, n_features]`.
///
/// # Errors
///
/// Returns [`ClearGbmError::IntegerConversion`] if `n_features` exceeds
/// `u32::MAX` (the same ceiling row subsampling imposes on `n_samples`).
pub fn tree_column_budget(
    fraction: f64,
    n_features: usize,
) -> Result<usize, crate::error::ClearGbmError> {
    let n_features_u32 = match u32::try_from(n_features) {
        Ok(v) => v,
        Err(_) => {
            return Err(crate::error::ClearGbmError::IntegerConversion {
                context: format!("n_features = {n_features} exceeds u32::MAX"),
            })
        }
    };
    let k_f64 = (f64::from(n_features_u32) * fraction).floor();
    let k = match crate::training::subsampling::f64_to_usize_checked(k_f64, "tree column budget") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok(k.max(1_usize))
}
