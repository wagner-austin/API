//! Per-split feature subsampling for `max_features`.
//!
//! Each node considers only a random subset of features when searching for
//! its best split — the classic random-forest-style regularizer, applied
//! per split as the config documents. Histograms are still built for every
//! feature (sibling subtraction needs complete parent histograms); only the
//! split *search* is restricted, so the knob changes which splits win, not
//! how histograms are constructed.
//!
//! Determinism is stream-free: each node's subset is a pure function of
//! `(seed, node_id)` via a per-node [`SimpleRng`], so drawing subsets never
//! advances the run RNG that row subsampling reads. With `max_features`
//! unset no subset is ever derived, which keeps unsubsampled training
//! bit-identical to the history before this axis existed.

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
) -> Vec<bool> {
    // usize -> u64 is lossless on every supported target; the error arm is
    // statically dead (same rationale as `crate::narrow::index_widen`).
    let node_u64 = u64::try_from(node_id).unwrap_or(u64::MAX);
    let mut rng = SimpleRng::new(subsample.seed ^ node_u64.wrapping_mul(NODE_MIX));

    let mut order: Vec<usize> = (0_usize..n_features).collect();
    let k = subsample.k;
    for slot in 0_usize..k {
        let remaining = n_features - slot;
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
