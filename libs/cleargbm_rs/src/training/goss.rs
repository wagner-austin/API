//! Gradient-based one-side sampling (GOSS) for the single-score loop.
//!
//! The formulation is Ke et al. 2017's Algorithm 2 as LightGBM ships it
//! (`goss.hpp` @ 3ec5b99b, pinned in the tech-wiki), including the shipped
//! code's two divergences from the paper: rows rank by |gradient x
//! hessian| rather than |gradient|, and the caller skips sampling entirely
//! during the first `1 / learning_rate` rounds. Rows at or above the
//! top-k threshold are kept outright; the rest are sampled streaming with
//! an adaptive probability (what the quota still needs over what remains),
//! and each sampled row's gradient AND hessian are multiplied by
//! `(cnt - top_k) / other_k` in place — the unbiasing constant applied to
//! both derivative streams.
//!
//! One stated divergence of our own: `other_k` floors at 1. LightGBM's
//! expression divides by `cnt * other_rate` truncated toward zero, which
//! on a small-enough dataset is a division by zero; a floor of one row is
//! the honest guard, and it changes nothing whenever
//! `cnt * other_rate >= 1`.

use crate::error::ClearGbmError;
use crate::training::rng::SimpleRng;

/// The validated GOSS rates (both set, each in (0, 1), sum at most 1).
#[derive(Debug, Clone, Copy)]
pub(super) struct GossRates {
    /// Fraction of rows kept outright, by |gradient x hessian| rank.
    pub top_rate: f64,
    /// Fraction of the remaining rows sampled and reweighted.
    pub other_rate: f64,
}

/// Converts a count bounded by the row count to `f64`.
///
/// Row counts are bounded by the crate's u32 index ceiling (the
/// subsampler enforces it on every non-GOSS round, and a dataset past
/// 4 billion rows is out of scope for a single-node histogram GBM), so
/// the saturating arm is statically dead — the crate's dead-arm idiom.
fn count_to_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

/// Runs one round's GOSS pass: selects the training rows and reweights
/// the sampled low-gradient rows' gradients and hessians in place.
///
/// # Args
///
/// * `gradients` - This round's gradients; sampled rows are scaled.
/// * `hessians` - This round's hessians, same length; sampled rows are
///   scaled.
/// * `rates` - The validated GOSS rates.
/// * `rng` - The run's row-sampling RNG (GOSS is its only consumer when
///   active, because GOSS excludes `subsample < 1`).
///
/// # Returns
///
/// The kept row indices, ascending — the top rows plus the sampled rest.
///
/// # Errors
///
/// Returns `ClearGbmError::IntegerConversion` if a computed count fails
/// the float-to-index conversion (bounded inputs make this unreachable
/// in practice).
pub(super) fn goss_sample_indices(
    gradients: &mut [f64],
    hessians: &mut [f64],
    rates: GossRates,
    rng: &mut SimpleRng,
) -> Result<Vec<u32>, ClearGbmError> {
    let cnt = gradients.len();
    let cnt_f64 = count_to_f64(cnt);

    // Importance per row: |gradient x hessian| (the shipped ranking key).
    let scores: Vec<f64> = gradients
        .iter()
        .zip(hessians.iter())
        .map(|(&g, &h)| (g * h).abs())
        .collect();

    let top_k_raw = propagate!(super::subsampling::f64_to_usize_checked(
        (cnt_f64 * rates.top_rate).floor(),
        "GOSS top count"
    ));
    let top_k = top_k_raw.max(1_usize);
    let other_k_raw = propagate!(super::subsampling::f64_to_usize_checked(
        (cnt_f64 * rates.other_rate).floor(),
        "GOSS other count"
    ));
    let other_k = other_k_raw.max(1_usize);

    // The top-k threshold from a partial selection (values only; the
    // selection's internal order does not affect the result).
    let mut sorted = scores.clone();
    let pivot = top_k - 1_usize;
    sorted.select_nth_unstable_by(pivot, |a, b| {
        // Scores are finite by construction; the Equal default is
        // statically dead (the crate's dead-arm idiom).
        b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal)
    });
    let threshold = sorted[pivot];

    let top_k_f64 = count_to_f64(top_k);
    let other_k_f64 = count_to_f64(other_k);
    let multiply = (cnt_f64 - top_k_f64) / other_k_f64;

    let mut kept: Vec<u32> = Vec::with_capacity(top_k + other_k);
    let mut big_weight_cnt: usize = 0_usize;
    for (i, &score) in scores.iter().enumerate() {
        // Indices are bounded by the row count, so the saturating arm is
        // statically dead (the crate's dead-arm idiom).
        let i_u32 = u32::try_from(i).unwrap_or(u32::MAX);
        if score >= threshold {
            kept.push(i_u32);
            big_weight_cnt += 1_usize;
        } else {
            // Adaptive draw: what the quota still needs over what remains
            // below the threshold. `rest_all` counts the remaining
            // below-threshold rows INCLUDING this one, so it is always at
            // least 1 here: at least top_k rows sit at or above the
            // threshold by construction, and ties only push the kept
            // count higher, which makes the subtraction smaller. A
            // negative `rest_need` (quota already met) yields a negative
            // probability and the draw never fires — the shipped
            // semantics, without its signed-counter detour.
            let sampled = count_to_f64(kept.len() - big_weight_cnt);
            let rest_need = other_k_f64 - sampled;
            let rest_all = (cnt_f64 - count_to_f64(i)) - (top_k_f64 - count_to_f64(big_weight_cnt));
            let prob = rest_need / rest_all;
            if rng.next_f64() < prob {
                kept.push(i_u32);
                gradients[i] *= multiply;
                hessians[i] *= multiply;
            }
        }
    }
    Ok(kept)
}
