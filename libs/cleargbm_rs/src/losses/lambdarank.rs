//! LambdaMART ranking loss: pair lambdas, DCG tables, and the NDCG metric.
//!
//! The formulation is Burges 2010 (MSR-TR-2010-82) as implemented by
//! LightGBM's `LambdarankNDCG` (`rank_objective.hpp` @ 3ec5b99b) and
//! `DCGCalculator` (`dcg_calculator.cpp` @ 3ec5b99b), with three deliberate,
//! documented divergences:
//!
//! * the pair sigmoid is evaluated EXACTLY instead of through LightGBM's
//!   million-entry lookup table — the table is a speed cache with
//!   quantization error, not semantics;
//! * the sigmoid shape parameter is fixed at 1.0 (LightGBM's default) with
//!   no config knob — adding one later is a stated knob, never a hidden
//!   default;
//! * lambda normalization (LightGBM's `lambdarank_norm`, default on) is
//!   always applied — the score-distance division and the
//!   `log2(1 + sum) / sum` row rescale are part of this objective's one
//!   stated behavior.
//!
//! Everything else is parity: `2^label - 1` gains capped at label 31,
//! `1 / log2(rank + 2)` discounts, the counting-sort max DCG, the
//! truncation-bounded pair loop that skips equal labels, and per-row weights
//! multiplying both lambda and hessian after the query computation.

use crate::error::ClearGbmError;
use crate::narrow::index_widen;

/// Exclusive upper bound on relevance labels: gains are `2^label - 1`, and
/// past label 31 the gain table would overflow useful range (LightGBM's
/// `max_label = 31` cap, same reasoning).
pub(crate) const MAX_RANKING_LABEL: u32 = 31_u32;

/// Maximum documents in one query, bounding the position-discount table
/// (LightGBM's `kMaxPosition`).
pub(crate) const MAX_QUERY_LENGTH: usize = 10_000_usize;

/// Builds the label-gain table: `gains[l] = 2^l - 1` for labels 0..=31.
///
/// # Returns
///
/// A 32-entry table; `gains[0] == 0.0`.
#[must_use]
pub(crate) fn label_gains() -> Vec<f64> {
    let mut gains = Vec::with_capacity(32_usize);
    let mut power = 1.0_f64;
    for _ in 0_u32..32_u32 {
        gains.push(power - 1.0_f64);
        power *= 2.0_f64;
    }
    gains
}

/// Returns the position discount `1 / log2(rank + 2)` for a 0-based rank.
///
/// # Args
///
/// * `rank` - 0-based position, `< MAX_QUERY_LENGTH` (validated at the
///   query-group boundary).
#[must_use]
pub(crate) fn position_discount(rank: usize) -> f64 {
    // Bounded by MAX_QUERY_LENGTH, so the u32 conversion's error arm is
    // statically dead (the crate's dead-arm idiom).
    let r = u32::try_from(rank).unwrap_or(u32::MAX);
    1.0_f64 / f64::from(r + 2_u32).log2()
}

/// Validates ranking relevance labels: every label must be `< 32`.
///
/// # Args
///
/// * `y` - Relevance labels.
/// * `name` - Argument name, used in the error message.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` naming the first offending
/// index and value.
pub(crate) fn validate_ranking_labels(y: &[u32], name: &str) -> Result<(), ClearGbmError> {
    for (i, &label) in y.iter().enumerate() {
        if label > MAX_RANKING_LABEL {
            return Err(ClearGbmError::InvalidParameter {
                name: name.to_string(),
                reason: format!(
                    "relevance labels must be <= {MAX_RANKING_LABEL} (gain = 2^label - 1), \
                     got {label} at index {i}"
                ),
            });
        }
    }
    Ok(())
}

/// Validates query group sizes against the row count they partition.
///
/// # Args
///
/// * `groups` - Documents per query, in row order.
/// * `n_rows` - Total rows the groups must partition exactly.
/// * `name` - Argument name, used in error messages.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` if the group list is empty, a
/// group is empty or longer than [`MAX_QUERY_LENGTH`], or the sizes do not
/// sum to `n_rows`.
pub(crate) fn validate_query_groups(
    groups: &[usize],
    n_rows: usize,
    name: &str,
) -> Result<(), ClearGbmError> {
    if groups.is_empty() {
        return Err(ClearGbmError::InvalidParameter {
            name: name.to_string(),
            reason: "must be non-empty: ranking requires query groups".to_string(),
        });
    }
    let mut total: usize = 0_usize;
    for (i, &cnt) in groups.iter().enumerate() {
        if cnt == 0_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: name.to_string(),
                reason: format!("group at index {i} is empty; every query needs >= 1 document"),
            });
        }
        if cnt > MAX_QUERY_LENGTH {
            return Err(ClearGbmError::InvalidParameter {
                name: name.to_string(),
                reason: format!(
                    "group at index {i} has {cnt} documents, over the {MAX_QUERY_LENGTH} cap"
                ),
            });
        }
        total = total.saturating_add(cnt);
    }
    if total != n_rows {
        return Err(ClearGbmError::InvalidParameter {
            name: name.to_string(),
            reason: format!("group sizes sum to {total} but there are {n_rows} rows"),
        });
    }
    Ok(())
}

/// Computes `1 / maxDCG@k` for one query by counting sort over labels.
///
/// Never sorts documents: gains are monotone in the label, so the ideal
/// ordering is by label descending, and a per-label count suffices
/// (LightGBM's `CalMaxDCGAtK`).
///
/// # Args
///
/// * `labels` - The query's relevance labels (each `<= MAX_RANKING_LABEL`).
/// * `k` - Truncation position.
/// * `gains` - The label-gain table from [`label_gains`].
///
/// # Returns
///
/// `1 / maxDCG@k`, or `0.0` when the query's max DCG is zero (every label
/// 0) — which zeroes every pair delta, so a query with nothing to rank
/// contributes nothing.
#[must_use]
pub(crate) fn inverse_max_dcg_at_k(labels: &[u32], k: usize, gains: &[f64]) -> f64 {
    let mut counts = [0_usize; 32];
    for &label in labels {
        counts[index_widen(label)] += 1_usize;
    }
    let mut max_dcg = 0.0_f64;
    let mut position = 0_usize;
    for label in (0_usize..32_usize).rev() {
        let mut remaining = counts[label];
        while remaining > 0_usize && position < k && position < labels.len() {
            max_dcg += gains[label] * position_discount(position);
            position += 1_usize;
            remaining -= 1_usize;
        }
    }
    if max_dcg > 0.0_f64 {
        1.0_f64 / max_dcg
    } else {
        0.0_f64
    }
}

/// Returns the query's document indices stable-sorted by score descending.
///
/// Stable so ties keep row order — the same determinism LightGBM's
/// `std::stable_sort` provides.
fn sorted_by_score_desc(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0_usize..scores.len()).collect();
    // Scores are sums of finite leaf values from finite inputs; NaN cannot
    // reach here, so the Equal default is statically dead (the crate's
    // dead-arm idiom).
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    idx
}

/// Fills one query's lambdas and hessians from the truncation-bounded pair
/// scan (LightGBM's `GetGradientsForOneQuery`, sigma = 1, norm always on).
///
/// For each counted pair the more relevant document is `high`; the pair
/// lambda starts as `1 / (1 + e^(s_high - s_low))`, the pair hessian as
/// `p (1 - p)`; both scale by the pair's |delta NDCG| (gain gap x discount
/// gap x `inverse_max_dcg`, divided by `0.01 + |delta score|` when scores
/// are non-degenerate); lambdas accumulate with opposite signs into the two
/// rows and the whole query rescales by `log2(1 + sum) / sum`.
///
/// # Args
///
/// * `scores` - The query's raw scores.
/// * `labels` - The query's relevance labels, same length.
/// * `inverse_max_dcg` - From [`inverse_max_dcg_at_k`].
/// * `truncation_level` - The configured cutoff bounding the outer loop.
/// * `gains` - The label-gain table.
/// * `grad_out` - Output lambda slice (the boosting loop's gradients),
///   same length; overwritten.
/// * `hess_out` - Output hessian slice, same length; overwritten.
pub(crate) fn fill_query_lambdas(
    scores: &[f64],
    labels: &[u32],
    inverse_max_dcg: f64,
    truncation_level: usize,
    gains: &[f64],
    grad_out: &mut [f64],
    hess_out: &mut [f64],
) {
    let cnt = scores.len();
    for i in 0_usize..cnt {
        grad_out[i] = 0.0_f64;
        hess_out[i] = 0.0_f64;
    }
    if cnt < 2_usize {
        return;
    }
    let sorted_idx = sorted_by_score_desc(scores);
    let best_score = scores[sorted_idx[0]];
    let worst_score = scores[sorted_idx[cnt - 1_usize]];
    let mut sum_lambdas = 0.0_f64;

    let mut i = 0_usize;
    while i + 1_usize < cnt && i < truncation_level {
        for j in (i + 1_usize)..cnt {
            if labels[sorted_idx[i]] == labels[sorted_idx[j]] {
                continue;
            }
            let (high_rank, low_rank) = if labels[sorted_idx[i]] > labels[sorted_idx[j]] {
                (i, j)
            } else {
                (j, i)
            };
            let high = sorted_idx[high_rank];
            let low = sorted_idx[low_rank];
            let delta_score = scores[high] - scores[low];

            let dcg_gap = gains[index_widen(labels[high])] - gains[index_widen(labels[low])];
            let paired_discount =
                (position_discount(high_rank) - position_discount(low_rank)).abs();
            let mut delta_pair_ndcg = dcg_gap * paired_discount * inverse_max_dcg;
            // The score-distance regularization, applied whenever the
            // query's scores are non-degenerate.
            if best_score != worst_score {
                delta_pair_ndcg /= 0.01_f64 + delta_score.abs();
            }
            let p = 1.0_f64 / (1.0_f64 + delta_score.exp());
            let mut p_lambda = p;
            let mut p_hessian = p * (1.0_f64 - p);
            p_lambda *= -delta_pair_ndcg;
            p_hessian *= delta_pair_ndcg;
            grad_out[low] -= p_lambda;
            hess_out[low] += p_hessian;
            grad_out[high] += p_lambda;
            hess_out[high] += p_hessian;
            // p_lambda is negative, so subtracting accumulates a positive
            // total.
            sum_lambdas -= 2.0_f64 * p_lambda;
        }
        i += 1_usize;
    }

    if sum_lambdas > 0.0_f64 {
        let norm_factor = (1.0_f64 + sum_lambdas).log2() / sum_lambdas;
        for r in 0_usize..cnt {
            grad_out[r] *= norm_factor;
            hess_out[r] *= norm_factor;
        }
    }
}

/// Computes NDCG@k for one query.
///
/// # Args
///
/// * `scores` - The query's raw scores.
/// * `labels` - The query's relevance labels, same length.
/// * `k` - Truncation position.
/// * `gains` - The label-gain table.
///
/// # Returns
///
/// DCG@k of the score ordering divided by the ideal DCG@k, or `1.0` when
/// the ideal DCG is zero — a query whose every label is 0 has no ranking to
/// get wrong.
#[must_use]
pub(crate) fn ndcg_at_k(scores: &[f64], labels: &[u32], k: usize, gains: &[f64]) -> f64 {
    let inverse_max = inverse_max_dcg_at_k(labels, k, gains);
    if inverse_max == 0.0_f64 {
        return 1.0_f64;
    }
    let sorted_idx = sorted_by_score_desc(scores);
    let mut dcg = 0.0_f64;
    for (position, &doc) in sorted_idx.iter().enumerate().take(k) {
        dcg += gains[index_widen(labels[doc])] * position_discount(position);
    }
    dcg * inverse_max
}

/// Computes the mean NDCG@k over every query of a grouped dataset.
///
/// # Args
///
/// * `scores` - Raw scores for all rows.
/// * `labels` - Relevance labels for all rows, same length.
/// * `groups` - Documents per query, partitioning the rows exactly
///   (validated at the training boundary).
/// * `k` - Truncation position.
/// * `gains` - The label-gain table.
///
/// # Returns
///
/// The unweighted mean of the per-query NDCG@k values.
#[must_use]
pub(crate) fn mean_ndcg_at_k(
    scores: &[f64],
    labels: &[u32],
    groups: &[usize],
    k: usize,
    gains: &[f64],
) -> f64 {
    let mut total = 0.0_f64;
    let mut start = 0_usize;
    for &cnt in groups {
        let end = start + cnt;
        total += ndcg_at_k(&scores[start..end], &labels[start..end], k, gains);
        start = end;
    }
    // Group lists are validated non-empty and bounded by MAX_QUERY_LENGTH
    // per query, so the count fits u32 (the crate's dead-arm idiom).
    let n_queries = u32::try_from(groups.len()).unwrap_or(u32::MAX);
    total / f64::from(n_queries)
}
