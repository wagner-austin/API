//! Many-vs-many categorical split search.
//!
//! Sorts a node's categories by their gradient-to-hessian ratio (the
//! ordering Fisher 1958 proves sufficient for an optimal binary partition
//! under a convex loss) and prefix-scans it, so the best subset split is
//! found in O(K log K) rather than over 2^K subsets. The winning subset
//! is carried as a [`CategoryBinSet`] over histogram bins.

use crate::error::ClearGbmError;
use crate::types::{HistogramBuffer, SplitConfig};

use super::{
    compute_split_gain, NanDirection, SplitDecision, SplitResult, SplitResultConfig, EPSILON,
};

/// A set of histogram bins routed to the left child by a categorical split.
///
/// A 256-bit mask indexed by bin, matching the `u8` bin-index invariant
/// (`max_bins <= 255` plus the NaN bin). The NaN bin is never a member —
/// missing values route by the split's [`NanDirection`], exactly as for
/// threshold splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CategoryBinSet {
    /// Four 64-bit words covering bins 0..256; bin `b` lives at
    /// `words[b / 64]`, bit `b % 64`.
    words: [u64; 4],
}

impl CategoryBinSet {
    /// Creates an empty set.
    #[must_use]
    pub const fn new() -> Self {
        Self { words: [0_u64; 4] }
    }

    /// Inserts a bin into the set.
    ///
    /// Bins at or above 256 are ignored — unrepresentable by construction,
    /// since the config layer caps `max_bins <= 255`.
    pub fn insert(&mut self, bin: usize) {
        if bin < 256_usize {
            self.words[bin / 64_usize] |= 1_u64 << (bin % 64_usize);
        }
    }

    /// Returns whether the set contains a bin.
    #[must_use]
    pub const fn contains(&self, bin: usize) -> bool {
        if bin >= 256_usize {
            return false;
        }
        (self.words[bin / 64_usize] >> (bin % 64_usize)) & 1_u64 == 1_u64
    }

    /// Returns the bins in the set, ascending.
    #[must_use]
    pub fn bins(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for bin in 0_usize..256_usize {
            if self.contains(bin) {
                out.push(bin);
            }
        }
        out
    }

    /// Returns the number of bins in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        let mut total = 0_u32;
        for word in self.words {
            total += word.count_ones();
        }
        crate::narrow::index_widen(total)
    }

    /// Returns whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0_usize
    }
}

/// Internal record for one non-empty category bin during the sorted scan.
#[derive(Debug, Clone, Copy)]
struct CategoryStat {
    /// The histogram bin holding this category.
    bin: usize,
    /// Gradient sum over the category's samples at this node.
    g: f64,
    /// Hessian sum over the category's samples at this node.
    h: f64,
    /// Sample count for the category at this node.
    n: usize,
}

/// Finds the best many-vs-many categorical split for one feature.
///
/// The LightGBM mechanism: sort the node's non-empty categories by their
/// gradient-to-hessian ratio (the per-category optimal-value ordering that
/// Fisher 1958 proves sufficient for an optimal binary partition under a
/// convex loss), then prefix-scan the sorted order — every "first k sorted
/// categories go left" subset is a candidate, and the best subset overall
/// is among them. Missing values are tried on both sides exactly as in the
/// threshold scan; categories absent from this node route right at
/// prediction time by not being members of the left set.
///
/// The sort key floors near-zero hessians at [`EPSILON`] rather than
/// applying LightGBM's `cat_smooth` prior — no smoothing constant exists in
/// the config, so none is silently applied. Ties order by bin index, which
/// keeps the scan a pure function of the histogram.
///
/// # Args
///
/// * `histogram` - Histogram for this feature (NaN bin last, when present).
/// * `feature_index` - Index of this feature.
/// * `config` - Split configuration (min_samples_leaf, reg_lambda, min_gain).
/// * `n_categories` - Number of category bins this feature uses; bins at or
///   beyond this count belong to other features' wider ranges and are
///   guaranteed empty for this one.
/// * `n_regular_bins` - Number of regular bins (the NaN bin sits here).
///
/// # Returns
///
/// * `Ok(Some(SplitResult))` - Best subset split meeting all constraints.
/// * `Ok(None)` - No valid split found.
///
/// # Errors
///
/// Returns [`ClearGbmError::InvalidParameter`] if `n_categories` or
/// `n_regular_bins` exceeds the histogram's bin count.
pub fn find_best_categorical_split_from_histogram(
    histogram: &HistogramBuffer,
    feature_index: usize,
    config: &SplitConfig,
    n_categories: usize,
    n_regular_bins: usize,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let n_bins = histogram.n_bins();
    if n_regular_bins > n_bins {
        return Err(ClearGbmError::InvalidParameter {
            name: "n_regular_bins".to_string(),
            reason: format!(
                "n_regular_bins ({n_regular_bins}) cannot exceed histogram n_bins ({n_bins})"
            ),
        });
    }
    if n_categories > n_regular_bins {
        return Err(ClearGbmError::InvalidParameter {
            name: "n_categories".to_string(),
            reason: format!(
                "n_categories ({n_categories}) cannot exceed n_regular_bins ({n_regular_bins})"
            ),
        });
    }

    // NaN statistics, same layout as the threshold scan.
    let has_nan_bin = n_bins > n_regular_bins;
    let (g_nan, h_nan, n_nan) = if has_nan_bin {
        let acc = histogram.bins[n_regular_bins];
        (acc.gradient_sum, acc.hessian_sum, acc.count)
    } else {
        (0.0_f64, 0.0_f64, 0_usize)
    };

    // Collect the node's non-empty categories and the totals.
    let mut stats: Vec<CategoryStat> = Vec::with_capacity(n_categories);
    let mut g_regular = 0.0_f64;
    let mut h_regular = 0.0_f64;
    let mut n_regular = 0_usize;
    for (bin, acc) in histogram.bins.iter().enumerate().take(n_categories) {
        g_regular += acc.gradient_sum;
        h_regular += acc.hessian_sum;
        n_regular += acc.count;
        if acc.count > 0_usize {
            stats.push(CategoryStat {
                bin,
                g: acc.gradient_sum,
                h: acc.hessian_sum,
                n: acc.count,
            });
        }
    }

    let g_total = g_regular + g_nan;
    let h_total = h_regular + h_nan;
    let n_total = n_regular + n_nan;

    let min_samples_leaf = config.min_samples_leaf();
    if n_total < 2_usize * min_samples_leaf || stats.len() < 2_usize {
        return Ok(None);
    }

    // The Fisher order: ascending per-category optimal value (-g/h reversed
    // is g/h; either direction works since every prefix of one ordering is a
    // suffix-complement of the other). Ties break by bin index so the scan
    // is deterministic.
    stats.sort_by(|a, b| {
        let h_a = if a.h.abs() < EPSILON { EPSILON } else { a.h };
        let h_b = if b.h.abs() < EPSILON { EPSILON } else { b.h };
        let key_a = a.g / h_a;
        let key_b = b.g / h_b;
        match key_a.total_cmp(&key_b) {
            core::cmp::Ordering::Equal => a.bin.cmp(&b.bin),
            other => other,
        }
    });

    let reg_lambda = config.reg_lambda();
    let min_gain = config.min_gain();

    // Prefix scan over the sorted categories: the first k go left.
    let mut best: Option<(usize, f64, f64, f64, usize, NanDirection)> = None;
    let mut g_left_base = 0.0_f64;
    let mut h_left_base = 0.0_f64;
    let mut n_left_base = 0_usize;

    for (prefix_len, stat) in stats
        .iter()
        .enumerate()
        .take(stats.len().saturating_sub(1_usize))
    {
        g_left_base += stat.g;
        h_left_base += stat.h;
        n_left_base += stat.n;

        for nan_dir in [NanDirection::Left, NanDirection::Right] {
            let (g_left, h_left, n_left) = if nan_dir.goes_left() {
                (
                    g_left_base + g_nan,
                    h_left_base + h_nan,
                    n_left_base + n_nan,
                )
            } else {
                (g_left_base, h_left_base, n_left_base)
            };

            let n_right = n_total.saturating_sub(n_left);
            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }

            let g_right = g_total - g_left;
            let h_right = h_total - h_left;

            let gain = compute_split_gain(
                g_left, h_left, g_right, h_right, g_total, h_total, reg_lambda,
            );
            if gain <= min_gain {
                continue;
            }

            let dominated = match best {
                Some((_, best_gain, _, _, _, _)) => gain <= best_gain,
                None => false,
            };
            if !dominated {
                best = Some((prefix_len + 1_usize, gain, g_left, h_left, n_left, nan_dir));
            }
        }
    }

    let Some((k, gain, g_left, h_left, n_left, nan_direction)) = best else {
        return Ok(None);
    };

    let mut left_bins = CategoryBinSet::new();
    for stat in stats.iter().take(k) {
        left_bins.insert(stat.bin);
    }

    let g_right = g_total - g_left;
    let h_right = h_total - h_left;
    let n_right = n_total.saturating_sub(n_left);

    Ok(Some(SplitResult::new(SplitResultConfig {
        feature_index,
        decision: SplitDecision::CategorySubset { left_bins },
        gain,
        left_gradient_sum: g_left,
        left_hessian_sum: h_left,
        left_count: n_left,
        right_gradient_sum: g_right,
        right_hessian_sum: h_right,
        right_count: n_right,
        nan_direction,
    })))
}
