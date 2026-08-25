//! Bin edge computation for feature discretization.
//!
//! Computes count-aware bin edges for each feature, converting continuous
//! values into discrete bins for histogram-based split finding.
//!
//! The algorithm works on (distinct value, count) pairs, following the
//! shipped semantics of LightGBM's `GreedyFindBin` (src/io/bin.cpp, archived
//! in the tech-wiki as `lightgbm-bin-cpp.html`). The `min_data_in_bin`
//! floor (config field 25; unset = 1) is a binning-coarseness regularizer:
//! it merges rare adjacent values until every bin holds at least the floor,
//! and caps the greedy budget at `n / floor`. With no floor:
//!
//! - When a feature has at most `max_bins` distinct values, EVERY distinct
//!   value gets its own bin (edges at midpoints between neighbours).
//! - Otherwise, bins are formed greedily to equal sample counts, with any
//!   single value heavier than the mean bin size taking a bin of its own
//!   and the remaining budget re-spread over the rest.
//!
//! This replaced a quantile-of-multiset rule (edges at quantile positions of
//! the sorted value array, deduplicated) on 2026-08-24: on zero-inflated
//! features that rule collapsed thousands of distinct values into a handful
//! of bins — weather_tmax's `hot_excess` (2,359 distinct values) got SIX
//! bins because ~95% of days have zero excess and every quantile position
//! landed on the zero — and on low-cardinality features it merged distinct
//! values even when bins were free (41 species into 36 bins at 64 budget).
//! One deliberate divergence from the shipped code: the running mean is
//! not refreshed once the rest-bin budget hits zero (the shipped code
//! divides by zero there).

use crate::error::ClearGbmError;

/// Sorted thresholds defining bin boundaries for one feature.
///
/// `K - 1` edges define `K` regular bins:
/// - Bin 0: values ≤ edges\[0\]
/// - Bin i: edges\[i-1\] < values ≤ edges\[i\]
/// - Bin K-1: values > edges\[K-2\]
///
/// Empty edges (e.g. all-NaN feature or single unique value) → 1 regular bin.
#[derive(Debug, Clone, PartialEq)]
pub struct BinEdges {
    /// Sorted, deduplicated, finite thresholds.
    edges: Vec<f64>,
}

impl BinEdges {
    /// Creates new bin edges from a vector of thresholds.
    ///
    /// # Args
    ///
    /// * `edges` - Sorted, finite thresholds. Empty is valid (1 bin).
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if edges are not sorted or contain
    /// non-finite values.
    pub fn new(edges: Vec<f64>) -> Result<Self, ClearGbmError> {
        for (i, &e) in edges.iter().enumerate() {
            if !e.is_finite() {
                return Err(ClearGbmError::InvalidParameter {
                    name: "edges".to_string(),
                    reason: format!("edge at index {i} is not finite: {e}"),
                });
            }
            if i > 0_usize {
                let prev = edges[i - 1_usize];
                if e <= prev {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "edges".to_string(),
                        reason: format!(
                            "edges not strictly sorted: edges[{}]={prev} >= edges[{i}]={e}",
                            i - 1_usize
                        ),
                    });
                }
            }
        }
        Ok(Self { edges })
    }

    /// Returns the edge thresholds.
    #[must_use]
    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    /// Returns the number of regular bins (excluding NaN bin).
    ///
    /// Equal to `edges.len() + 1`.
    #[must_use]
    pub fn n_regular_bins(&self) -> usize {
        self.edges.len() + 1_usize
    }
}

/// Converts a sample count to f64, saturating at `u32::MAX`.
///
/// Counts are bounded by the row count; datasets beyond `u32::MAX` rows
/// are outside every supported scale, so saturation is unreachable in
/// practice. The idiom matches the loss modules' count conversions.
fn count_to_f64(v: usize) -> f64 {
    f64::from(u32::try_from(v).unwrap_or(u32::MAX))
}

/// Picks the bin edge between two adjacent distinct values.
///
/// The midpoint, unless floating-point rounding lands it on `b` (possible
/// for adjacent doubles), in which case the edge sits on `a`. Either way
/// `a <= edge < b`, so under the `value <= edge` bin rule `a` always
/// routes left and `b` always routes right.
fn midpoint_edge(a: f64, b: f64) -> f64 {
    let m = a * 0.5_f64 + b * 0.5_f64;
    if m >= b {
        a
    } else {
        m
    }
}

/// Computes count-aware bin edges for each feature.
///
/// For each feature column, collects non-NaN values, folds them into
/// (distinct value, count) pairs, and picks up to `max_bins - 1` edges —
/// one bin per distinct value when they fit the budget, greedy
/// equal-count bins otherwise. See the module docs for the algorithm.
///
/// # Args
///
/// * `x` - Row-major feature matrix `[n_samples][n_features]`.
/// * `max_bins` - Maximum number of bins per feature (>= 2, ≤ u32::MAX).
/// * `min_data_in_bin` - Minimum samples per bin (>= 1; 1 = no floor).
///
/// # Returns
///
/// One `BinEdges` per feature.
///
/// # Errors
///
/// * `ClearGbmError::InvalidParameter` if `max_bins < 2`,
///   `max_bins > u32::MAX`, or `min_data_in_bin < 1`.
/// * `ClearGbmError::EmptyInput` if `x` is empty.
/// * `ClearGbmError::ShapeMismatch` if rows have inconsistent lengths.
pub fn compute_bin_edges(
    x: &[&[f64]],
    max_bins: usize,
    min_data_in_bin: usize,
) -> Result<Vec<BinEdges>, ClearGbmError> {
    if min_data_in_bin < 1_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "min_data_in_bin".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }
    if max_bins < 2_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "max_bins".to_string(),
            reason: "must be >= 2".to_string(),
        });
    }
    // Validate max_bins fits in u32 so count_to_f64 conversions stay exact.
    let _ = match u32::try_from(max_bins) {
        Ok(v) => v,
        Err(_) => {
            return Err(ClearGbmError::InvalidParameter {
                name: "max_bins".to_string(),
                reason: format!("{max_bins} exceeds maximum supported value (u32::MAX)"),
            })
        }
    };
    if x.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "x must not be empty".to_string(),
        });
    }
    let n_features = x[0].len();
    if n_features == 0_usize {
        return Err(ClearGbmError::EmptyInput {
            context: "x has zero features".to_string(),
        });
    }
    // Validate all rows have the same number of features
    for (i, row) in x.iter().enumerate() {
        if row.len() != n_features {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("all rows with {n_features} features"),
                got: format!("row {i} has {} features", row.len()),
            });
        }
    }

    let mut result = Vec::with_capacity(n_features);
    for feat_idx in 0_usize..n_features {
        result.push(compute_feature_edges(
            x,
            feat_idx,
            max_bins,
            min_data_in_bin,
        ));
    }

    Ok(result)
}

/// Computes the count-aware bin edges for one feature column.
///
/// The per-feature body of [`compute_bin_edges`], extracted so mixed
/// numeric/categorical binning can compute numeric edges only for the
/// features that need them. See the module docs for the algorithm and
/// its provenance.
///
/// # Args
///
/// * `x` - Row-major feature matrix, already shape-validated by the caller.
/// * `feat_idx` - The feature to bin.
/// * `max_bins` - Bin budget (>= 2, validated by the caller).
/// * `min_data_in_bin` - Minimum samples per bin (>= 1, validated by the
///   caller; 1 = no floor).
///
/// # Returns
///
/// The feature's `BinEdges` (empty for all-NaN or single-valued columns).
pub(super) fn compute_feature_edges(
    x: &[&[f64]],
    feat_idx: usize,
    max_bins: usize,
    min_data_in_bin: usize,
) -> BinEdges {
    // Collect non-NaN values for this feature
    let mut valid_values: Vec<f64> = Vec::new();
    for row in x {
        let val = row[feat_idx];
        if !val.is_nan() {
            valid_values.push(val);
        }
    }

    // All NaN → empty edges (1 bin)
    if valid_values.is_empty() {
        return BinEdges { edges: Vec::new() };
    }

    // Sort valid values
    valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n_valid = valid_values.len();

    // Single unique value (or a range below f64 resolution) → empty edges
    if n_valid == 1_usize
        || (valid_values[0] - valid_values[n_valid - 1_usize]).abs() < f64::EPSILON
    {
        return BinEdges { edges: Vec::new() };
    }

    // Fold the sorted values into (distinct value, count) pairs. Equal
    // values are adjacent after the sort, and NaN was already excluded,
    // so exact equality is the right test.
    let mut distinct: Vec<f64> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for &val in &valid_values {
        if distinct.last() == Some(&val) {
            let last_idx = counts.len() - 1_usize;
            counts[last_idx] += 1_usize;
        } else {
            distinct.push(val);
            counts.push(1_usize);
        }
    }

    if distinct.len() <= max_bins {
        return BinEdges {
            edges: per_value_edges(&distinct, &counts, min_data_in_bin),
        };
    }
    BinEdges {
        edges: greedy_edges(&distinct, &counts, n_valid, max_bins, min_data_in_bin),
    }
}

/// One bin per qualifying distinct value: edges at midpoints between
/// neighbours, closed only once a bin holds `min_data_in_bin` samples.
///
/// The exact-resolution case — the feature has no more distinct values
/// than the bin budget. With no floor (`min_data_in_bin = 1`) every
/// distinct value gets its own bin; a floor merges rare adjacent values
/// until each bin holds at least the floor, with the final bin taking
/// the remainder.
///
/// # Args
///
/// * `distinct` - Sorted distinct values, at least two.
/// * `counts` - Sample count per distinct value.
/// * `min_data_in_bin` - Minimum samples per bin (>= 1).
fn per_value_edges(distinct: &[f64], counts: &[usize], min_data_in_bin: usize) -> Vec<f64> {
    let mut edges_vec: Vec<f64> = Vec::with_capacity(distinct.len() - 1_usize);
    let mut cur_count = 0_usize;
    for i in 0_usize..distinct.len() - 1_usize {
        cur_count += counts[i];
        if cur_count >= min_data_in_bin {
            // Each edge lies in [distinct[i], distinct[i+1]) — disjoint,
            // ordered intervals — so edges are strictly increasing by
            // construction and need no dedup.
            edges_vec.push(midpoint_edge(distinct[i], distinct[i + 1_usize]));
            cur_count = 0_usize;
        }
    }
    edges_vec
}

/// Greedy equal-count binning over (distinct value, count) pairs.
///
/// The shipped `GreedyFindBin` walk: values heavier than the mean bin
/// size take a bin of their own up front and the mean re-spreads over
/// the rest; a bin closes when it reaches the running mean, on a heavy
/// value, or just before one once half-full.
///
/// # Args
///
/// * `distinct` - Sorted distinct values, more than `max_bins` of them.
/// * `counts` - Sample count per distinct value.
/// * `n_valid` - Total sample count.
/// * `max_bins` - Bin budget (>= 2).
/// * `min_data_in_bin` - Minimum samples per bin (>= 1); caps the
///   effective budget at `n_valid / min_data_in_bin` per the shipped
///   semantics, so no bin can be forced below the floor by the budget.
fn greedy_edges(
    distinct: &[f64],
    counts: &[usize],
    n_valid: usize,
    max_bins: usize,
    min_data_in_bin: usize,
) -> Vec<f64> {
    let n_distinct = distinct.len();
    let max_bins = max_bins.min(n_valid / min_data_in_bin).max(1_usize);
    let mut mean_bin_size = count_to_f64(n_valid) / count_to_f64(max_bins);

    // Values heavier than the mean take a bin of their own; the mean
    // re-spreads over what remains.
    let mut is_big: Vec<bool> = vec![false; n_distinct];
    let mut rest_bins = max_bins;
    let mut rest_samples = n_valid;
    for i in 0_usize..n_distinct {
        if count_to_f64(counts[i]) >= mean_bin_size {
            is_big[i] = true;
            rest_bins = rest_bins.saturating_sub(1_usize);
            rest_samples -= counts[i];
        }
    }
    // At least one rest bin always remains here: a big value holds at
    // least the mean bin size, so all bins going big would need every
    // sample — yet the greedy case has more distinct values than bins,
    // so a non-big value always exists and the division is safe.
    mean_bin_size = count_to_f64(rest_samples) / count_to_f64(rest_bins);

    // Walk the distinct values, closing bins at the running mean. Each
    // closed bin records its last value and the next bin's first value;
    // the edge lands at their midpoint.
    let mut upper: Vec<f64> = Vec::new();
    let mut lower: Vec<f64> = vec![distinct[0]];
    let mut cur_count = 0_usize;
    for i in 0_usize..n_distinct - 1_usize {
        if !is_big[i] {
            rest_samples -= counts[i];
        }
        cur_count += counts[i];
        let half_full = count_to_f64(cur_count) >= (mean_bin_size * 0.5_f64).max(1.0_f64);
        let close = is_big[i]
            || count_to_f64(cur_count) >= mean_bin_size
            || (is_big[i + 1_usize] && half_full);
        if close {
            upper.push(distinct[i]);
            lower.push(distinct[i + 1_usize]);
            if upper.len() >= max_bins - 1_usize {
                break;
            }
            cur_count = 0_usize;
            if !is_big[i] {
                rest_bins = rest_bins.saturating_sub(1_usize);
                if rest_bins > 0_usize {
                    mean_bin_size = count_to_f64(rest_samples) / count_to_f64(rest_bins);
                }
            }
        }
    }

    let mut edges_vec: Vec<f64> = Vec::with_capacity(upper.len());
    for i in 0_usize..upper.len() {
        // Each edge lies in [upper[i], lower[i+1]) and the next bin
        // starts at or above lower[i+1] — disjoint, ordered intervals —
        // so edges are strictly increasing by construction.
        edges_vec.push(midpoint_edge(upper[i], lower[i + 1_usize]));
    }
    edges_vec
}

/// Converts a non-negative integer-valued f64 to usize without `as` casts.
///
/// Decomposes the f64 into binary digits and builds the usize bit-by-bit.
/// O(32) constant time for values up to u32::MAX.
///
/// # Args
///
/// * `x` - Non-negative integer-valued f64 in `[0, u32::MAX]`.
/// * `context` - Description for error messages.
///
/// # Errors
///
/// Returns `ClearGbmError::IntegerConversion` if `x` is negative, non-integer,
/// NaN, infinite, or exceeds u32::MAX.
#[cfg(test)]
pub(crate) fn f64_to_usize_checked(x: f64, context: &str) -> Result<usize, ClearGbmError> {
    if x < 0.0_f64 || !x.is_finite() {
        return Err(ClearGbmError::IntegerConversion {
            context: format!("{context}: {x} is not a valid non-negative finite number"),
        });
    }
    if x != x.floor() {
        return Err(ClearGbmError::IntegerConversion {
            context: format!("{context}: {x} is not an integer"),
        });
    }
    if x > f64::from(u32::MAX) {
        return Err(ClearGbmError::IntegerConversion {
            context: format!("{context}: {x} exceeds u32::MAX"),
        });
    }

    // x == 0.0 handled separately (loop body requires x >= 1.0).
    if x == 0.0_f64 {
        return Ok(0_usize);
    }

    // Decompose x into binary digits and build the usize.
    // x is a non-negative integer f64 in [1.0, u32::MAX].
    // f64 represents integers ≤ 2^53 exactly; u32::MAX < 2^53, so all
    // intermediate values (halves) are exact.
    let mut remaining = x;
    let mut result = 0_usize;
    let mut place = 1_usize;

    loop {
        // Extract least-significant bit: remainder of division by 2
        let half = (remaining / 2.0_f64).floor();
        let is_odd = remaining - 2.0_f64 * half;
        if is_odd >= 0.5_f64 {
            result += place;
        }
        remaining = half;
        if remaining < 1.0_f64 {
            break;
        }
        place *= 2_usize;
    }

    Ok(result)
}
