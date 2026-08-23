//! Bin edge computation for feature discretization.
//!
//! Computes quantile-based bin edges for each feature, converting continuous
//! values into discrete bins for histogram-based split finding.

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

/// Computes `floor(a * b / c)` without usize overflow, given `a < c`.
///
/// Uses the identity `a*b/c = a*(b/c) + a*(b%c)/c` when `a*b` would overflow.
/// The fallback requires `a < c` and `c² ≤ usize::MAX` (guaranteed when
/// `max_bins ≤ u32::MAX` on 64-bit targets).
///
/// # Args
///
/// * `a` - Numerator factor, must be < `c`.
/// * `b` - Numerator factor.
/// * `c` - Denominator, must be > 0.
pub(crate) fn quantile_position(a: usize, b: usize, c: usize) -> usize {
    match a.checked_mul(b) {
        Some(product) => product / c,
        None => {
            // a * b overflows usize. Split using: a*b/c = a*(b/c) + a*(b%c)/c.
            // Safe because a < c guarantees:
            //   a*(b/c) < c*(b/c) ≤ b (fits in usize)
            //   a*(b%c) < c*c = c² (fits when c ≤ u32::MAX on 64-bit)
            a * (b / c) + a * (b % c) / c
        }
    }
}

/// Computes quantile-based bin edges for each feature.
///
/// For each feature column, collects non-NaN values, sorts them, and picks
/// up to `max_bins - 1` quantile thresholds. Duplicate edges are removed.
///
/// # Args
///
/// * `x` - Row-major feature matrix `[n_samples][n_features]`.
/// * `max_bins` - Maximum number of bins per feature (>= 2, ≤ u32::MAX).
///
/// # Returns
///
/// One `BinEdges` per feature.
///
/// # Errors
///
/// * `ClearGbmError::InvalidParameter` if `max_bins < 2` or `max_bins > u32::MAX`.
/// * `ClearGbmError::EmptyInput` if `x` is empty.
/// * `ClearGbmError::ShapeMismatch` if rows have inconsistent lengths.
pub fn compute_bin_edges(x: &[&[f64]], max_bins: usize) -> Result<Vec<BinEdges>, ClearGbmError> {
    if max_bins < 2_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "max_bins".to_string(),
            reason: "must be >= 2".to_string(),
        });
    }
    // Validate max_bins fits in u32 to guarantee overflow safety in quantile_position.
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
        result.push(compute_feature_edges(x, feat_idx, max_bins));
    }

    Ok(result)
}

/// Computes the quantile bin edges for one feature column.
///
/// The per-feature body of [`compute_bin_edges`], extracted so mixed
/// numeric/categorical binning can compute numeric edges only for the
/// features that need them. Byte-identical to the pre-extraction loop.
///
/// # Args
///
/// * `x` - Row-major feature matrix, already shape-validated by the caller.
/// * `feat_idx` - The feature to bin.
/// * `max_bins` - Bin budget (>= 2, validated by the caller).
///
/// # Returns
///
/// The feature's `BinEdges` (empty for all-NaN or single-valued columns).
pub(super) fn compute_feature_edges(x: &[&[f64]], feat_idx: usize, max_bins: usize) -> BinEdges {
    let n_edges = max_bins - 1_usize;

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

    // Single unique value → empty edges
    if n_valid == 1_usize
        || (valid_values[0] - valid_values[n_valid - 1_usize]).abs() < f64::EPSILON
    {
        return BinEdges { edges: Vec::new() };
    }

    // Compute quantile edges using integer arithmetic.
    // pos = floor(edge_idx / max_bins * (n_valid - 1))
    // Equivalent to: edge_idx * (n_valid - 1) / max_bins (integer division).
    let n_valid_minus_one = n_valid - 1_usize;
    let mut edges_vec: Vec<f64> = Vec::new();

    for edge_idx in 1_usize..=n_edges {
        let pos = quantile_position(edge_idx, n_valid_minus_one, max_bins);
        let edge_value = valid_values[pos];

        // Deduplicate: only add if strictly greater than last edge
        let should_add = match edges_vec.last() {
            Some(&last) => edge_value > last,
            None => true,
        };
        if should_add {
            edges_vec.push(edge_value);
        }
    }

    BinEdges { edges: edges_vec }
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
