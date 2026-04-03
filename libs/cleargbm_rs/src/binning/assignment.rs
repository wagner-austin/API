//! Sample bin assignment for feature discretization.
//!
//! Assigns each sample to a bin index per feature using binary search
//! on precomputed bin edges. NaN values go to the dedicated NaN bin.

use crate::error::ClearGbmError;

use super::edges::BinEdges;

/// Assigns each sample to a bin index per feature.
///
/// For each (sample, feature) pair, performs binary search on the feature's
/// bin edges to find the correct bin index. NaN values are assigned to the
/// NaN bin at index `n_regular_bins`.
///
/// # Args
///
/// * `x` - Row-major feature matrix `[n_samples][n_features]`.
/// * `bin_edges` - One `BinEdges` per feature (from `compute_bin_edges`).
/// * `n_regular_bins` - Uniform bin count (= `max_bins`). NaN bin is at this index.
///
/// # Returns
///
/// Bin assignments `[n_samples][n_features]`, values in `0..=n_regular_bins`.
/// Compatible with `BuildTreeInput.bins`.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` if `x` is empty.
/// * `ClearGbmError::ShapeMismatch` if `bin_edges.len() != n_features` or rows differ in length.
pub fn bin_samples(
    x: &[&[f64]],
    bin_edges: &[BinEdges],
    n_regular_bins: usize,
) -> Result<Vec<Vec<usize>>, ClearGbmError> {
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
    if bin_edges.len() != n_features {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("bin_edges length {n_features}"),
            got: format!("bin_edges length {}", bin_edges.len()),
        });
    }
    // Validate consistent row lengths
    for (i, row) in x.iter().enumerate() {
        if row.len() != n_features {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("all rows with {n_features} features"),
                got: format!("row {i} has {} features", row.len()),
            });
        }
    }

    let nan_bin = n_regular_bins;
    let mut result = Vec::with_capacity(x.len());

    for row in x {
        let mut sample_bins = Vec::with_capacity(n_features);
        for (feat_idx, &val) in row.iter().enumerate() {
            if val.is_nan() {
                sample_bins.push(nan_bin);
            } else {
                let bin = assign_bin(val, bin_edges[feat_idx].edges());
                sample_bins.push(bin);
            }
        }
        result.push(sample_bins);
    }

    Ok(result)
}

/// Assigns a single non-NaN value to a bin using binary search on edges.
///
/// - value ≤ edges\[0\] → bin 0
/// - edges\[i-1\] < value ≤ edges\[i\] → bin i
/// - value > edges\[last\] → bin edges.len()
pub(super) fn assign_bin(value: f64, edges: &[f64]) -> usize {
    if edges.is_empty() {
        return 0_usize;
    }

    // Binary search: find first edge where value <= edge
    let mut lo = 0_usize;
    let mut hi = edges.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2_usize;
        if value <= edges[mid] {
            hi = mid;
        } else {
            lo = mid + 1_usize;
        }
    }
    lo
}
