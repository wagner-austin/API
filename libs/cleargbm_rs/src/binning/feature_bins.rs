//! Combined binning result for training.
//!
//! Wraps bin edges and sample bin assignments into a single struct
//! that produces output compatible with `BuildTreeInput`.

use crate::error::ClearGbmError;

use super::assignment::assign_bin;
use super::edges::{compute_bin_edges, BinEdges};

/// Combined binning result: edges + sample assignments.
///
/// Produced once per training run by `precompute_feature_bins`, then
/// referenced by every boosting iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureBins {
    /// One `BinEdges` per feature.
    bin_edges: Vec<BinEdges>,

    /// Bin index per (sample, feature). Shape: `[n_samples][n_features]`.
    /// Values in `0..=n_regular_bins` (NaN bin at `n_regular_bins`).
    sample_bins: Vec<Vec<usize>>,

    /// Uniform regular bin count (= `max_bins`).
    n_regular_bins: usize,
}

impl FeatureBins {
    /// Returns the bin edges per feature.
    #[must_use]
    pub fn bin_edges(&self) -> &[BinEdges] {
        &self.bin_edges
    }

    /// Returns sample bin assignments `[n_samples][n_features]`.
    #[must_use]
    pub fn sample_bins(&self) -> &[Vec<usize>] {
        &self.sample_bins
    }

    /// Returns the uniform regular bin count.
    #[must_use]
    pub fn n_regular_bins(&self) -> usize {
        self.n_regular_bins
    }

    /// Converts bin edges to the `BuildTreeInput.bin_thresholds` format.
    ///
    /// Returns `[n_features][n_regular_bins]` where each inner Vec contains
    /// the actual edge thresholds padded with `f64::INFINITY` to length
    /// `n_regular_bins`.
    ///
    /// The tree builder reads `bin_thresholds[feature][split_bin]` to get
    /// the split threshold. Unused bin slots (from features with fewer
    /// unique values) get `f64::INFINITY` — these bins have zero histogram
    /// counts and are never selected as split points.
    #[must_use]
    pub fn bin_thresholds(&self) -> Vec<Vec<f64>> {
        let mut thresholds = Vec::with_capacity(self.bin_edges.len());
        for be in &self.bin_edges {
            let edges = be.edges();
            let mut feat_thresholds = Vec::with_capacity(self.n_regular_bins);
            // First edges.len() entries are the actual thresholds
            for &e in edges {
                feat_thresholds.push(e);
            }
            // Pad remaining with INFINITY (last regular bin + unused bins)
            while feat_thresholds.len() < self.n_regular_bins {
                feat_thresholds.push(f64::INFINITY);
            }
            thresholds.push(feat_thresholds);
        }
        thresholds
    }
}

/// Precomputes bin edges and sample bin assignments for all features.
///
/// This is the single entry point called once per training run. It computes
/// quantile-based bin edges, then assigns each sample to its bin.
///
/// # Args
///
/// * `x` - Row-major feature matrix `[n_samples][n_features]`.
/// * `max_bins` - Maximum number of bins per feature (>= 2).
///
/// # Returns
///
/// A `FeatureBins` containing edges, sample assignments, and the bin count.
///
/// # Errors
///
/// Propagates errors from `compute_bin_edges`.
pub fn precompute_feature_bins(
    x: &[&[f64]],
    max_bins: usize,
) -> Result<FeatureBins, ClearGbmError> {
    let bin_edges = match compute_bin_edges(x, max_bins) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };

    // Assign bins directly using assign_bin (validation already done by compute_bin_edges).
    let nan_bin = max_bins;
    let mut sample_bins = Vec::with_capacity(x.len());
    for row in x {
        let mut row_bins = Vec::with_capacity(bin_edges.len());
        for (feat_idx, be) in bin_edges.iter().enumerate() {
            let val = row[feat_idx];
            if val.is_nan() {
                row_bins.push(nan_bin);
            } else {
                row_bins.push(assign_bin(val, be.edges()));
            }
        }
        sample_bins.push(row_bins);
    }

    Ok(FeatureBins {
        bin_edges,
        sample_bins,
        n_regular_bins: max_bins,
    })
}
