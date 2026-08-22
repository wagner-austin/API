//! Combined binning result for training.
//!
//! Wraps bin edges and sample bin assignments into a single struct
//! that produces output compatible with `BuildTreeInput`.
//!
//! # Storage layout
//!
//! `sample_bins` is a flat, row-major `Vec<u8>`: bin `[sample_idx, feat_idx]`
//! lives at `sample_bins[sample_idx * n_features + feat_idx]`. The
//! single-pass node build reads one sample's bins for every feature as one
//! contiguous `n_features`-byte row, so a node walk touches each sample's
//! row once instead of visiting `n_features` separate columns (measured
//! 2026-08-21: fusing the walks cut ClearGBM fit time ~13%). The
//! column-major layout this replaced was itself measured in against a
//! jagged `Vec<Vec>` on 2026-07-21 — see the wiki page
//! `cleargbm-perf-column-major-sample-bins` for that history. Bin values
//! are in `0..=n_regular_bins`; the NaN bin is at `n_regular_bins` and is
//! representable in `u8` because the training config caps `max_bins ≤ 255`.

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

    /// Flat row-major bin index storage.
    ///
    /// `sample_bins[sample_idx * n_features + feat_idx]`. Values in
    /// `0..=n_regular_bins` (NaN bin at `n_regular_bins`).
    sample_bins: Vec<u8>,

    /// Number of samples (row count in the original feature matrix).
    n_samples: usize,

    /// Number of features (column count in the original feature matrix).
    n_features: usize,

    /// Uniform regular bin count (= `max_bins`).
    n_regular_bins: usize,
}

impl FeatureBins {
    /// Returns the bin edges per feature.
    #[must_use]
    pub fn bin_edges(&self) -> &[BinEdges] {
        &self.bin_edges
    }

    /// Returns the flat row-major bin storage.
    ///
    /// Access as `bins()[sample_idx * n_features() + feat_idx]`. Callers that
    /// want one sample's row should use [`Self::bins_for_sample`].
    #[must_use]
    pub fn bins(&self) -> &[u8] {
        &self.sample_bins
    }

    /// Returns the contiguous bin row for a single sample.
    ///
    /// Returns `&sample_bins[sample_idx * n_features..][..n_features]`.
    /// The slice is empty (and the return still valid) when the sample index
    /// is out of range; callers should validate before use.
    #[must_use]
    pub fn bins_for_sample(&self, sample_idx: usize) -> &[u8] {
        if sample_idx >= self.n_samples {
            return &[];
        }
        let start = sample_idx * self.n_features;
        let end = start + self.n_features;
        &self.sample_bins[start..end]
    }

    /// Returns the sample count (row count of the original feature matrix).
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Returns the feature count (column count of the original feature matrix).
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
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
/// quantile-based bin edges, then assigns each sample to its bin. The output
/// is a flat column-major `Vec<u8>` — see the module-level docs for the
/// layout rationale.
///
/// # Args
///
/// * `x` - Row-major feature matrix `[n_samples][n_features]`.
/// * `max_bins` - Maximum number of bins per feature (`2 ≤ max_bins ≤ 255`).
///   The upper bound is enforced by
///   [`GradientBoostingConfig::new`](crate::training::GradientBoostingConfig::new)
///   before this function is called; this function relies on the invariant to
///   pack bin indices into `u8`.
///
/// # Returns
///
/// A `FeatureBins` containing edges, flat row-major u8 assignments, and
/// the bin count.
///
/// # Errors
///
/// * Propagates errors from `compute_bin_edges`.
/// * Returns `ClearGbmError::InvalidParameter` if `max_bins > 255` — the u8
///   bin-index invariant is broken.
pub fn precompute_feature_bins(
    x: &[&[f64]],
    max_bins: usize,
) -> Result<FeatureBins, ClearGbmError> {
    if max_bins > 255_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "max_bins".to_string(),
            reason: format!("must be <= 255 (u8 bin index), got {max_bins}"),
        });
    }

    let bin_edges = match compute_bin_edges(x, max_bins) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };

    let n_samples = x.len();
    let n_features = bin_edges.len();
    let nan_bin_usize = max_bins;

    // Flat row-major storage: sample_bins[sample_idx * n_features + feat_idx].
    // Filled rows-outer, which also matches the input matrix's own row-major
    // shape: each input row is read once, in order.
    let mut sample_bins = vec![0_u8; n_samples * n_features];

    for (sample_idx, row) in x.iter().enumerate() {
        let row_start = sample_idx * n_features;
        for (feat_idx, be) in bin_edges.iter().enumerate() {
            let val = row[feat_idx];
            let bin_idx_usize = if val.is_nan() {
                nan_bin_usize
            } else {
                assign_bin(val, be.edges())
            };
            // `max_bins <= 255` is rejected at the top of this function, and
            // the NaN bin sits at `max_bins` — the largest index either branch
            // above can produce — so every value here is already in u8 range.
            // Written as a saturating conversion rather than a fallible one
            // because the error arm would be statically dead: the guard has
            // already run, so there is no input that reaches it.
            let bin_idx: u8 = u8::try_from(bin_idx_usize).unwrap_or(u8::MAX);
            sample_bins[row_start + feat_idx] = bin_idx;
        }
    }

    Ok(FeatureBins {
        bin_edges,
        sample_bins,
        n_samples,
        n_features,
        n_regular_bins: max_bins,
    })
}
