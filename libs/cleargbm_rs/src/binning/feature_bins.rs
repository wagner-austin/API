//! Combined binning result for training.
//!
//! Wraps bin edges and sample bin assignments into a single struct
//! that produces output compatible with `BuildTreeInput`.
//!
//! # Storage layout
//!
//! `sample_bins` is a flat, column-major `Vec<u8>`: bin `[feat_idx, sample_idx]`
//! lives at `sample_bins[feat_idx * n_samples + sample_idx]`. A per-feature
//! histogram scan walks `n_samples` contiguous bytes, so cache prefetching
//! hits and a 256-bit AVX load pulls 32 bin values instead of 4. Bin values
//! are in `0..=n_regular_bins`; the NaN bin is at `n_regular_bins` and is
//! representable in `u8` because the training config caps `max_bins ≤ 255`.

use crate::error::ClearGbmError;

use super::assignment::assign_bin;
use super::edges::{compute_bin_edges, BinEdges};

/// Converts a `usize` to `u8`, returning an `IntegerConversion` error on overflow.
///
/// Used at the storage boundary where bin indices are packed into `u8` for
/// cache density. Under the `max_bins ≤ 255` invariant enforced upstream by
/// the training config, the `try_from` call cannot fail — the Err arm is a
/// defense-in-depth guard against callers that bypass the invariant.
fn usize_to_u8(value: usize, context: &str) -> Result<u8, ClearGbmError> {
    match u8::try_from(value) {
        Ok(v) => Ok(v),
        Err(_) => Err(ClearGbmError::IntegerConversion {
            context: format!("{context}: {value} does not fit in u8"),
        }),
    }
}

/// Combined binning result: edges + sample assignments.
///
/// Produced once per training run by `precompute_feature_bins`, then
/// referenced by every boosting iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureBins {
    /// One `BinEdges` per feature.
    bin_edges: Vec<BinEdges>,

    /// Flat column-major bin index storage.
    ///
    /// `sample_bins[feat_idx * n_samples + sample_idx]`. Values in
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

    /// Returns the flat column-major bin storage.
    ///
    /// Access as `bins()[feat_idx * n_samples() + sample_idx]`. Callers that
    /// want a per-feature slice should use [`Self::bins_for_feature`].
    #[must_use]
    pub fn bins(&self) -> &[u8] {
        &self.sample_bins
    }

    /// Returns the contiguous bin slice for a single feature.
    ///
    /// Returns `&sample_bins[feat_idx * n_samples..(feat_idx + 1) * n_samples]`.
    /// The slice is empty (and the return still valid) when the feature index
    /// is out of range; callers should validate before use.
    #[must_use]
    pub fn bins_for_feature(&self, feat_idx: usize) -> &[u8] {
        if feat_idx >= self.n_features {
            return &[];
        }
        let start = feat_idx * self.n_samples;
        let end = start + self.n_samples;
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
/// A `FeatureBins` containing edges, flat column-major u8 assignments, and
/// the bin count.
///
/// # Errors
///
/// * Propagates errors from `compute_bin_edges`.
/// * Returns `ClearGbmError::InvalidParameter` if `max_bins > 255` — the u8
///   bin-index invariant is broken.
/// * Returns `ClearGbmError::IntegerConversion` if a computed bin index cannot
///   be represented in `u8` (dead branch under the `max_bins ≤ 255` guard, but
///   the check is kept as a defense in depth).
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
    let nan_bin: u8 = match usize_to_u8(nan_bin_usize, "nan_bin_index") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // Flat column-major storage: sample_bins[feat_idx * n_samples + sample_idx]
    let mut sample_bins = vec![0_u8; n_samples * n_features];

    for (feat_idx, be) in bin_edges.iter().enumerate() {
        let feat_edges = be.edges();
        let feat_col_start = feat_idx * n_samples;
        for (sample_idx, row) in x.iter().enumerate() {
            let val = row[feat_idx];
            let bin_idx_usize = if val.is_nan() {
                nan_bin_usize
            } else {
                assign_bin(val, feat_edges)
            };
            let bin_idx: u8 = if bin_idx_usize == nan_bin_usize {
                nan_bin
            } else {
                match usize_to_u8(bin_idx_usize, "bin_index") {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                }
            };
            sample_bins[feat_col_start + sample_idx] = bin_idx;
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
