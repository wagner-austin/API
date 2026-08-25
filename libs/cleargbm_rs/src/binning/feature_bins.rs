//! Combined binning result for training.
//!
//! Wraps per-feature binning (quantile edges for numeric features, category
//! maps for categorical ones) and sample bin assignments into a single
//! struct that produces output compatible with `BuildTreeInput`.
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
use super::categorical::{categorical_column_bins, CategoryMap};
use super::edges::{compute_feature_edges, BinEdges};

/// How one feature is binned.
///
/// An enum rather than parallel optional tables so a feature can never
/// carry both quantile edges and a category map, and a consumer can never
/// read the wrong one silently.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureBinning {
    /// Ordered quantile bins over a numeric feature.
    Numeric(BinEdges),
    /// One bin per distinct category code.
    Categorical(CategoryMap),
}

/// Combined binning result: per-feature binning + sample assignments.
///
/// Produced once per training run by `precompute_feature_bins`, then
/// referenced by every boosting iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureBins {
    /// One binning per feature.
    per_feature: Vec<FeatureBinning>,

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
    /// Returns the per-feature binning.
    #[must_use]
    pub fn per_feature(&self) -> &[FeatureBinning] {
        &self.per_feature
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

    /// Converts per-feature binning to the `BuildTreeInput.bin_thresholds` format.
    ///
    /// Returns `[n_features][n_regular_bins]` where each inner Vec contains
    /// the actual edge thresholds padded with `f64::INFINITY` to length
    /// `n_regular_bins`.
    ///
    /// The tree builder reads `bin_thresholds[feature][split_bin]` to get a
    /// THRESHOLD split's boundary. Unused bin slots (from features with
    /// fewer unique values) get `f64::INFINITY` — these bins have zero
    /// histogram counts and are never selected as split points. A
    /// categorical feature's row is entirely `f64::INFINITY`: no threshold
    /// split can ever name it, because the split search partitions its bins
    /// by set membership and node finalization reads the category map, not
    /// this table.
    #[must_use]
    pub fn bin_thresholds(&self) -> Vec<Vec<f64>> {
        let mut thresholds = Vec::with_capacity(self.per_feature.len());
        for binning in &self.per_feature {
            let mut feat_thresholds = Vec::with_capacity(self.n_regular_bins);
            if let FeatureBinning::Numeric(be) = binning {
                for &e in be.edges() {
                    feat_thresholds.push(e);
                }
            }
            // Pad remaining with INFINITY (last regular bin + unused bins;
            // the whole row for categorical features).
            while feat_thresholds.len() < self.n_regular_bins {
                feat_thresholds.push(f64::INFINITY);
            }
            thresholds.push(feat_thresholds);
        }
        thresholds
    }
}

/// Precomputes per-feature binning and sample bin assignments.
///
/// This is the single entry point called once per training run. Numeric
/// features get quantile-based bin edges; features flagged in
/// `categorical_mask` get one bin per distinct category code. Each sample
/// is then assigned to its bin. The output is a flat row-major `Vec<u8>` —
/// see the module-level docs for the layout rationale.
///
/// # Args
///
/// * `x` - Row-major feature matrix `[n_samples][n_features]`.
/// * `max_bins` - Maximum number of bins per feature (`2 ≤ max_bins ≤ 255`).
///   The upper bound is enforced by
///   [`GradientBoostingConfig::new`](crate::training::GradientBoostingConfig::new)
///   before this function is called; this function relies on the invariant to
///   pack bin indices into `u8`.
/// * `categorical_mask` - `Some(mask)` flags categorical features by
///   position; `None` bins every feature numerically, bit-identical to the
///   history before the categorical axis existed.
///
/// # Returns
///
/// A `FeatureBins` containing per-feature binning, flat row-major u8
/// assignments, and the bin count.
///
/// # Errors
///
/// * Returns `ClearGbmError::InvalidParameter` if `max_bins < 2` or
///   `max_bins > 255`, if the mask length disagrees with the feature count,
///   or if a categorical column holds a non-integer value or more distinct
///   categories than `max_bins`.
/// * Returns `ClearGbmError::EmptyInput` / `ClearGbmError::ShapeMismatch`
///   for an empty or ragged matrix.
pub fn precompute_feature_bins(
    x: &[&[f64]],
    max_bins: usize,
    min_data_in_bin: usize,
    categorical_mask: Option<&[bool]>,
) -> Result<FeatureBins, ClearGbmError> {
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
    if max_bins > 255_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "max_bins".to_string(),
            reason: format!("must be <= 255 (u8 bin index), got {max_bins}"),
        });
    }
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
    for (i, row) in x.iter().enumerate() {
        if row.len() != n_features {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("all rows with {n_features} features"),
                got: format!("row {i} has {} features", row.len()),
            });
        }
    }
    if let Some(mask) = categorical_mask {
        if mask.len() != n_features {
            return Err(ClearGbmError::InvalidParameter {
                name: "categorical_features".to_string(),
                reason: format!(
                    "categorical mask covers {} features but the matrix has {n_features}",
                    mask.len()
                ),
            });
        }
    }

    let n_samples = x.len();
    let nan_bin = u8::try_from(max_bins).unwrap_or(u8::MAX);

    // Flat row-major storage: sample_bins[sample_idx * n_features + feat_idx].
    // Initialized to the NaN bin so each feature's pass only writes the
    // non-missing rows it assigned.
    let mut sample_bins = vec![nan_bin; n_samples * n_features];
    let mut per_feature: Vec<FeatureBinning> = Vec::with_capacity(n_features);

    for feat_idx in 0_usize..n_features {
        let is_categorical = categorical_mask.is_some_and(|mask| mask[feat_idx]);
        if is_categorical {
            let column: Vec<f64> = x.iter().map(|row| row[feat_idx]).collect();
            let (map, column_bins) = match categorical_column_bins(&column, feat_idx, max_bins) {
                Ok(pair) => pair,
                Err(e) => return Err(e),
            };
            for (sample_idx, bin) in column_bins.iter().enumerate() {
                if let Some(bin_usize) = bin {
                    // Bounded by max_bins <= 255 via the distinct-count check
                    // inside categorical_column_bins, so the u8 conversion's
                    // error arm is statically dead — saturate like the
                    // numeric path below.
                    sample_bins[sample_idx * n_features + feat_idx] =
                        u8::try_from(*bin_usize).unwrap_or(u8::MAX);
                }
            }
            per_feature.push(FeatureBinning::Categorical(map));
        } else {
            let be = compute_feature_edges(x, feat_idx, max_bins, min_data_in_bin);
            for (sample_idx, row) in x.iter().enumerate() {
                let val = row[feat_idx];
                if !val.is_nan() {
                    let bin_idx_usize = assign_bin(val, be.edges());
                    // max_bins <= 255 is enforced above and assign_bin's
                    // result is < max_bins, so every value here is already
                    // in u8 range; saturating conversion because the error
                    // arm would be statically dead.
                    sample_bins[sample_idx * n_features + feat_idx] =
                        u8::try_from(bin_idx_usize).unwrap_or(u8::MAX);
                }
            }
            per_feature.push(FeatureBinning::Numeric(be));
        }
    }

    Ok(FeatureBins {
        per_feature,
        sample_bins,
        n_samples,
        n_features,
        n_regular_bins: max_bins,
    })
}
