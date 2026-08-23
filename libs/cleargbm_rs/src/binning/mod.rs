//! Feature binning for histogram-based gradient boosting.
//!
//! Converts continuous feature values into discrete bin indices used by
//! the tree builder. Produces output compatible with `BuildTreeInput`.
//!
//! # Overview
//!
//! - [`compute_bin_edges`] computes quantile-based bin thresholds per feature
//! - [`bin_samples`] assigns each sample to a bin per feature
//! - [`precompute_feature_bins`] combines both into a single `FeatureBins`
//!   result, giving features flagged categorical one bin per distinct code

mod assignment;
mod categorical;
mod edges;
mod feature_bins;

#[cfg(test)]
mod tests;

pub use assignment::bin_samples;
pub use categorical::CategoryMap;
pub use edges::{compute_bin_edges, BinEdges};
pub use feature_bins::{precompute_feature_bins, FeatureBinning, FeatureBins};
