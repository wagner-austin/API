//! Shared test helpers for split module tests.

use crate::error::ClearGbmError;
use crate::split::{find_best_split_across_features, find_best_split_from_histogram};
use crate::split::{MonotonicConstraint, SplitResult};
use crate::types::{HistogramBuffer, SplitConfig};

/// Epsilon value for floating-point comparisons in tests.
pub const EPSILON: f64 = 1e-10_f64;

/// Parameters for creating a `SplitConfig` in tests.
///
/// Used to avoid too_many_arguments lint in test helper functions.
#[derive(Debug, Clone, Copy)]
pub struct TestSplitParams {
    /// Minimum samples required to split a node.
    pub min_samples_split: usize,
    /// Minimum samples required in each leaf.
    pub min_samples_leaf: usize,
    /// Maximum number of bins.
    pub max_bins: usize,
    /// L2 regularization parameter.
    pub reg_lambda: f64,
    /// Minimum gain required for a split.
    pub min_gain: f64,
}

impl TestSplitParams {
    /// Creates a `SplitConfig` from these parameters.
    pub fn to_config(self) -> Result<SplitConfig, ClearGbmError> {
        SplitConfig::new(
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_bins,
            self.reg_lambda,
            self.min_gain,
        )
    }
}

/// Helper for find_best_split_from_histogram tests with full config params.
/// Called with both valid and invalid inputs to cover ? branches.
pub fn helper_find_split_with_config(
    histogram: &HistogramBuffer,
    feature_index: usize,
    n_regular_bins: usize,
    constraint: MonotonicConstraint,
    params: TestSplitParams,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let config = match params.to_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    find_best_split_from_histogram(
        histogram,
        feature_index,
        &config,
        n_regular_bins,
        constraint,
    )
}

/// Helper for find_best_split_across_features tests with full config params.
pub fn helper_find_split_across_with_config(
    histograms: &[HistogramBuffer],
    n_regular_bins: usize,
    constraints: Option<&[MonotonicConstraint]>,
    params: TestSplitParams,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let config = match params.to_config() {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    find_best_split_across_features(histograms, &config, n_regular_bins, constraints)
}
