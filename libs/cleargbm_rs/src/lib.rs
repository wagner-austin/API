//! `ClearGBM` Rust core with Python bindings.
//!
//! Provides high-performance gradient boosting primitives callable from Python.
//!
//! # Overview
//!
//! This crate implements the performance-critical components of `ClearGBM`:
//! - Histogram building (O(n) gradient/hessian accumulation)
//! - Split finding (O(K) scan over histogram bins)
//! - Tree construction
//! - Prediction/inference
//!
//! # Safety
//!
//! This crate forbids unsafe code. All operations use checked arithmetic
//! and explicit error handling via `Result<T, ClearGbmError>`.

/// Propagates `Result` errors, equivalent to `?` but compatible with
/// `clippy::question_mark_used`. The macro expansion maps both match arms
/// to the invocation line for LLVM coverage.
macro_rules! propagate {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(err) => return Err(err),
        }
    };
}

/// Like [`propagate!`] but converts the error via `.into()` before returning.
/// Useful when the error type needs conversion (e.g., `ClearGbmError` → `PyErr`).
macro_rules! propagate_into {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(err) => return Err(err.into()),
        }
    };
}

pub mod binning;
pub mod error;
pub mod histogram;
pub mod hooks;
pub mod losses;
pub mod narrow;
pub mod predict;
#[cfg(feature = "extension-module")]
pub mod pyo3_module;
pub mod split;
pub mod training;
pub mod tree;
pub mod types;

#[cfg(test)]
pub mod testkit;

pub use binning::{bin_samples, compute_bin_edges, precompute_feature_bins, BinEdges, FeatureBins};
pub use error::ClearGbmError;
pub use histogram::{build_histogram, subtract_histogram};
pub use hooks::Hooks;
pub use losses::{
    binary_log_loss, binary_log_loss_gradients, binary_log_loss_hessians,
    binary_log_loss_initial_prediction, sigmoid_array,
};
pub use predict::{
    predict_ensemble, predict_proba, predict_single, predict_tree, sigmoid, PredictEnsembleConfig,
};
pub use split::{
    check_monotonicity_constraint, compute_split_gain, find_best_split_across_features,
    find_best_split_from_histogram, MonotonicConstraint, NanDirection, SplitResult,
    SplitResultConfig,
};
pub use training::{
    train_gradient_boosting, GradientBoostingConfig, GradientBoostingConfigParams,
    GradientBoostingModel,
};
pub use tree::{build_tree, compute_leaf_value, BuildTreeInput, Tree, TreeBuildConfig};
pub use types::{HistogramBuffer, SplitConfig, TreeNode, TreeNodeConfig};
