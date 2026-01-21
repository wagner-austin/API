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

pub mod error;
pub mod histogram;
pub mod hooks;
pub mod split;
pub mod tree;
pub mod types;

#[cfg(test)]
pub mod testkit;

pub use error::ClearGbmError;
pub use histogram::{build_histogram, subtract_histogram};
pub use hooks::Hooks;
pub use split::{
    check_monotonicity_constraint, compute_split_gain, find_best_split_across_features,
    find_best_split_from_histogram, MonotonicConstraint, NanDirection, SplitResult,
    SplitResultConfig,
};
pub use tree::{build_tree, compute_leaf_value, BuildTreeInput, Tree, TreeBuildConfig};
pub use types::{HistogramBuffer, SplitConfig, TreeNode, TreeNodeConfig};
