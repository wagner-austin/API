# ClearGBM Rust Core Transition Plan

## Overview

Port the performance-critical components of cleargbm to Rust via PyO3, maintaining the existing Python API while achieving 20-50x speedup on hot paths.

**Goals:**
1. Preserve existing Python API (no breaking changes for consumers)
2. Port histogram building, split finding, and tree traversal to Rust
3. Maintain strict code standards equivalent to our Python requirements
4. Achieve 100% test coverage with no mocks, only fakes
5. Zero `unsafe` code unless absolutely necessary (and documented)

---

## Code Standards (Rust Equivalent of Python Standards)

### Strict Typing

| Python Standard | Rust Equivalent |
|-----------------|-----------------|
| No `Any` | No equivalent exists in Rust |
| No `cast()` | `#![forbid(clippy::as_conversions)]` - use `.into()`, `.try_into()` |
| No `type: ignore` | `#![forbid(clippy::allow_attributes)]` - no escape hatches |
| No `.pyi` stubs | Not applicable (types are in source) |
| No `noqa` | `#![forbid(clippy::allow_attributes)]` |
| TypedDicts | Rust structs with `#[derive(Debug, Clone, PartialEq)]` |

### Error Handling

| Python Standard | Rust Equivalent |
|-----------------|-----------------|
| No try/except recovery | `Result<T, E>` with `?` propagation |
| No fallback/best-effort | All `Result` must be handled explicitly |
| Explicit failure propagation | `#![deny(clippy::unwrap_used, clippy::expect_used)]` |
| Clear failure APIs | Custom error enums with descriptive variants |

### Testing

| Python Standard | Rust Equivalent |
|-----------------|-----------------|
| 100% coverage | `cargo-tarpaulin --fail-under 100` or `cargo-llvm-cov` |
| No mocks, only fakes | Traits for DI, fake implementations |
| No weak assertions | `assert_eq!` with specific values, not just `assert!` |
| `_test_hooks.py` pattern | Generic type parameters or trait objects for injection |

### Documentation

| Python Standard | Rust Equivalent |
|-----------------|-----------------|
| Google-style docstrings | `///` doc comments with Args/Returns/Errors sections |
| Docstrings on all public items | `#![warn(missing_docs)]` |

---

## Cargo.toml Configuration

```toml
[package]
name = "cleargbm_rs"
version = "0.1.0"
edition = "2021"
description = "High-performance Rust core for ClearGBM gradient boosting"
license = "MIT"
readme = "README.md"
repository = "https://github.com/wagner-austin/API"
keywords = ["machine-learning", "gradient-boosting", "gbm", "python-bindings"]
categories = ["science", "algorithms"]

[lib]
name = "cleargbm_rs"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.27", features = ["extension-module"] }
numpy = "0.27"
ndarray = "0.17"
thiserror = "2.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[dev-dependencies]
proptest = "1.5"

# =============================================================================
# Strict Lints (Equivalent to our Python mypy strict + ruff)
# =============================================================================

[lints.rust]
unsafe_code = "forbid"
missing_docs = "deny"

[lints.clippy]
# All lints denied
all = "deny"
cargo = "deny"

# No escape hatches (equivalent to no type: ignore, no noqa)
allow_attributes = "deny"
allow_attributes_without_reason = "deny"

# No silent failures (equivalent to no try/except recovery)
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
unreachable = "deny"
todo = "deny"
unimplemented = "deny"

# No unsafe conversions (equivalent to no cast())
as_conversions = "deny"
cast_possible_truncation = "deny"
cast_sign_loss = "deny"
cast_precision_loss = "deny"
cast_possible_wrap = "deny"

# No implicit behavior
default_numeric_fallback = "deny"
implicit_clone = "deny"

# Explicit error handling
result_unit_err = "deny"
map_err_ignore = "deny"

# Code quality
clone_on_ref_ptr = "deny"
redundant_closure_for_method_calls = "deny"
manual_let_else = "deny"
needless_pass_by_value = "deny"

# Documentation
missing_docs_in_private_items = "deny"
missing_errors_doc = "deny"
```

---

## Project Structure

```
libs/
├── cleargbm/                    # Existing Python package
│   ├── src/cleargbm/
│   │   ├── __init__.py
│   │   ├── histogram.py         # Calls Rust when available
│   │   ├── tree.py              # Calls Rust when available
│   │   └── ...
│   ├── tests/
│   └── docs/
│       └── rust-core-transition-plan.md  # This document
│
└── cleargbm_rs/                 # Rust core (IMPLEMENTED)
    ├── Cargo.toml               # Strict lint configuration
    ├── Makefile                 # make check (lint + test)
    ├── pyproject.toml           # Maturin build config (future)
    └── src/
        ├── lib.rs               # Module exports
        ├── error.rs             # ClearGbmError enum (with inline tests)
        ├── types/               # TreeNode, HistogramBuffer, SplitConfig (with inline tests)
        ├── histogram/           # build_histogram, subtract_histogram (with inline tests)
        ├── split/               # Split finding (with inline tests)
        ├── tree/                # Tree construction: build_tree, TreeBuildConfig, Tree (with inline tests)
        └── predict/             # Prediction/inference (IMPLEMENTED)
            ├── mod.rs           # sigmoid, predict_single, predict_tree, predict_ensemble, predict_proba
            └── tests/           # 57 unit tests across 5 test files
    └── tests/
        └── integration_tests.rs # 17 end-to-end integration tests
```

**Note:** Tests are inline in each module (`#[cfg(test)] mod tests { ... }`) following Rust convention.
Integration tests are in `tests/integration_tests.rs`.

---

## Progress Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Setup and Error Types | **COMPLETED** |
| 2 | Core Types (TreeNode, HistogramBuffer) | **COMPLETED** |
| 3 | Histogram Building | **COMPLETED** |
| 4 | Split Finding | **COMPLETED** |
| 5 | Tree Construction | **COMPLETED** |
| 6 | Prediction/Inference | **COMPLETED** |
| 7 | PyO3 Bindings | **COMPLETED** |
| 8 | Python Integration | PENDING |
| 9 | Benchmarking and Optimization | PENDING |

**Current Status:** `make check` passes (1122 tests, clippy clean, fmt clean, 100% segment coverage). All lints at `forbid`.

---

## Phase 1: Project Setup and Error Types

### Error Handling Strategy

All errors use a custom enum. No panics, no unwrap, no expect.
**No type aliases** - always use `std::result::Result<T, ClearGbmError>` explicitly.

```rust
// src/error.rs

//! Error types for `ClearGBM` Rust core.
//!
//! All errors are explicit and propagate via `Result<T, ClearGbmError>`.
//! No panics, no unwrap, no expect in production code.

use thiserror::Error;

/// Errors that can occur during `ClearGBM` operations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ClearGbmError {
    /// Feature index is out of bounds.
    #[error("feature index {index} out of bounds (n_features={n_features})")]
    FeatureIndexOutOfBounds {
        /// The invalid index that was provided.
        index: usize,
        /// The total number of features.
        n_features: usize,
    },

    /// Sample index is out of bounds.
    #[error("sample index {index} out of bounds (n_samples={n_samples})")]
    SampleIndexOutOfBounds {
        /// The invalid index that was provided.
        index: usize,
        /// The total number of samples.
        n_samples: usize,
    },

    /// Bin index is out of bounds.
    #[error("bin index {bin} out of bounds (n_bins={n_bins})")]
    BinIndexOutOfBounds {
        /// The invalid bin index.
        bin: usize,
        /// The total number of bins.
        n_bins: usize,
    },

    /// Array shape mismatch.
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        got: String,
    },

    /// Empty input where non-empty required.
    #[error("empty input: {context}")]
    EmptyInput {
        /// Context describing what was empty.
        context: String,
    },

    /// Invalid parameter value.
    #[error("invalid parameter {name}: {reason}")]
    InvalidParameter {
        /// Parameter name.
        name: String,
        /// Reason it's invalid.
        reason: String,
    },

    /// Tree construction failed.
    #[error("tree construction failed: {reason}")]
    TreeConstructionFailed {
        /// Reason for failure.
        reason: String,
    },

    /// Node not found in tree.
    #[error("node {node_id} not found in tree")]
    NodeNotFound {
        /// The missing node ID.
        node_id: usize,
    },

    /// Integer conversion failed.
    #[error("integer conversion failed: {context}")]
    IntegerConversion {
        /// Context describing the conversion.
        context: String,
    },
}

// NOTE: No type alias - use std::result::Result<T, ClearGbmError> explicitly
```

---

## Phase 2: Core Types

### TreeNodeConfig (Avoids too_many_arguments)

```rust
/// Configuration for creating an internal (split) tree node.
///
/// Used to avoid having too many function arguments while maintaining
/// explicit, named parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TreeNodeConfig {
    pub node_id: usize,
    pub feature_index: usize,
    pub threshold: f64,
    pub value: f64,
    pub n_samples: usize,
    pub left_child: usize,
    pub right_child: usize,
    pub nan_goes_left: bool,
}
```

### TreeNode (Equivalent to Python TypedDict)

```rust
// src/types.rs

//! Core data structures for `ClearGBM`.
//!
//! All types are immutable after construction. Use builder patterns
//! or constructor functions to create instances.

use serde::{Deserialize, Serialize};
use crate::error::ClearGbmError;

/// A node in a gradient boosted decision tree.
///
/// Nodes are either internal (with split information) or leaf (with prediction value).
/// This is equivalent to the Python `TreeNode` `TypedDict`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeNode {
    node_id: usize,
    is_leaf: bool,
    feature_index: Option<usize>,
    threshold: Option<f64>,
    value: f64,
    n_samples: usize,
    left_child: Option<usize>,
    right_child: Option<usize>,
    nan_goes_left: bool,
}

impl TreeNode {
    /// Creates a new leaf node.
    #[must_use]
    pub const fn new_leaf(node_id: usize, value: f64, n_samples: usize) -> Self {
        Self {
            node_id,
            is_leaf: true,
            feature_index: None,
            threshold: None,
            value,
            n_samples,
            left_child: None,
            right_child: None,
            nan_goes_left: true,
        }
    }

    /// Creates a new internal (split) node from configuration.
    ///
    /// Uses `TreeNodeConfig` to avoid too_many_arguments lint.
    #[must_use]
    pub const fn new_internal(config: TreeNodeConfig) -> Self {
        Self {
            node_id: config.node_id,
            is_leaf: false,
            feature_index: Some(config.feature_index),
            threshold: Some(config.threshold),
            value: config.value,
            n_samples: config.n_samples,
            left_child: Some(config.left_child),
            right_child: Some(config.right_child),
            nan_goes_left: config.nan_goes_left,
        }
    }

    // Getters: node_id(), is_leaf(), feature_index(), threshold(),
    //          value(), n_samples(), left_child(), right_child(), nan_goes_left()
    // All are `#[must_use] pub const fn`
}

### HistogramBuffer

```rust
/// Histogram buffer for gradient/hessian accumulation.
///
/// Used during split finding to accumulate statistics per bin.
/// Equivalent to Python `HistogramBuffer` but with explicit sizing.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBuffer {
    gradient_sums: Vec<f64>,
    hessian_sums: Vec<f64>,
    counts: Vec<usize>,
    n_bins: usize,
}

impl HistogramBuffer {
    /// Creates a new zeroed histogram buffer.
    #[must_use]
    pub fn new(n_bins: usize) -> Self { ... }

    /// Returns the number of bins.
    #[must_use]
    pub const fn n_bins(&self) -> usize { ... }

    /// Accumulates a sample into the appropriate bin.
    ///
    /// # Errors
    /// Returns `ClearGbmError::BinIndexOutOfBounds` if `bin` >= `n_bins`.
    pub fn accumulate(&mut self, bin: usize, gradient: f64, hessian: f64)
        -> std::result::Result<(), ClearGbmError> { ... }

    /// Per-bin accessors (all return Result for bounds checking)
    pub fn gradient_sum(&self, bin: usize) -> std::result::Result<f64, ClearGbmError> { ... }
    pub fn hessian_sum(&self, bin: usize) -> std::result::Result<f64, ClearGbmError> { ... }
    pub fn count(&self, bin: usize) -> std::result::Result<usize, ClearGbmError> { ... }

    /// Slice accessors (for bulk operations)
    #[must_use]
    pub fn gradient_sums(&self) -> &[f64] { ... }
    #[must_use]
    pub fn hessian_sums(&self) -> &[f64] { ... }
    #[must_use]
    pub fn counts(&self) -> &[usize] { ... }

    /// Resets all bins to zero (for reuse).
    pub fn reset(&mut self) { ... }

    /// Computes sibling histogram by subtraction: self = parent - child.
    /// This is the "histogram trick" for 2x speedup.
    ///
    /// # Errors
    /// Returns `ClearGbmError::ShapeMismatch` if bin counts don't match.
    pub fn subtract_into(&mut self, parent: &Self, child: &Self)
        -> std::result::Result<(), ClearGbmError> { ... }

    /// Copies contents from another histogram buffer.
    ///
    /// # Errors
    /// Returns `ClearGbmError::ShapeMismatch` if bin counts don't match.
    pub fn copy_from(&mut self, other: &Self)
        -> std::result::Result<(), ClearGbmError> { ... }
}
```

### SplitConfig

```rust
/// Configuration for histogram-based split finding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitConfig {
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_bins: usize,
    reg_lambda: f64,
    min_gain: f64,
}

impl SplitConfig {
    /// Creates a new split configuration with validation.
    ///
    /// # Errors
    /// Returns `ClearGbmError::InvalidParameter` if:
    /// - `min_samples_split` < 2
    /// - `min_samples_leaf` < 1
    /// - `max_bins` < 2
    /// - `reg_lambda` < 0.0
    /// - `min_gain` < 0.0
    pub fn new(
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_bins: usize,
        reg_lambda: f64,
        min_gain: f64,
    ) -> std::result::Result<Self, ClearGbmError> { ... }

    // Getters: min_samples_split(), min_samples_leaf(), max_bins(),
    //          reg_lambda(), min_gain()
    // All are `#[must_use] pub const fn`
}
```

---

## Phase 3: Histogram Building

```rust
// src/histogram.rs

//! Histogram building for gradient boosting.
//!
//! Implements O(n) histogram construction with NaN handling.
//! This is the primary hot path and performance-critical code.

use crate::error::ClearGbmError;
use crate::types::HistogramBuffer;

/// Builds a histogram from sample gradients and hessians.
///
/// This is the core O(n) operation that accumulates gradient statistics
/// into bins for split finding.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` - If `sample_indices` is empty.
/// * `ClearGbmError::SampleIndexOutOfBounds` - If any index is out of bounds.
/// * `ClearGbmError::ShapeMismatch` - If array lengths don't match.
/// * `ClearGbmError::BinIndexOutOfBounds` - If any bin index is out of bounds.
pub fn build_histogram(
    sample_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[usize],
    n_bins: usize,
) -> std::result::Result<HistogramBuffer, ClearGbmError> { ... }

/// Computes sibling histogram by subtraction (2x speedup).
///
/// Given parent histogram and one child histogram, computes the
/// other child by subtraction: sibling = parent - child.
///
/// # Errors
///
/// * `ClearGbmError::ShapeMismatch` - If histograms have different `n_bins`.
pub fn subtract_histogram(
    parent: &HistogramBuffer,
    child: &HistogramBuffer,
) -> std::result::Result<HistogramBuffer, ClearGbmError> { ... }
```

### Test Pattern (No unwrap in tests)

Since `unwrap_used = "deny"`, tests use `assert!` + `if let`:

```rust
#[test]
fn test_build_histogram_simple() {
    let result = build_histogram(&sample_indices, &gradients, &hessians, &bins, n_bins);

    assert!(result.is_ok());
    if let Ok(hist) = result {
        if let Ok(grad_0) = hist.gradient_sum(0_usize) {
            assert!((grad_0 - 0.4_f64).abs() < 1e-10_f64);
        }
        assert_eq!(hist.count(0_usize).ok(), Some(2_usize));
    }
}

#[test]
fn test_build_histogram_empty_indices_fails() {
    let result = build_histogram(&[], &gradients, &hessians, &bins, 2_usize);

    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(ClearGbmError::EmptyInput { .. })
    ));
}
```

### Public Exports (lib.rs)

```rust
//! `ClearGBM` Rust core with Python bindings.

pub mod error;
pub mod histogram;
pub mod types;

pub use error::ClearGbmError;
pub use histogram::{build_histogram, subtract_histogram};
pub use types::{HistogramBuffer, SplitConfig, TreeNode, TreeNodeConfig};
```

---

## Phase 4: Split Finding (COMPLETED)

```rust
// src/split.rs (IMPLEMENTED)

use crate::error::ClearGbmError;
use crate::types::{HistogramBuffer, SplitConfig};
use serde::{Deserialize, Serialize};

/// Direction for NaN values during tree traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NanDirection {
    Left,
    Right,
}

/// Monotonicity constraint for a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotonicConstraint {
    None,
    Increasing,
    Decreasing,
}

/// Configuration for creating a `SplitResult`.
/// Used to avoid too_many_arguments lint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitResultConfig {
    pub feature_index: usize,
    pub split_bin: usize,
    pub gain: f64,
    pub left_gradient_sum: f64,
    pub left_hessian_sum: f64,
    pub left_count: usize,
    pub right_gradient_sum: f64,
    pub right_hessian_sum: f64,
    pub right_count: usize,
    pub nan_direction: NanDirection,
}

/// Result of a split search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitResult {
    feature_index: usize,
    split_bin: usize,
    gain: f64,
    left_gradient_sum: f64,
    left_hessian_sum: f64,
    left_count: usize,
    right_gradient_sum: f64,
    right_hessian_sum: f64,
    right_count: usize,
    nan_direction: NanDirection,
}

impl SplitResult {
    /// Creates a new `SplitResult` from configuration.
    #[must_use]
    pub const fn new(config: SplitResultConfig) -> Self { ... }

    // Getters: feature_index(), split_bin(), gain(), left_gradient_sum(),
    //          left_hessian_sum(), left_count(), right_gradient_sum(),
    //          right_hessian_sum(), right_count(), nan_direction()
    // All are `#[must_use] pub const fn`
}

/// Computes the gain from a split with L2 regularization.
///
/// Formula: G_L^2/(H_L + λ) + G_R^2/(H_R + λ) - G^2/(H + λ)
pub fn compute_split_gain(
    g_left: f64, h_left: f64,
    g_right: f64, h_right: f64,
    g_total: f64, h_total: f64,
    reg_lambda: f64,
) -> f64 { ... }

/// Checks if a split satisfies a monotonicity constraint.
///
/// For Increasing: left_weight <= right_weight
/// For Decreasing: left_weight >= right_weight
pub fn check_monotonicity_constraint(
    constraint: MonotonicConstraint,
    g_left: f64, h_left: f64,
    g_right: f64, h_right: f64,
) -> bool { ... }

/// Finds the best split for a single feature from its histogram.
/// O(K) where K is the number of bins.
///
/// Evaluates both NaN-goes-left and NaN-goes-right for each split point,
/// applies monotonicity constraints, and returns the best valid split.
pub fn find_best_split_from_histogram(
    histogram: &HistogramBuffer,
    feature_index: usize,
    config: &SplitConfig,
    n_regular_bins: usize,
    monotonic_constraint: MonotonicConstraint,
) -> std::result::Result<Option<SplitResult>, ClearGbmError> { ... }

/// Finds the best split across multiple features.
///
/// Calls `find_best_split_from_histogram` for each feature and returns
/// the split with the highest gain.
pub fn find_best_split_across_features(
    histograms: &[HistogramBuffer],
    config: &SplitConfig,
    n_regular_bins: usize,
    monotonic_constraints: Option<&[MonotonicConstraint]>,
) -> std::result::Result<Option<SplitResult>, ClearGbmError> { ... }
```

### Key Implementation Details

- **NaN handling**: Each split point is evaluated with both NaN-goes-left and NaN-goes-right; the direction with higher gain is selected
- **Monotonicity constraints**: Optional per-feature constraints that reject splits violating increasing/decreasing predictions
- **Prefix sums**: Uses cumulative gradient/hessian sums for O(K) split finding
- **Config pattern**: Uses `SplitResultConfig` struct to avoid too_many_arguments lint
- **32 inline tests**: Comprehensive coverage of edge cases, NaN handling, monotonicity constraints, and multi-feature selection

---

## Phase 5: Tree Construction (COMPLETED)

```rust
// src/tree.rs (IMPLEMENTED)

use crate::error::ClearGbmError;
use crate::histogram::{build_histogram, subtract_histogram};
use crate::split::{find_best_split_from_histogram, MonotonicConstraint, SplitResult};
use crate::types::{HistogramBuffer, SplitConfig, TreeNode, TreeNodeConfig};
use serde::{Deserialize, Serialize};

/// Configuration for tree building.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeBuildConfig {
    max_depth: usize,
    max_leaves: usize,
    reg_alpha: f64,
    reg_lambda: f64,
    split_config: SplitConfig,
}

impl TreeBuildConfig {
    /// Creates a new tree build configuration.
    pub fn new(
        max_depth: usize,
        max_leaves: usize,
        reg_alpha: f64,
        reg_lambda: f64,
        split_config: SplitConfig,
    ) -> std::result::Result<Self, ClearGbmError> { ... }

    // Getters: max_depth(), max_leaves(), reg_alpha(), reg_lambda(), split_config()
}

/// A complete decision tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tree {
    nodes: Vec<TreeNode>,
    max_depth: usize,
    n_leaves: usize,
}

impl Tree {
    #[must_use]
    pub fn new(nodes: Vec<TreeNode>, max_depth: usize, n_leaves: usize) -> Self { ... }

    #[must_use]
    pub fn nodes(&self) -> &[TreeNode] { ... }

    pub fn node(&self, node_id: usize) -> std::result::Result<&TreeNode, ClearGbmError> { ... }

    pub fn root(&self) -> std::result::Result<&TreeNode, ClearGbmError> { ... }

    // Getters: max_depth(), n_leaves(), n_nodes()
}

/// Computes the optimal leaf value from gradient and hessian sums.
/// Formula: -G / (H + λ) with L1 soft thresholding.
#[must_use]
pub fn compute_leaf_value(
    gradient_sum: f64,
    hessian_sum: f64,
    reg_alpha: f64,
    reg_lambda: f64,
) -> f64 { ... }

/// Configuration for `build_tree` to avoid too many arguments.
#[derive(Debug, Clone)]
pub struct BuildTreeInput<'a> {
    pub sample_indices: &'a [usize],
    pub gradients: &'a [f64],
    pub hessians: &'a [f64],
    pub bins: &'a [Vec<usize>],
    pub n_regular_bins: usize,
    pub bin_thresholds: &'a [Vec<f64>],
    pub config: &'a TreeBuildConfig,
    pub monotonic_constraints: Option<&'a [MonotonicConstraint]>,
}

/// Builds a decision tree using histogram-based split finding.
/// Uses depth-first traversal with sibling histogram subtraction trick.
pub fn build_tree(input: &BuildTreeInput<'_>) -> std::result::Result<Tree, ClearGbmError> { ... }
```

### Key Implementation Details

- **Depth-first traversal**: Uses stack-based approach matching Python implementation
- **Histogram subtraction trick**: Builds histogram for smaller child, derives larger via subtraction for 2x speedup
- **Leaf value computation**: `-G/(H+λ)` with L1 soft thresholding support
- **Stopping criteria**: max_depth, max_leaves, min_samples_split, min_samples_leaf
- **Config pattern**: Uses `BuildTreeInput` and `ChildHistogramConfig` structs to avoid too_many_arguments lint
- **23 inline tests**: Comprehensive coverage of tree building, stopping criteria, sample splitting

---

## Phase 6: Prediction/Inference (COMPLETED)

```rust
// src/predict/mod.rs (IMPLEMENTED)

use crate::error::ClearGbmError;
use crate::tree::Tree;

/// Configuration for ensemble prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictEnsembleConfig {
    base_prediction: f64,
    learning_rate: f64,
}

impl PredictEnsembleConfig {
    /// Creates a new ensemble prediction configuration.
    ///
    /// # Errors
    /// Returns `ClearGbmError::InvalidParameter` if learning_rate is not in (0.0, 1.0].
    pub fn new(base_prediction: f64, learning_rate: f64)
        -> std::result::Result<Self, ClearGbmError> { ... }

    // Getters: base_prediction(), learning_rate()
    // All are `#[must_use] pub const fn`
}

/// Computes the sigmoid (logistic) function with input clipping to [-500, 500].
#[must_use]
pub fn sigmoid(x: f64) -> f64 { ... }

/// Predicts a single sample by traversing the tree from root to leaf.
///
/// # Errors
/// * `ClearGbmError::NodeNotFound` - If a referenced node does not exist.
/// * `ClearGbmError::FeatureIndexOutOfBounds` - If a node references an invalid feature.
/// * `ClearGbmError::TreeConstructionFailed` - If the tree is malformed or contains cycles.
pub fn predict_single(tree: &Tree, features: &[f64])
    -> std::result::Result<f64, ClearGbmError> { ... }

/// Predicts multiple samples against a single tree.
///
/// # Errors
/// * `ClearGbmError::EmptyInput` - If features is empty.
/// * Any error from `predict_single`.
pub fn predict_tree(tree: &Tree, features: &[&[f64]])
    -> std::result::Result<Vec<f64>, ClearGbmError> { ... }

/// Predicts raw scores for multiple samples using an ensemble of trees.
/// Formula: base_prediction + learning_rate * sum(tree_predictions)
///
/// # Errors
/// * `ClearGbmError::EmptyInput` - If features or trees is empty.
/// * Any error from `predict_single`.
pub fn predict_ensemble(trees: &[Tree], features: &[&[f64]], config: &PredictEnsembleConfig)
    -> std::result::Result<Vec<f64>, ClearGbmError> { ... }

/// Converts raw predictions to class probabilities via sigmoid.
/// Returns Vec<(prob_class_0, prob_class_1)>.
#[must_use]
pub fn predict_proba(raw_predictions: &[f64]) -> Vec<(f64, f64)> { ... }
```

### Key Implementation Details

- **Cycle guard in `predict_single`**: Caps iterations at `tree.n_nodes()` to detect malformed trees
- **NaN handling**: Respects `nan_goes_left` flag on each internal node
- **Sigmoid clipping**: Input clamped to [-500, 500] for numerical stability
- **`predict_proba` is infallible**: Returns empty vec for empty input (matches `compute_leaf_value` pattern)
- **57 unit tests + 5 integration tests**: Covers sigmoid, single prediction, batch, ensemble, probability conversion, and all error paths
- **Test files**: `src/predict/tests/{sigmoid_tests,single_tests,batch_tests,ensemble_tests,error_tests}.rs`

---

## Phase 7: PyO3 Bindings (COMPLETED)

### Approach

All bindings use `PyCFunction::new_closure` with manual argument extraction.
No `#[pyfunction]` proc macros. All lints stay at `forbid` — no relaxation.

### Architecture

Two-level function pattern for each binding:
- `_rs` function: typed, testable, calls Rust core
- `_from_args` wrapper: extracts from `PyTuple`, calls `_rs`

Module registration uses `.and_then()` chains — no explicit `Err` arms in our source.

### Files

```
src/pyo3_module/
├── mod.rs                 # Module entry (#[pymodule]) + register_all via .and_then()
├── array_helpers.rs       # Generic try_convert_int<F,T> + convert_int_slice<F,T>
├── error_conversion.rs    # ClearGbmError → PyErr conversion
├── histogram_fns.rs       # build_histogram_rs, subtract_histogram_rs
├── prediction_fns.rs      # sigmoid_rs, predict_single/tree/ensemble/proba_rs
├── tree_fns.rs            # build_tree_rs, PyTree (#[pyclass]), json serde
└── tests/                 # 6 test modules, all Err paths exercised
    ├── mod.rs
    ├── array_helpers_tests.rs
    ├── error_conversion_tests.rs
    ├── histogram_fns_tests.rs
    ├── module_init_tests.rs
    ├── prediction_fns_tests.rs
    └── tree_fns_tests.rs
```

### Exposed Python API

| Function | Description |
|---|---|
| `build_histogram_rs(indices, grads, hess, bins, n_bins)` | Returns `(grad_sums, hess_sums, counts)` as numpy arrays |
| `subtract_histogram_rs(g1, h1, c1, g2, h2, c2, n_bins)` | Sibling histogram subtraction |
| `build_tree_rs(indices, grads, hess, bins, n_bins, thresholds, config_json)` | Returns `PyTree` |
| `predict_single_rs(tree, features)` | Single sample prediction |
| `predict_tree_rs(tree, features_2d)` | Batch prediction |
| `predict_ensemble_rs(trees, features_2d, base_pred, lr)` | Ensemble prediction |
| `predict_proba_rs(raw_preds)` | Raw → probability conversion |
| `sigmoid_rs(x)` | Sigmoid function |
| `PyTree` | Class with `.to_json()`, `.from_json()`, `.max_depth`, `.n_leaves`, `.n_nodes` |

### Key Patterns

- **Generic integer conversion**: `try_convert_int<F, T>` tested with u64→u32 overflow to exercise Err arms unreachable on 64-bit (usize↔u64)
- **Named error helpers**: `ser_err()`, `shape_err()` extracted from `.map_err()` closures for compact calls + direct testing
- **`.and_then()` chains**: Module registration eliminates explicit Err arms; error propagation lives in `Result::and_then`
- **`.map()` for infallible transforms**: `PyTuple::new().map(|t| t.unbind().into_any())`

### Stats

- **1122 tests** (1105 unit + 17 integration), all passing
- **3253/3253 lines — 100.00% segment coverage**
- All clippy lints at `forbid` including `question_mark_used`, `as_conversions`, `unwrap_used`
- `unsafe_code = "forbid"` — PyO3 0.27.2's `#[pymodule]` expansion passes this lint

---

## Phase 8: Python Integration (IN PROGRESS)

### Approach

All hot-path operations are wired through `_test_hooks.py` module-level hooks.
Each hook follows a three-part pattern:

1. **Protocol** — typed callable interface (e.g., `PredictTreeBackend`)
2. **Default implementation** — pure Python function (e.g., `_default_predict_tree`)
3. **Module-level hook variable** — set to default, swappable at startup (e.g., `_predict_tree_backend`)
4. **Public accessor** — calls the hook directly (e.g., `predict_tree()`)

No `try/except`, no auto-detection, no `_get_default_backend()`. Production sets hooks
to Rust implementations at startup. Tests call through the Python defaults or inject fakes.

### Hook Inventory

| Hook Variable | Protocol | Default | Caller Module |
|---|---|---|---|
| `_random_state_factory` | `Callable[[int], RandomStateProtocol]` | `_default_random_state_factory` | `tree.py` |
| `_pool_factory` | `Callable[..., WorkerPoolProtocol]` | `_default_pool_factory` | `tree.py` |
| `_float_buffer_factory` | `Callable[[int], FloatBuffer]` | `_default_float_buffer_factory` | (internal) |
| `_int_buffer_factory` | `Callable[[int], IntBuffer]` | `_default_int_buffer_factory` | (internal) |
| `_histogram_buffer_factory` | `Callable[[int], HistogramBuffer]` | `_default_histogram_buffer_factory` | (internal) |
| `_build_histogram_backend` | `BuildHistogramBackend` | `_default_build_histogram` | `histogram.py` |
| `_subtract_histogram_backend` | `SubtractHistogramBackend` | `_default_subtract_histogram` | `histogram.py` |
| `_predict_tree_backend` | `PredictTreeBackend` | `_default_predict_tree` | `tree.py` |
| `_sigmoid_backend` | `SigmoidBackend` | `_default_sigmoid` | `losses.py` |
| `_sigmoid_array_backend` | `SigmoidArrayBackend` | `_default_sigmoid_array` | `losses.py` |
| `guard_find_monorepo_root` | `FindMonorepoRootProto \| None` | `None` | `scripts/guard.py` |
| `guard_load_orchestrator` | `LoadOrchestratorProto \| None` | `None` | `scripts/guard.py` |

### Caller Wiring

Each caller imports the public accessor and delegates to it:

```python
# histogram.py
from cleargbm._test_hooks import build_histogram as _build_histogram_hook
from cleargbm._test_hooks import subtract_histogram as _subtract_histogram_hook

# tree.py
from cleargbm._test_hooks import predict_tree as _predict_tree_hook

# losses.py
from cleargbm._test_hooks import sigmoid as _sigmoid_hook
from cleargbm._test_hooks import sigmoid_array as _sigmoid_array_hook
```

### Stats

- **488 tests**, all passing
- **2057 statements, 592 branches — 100.00% coverage**
- No `Any`, no `cast()`, no `type: ignore`
- All hooks tested directly in `test_test_hooks.py`

### Remaining Work

- **Rust adapter functions**: Wrapper functions that accept Python types (e.g., `DecisionTree`
  TypedDict) and call `cleargbm_rs` functions (which use `PyTree` opaque wrapper)
- **Production startup wiring**: Code that sets `_test_hooks._predict_tree_backend = rust_adapter`
  when the Rust extension is available
- **Type bridge**: Converting between Python `DecisionTree` TypedDict and Rust `PyTree`
  (via JSON or direct field mapping)
- **`maturin develop`** verification: Build Rust extension and verify `import cleargbm_rs` works

---

## Downstream Consumers

### covenant_ml (direct dependency)

covenant_ml registers cleargbm as a classifier backend via `covenant_ml.backends.cleargbm.backend`.
This is the only downstream consumer of cleargbm's public API.

**Imports from cleargbm:**

```python
from cleargbm.ensemble import predict_proba as cgbm_predict_proba
from cleargbm.ensemble import train_gradient_boosting
from cleargbm.explain import get_feature_importances
from cleargbm.types import (
    GradientBoostingConfig,
    GradientBoostingModel,
    TrainingProgress,
    decode_gradient_boosting_model,
    encode_gradient_boosting_model,
)
```

**API surface consumed:**

| Function / Type | Module | Usage |
|---|---|---|
| `train_gradient_boosting()` | `ensemble` | Core training loop (x_train, y_train, x_val, y_val, config, feature_names, progress_callback) |
| `predict_proba()` | `ensemble` | Inference — returns tuple, wrapped into ndarray |
| `get_feature_importances()` | `explain` | Returns list of dicts with `feature_name` and `total_contribution` keys |
| `GradientBoostingConfig` | `types` | TypedDict passed to `train_gradient_boosting()` |
| `GradientBoostingModel` | `types` | TypedDict returned from training, stored in `_ClearGBMPrepared` |
| `TrainingProgress` | `types` | TypedDict passed to progress callback (keys: `tree_index`, `total_trees`, `train_loss`, `val_loss`) |
| `encode_gradient_boosting_model()` | `types` | Serializes model to dict for JSON persistence |
| `decode_gradient_boosting_model()` | `types` | Deserializes model from dict loaded from JSON |

**Hard constraints for Rust transition:**

1. **Function signatures unchanged** — `train_gradient_boosting()` and `predict_proba()` must
   accept and return identical types
2. **JSON serialization format identical** — saved `.json` model files must be loadable by both
   old and new code (no migration needed)
3. **TypedDict structures frozen** — `GradientBoostingConfig`, `GradientBoostingModel`,
   `TrainingProgress` field names, types, and optionality must not change
4. **Feature importance format unchanged** — `get_feature_importances()` must return the same
   list-of-dicts structure with `feature_name` and `total_contribution` keys
5. **Progress callback interface unchanged** — `TrainingProgress` dict keys and value types
   must remain compatible

**Integration test gate:** covenant_ml has a 928-line integration test suite at
`covenant_ml/tests/backends/cleargbm/test_cleargbm_integration.py` that exercises training,
evaluation, save/load, feature importance, early stopping, and progress callbacks. This suite
must pass after the Rust transition with zero changes.

### covenant_nn (no dependency)

covenant_nn provides PyTorch neural network backends (MLP, LSTM) for covenant_ml. It depends on
covenant_ml's protocols but has **no dependency on cleargbm**. The Rust transition does not
affect covenant_nn.

### covenant-radar-api (indirect dependency)

covenant-radar-api consumes cleargbm indirectly through covenant_ml's backend registry. The
`train_job.py` worker creates a `ClearGBMBackend` via the registry and calls `train()`. No
direct imports from cleargbm exist in the service — all interaction flows through covenant_ml's
`ClassifierBackend` protocol. The Rust transition is transparent to the service as long as
covenant_ml's API surface is preserved.

---

## Testing Strategy

### Rust Tests (Inline with `#[cfg(test)]`)

Tests are inline in each module. Since `unwrap_used = "deny"`, use `assert!` + `if let`:

```rust
// In src/histogram.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_histogram_simple() {
        let result = build_histogram(&indices, &gradients, &hessians, &bins, n_bins);

        assert!(result.is_ok());
        if let Ok(hist) = result {
            // Use slice accessor for bulk check
            let total: f64 = hist.gradient_sums().iter().sum();
            assert!((total - expected).abs() < 1e-10_f64);
        }
    }
}
```

**Property-based testing** with proptest is available via `[dev-dependencies]` but
must also follow the no-unwrap pattern.

### Python Integration Tests

Integration tests will verify Rust backends match Python defaults by:

1. Saving the current (Python default) hook
2. Setting the hook to the Rust adapter
3. Running the same inputs through both and comparing results
4. Restoring the original hook

```python
# tests/test_rust_integration.py (planned — requires maturin build)

"""Integration tests verifying Rust backend matches Python backend."""

from cleargbm import _test_hooks

def test_rust_histogram_matches_python() -> None:
    """Verify Rust histogram produces same results as Python."""
    # Save Python default
    python_backend = _test_hooks._build_histogram_backend

    # Set Rust adapter (from production startup wiring)
    _test_hooks._build_histogram_backend = rust_histogram_adapter

    # ... run same inputs through both, compare results ...

    # Restore
    _test_hooks._build_histogram_backend = python_backend
```

---

## Benchmarking (Phase 9)

Benchmarking will be added in Phase 9 after PyO3 integration is complete.
Will use criterion with safe integer conversions (`f64::from(u32)` instead of `as f64`).

---

## Validation Checklist

### Code Quality (Rust) - Phases 1-7 COMPLETED
- [x] `cargo clippy` passes with all lints at `forbid`
- [x] `cargo fmt --check` passes
- [x] `cargo test --all-features` passes (1122 tests)
- [x] `cargo llvm-cov` + `check_segment_coverage.py` passes (100.00%, 3253/3253 lines)
- [x] No `unsafe` code (`unsafe_code = "forbid"`)
- [x] No `unwrap()`, `expect()`, or `panic!()` (all at `forbid`)
- [x] No `?` operator (`question_mark_used = "forbid"`)
- [x] No `as` casts (`as_conversions = "forbid"`)
- [x] All public items have doc comments (`missing_docs = "forbid"`)
- [x] All error variants documented

### Code Quality (Python Integration) - IN PROGRESS
- [x] `make check` passes in cleargbm (488 tests, 100.00% coverage)
- [x] 100% test coverage maintained (2057 statements, 592 branches)
- [x] No `Any`, `cast()`, or `type: ignore`
- [x] `_test_hooks.py` pattern used for DI (12 hooks, all Protocol-typed)
- [x] All hot-path callers wired through hooks (histogram, predict_tree, sigmoid, sigmoid_array)
- [ ] Rust adapter functions (convert Python types to Rust types)
- [ ] Production startup wiring (set hooks to Rust implementations)
- [ ] `maturin develop` builds and `import cleargbm_rs` works

### Performance - PENDING
- [ ] Benchmark shows 10x+ improvement over Python
- [ ] No regression in existing tests
- [ ] Memory usage reasonable

### Integration - PENDING
- [ ] Python API unchanged (no breaking changes)
- [ ] Rust backend set via hooks at startup (no try/except, no auto-detection)
- [ ] CI builds both Python and Rust

### Downstream Compatibility - PENDING
- [ ] covenant_ml `test_cleargbm_integration.py` passes with zero changes (928-line suite)
- [ ] JSON model serialization format unchanged (encode/decode round-trip)
- [ ] `TrainingProgress` callback interface unchanged
- [ ] `get_feature_importances()` output format unchanged

---

## Implementation Order

1. **Phase 1**: Project setup, Cargo.toml with strict lints, error types ✅ COMPLETED
2. **Phase 2**: Core types (TreeNode, HistogramBuffer, SplitConfig) ✅ COMPLETED
3. **Phase 3**: Histogram building with tests ✅ COMPLETED
4. **Phase 4**: Split finding ✅ COMPLETED
5. **Phase 5**: Tree construction ✅ COMPLETED
6. **Phase 6**: Prediction/inference ✅ COMPLETED
7. **Phase 7**: PyO3 bindings (PyCFunction::new_closure, manual extraction) ✅ COMPLETED
8. **Phase 8**: Python integration with _test_hooks ⏳ IN PROGRESS (hooks wired, adapters pending)
9. **Phase 9**: Final benchmarking and optimization

Each phase must pass `make check` before proceeding.

---

*Last updated: March 2026*
