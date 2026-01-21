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
        ├── types.rs             # TreeNode, HistogramBuffer, SplitConfig (with inline tests)
        ├── histogram.rs         # build_histogram, subtract_histogram (with inline tests)
        ├── split.rs             # Split finding: find_best_split_from_histogram, find_best_split_across_features (with inline tests)
        └── tree.rs              # Tree construction: build_tree, TreeBuildConfig, Tree (with inline tests)
```

**Note:** Tests are inline in each module (`#[cfg(test)] mod tests { ... }`) following Rust convention.
No separate tests/ or benches/ directories - keeps tests co-located with implementation.

---

## Progress Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Setup and Error Types | **COMPLETED** |
| 2 | Core Types (TreeNode, HistogramBuffer) | **COMPLETED** |
| 3 | Histogram Building | **COMPLETED** |
| 4 | Split Finding | **COMPLETED** |
| 5 | Tree Construction | **COMPLETED** |
| 6 | Prediction/Inference | PENDING |
| 7 | PyO3 Bindings | PENDING |
| 8 | Python Integration | PENDING |
| 9 | Benchmarking and Optimization | PENDING |

**Current Status:** `make check` passes (103 tests, clippy clean, fmt clean).

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

## Phase 6: Prediction/Inference (PENDING)

To be implemented following same patterns (no type alias, explicit Result types).

---

## Phase 7: PyO3 Bindings (PENDING)

**Note:** PyO3 bindings will need to use `usize::try_from()` instead of `as usize`
to comply with our `as_conversions = "deny"` lint. Example pattern:

```rust
// Convert i64 to usize safely (no `as` casts)
let indices: std::result::Result<Vec<usize>, _> = sample_indices
    .as_slice()?
    .iter()
    .map(|&x| usize::try_from(x).map_err(|_| ClearGbmError::IntegerConversion {
        context: "i64 to usize".to_string(),
    }))
    .collect();

// Use slice accessors instead of unwrap_or
let grad_sums: Vec<f64> = result.gradient_sums().to_vec();
let hess_sums: Vec<f64> = result.hessian_sums().to_vec();
let counts: Vec<usize> = result.counts().to_vec();
```

---

## Phase 8: Python Integration (PENDING)

```python
# cleargbm/histogram.py (updated to use Rust when available)

"""Histogram building for gradient boosting.

Uses Rust implementation when available, falls back to Python otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cleargbm._test_hooks import histogram_backend

if TYPE_CHECKING:
    from cleargbm.types import HistogramResult


def build_histogram(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    bins: NDArray[np.int64],
    n_bins: int,
) -> HistogramResult:
    """Build gradient/hessian histogram for split finding.

    Args:
        sample_indices: Indices of samples at this node.
        gradients: Gradient values for all samples.
        hessians: Hessian values for all samples.
        bins: Pre-computed bin assignments.
        n_bins: Number of histogram bins.

    Returns:
        HistogramResult with gradient_sums, hessian_sums, counts.
    """
    return histogram_backend(sample_indices, gradients, hessians, bins, n_bins)
```

```python
# cleargbm/_test_hooks.py

"""Hooks for dependency injection in cleargbm.

Production sets hooks to Rust implementations at startup.
Tests set hooks to Python fakes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm.types import HistogramResult


class HistogramBackendProtocol(Protocol):
    """Protocol for histogram building backend."""

    def __call__(
        self,
        sample_indices: NDArray[np.int64],
        gradients: NDArray[np.float64],
        hessians: NDArray[np.float64],
        bins: NDArray[np.int64],
        n_bins: int,
    ) -> HistogramResult:
        """Build histogram from sample data."""
        ...


def _python_histogram_backend(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    bins: NDArray[np.int64],
    n_bins: int,
) -> HistogramResult:
    """Pure Python implementation (fallback/testing)."""
    # Existing Python implementation
    ...


def _rust_histogram_backend(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    bins: NDArray[np.int64],
    n_bins: int,
) -> HistogramResult:
    """Rust implementation via PyO3."""
    from cleargbm_rs import build_histogram_rs

    grad_sums, hess_sums, counts = build_histogram_rs(
        sample_indices, gradients, hessians, bins, n_bins
    )
    return HistogramResult(
        gradient_sums=grad_sums,
        hessian_sums=hess_sums,
        counts=counts,
    )


def _get_default_backend() -> HistogramBackendProtocol:
    """Get default backend (Rust if available, else Python)."""
    try:
        import cleargbm_rs  # noqa: F401
        return _rust_histogram_backend
    except ImportError:
        return _python_histogram_backend


# Production code calls this directly
# Tests override: _test_hooks.histogram_backend = fake_backend
histogram_backend: HistogramBackendProtocol = _get_default_backend()
```

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

```python
# tests/test_rust_integration.py

"""Integration tests verifying Rust backend matches Python backend."""

import numpy as np
import pytest

from cleargbm import _test_hooks
from cleargbm._test_hooks import (
    _python_histogram_backend,
    _rust_histogram_backend,
)


def test_rust_matches_python() -> None:
    """Verify Rust implementation produces same results as Python."""
    rng = np.random.default_rng(42)
    n_samples = 10000
    n_bins = 64

    sample_indices = np.arange(n_samples, dtype=np.int64)
    gradients = rng.standard_normal(n_samples)
    hessians = np.ones(n_samples, dtype=np.float64)
    bins = rng.integers(0, n_bins, size=n_samples, dtype=np.int64)

    python_result = _python_histogram_backend(
        sample_indices, gradients, hessians, bins, n_bins
    )
    rust_result = _rust_histogram_backend(
        sample_indices, gradients, hessians, bins, n_bins
    )

    np.testing.assert_allclose(
        python_result.gradient_sums,
        rust_result.gradient_sums,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        python_result.hessian_sums,
        rust_result.hessian_sums,
        rtol=1e-10,
    )
    np.testing.assert_array_equal(
        python_result.counts,
        rust_result.counts,
    )
```

---

## Benchmarking (Phase 9)

Benchmarking will be added in Phase 9 after PyO3 integration is complete.
Will use criterion with safe integer conversions (`f64::from(u32)` instead of `as f64`).

---

## Validation Checklist

### Code Quality (Rust) - Phases 1-5 COMPLETED
- [x] `cargo clippy` passes with all lints enabled
- [x] `cargo fmt --check` passes
- [x] `cargo test` passes (103 tests)
- [ ] `cargo tarpaulin --fail-under 100` passes (future)
- [x] No `unsafe` code (`unsafe_code = "forbid"`)
- [x] No `unwrap()`, `expect()`, or `panic!()` (`unwrap_used = "deny"`)
- [x] All public items have doc comments (`missing_docs = "deny"`)
- [x] All error variants documented

### Code Quality (Python Integration) - PENDING
- [ ] `make check` passes in cleargbm
- [ ] 100% test coverage maintained
- [ ] No `Any`, `cast()`, or `type: ignore`
- [ ] `_test_hooks.py` pattern used for DI

### Performance - PENDING
- [ ] Benchmark shows 10x+ improvement over Python
- [ ] No regression in existing tests
- [ ] Memory usage reasonable

### Integration - PENDING
- [ ] Python API unchanged (no breaking changes)
- [ ] Rust backend optional (graceful fallback)
- [ ] CI builds both Python and Rust

---

## Implementation Order

1. **Phase 1**: Project setup, Cargo.toml with strict lints, error types ✅ COMPLETED
2. **Phase 2**: Core types (TreeNode, HistogramBuffer, SplitConfig) ✅ COMPLETED
3. **Phase 3**: Histogram building with tests ✅ COMPLETED
4. **Phase 4**: Split finding ✅ COMPLETED
5. **Phase 5**: Tree construction ✅ COMPLETED
6. **Phase 6**: Prediction/inference ⏳ NEXT
7. **Phase 7**: PyO3 bindings
8. **Phase 8**: Python integration with _test_hooks
9. **Phase 9**: Final benchmarking and optimization

Each phase must pass `make check` before proceeding.

---

*Last updated: January 2026*
