# cleargbm_rs

High-performance Rust core for ClearGBM gradient boosting.

## Overview

This crate provides Rust implementations of ClearGBM's performance-critical algorithms:

- **Histogram building** - O(n) gradient/hessian accumulation
- **Split finding** - O(K) scan over histogram bins
- **Tree construction** - Recursive tree building
- **Prediction** - Fast tree traversal

## Installation

### From source (development)

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and install
cd libs/cleargbm_rs
maturin develop --release
```

### Python usage

```python
from cleargbm_rs import build_histogram_rs

# Build histogram from sample data
grad_sums, hess_sums, counts = build_histogram_rs(
    sample_indices,
    gradients,
    hessians,
    bins,
    n_bins,
)
```

## Development

### Setup

```bash
make setup
```

### Running checks

```bash
# Full check (Rust + Python lint and test)
make check

# Rust only
make rust-lint
make rust-test

# Python only
make python-lint
make python-test
```

## Code Standards

### Rust

- `#![forbid(unsafe_code)]` - No unsafe code
- `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` - No panics
- `#![deny(clippy::as_conversions)]` - No unsafe casts
- All errors via `Result<T, ClearGbmError>`
- 100% test coverage

### Python

- mypy strict mode with `disallow_any_*`
- ruff with banned `typing.Any` and `typing.cast`
- 100% test coverage
- `_test_hooks.py` pattern for dependency injection

## License

MIT
