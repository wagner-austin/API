# ClearGBM

*Gradient Boosting You Can See Through*

From-scratch interpretable gradient boosting with zero external dependencies. Built for transparency and explainability using only Python stdlib.

## Features

- **From Scratch**: Pure Python stdlib implementation - no numpy, sklearn, xgboost, or lightgbm
- **Interpretable by Design**: Rule extraction, contribution breakdown, feature interactions
- **Histogram-Based Optimization**: LightGBM-style O(K) split finding instead of O(n log n) sorting
- **Strict Typing**: 100% typed with TypedDicts, Protocols, no Any/cast/ignore
- **Production Ready**: Plugs into covenant_ml as a ClassifierBackend

## Installation

```bash
poetry add cleargbm
```

## Quick Start

```python
from cleargbm.ensemble import train_gradient_boosting, predict_proba
from cleargbm.explain import explain_prediction, extract_rules
from cleargbm.types import GradientBoostingConfig

# Configure
config: GradientBoostingConfig = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": None,
    "max_bins": 64,  # histogram bins for O(K) split finding
    "subsample": 1.0,
    "random_state": 42,
    "track_contributions": True,
    "monotonic_constraints": None,
    "reg_alpha": 0.0,  # L1 regularization
    "reg_lambda": 0.0,  # L2 regularization
    "n_jobs": 1,  # parallel workers (-1 = all cores)
}

# Train
model = train_gradient_boosting(
    x_train, y_train,
    x_val, y_val,
    config,
    feature_names=("debt_ratio", "coverage", "current_ratio"),
    progress_callback=None,
)

# Predict
proba = predict_proba(model, x_test)

# Explain
explanation = explain_prediction(model, x_test[0])
rules = extract_rules(model, min_samples=10, max_rules=20)
```

## Histogram-Based Split Finding

ClearGBM uses LightGBM-style histogram binning for efficient split finding:

- **O(K) Complexity**: Instead of O(n log n) sorting per split, bins features into K buckets and scans with prefix sums
- **Configurable Bins**: `max_bins` parameter (default: 64) controls granularity vs speed tradeoff
- **Gradient/Hessian Histograms**: Accumulates gradients and hessians per bin for split evaluation
- **Monotonic Constraint Support**: Constraints enforced during single bin scan
- **Precomputed Bins**: Feature bins computed once at training start, reused across all trees
- **Micro-Optimizations**: `map`/`itemgetter` for tuple creation, delayed index allocation

```python
# For faster training on large datasets, reduce max_bins
config["max_bins"] = 32  # Faster but coarser splits

# For more accurate splits on smaller datasets, increase max_bins
config["max_bins"] = 128  # Slower but finer splits
```

## Parallel Training

ClearGBM supports parallel histogram building across CPU cores:

```python
# Use all available cores
config["n_jobs"] = -1

# Use specific number of workers
config["n_jobs"] = 4

# Sequential (default, best for small datasets)
config["n_jobs"] = 1
```

**Implementation Details**:
- Pool reuse: Single `multiprocessing.Pool` across all trees
- Shared memory: Gradients/hessians passed via `SharedMemory` (not pickled per-batch)
- Pool initializer: Feature bins broadcast once at pool creation
- Batched workers: Features grouped to reduce IPC calls from O(n_features) to O(n_jobs)

**When to Use Parallel**:
- `n_jobs=1` is fastest for datasets < 10K samples (pool overhead dominates)
- `n_jobs=-1` helps for datasets ≥ 10K samples with 50+ features
- Use `scripts/autotune.py` to find optimal settings for your hardware

```bash
# Find optimal n_jobs and max_bins for your data
poetry run python -m scripts.autotune --samples 10000 --features 50
```

## Why ClearGBM?

| Feature | XGBoost | LightGBM | ClearGBM |
|---------|---------|----------|----------|
| Speed | Fast | Faster | Moderate |
| Accuracy | Excellent | Excellent | Good |
| Interpretability | Limited | Limited | First-class |
| Dependencies | C++ lib | C++ lib | Python stdlib only |
| Rule extraction | Post-hoc | Post-hoc | Built-in |
| Split algorithm | Exact/Histogram | Histogram | Histogram |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | - | Number of boosting rounds |
| `max_depth` | int | - | Maximum tree depth |
| `learning_rate` | float | - | Shrinkage factor for updates |
| `min_samples_split` | int | - | Minimum samples to split a node |
| `min_samples_leaf` | int | - | Minimum samples in a leaf |
| `max_features` | int \| None | None | Max features per split (None = all) |
| `max_bins` | int | 64 | Histogram bins for split finding |
| `subsample` | float | 1.0 | Row subsampling ratio |
| `random_state` | int | - | Random seed for reproducibility |
| `track_contributions` | bool | - | Store per-tree contributions |
| `monotonic_constraints` | tuple \| None | None | +1=increasing, -1=decreasing, 0=none |
| `reg_alpha` | float | 0.0 | L1 regularization (soft thresholding) |
| `reg_lambda` | float | 0.0 | L2 regularization (leaf shrinkage) |
| `n_jobs` | int | 1 | Parallel workers (-1 = all cores, 1 = sequential) |

## Development

```bash
make lint   # guards + ruff + mypy
make test   # pytest with 100% coverage
make check  # lint + test
```

## Architecture

See [docs/plan.md](docs/plan.md) for detailed architecture and implementation plan.

## Requirements

- Python 3.11+
- No external dependencies (pure Python stdlib)
- 100% test coverage enforced
