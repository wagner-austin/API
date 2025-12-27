# Architecture: cleargbm Library

## Overview

ClearGBM is a high-performance gradient boosting implementation built on numpy with first-class interpretability. Designed for transparency and explainability with strict typing (`disallow_any_expr = true`) and 100% test coverage.

## Dependencies

- `numpy` ^2.3.5 - Array operations, vectorized histogram building

## Directory Structure

```
libs/cleargbm/
├── pyproject.toml
├── README.md
├── Makefile
├── docs/
│   ├── architecture.md      # This file
│   └── hybrid_model.md      # Future roadmap (GOSS, oblivious trees, etc.)
├── scripts/
│   ├── __init__.py
│   ├── guard.py             # Monorepo guard runner
│   ├── benchmark.py         # Performance benchmark suite
│   └── autotune.py          # Grid search for optimal n_jobs/max_bins
├── src/cleargbm/
│   ├── __init__.py          # Public exports
│   ├── types.py             # TypedDicts, Protocols, encode/decode
│   ├── losses.py            # Loss functions with gradients/hessians
│   ├── histogram.py         # LightGBM-style histogram binning
│   ├── buffers.py           # Mutable numpy buffer classes
│   ├── split.py             # Split computation, leaf values, gain
│   ├── parallel.py          # Parallel histogram building with shared memory
│   ├── tree.py              # Decision tree building and prediction
│   ├── ensemble.py          # Gradient boosting ensemble
│   ├── explain.py           # Rule extraction, contributions
│   └── _test_hooks.py       # Internal hooks for DI (private)
└── tests/
    ├── __init__.py
    ├── conftest.py          # Shared fixtures (make_config helper)
    ├── test_types.py
    ├── test_losses.py
    ├── test_histogram.py
    ├── test_buffers.py
    ├── test_split.py
    ├── test_parallel.py
    ├── test_tree.py
    ├── test_ensemble.py
    ├── test_explain.py
    └── test_test_hooks.py
```

## Core Types (types.py)

All structures are immutable TypedDicts with encode/decode functions.

### Array Types

```python
from numpy.typing import NDArray
import numpy as np

# Feature matrix and targets
x: NDArray[np.float64]           # Shape: (n_samples, n_features)
y: NDArray[np.int64]             # Shape: (n_samples,)

# Gradients and hessians
gradients: NDArray[np.float64]   # Shape: (n_samples,)
hessians: NDArray[np.float64]    # Shape: (n_samples,)

# Sample indices for node partitioning
sample_indices: NDArray[np.int64] # Shape: (n_node_samples,)
```

### Key TypedDicts

```python
class TreeNode(TypedDict):
    node_id: int
    is_leaf: bool
    feature_index: int | None
    feature_name: str | None
    threshold: float | None
    nan_direction: Literal["left", "right"] | None
    value: float | None
    n_samples: int
    left_child: int | None
    right_child: int | None

class DecisionTree(TypedDict):
    nodes: tuple[TreeNode, ...]
    max_depth: int
    n_leaves: int
    feature_names: tuple[str, ...]

class GradientBoostingConfig(TypedDict):
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_samples_split: int
    min_samples_leaf: int
    max_features: int | None
    max_bins: int
    subsample: float
    random_state: int
    track_contributions: bool
    monotonic_constraints: tuple[int, ...] | None
    reg_alpha: float   # L1 regularization
    reg_lambda: float  # L2 regularization
    n_jobs: int

class SplitCandidate(TypedDict):
    feature_index: int
    threshold: float
    gain: float
    left_indices: NDArray[np.int64]
    right_indices: NDArray[np.int64]
    nan_direction: Literal["left", "right"]
```

### Protocols

```python
class RandomStateProtocol(Protocol):
    def permutation(self, n: int) -> NDArray[np.int64]: ...
    def choice(self, n: int, size: int, replace: bool) -> NDArray[np.int64]: ...

class LossFunction(Protocol):
    def loss(self, y_true: NDArray[np.int64], y_pred: NDArray[np.float64]) -> float: ...
    def gradients(self, y_true: NDArray[np.int64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def hessians(self, y_true: NDArray[np.int64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def initial_prediction(self, y_true: NDArray[np.int64]) -> float: ...

class WorkerPoolProtocol(Protocol):
    def map_batched(self, func: Callable[..., list[tuple[int, Histogram]]], args_list: list[tuple[...]]) -> list[list[tuple[int, Histogram]]]: ...
    def close(self) -> None: ...
    def join(self) -> None: ...
```

## Key Modules

### 1. buffers.py - Mutable Numpy Buffers

Pre-allocated, reusable buffers to avoid allocation overhead:

```python
class HistogramBuffer:
    """Pre-allocated histogram accumulator using numpy arrays."""
    _grad_sums: NDArray[np.float64]
    _hess_sums: NDArray[np.float64]
    _counts: NDArray[np.int64]

    def accumulate_batch(
        self,
        bin_indices: NDArray[np.int64],
        gradients: NDArray[np.float64],
        hessians: NDArray[np.float64],
    ) -> None:
        """Vectorized accumulation using np.add.at()."""
        np.add.at(self._grad_sums, bin_indices, gradients)
        np.add.at(self._hess_sums, bin_indices, hessians)
        np.add.at(self._counts, bin_indices, 1)

    def subtract_into(self, parent: HistogramBuffer, child: HistogramBuffer) -> None:
        """Sibling subtraction: self = parent - child."""
        np.subtract(parent._grad_sums, child._grad_sums, out=self._grad_sums)
        np.subtract(parent._hess_sums, child._hess_sums, out=self._hess_sums)
        np.subtract(parent._counts, child._counts, out=self._counts)
```

Also includes `FloatBuffer` and `IntBuffer` for general-purpose mutable arrays.

### 2. histogram.py - Histogram-Based Split Finding

LightGBM-style O(K) split finding instead of O(n log n) sorting:

```python
def build_histogram(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    sample_bins: NDArray[np.int64],
    n_bins: int,
) -> Histogram:
    """Build gradient/hessian histogram for a feature."""

def find_best_split_from_histogram(
    histogram: Histogram,
    feature_idx: int,
    g_total: float,
    h_total: float,
    min_samples_leaf: int,
    monotonic_constraint: int,
    reg_lambda: float,
) -> HistogramSplit | None:
    """Find best split by scanning histogram bins."""
```

NaN values get a dedicated bin (`NAN_BIN_OFFSET = 1`). Both NaN-goes-left and NaN-goes-right are evaluated; the direction yielding higher gain is chosen.

### 3. parallel.py - Parallel Histogram Building

Multiprocessing with shared memory optimization:

```python
def _find_best_histogram_split_parallel(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    pool: WorkerPoolProtocol,
    parent_histograms: dict[int, HistogramBuffer] | None,
) -> tuple[SplitCandidate | None, dict[int, HistogramBuffer]]:
    """Parallel split finding with shared memory for gradients/hessians."""
```

**Optimizations:**
1. **Pool reuse** - Single pool across all trees
2. **Batched workers** - Features grouped per IPC call
3. **Pool initializer** - `feature_bins` broadcast once via `_worker_initializer`
4. **Shared memory** - Gradients/hessians via `multiprocessing.shared_memory`

### 4. tree.py - Decision Tree Building

```python
def build_tree(
    x: NDArray[np.float64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    feature_bins: FeatureBins | None = None,
    pool: WorkerPoolProtocol | None = None,
) -> DecisionTree:
    """Build a single decision tree."""

def predict_tree(tree: DecisionTree, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Get predictions from tree for all samples."""

def explain_tree_prediction(tree: DecisionTree, x_single: NDArray[np.float64]) -> TreePredictionExplanation:
    """Explain prediction for a single sample."""
```

**Sibling histogram subtraction:** After partitioning, smaller child's histogram is built directly; larger child's histogram is derived via `sibling = parent - child` (2x speedup on histogram building).

### 5. split.py - Split Computation

```python
def _compute_leaf_value(
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> float:
    """Compute optimal leaf value with regularization.

    L1 (alpha): leaf = -sign(G) * max(|G| - alpha, 0) / (H + lambda)
    L2 (lambda): leaf = -G / (H + lambda)
    """

def _compute_split_gain(
    g_left: float, h_left: float,
    g_right: float, h_right: float,
    g_total: float, h_total: float,
    reg_lambda: float = 0.0,
) -> float:
    """Compute gain with L2 regularization.

    Gain = G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - G^2/(H + lambda)
    """
```

### 6. ensemble.py - Gradient Boosting Training

```python
def train_gradient_boosting(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    progress_callback: Callable[[TrainingProgress], None] | None = None,
) -> GradientBoostingModel:
    """Train gradient boosting classifier."""

def predict_proba(model: GradientBoostingModel, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Predict class probabilities."""
```

## Design Decisions

### Strict Typing with Numpy

To maintain `disallow_any_expr = true` with numpy:

```python
# Scalar extraction (avoids Any from indexing)
value: float = array.item(i)

# Shape access (avoids Any)
n_samples: int = int(array.shape[0])

# Array construction with tuples (avoids Any from list inference)
arr: NDArray[np.int64] = np.array((1, 2, 3), dtype=np.int64)

# 2D access pattern
row: NDArray[np.float64] = x[i, :]
value: float = row.item(j)
```

### Determinism

- Equal-size children: left child's histogram is built (consistent tie-breaking)
- Equal-gain splits: lower feature_index wins
- Parallel histograms: features sorted by index before selection

### Pool Optimization

Shared memory eliminates pickle overhead for gradients/hessians. The autotune script helps find when multiprocessing actually helps:
- `max_bins=32` is fastest across all tested sizes
- `n_jobs=1` wins for datasets < 10K samples (pool overhead dominates)
- Speedup from multiprocessing is modest (1.1-1.3x)

## Testing Strategy (_test_hooks.py)

### Hooks Pattern

Production code calls hooks directly. Production sets real implementations at startup. Tests set fakes.

```python
# _test_hooks.py
_random_state_factory: Callable[[int], RandomStateProtocol] = _default_random_state_factory
_pool_factory: PoolFactoryProtocol = _default_pool_factory

def get_random_state(seed: int) -> RandomStateProtocol:
    return _random_state_factory(seed)

def create_worker_pool(...) -> WorkerPoolProtocol:
    return _pool_factory(...)

def set_random_state_factory(factory: Callable[[int], RandomStateProtocol]) -> None:
    global _random_state_factory
    _random_state_factory = factory
```

### Fake Implementations (in tests)

```python
class FakeRandomState:
    def permutation(self, n: int) -> NDArray[np.int64]:
        return np.arange(n, dtype=np.int64)  # Identity permutation

    def choice(self, n: int, size: int, replace: bool) -> NDArray[np.int64]:
        return np.arange(size, dtype=np.int64) % n

class _FakeSequentialPool:
    """Fake pool that runs workers sequentially for testing."""
    def map_batched(self, func, args_list):
        return [func(*args) for args in args_list]
```

## Public API (__init__.py)

```python
# Training
from cleargbm.ensemble import train_gradient_boosting, predict_proba

# Types
from cleargbm.types import (
    GradientBoostingConfig,
    GradientBoostingModel,
    DecisionTree,
    TreeNode,
    # ... encode/decode functions
)

# Explainability
from cleargbm.explain import explain_prediction, extract_rules
```

## Test Coverage

- 459 tests passing
- 100% statement and branch coverage required
- Tests for each encode/decode pair with round-trip validation
- Tests for parallel equivalence (n_jobs=1 vs n_jobs=2 produce identical results)
- Tests for NaN handling in all paths
- Tests for regularization (L1 soft threshold, L2 shrinkage)
- No mocks - explicit fakes via `_test_hooks`

## Scripts

### benchmark.py

```bash
poetry run python -m scripts.benchmark
poetry run python -m scripts.benchmark --samples 10000 --features 20 --trees 50
```

### autotune.py

```bash
poetry run python -m scripts.autotune
poetry run python -m scripts.autotune --samples 10000 --features 50
```

## Future Roadmap

See [hybrid_model.md](hybrid_model.md) for planned features:
- Early stopping
- Gradient quantization
- Leaf-wise tree growth
- GOSS sampling
- Oblivious trees
