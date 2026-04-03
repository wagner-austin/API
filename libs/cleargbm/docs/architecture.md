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
│   ├── architecture.md              # This file
│   └── rust-core-transition-plan.md # Rust core transition plan (phases 1-9)
├── scripts/
│   ├── __init__.py
│   ├── _test_hooks.py       # Guard hook protocols for testability
│   ├── guard.py             # Monorepo guard runner
│   ├── benchmark.py         # Performance benchmark suite
│   └── autotune.py          # Grid search for optimal n_jobs/max_bins
├── src/cleargbm/
│   ├── __init__.py          # Public exports
│   ├── types.py             # Re-export layer for all TypedDicts/encode/decode
│   ├── _types_json.py       # JSON aliases, validators, extractors
│   ├── _types_tree.py       # BinEdges, FeatureBins, tree structures, SplitCandidate
│   ├── _types_model.py      # GradientBoostingConfig, GradientBoostingModel, TrainingProgress
│   ├── _types_explain.py    # FeatureContribution, PredictionExplanation, Rule
│   ├── _types_tuning.py     # TimingResult, TuningReport
│   ├── _types_buffer.py     # FloatBufferData, IntBufferData, HistogramBufferData
│   ├── losses.py            # Loss functions with gradients/hessians
│   ├── histogram.py         # LightGBM-style histogram binning
│   ├── buffers.py           # Mutable numpy buffer classes
│   ├── split.py             # Split computation, leaf values, gain
│   ├── parallel.py          # Parallel histogram building with shared memory
│   ├── tree.py              # Decision tree building and prediction
│   ├── ensemble.py          # Gradient boosting ensemble + native training API
│   ├── explain.py           # Rule extraction, contributions
│   ├── _test_hooks.py       # Re-export layer for all hook protocols/accessors
│   ├── _hooks_infra.py      # Hooks: random state, worker pool, buffer factories
│   ├── _hooks_histogram.py  # Hooks: build_histogram, subtract_histogram
│   ├── _hooks_prediction.py # Hooks: predict_tree (single-tree traversal)
│   ├── _hooks_sigmoid.py    # Hooks: sigmoid (scalar + array)
│   ├── _hooks_loss.py       # Hooks: binary_log_loss, gradients, hessians, initial
│   ├── _hooks_binning.py    # Hooks: precompute_feature_bins
│   ├── _hooks_ensemble.py   # Hooks: predict_raw_ensemble, predict_proba_from_raw
│   ├── _hooks_compute.py    # Re-export layer for all compute hooks
│   ├── _hooks_native.py     # Hooks: native (Rust full-loop) training + model predict
│   ├── _hooks_guard.py      # Hooks: guard script protocols
│   ├── _rust_adapters.py    # Per-operation Rust adapters + use_rust_backend()
│   └── _rust_native_adapters.py # Full-loop Rust training + model predict adapters
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures (make_config helper)
    ├── test_types.py
    ├── test_losses.py
    ├── test_histogram.py
    ├── test_buffers.py
    ├── test_split.py
    ├── test_parallel.py
    ├── test_tree.py
    ├── test_ensemble.py
    ├── test_explain.py
    ├── test_test_hooks.py
    ├── test_rust_adapters.py
    └── test_rust_native_adapters.py
```

## Core Types (types.py — re-export layer)

All structures are immutable TypedDicts with encode/decode functions. The
`types.py` module is a thin re-export layer; actual definitions live in focused
sub-modules:

- `_types_json.py` — JSON aliases (`JSONValue`, `JSONDict`), `JSONTypeError`,
  validation helpers (`require_positive_int`, etc.), raw dict extractors
- `_types_tree.py` — `BinEdges`, `FeatureBins`, `SplitCondition`, `TreeNode`,
  `DecisionTree`, `TreePredictionExplanation`, `SplitCandidate`
- `_types_model.py` — `GradientBoostingConfig`, `GradientBoostingModel`,
  `TrainingProgress`
- `_types_explain.py` — `FeatureContribution`, `PredictionExplanation`, `Rule`
- `_types_tuning.py` — `TimingResult`, `TuningReport`
- `_types_buffer.py` — `FloatBufferData`, `IntBufferData`, `HistogramBufferData`

No mutable module-level variables exist in these modules, so re-export via
`from module import name` is safe. Consumers import from `cleargbm.types`.

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
    early_stopping_rounds: int | None  # None = disabled

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
    """Train gradient boosting classifier (Python orchestration)."""

def train_gradient_boosting_native(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
) -> NativeModel:
    """Train via Rust full training loop (single native call)."""

def predict_proba(model: GradientBoostingModel, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Predict class probabilities (Python model)."""

def predict_proba_native(model: NativeModel, x: NDArray[np.float64]) -> tuple[tuple[float, float], ...]:
    """Predict class probabilities (Rust native model)."""
```

**Two training paths:**
- `train_gradient_boosting` — Python orchestrates the training loop; individual operations (histogram, sigmoid, loss) delegate to active hooks (Python or Rust). Compatible with per-tree progress callbacks.
- `train_gradient_boosting_native` — Entire training loop runs in a single Rust call via `cleargbm_rs.train_gradient_boosting_rs`. Maximum speed, no per-iteration FFI overhead.

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

## Testing Strategy (hooks pattern)

### Hooks Architecture

Hook definitions are split across focused sub-modules by concern:

- `_hooks_infra.py` — random state, worker pool, buffer factories
- `_hooks_histogram.py` — build_histogram, subtract_histogram
- `_hooks_prediction.py` — predict_tree (single-tree traversal)
- `_hooks_sigmoid.py` — sigmoid (scalar + array)
- `_hooks_loss.py` — binary_log_loss, gradients, hessians, initial_prediction
- `_hooks_binning.py` — precompute_feature_bins
- `_hooks_ensemble.py` — predict_raw_ensemble, predict_proba_from_raw
- `_hooks_native.py` — train_native, predict_raw_native, predict_proba_native
- `_hooks_guard.py` — guard script hook protocols

**Re-export layers:**
- `_hooks_compute.py` — re-exports all compute hook protocols + public functions
- `_test_hooks.py` — re-exports all hook protocols/accessors from all sub-modules

Mutable hook variables (`_*_backend`) live in their originating sub-module and
must be referenced there directly. Re-export modules only export immutable names
(Protocol classes, public delegator functions, default implementations).

### Rust Backend Wiring

```python
# _hooks_histogram.py (sub-module owns the mutable hook)
_build_histogram_backend: BuildHistogramBackend = _default_build_histogram

def build_histogram(...) -> HistogramBuffer:
    return _build_histogram_backend(...)

# _rust_adapters.py (sets hooks on sub-modules directly)
def use_rust_backend() -> None:
    from cleargbm import _hooks_histogram, _hooks_sigmoid, ...
    _hooks_histogram._build_histogram_backend = _rust_build_histogram
    # ... all 12 per-operation hooks
    wire_native_hooks()  # sets 3 native training hooks

def use_python_backend() -> None:
    from cleargbm import _hooks_histogram, _hooks_sigmoid, ...
    _hooks_histogram._build_histogram_backend = _hooks_histogram._default_build_histogram
    # ... all 12 per-operation hooks
    unwire_native_hooks()  # clears 3 native training hooks
```

**Important:** Code that reads or writes mutable hook variables must reference
the sub-module directly (e.g. `_hooks_histogram`, `_hooks_sigmoid`), not the
re-export layers, because `from x import _mutable_var` creates a local binding
copy that doesn't track mutations on the source module.

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

- 549 tests passing
- 100% statement and branch coverage required (2444 statements, 608 branches)
- Tests for each encode/decode pair with round-trip validation
- Tests for parallel equivalence (n_jobs=1 vs n_jobs=2 produce identical results)
- Tests for NaN handling in all paths
- Tests for regularization (L1 soft threshold, L2 shrinkage)
- Tests for Rust adapter equivalence (Rust vs Python produce identical results)
- Tests for native Rust training loop (train, predict_raw, predict_proba)
- Tests for re-export layer identity (re-exported names are same objects)
- No mocks - explicit fakes via hooks

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

See [rust-core-transition-plan.md](rust-core-transition-plan.md) for the Rust core transition (phases 1-9).

Planned algorithm features (not yet implemented):
- Gradient quantization
- Leaf-wise tree growth
- GOSS sampling
- Oblivious trees
- Ordered boosting
