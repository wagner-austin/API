# ClearGBM Implementation Plan

**Tagline:** *Gradient Boosting You Can See Through*

ClearGBM is a from-scratch gradient boosting implementation with first-class interpretability. Zero external dependencies - pure Python stdlib only (no numpy, no ML libraries). Designed to plug into the existing `covenant_ml` backend system.

## 1. Goals

1. **From Scratch**: Implement gradient boosting using only Python stdlib - no numpy, sklearn, xgboost, lightgbm, or torch
2. **Interpretable by Design**: Rule extraction, contribution breakdown, and feature interactions baked into data structures
3. **Production Quality**: Strict typing, 100% test coverage, no shortcuts
4. **Pluggable**: Implement `ClassifierBackend` protocol from `covenant_ml`
5. **Benchmarkable**: Direct comparison against XGBoost/LightGBM on covenant datasets

## 2. Architecture Overview

```
libs/cleargbm/
├── src/cleargbm/
│   ├── __init__.py           # Public API exports
│   ├── types.py              # TypedDicts, Protocols, encode/decode
│   ├── losses.py             # Loss functions with gradients/hessians
│   ├── histogram.py          # LightGBM-style histogram binning for O(K) split finding
│   ├── split.py              # Split computation, leaf values, gain calculation
│   ├── parallel.py           # Parallel histogram building with shared memory
│   ├── tree.py               # Decision tree building and prediction
│   ├── ensemble.py           # Gradient boosting ensemble
│   ├── explain.py            # Rule extraction, contributions
│   ├── _test_hooks.py        # Internal hooks for DI (private)
│   └── backend.py            # ClassifierBackend adapter
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures (make_config helper)
│   ├── test_types.py
│   ├── test_losses.py
│   ├── test_histogram.py
│   ├── test_split.py
│   ├── test_parallel.py
│   ├── test_tree.py
│   ├── test_ensemble.py
│   ├── test_explain.py
│   └── test_backend.py
├── scripts/
│   ├── __init__.py
│   ├── guard.py              # Monorepo guard runner
│   ├── benchmark.py          # Performance benchmark suite
│   └── autotune.py           # Grid search for optimal n_jobs/max_bins
├── docs/
│   ├── plan.md               # This file
│   └── refactor.md           # Optimization roadmap
├── pyproject.toml
├── Makefile
└── README.md
```

## 3. Core Data Structures

### 3.1 Tree Structures

All structures are immutable TypedDicts. No dataclasses.

```python
# types.py

class SplitCondition(TypedDict):
    """A single split condition in a decision tree."""
    feature_index: int
    feature_name: str
    threshold: float
    direction: Literal["left", "right"]  # which way this sample went


class TreeNode(TypedDict):
    """A node in the decision tree."""
    node_id: int
    is_leaf: bool
    # Split info (only for non-leaf nodes)
    feature_index: int | None
    feature_name: str | None
    threshold: float | None
    # Leaf info (only for leaf nodes)
    value: float | None  # prediction value (gradient sum / hessian sum)
    n_samples: int
    # Tree structure
    left_child: int | None   # node_id of left child
    right_child: int | None  # node_id of right child


class DecisionTree(TypedDict):
    """Complete decision tree structure."""
    nodes: tuple[TreeNode, ...]
    max_depth: int
    n_leaves: int
    feature_names: tuple[str, ...]


class TreePredictionExplanation(TypedDict):
    """Explanation for a single tree's prediction."""
    tree_index: int
    prediction: float
    path: tuple[SplitCondition, ...]  # splits traversed to reach leaf
    leaf_node_id: int
    n_samples_in_leaf: int
```

### 3.2 Ensemble Structures

```python
class GradientBoostingConfig(TypedDict):
    """Configuration for gradient boosting training."""
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_samples_split: int
    min_samples_leaf: int
    max_features: int | None  # None = use all features
    max_bins: int  # histogram bins for O(K) split finding (default: 64)
    subsample: float  # row subsampling ratio (1.0 = no subsampling)
    random_state: int
    # Interpretability options
    track_contributions: bool  # store per-tree contributions
    # Monotonicity constraints: +1 = increasing, -1 = decreasing, 0 = none
    monotonic_constraints: tuple[int, ...] | None
    # Regularization
    reg_alpha: float  # L1 regularization (soft thresholding)
    reg_lambda: float  # L2 regularization (leaf shrinkage)
    # Parallelism
    n_jobs: int  # parallel workers (-1 = all cores, 1 = sequential)


class GradientBoostingModel(TypedDict):
    """Trained gradient boosting model."""
    trees: tuple[DecisionTree, ...]
    base_prediction: float  # initial prediction (log-odds for classification)
    learning_rate: float
    feature_names: tuple[str, ...]
    n_classes: int
    config: GradientBoostingConfig


class PredictionExplanation(TypedDict):
    """Full explanation for a gradient boosting prediction."""
    final_probability: float
    base_prediction: float
    tree_contributions: tuple[TreePredictionExplanation, ...]
    top_features: tuple[FeatureContribution, ...]


class FeatureContribution(TypedDict):
    """Contribution of a single feature to the prediction."""
    feature_name: str
    feature_index: int
    total_contribution: float  # sum across all trees
    n_splits: int  # how many times this feature was used in path
```

### 3.3 Training State Structures

```python
class TreeBuildState(TypedDict):
    """Mutable state during tree construction (internal only)."""
    nodes: list[TreeNode]
    next_node_id: int


class TrainingProgress(TypedDict):
    """Progress update during training."""
    tree_index: int
    total_trees: int
    train_loss: float
    val_loss: float | None


class SplitCandidate(TypedDict):
    """A potential split to evaluate."""
    feature_index: int
    threshold: float
    gain: float
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
```

## 4. Module Specifications

### 4.1 types.py

**Purpose**: All TypedDicts, Protocols, encode/decode functions, validation.

**Exports**:
- All TypedDict definitions
- `encode_*` functions for serialization
- `decode_*` functions with `require_*` validation
- `FeatureArray` and `LabelArray` type aliases

**Validation Pattern**:
```python
def require_positive_int(value: int, name: str) -> int:
    """Validate that value is a positive integer.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def decode_gradient_boosting_config(raw: dict[str, object]) -> GradientBoostingConfig:
    """Decode raw dict to GradientBoostingConfig.

    Args:
        raw: Raw dictionary from JSON/config.

    Returns:
        Validated GradientBoostingConfig.

    Raises:
        KeyError: If required key is missing.
        ValueError: If value fails validation.
        TypeError: If value has wrong type.
    """
    n_estimators = require_positive_int(
        _require_int(raw, "n_estimators"), "n_estimators"
    )
    # ... etc
```

### 4.2 losses.py

**Purpose**: Loss functions with first and second derivatives for gradient boosting.

**Exports**:
- `LossFunction` Protocol
- `BinaryLogLoss` - log loss for binary classification
- `sigmoid` - numerically stable sigmoid function
- `sigmoid_array` - sigmoid for arrays
- `compute_raw_predictions` - combine base + tree predictions
- `raw_to_proba` - convert log-odds to probabilities

**Key Functions**:
```python
class LossFunction(Protocol):
    """Protocol for loss functions used in gradient boosting."""

    def loss(
        self,
        y_true: tuple[int, ...],
        y_pred: FloatArray,
    ) -> float:
        """Compute mean loss."""
        ...

    def gradients(
        self,
        y_true: tuple[int, ...],
        y_pred: FloatArray,
    ) -> FloatArray:
        """Compute gradients (first derivative of loss w.r.t. predictions)."""
        ...

    def hessians(
        self,
        y_true: tuple[int, ...],
        y_pred: FloatArray,
    ) -> FloatArray:
        """Compute hessians (second derivative for Newton step)."""
        ...

    def initial_prediction(
        self,
        y_true: tuple[int, ...],
    ) -> float:
        """Compute initial prediction (log-odds for classification)."""
        ...


def sigmoid(x: float) -> float:
    """Compute sigmoid function with numerical stability.

    Args:
        x: Input value (log-odds).

    Returns:
        Probability in [0, 1].
    """
    x_clipped = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x_clipped))
```

**Binary Log Loss Implementation**:
```
loss = -[y * log(p) + (1-y) * log(1-p)]
gradient = p - y  (derivative of loss w.r.t. raw prediction)
hessian = p * (1-p)  (second derivative)
initial = log(p_mean / (1 - p_mean))  (log-odds of positive class rate)
```

### 4.3 tree.py

**Purpose**: Single decision tree implementation with interpretable structure.

**Exports**:
- `build_tree` - construct tree from gradients/hessians
- `predict_tree` - get predictions from tree
- `explain_tree_prediction` - get path explanation for single sample
- `extract_tree_rules` - get human-readable rules

**Algorithm**: LightGBM-style histogram binning with O(K) split finding
- Split criterion: Maximize gain = (G_L^2/H_L + G_R^2/H_R) - (G^2/H)
- Where G = sum of gradients, H = sum of hessians
- This is the standard XGBoost split criterion

**Histogram Optimization** (implemented in `histogram.py`):
- Bin each feature into K cuts once at training start using quantile-based edges
- At each node, build gradient/hessian histograms per feature and scan bins with prefix sums
- O(K) split finding instead of O(n log n) per-node sorting
- `max_bins` parameter controls number of bins (default: 64)
- Monotonic constraints are enforced during the single bin scan
- Precomputed feature bins can be passed to `build_tree` for reuse across trees in ensemble

**Micro-Optimizations** (pure Python, type-safe):
- `map(itemgetter(1), sorted_pairs)` for extracting indices from sorted tuples
- `map(sigmoid, x)` for applying functions to arrays
- `zip(..., strict=True)` with generators for element-wise operations
- Delayed tuple creation: only create left/right indices after best split is found

**Key Functions** (pure Python, no numpy):
```python
def find_best_split(
    x: FloatMatrix,
    gradients: FloatArray,
    hessians: FloatArray,
    sample_indices: tuple[int, ...],
    feature_indices: tuple[int, ...],
    min_samples_leaf: int,
    monotonic_constraint: int,
) -> SplitCandidate | None:
    """Find the best split for current node."""
    ...


def build_tree(
    x: FloatMatrix,
    gradients: FloatArray,
    hessians: FloatArray,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
) -> DecisionTree:
    """Build a single decision tree."""
    ...


def predict_tree(
    tree: DecisionTree,
    x: FloatMatrix,
) -> FloatArray:
    """Get predictions from tree for all samples."""
    ...


def explain_tree_prediction(
    tree: DecisionTree,
    x_single: FloatArray,
) -> TreePredictionExplanation:
    """Explain prediction for a single sample."""
    ...
```

### 4.4 ensemble.py

**Purpose**: Gradient boosting ensemble training and prediction.

**Exports**:
- `train_gradient_boosting` - train full ensemble
- `predict_proba` - get probability predictions
- `predict_raw` - get raw scores (log-odds)

**Algorithm**:
1. Initialize with base prediction (log-odds of positive class rate)
2. For each iteration:
   a. Compute current predictions
   b. Compute gradients and hessians
   c. Build tree to fit negative gradients
   d. Update predictions: pred += learning_rate * tree_pred
3. Return ensemble of trees

**Key Functions** (pure Python, no numpy):
```python
def train_gradient_boosting(
    x_train: FloatMatrix,
    y_train: tuple[int, ...],
    x_val: FloatMatrix | None,
    y_val: tuple[int, ...] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    progress_callback: Callable[[TrainingProgress], None] | None,
) -> GradientBoostingModel:
    """Train gradient boosting classifier."""
    ...


def predict_proba(
    model: GradientBoostingModel,
    x: FloatMatrix,
) -> tuple[tuple[float, float], ...]:
    """Predict class probabilities.

    Args:
        model: Trained model.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Probabilities array (n_samples, 2) for binary classification.
    """
    ...
```

### 4.5 explain.py

**Purpose**: Interpretability features - rule extraction, contribution breakdown.

**Exports**:
- `explain_prediction` - full explanation for single sample
- `extract_rules` - human-readable rules from model
- `get_feature_interactions` - detect feature co-occurrences
- `get_feature_importances` - aggregate importance scores

**Key Functions**:
```python
def explain_prediction(
    model: GradientBoostingModel,
    x_single: NDArray[np.float64],
) -> PredictionExplanation:
    """Generate full explanation for a single prediction.

    Args:
        model: Trained model.
        x_single: Single sample feature vector.

    Returns:
        Complete explanation with contributions from each tree.
    """
    ...


def extract_rules(
    model: GradientBoostingModel,
    min_samples: int = 10,
    max_rules: int = 20,
) -> tuple[Rule, ...]:
    """Extract human-readable rules from model.

    Finds the most common/important decision paths and converts
    them to readable rule format.

    Args:
        model: Trained model.
        min_samples: Minimum samples a rule must cover.
        max_rules: Maximum number of rules to return.

    Returns:
        Tuple of extracted rules, sorted by importance.
    """
    ...


class Rule(TypedDict):
    """Human-readable decision rule."""
    conditions: tuple[str, ...]  # ["debt_ratio > 2.5", "coverage < 1.2"]
    prediction_contribution: float
    n_samples: int
    importance: float
```

### 4.6 _test_hooks.py

**Purpose**: Dependency injection for testing. Private module (underscore prefix).

**Pattern**: Production code calls hooks directly. Production sets real implementations at startup. Tests set fakes.

```python
"""Internal hooks for dependency injection.

Production code sets hooks to real implementations at startup.
Tests set hooks to fakes before running.

This module is private (underscore prefix) - not for external use.
Built from scratch - uses only Python stdlib (no numpy).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol


class RandomStateProtocol(Protocol):
    """Protocol for random number generation."""

    def permutation(self, n: int) -> tuple[int, ...]:
        """Return random permutation of integers 0 to n-1."""
        ...

    def choice(
        self,
        n: int,
        size: int,
        replace: bool,
    ) -> tuple[int, ...]:
        """Return random sample of integers."""
        ...

    def rand_1d(self, size: int) -> tuple[float, ...]:
        """Return 1D tuple of random floats in [0, 1)."""
        ...

    def rand_2d(self, rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
        """Return 2D tuple of random floats in [0, 1)."""
        ...


class WorkerPoolProtocol(Protocol):
    """Protocol for worker pool with initialized feature bins.

    The pool is constructed with a feature-bins initializer so workers can read
    sample bin assignments from a module-global without receiving them via IPC.
    Gradients and hessians are passed via shared memory names in args.
    """

    def map_batched(
        self,
        func: Callable[..., list[tuple[int, Histogram]]],
        args_list: list[tuple[...]],  # includes shared memory names
    ) -> list[list[tuple[int, Histogram]]]:
        """Apply batched histogram worker to each batch."""
        ...

    def close(self) -> None:
        """Prevent any more tasks from being submitted."""
        ...

    def join(self) -> None:
        """Wait for worker processes to exit."""
        ...


# Default implementations (set at module load, overridden by tests)
_random_state_factory: Callable[[int], RandomStateProtocol] = (
    lambda seed: _PythonRandomStateWrapper(seed)
)


def get_random_state(seed: int) -> RandomStateProtocol:
    """Get random state instance."""
    return _random_state_factory(seed)


def create_worker_pool(
    n_workers: int,
    bin_edges: tuple[tuple[float, ...], ...],
    sample_bins: tuple[tuple[int, ...], ...],
) -> WorkerPoolProtocol:
    """Create a worker pool with feature_bins initialized in workers."""
    return _pool_factory(n_workers, bin_edges, sample_bins)
```

### 4.7 parallel.py

**Purpose**: Parallel histogram building with shared memory optimization.

**Exports**:
- `_find_best_histogram_split_parallel` - parallel split finding using worker pool
- `_find_best_histogram_split_sequential` - sequential split finding
- `_build_histogram_worker_batched` - top-level worker for Windows pickle compatibility
- `_worker_initializer` - sets up `_WORKER_FEATURE_BINS` global in worker processes

**Implementation**:
- Workers read `feature_bins` (bin edges + sample bins) from module-global `_WORKER_FEATURE_BINS`
- Gradients/hessians passed via `multiprocessing.shared_memory.SharedMemory` names (not pickled)
- `struct.pack_into` / `struct.unpack` for reading/writing floats to shared memory buffers
- Features batched by worker count to reduce IPC calls from O(n_features) to O(n_jobs)

```python
def _find_best_histogram_split_parallel(
    sample_indices: tuple[int, ...],
    feature_indices: tuple[int, ...],
    gradients: FloatArray,
    hessians: FloatArray,
    feature_bins: FeatureBins,
    config: GradientBoostingConfig,
    pool: WorkerPoolProtocol,
    parent_histograms: tuple[Histogram, ...] | None,
) -> SplitCandidate | None:
    """Find best split using parallel histogram building with shared memory."""
    # Create shared memory for gradients and hessians
    n_samples = len(gradients)
    shm_grad = shared_memory.SharedMemory(create=True, size=n_samples * 8)
    shm_hess = shared_memory.SharedMemory(create=True, size=n_samples * 8)
    try:
        # Write gradients/hessians to shared memory
        for i in range(n_samples):
            struct.pack_into("d", shm_grad.buf, i * 8, gradients[i])
            struct.pack_into("d", shm_hess.buf, i * 8, hessians[i])

        # Build batched args with shared memory names
        batched_args = _build_batched_args(
            feature_indices, sample_indices,
            shm_grad.name, shm_hess.name, n_samples, ...
        )

        # Workers read from shared memory by name
        batch_results = pool.map_batched(_build_histogram_worker_batched, batched_args)
        ...
    finally:
        # Clean up shared memory
        shm_grad.close()
        shm_hess.close()
        shm_grad.unlink()
        shm_hess.unlink()
```

### 4.8 backend.py

**Purpose**: Adapter to integrate with `covenant_ml.backends.protocol.ClassifierBackend`.

**Exports**:
- `ClearGBMBackend` - implements `ClassifierBackend` protocol
- `create_cleargbm_backend` - factory function

```python
class ClearGBMBackend:
    """ClearGBM backend implementing ClassifierBackend protocol."""

    def backend_name(self) -> BackendName:
        """Return backend identifier."""
        return "cleargbm"

    def capabilities(self) -> BackendCapabilities:
        """Return backend capabilities."""
        return {
            "supports_train": True,
            "supports_gpu": False,  # CPU only (numpy)
            "supports_early_stopping": True,
            "supports_feature_importance": True,
            "model_format": "json",
        }

    # ... implement full ClassifierBackend protocol
```

## 5. Serialization

### 5.1 Model Format

Models serialize to JSON for transparency and debuggability.

```python
def encode_model(model: GradientBoostingModel) -> dict[str, object]:
    """Encode model to JSON-serializable dict.

    Args:
        model: Trained model.

    Returns:
        Dictionary suitable for json.dumps().
    """
    ...


def decode_model(raw: dict[str, object]) -> GradientBoostingModel:
    """Decode model from JSON dict.

    Args:
        raw: Dictionary from json.loads().

    Returns:
        Validated model.

    Raises:
        KeyError: If required key missing.
        ValueError: If validation fails.
        TypeError: If type is wrong.
    """
    ...
```

### 5.2 Validation Functions

Every decode function has corresponding `require_*` validators:

```python
def _require_str(raw: dict[str, object], key: str) -> str:
    """Extract and validate string from raw dict."""
    value = raw[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be str, got {type(value).__name__}")
    return value


def _require_int(raw: dict[str, object], key: str) -> int:
    """Extract and validate int from raw dict."""
    value = raw[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int, got {type(value).__name__}")
    return value


def _require_float(raw: dict[str, object], key: str) -> float:
    """Extract and validate float from raw dict."""
    value = raw[key]
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise TypeError(f"{key} must be float, got {type(value).__name__}")
    return value
```

## 6. Testing Strategy

### 6.1 Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures, fake random state
├── test_types.py            # TypedDict encode/decode round-trips
├── test_losses.py           # Loss, gradient, hessian correctness
├── test_tree.py             # Single tree building and prediction
├── test_ensemble.py         # Full gradient boosting training
├── test_explain.py          # Interpretability features
├── test_backend.py          # ClassifierBackend integration
└── test_integration.py      # End-to-end with real data
```

### 6.2 Test Principles

1. **No mocks** - Use real implementations or explicit fakes via `_test_hooks`
2. **No weak assertions** - No `is not None`, `isinstance`, `hasattr`, `len > 0`
3. **Test actual behavior** - Assert on specific values, not just types
4. **100% coverage** - Statements and branches
5. **Deterministic** - Fixed seeds, no flaky tests

### 6.3 Example Tests

```python
# test_losses.py

def test_binary_log_loss_gradient_at_perfect_prediction() -> None:
    """Gradient should be zero when prediction equals label."""
    loss_fn = BinaryLogLoss()
    y_true = np.array([1, 0, 1], dtype=np.int64)
    y_pred = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    gradients = loss_fn.gradients(y_true, y_pred)

    np.testing.assert_array_almost_equal(
        gradients,
        np.array([0.0, 0.0, 0.0]),
        decimal=6,
    )


def test_binary_log_loss_gradient_direction() -> None:
    """Gradient should be positive when over-predicting, negative when under."""
    loss_fn = BinaryLogLoss()
    # Label is 0, prediction is 0.8 -> gradient should be positive
    y_true = np.array([0], dtype=np.int64)
    y_pred = np.array([0.8], dtype=np.float64)

    gradients = loss_fn.gradients(y_true, y_pred)

    assert gradients[0] > 0.0, "Gradient should be positive when over-predicting"


# test_tree.py

def test_build_tree_with_perfect_split() -> None:
    """Tree should find perfect split when one exists."""
    # Feature 0 perfectly separates classes
    x = np.array([
        [0.0, 1.0],
        [0.0, 2.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ], dtype=np.float64)

    # Gradients: negative for class 1, positive for class 0
    gradients = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64)
    hessians = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    config: GradientBoostingConfig = {
        "n_estimators": 1,
        "max_depth": 1,
        "learning_rate": 1.0,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "subsample": 1.0,
        "random_state": 42,
        "track_contributions": True,
        "monotonic_constraints": None,
    }

    tree = build_tree(
        x, gradients, hessians, config,
        feature_names=("feature_0", "feature_1"),
    )

    # Should split on feature 0 at threshold 0.5
    root = tree["nodes"][0]
    assert root["feature_index"] == 0
    assert root["threshold"] is not None
    assert 0.0 < root["threshold"] < 1.0
```

### 6.4 Fake Random State for Testing

```python
# conftest.py

class FakeRandomState:
    """Deterministic fake for RandomStateProtocol."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._counter = 0

    def permutation(self, n: int) -> NDArray[np.int64]:
        """Return identity permutation (no shuffling)."""
        return np.arange(n, dtype=np.int64)

    def choice(
        self,
        n: int,
        size: int,
        replace: bool,
    ) -> NDArray[np.int64]:
        """Return first `size` integers."""
        return np.arange(size, dtype=np.int64) % n


@pytest.fixture(autouse=True)
def setup_test_hooks() -> Generator[None, None, None]:
    """Set up fake hooks for all tests."""
    from cleargbm._test_hooks import set_random_state_factory

    original = cleargbm._test_hooks._random_state_factory
    set_random_state_factory(lambda seed: FakeRandomState(seed))
    yield
    set_random_state_factory(original)
```

## 7. Integration with covenant_ml

### 7.1 Backend Registration

Add ClearGBM to the backend registry in `covenant_ml`:

```python
# In covenant_ml/backends/registry.py (future integration)

from cleargbm.backend import create_cleargbm_backend

BACKENDS: dict[BackendName, Callable[[], ClassifierBackend]] = {
    "xgboost": create_xgboost_backend,
    "lightgbm": create_lightgbm_backend,
    "mlp": create_mlp_backend,
    "lstm": create_lstm_backend,
    "cleargbm": create_cleargbm_backend,  # NEW
}
```

### 7.2 Config Type

Add ClearGBM config to the union type:

```python
# In covenant_ml/types.py (future integration)

class ClearGBMConfig(TypedDict):
    """Configuration for ClearGBM backend."""
    backend: Literal["cleargbm"]
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_samples_split: int
    min_samples_leaf: int
    subsample: float
    random_state: int
    # Split ratios
    train_ratio: float
    val_ratio: float
    test_ratio: float


ClassifierTrainConfig = (
    XGBoostConfig
    | LightGBMConfig
    | MLPConfig
    | LSTMConfig
    | ClearGBMConfig  # NEW
)
```

## 8. Benchmarking Plan

### 8.1 Metrics to Compare

1. **AUC-ROC** - Primary metric for covenant prediction
2. **Training time** - Wall clock time to train
3. **Inference time** - Time per prediction
4. **Model size** - Serialized model bytes
5. **Interpretability** - Qualitative comparison of explanations

### 8.2 Benchmark Script

```python
# benchmarks/vs_xgboost.py

def run_benchmark(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
    feature_names: tuple[str, ...],
) -> BenchmarkResult:
    """Run head-to-head benchmark.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_test: Test features.
        y_test: Test labels.
        feature_names: Feature names.

    Returns:
        Benchmark results for both models.
    """
    # Train ClearGBM
    cleargbm_start = time.perf_counter()
    cleargbm_model = train_gradient_boosting(...)
    cleargbm_train_time = time.perf_counter() - cleargbm_start

    # Train XGBoost
    xgb_start = time.perf_counter()
    xgb_model = xgb.train(...)
    xgb_train_time = time.perf_counter() - xgb_start

    # Compare predictions, AUC, etc.
    ...
```

## 9. Implementation Order

1. **types.py** - Foundation, must be complete first
2. **losses.py** - Simple, self-contained
3. **_test_hooks.py** - Needed before tree.py
4. **tree.py** - Core algorithm, depends on losses
5. **ensemble.py** - Depends on tree
6. **explain.py** - Depends on ensemble
7. **backend.py** - Integration layer, depends on all above
8. **Tests** - In parallel with each module

## 10. Dependencies

### 10.1 Runtime Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.11"
# No external dependencies - built from scratch using only Python stdlib
```

That's it. Zero external runtime dependencies.

### 10.2 Dev Dependencies

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^9.0.0"
pytest-cov = "^7.0.0"
pytest-xdist = "^3.6.1"
mypy = "^1.13.0"
ruff = "^0.14.4"
```

### 10.3 Optional Integration Dependencies

For benchmarking (not required for core lib):

```toml
[tool.poetry.group.benchmark.dependencies]
xgboost = "^2.0"
covenant-ml = { path = "../covenant_ml", develop = true }
```

## 11. Success Criteria

1. **Correctness**: Gradient boosting produces reasonable predictions
2. **Performance**: Within 2x of XGBoost training time on small datasets
3. **Accuracy**: Within 0.02 AUC of XGBoost on covenant dataset
4. **Interpretability**: Can extract human-readable rules
5. **Integration**: Works as drop-in backend for covenant_ml
6. **Quality**: 100% test coverage, strict typing, no shortcuts
