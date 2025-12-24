# ClearGBM v2: Hybrid Model Architecture

A high-performance gradient boosting implementation combining the best features from XGBoost, LightGBM, and CatBoost. All changes maintain pure Python stdlib, strict typing (no Any/cast/ignore), and 100% test coverage.

## Design Principles

1. **Strict Typing**: No `Any`, no `cast`, no `type: ignore`, no `.pyi` stubs, no shims
2. **TypedDict Contracts**: Every TypedDict has encode/decode functions with `require_*` validation
3. **Dependency Injection**: `_test_hooks.py` for all external dependencies; production sets real implementations, tests set fakes
4. **No Mocks**: Tests use explicit fakes via hooks, not unittest.mock
5. **No Weak Assertions**: No `is not None`, `isinstance`, `hasattr`, `len > 0` as assertions
6. **No Best-Effort**: No try/except for recovery, no fallbacks, no backward compatibility shims
7. **100% Coverage**: Statements and branches, enforced by CI
8. **DRY/Modular**: Centralized utilities, no code duplication
9. **Google Docstrings**: Summary, Args, Returns, Raises with types

---

## Current Architecture (What Exists)

Based on actual code review of `tree.py`, `histogram.py`, `parallel.py`:

### Already Optimized

| Feature | Status | Location |
|---------|--------|----------|
| Histogram-based O(K) splits | ✅ Complete | `histogram.py` |
| Sibling histogram subtraction | ✅ Complete | `tree.py:_compute_child_histograms` |
| Worker pool with initializer | ✅ Complete | `parallel.py:_worker_initializer` |
| Shared memory for grad/hess | ✅ Complete | `parallel.py:_read_floats_from_shm` |
| Batched workers | ✅ Complete | `parallel.py:_build_batched_args` |
| Pool reuse across trees | ✅ Complete | `ensemble.py` |
| NaN bin handling | ✅ Complete | `histogram.py` |
| Cached histogram passing | ✅ Complete | `tree.py:_build_tree_with_histograms` |

### Current Data Flow

```
ensemble.py
  └─ Creates pool once with _worker_initializer(bin_edges, sample_bins)
  └─ For each tree:
      └─ tree.py:build_tree
          └─ _build_tree_with_histograms (stack-based depth-first)
              └─ _find_best_histogram_split_with_cache
                  └─ Routes to sequential or parallel
                  └─ Parallel: shared memory for grad/hess, workers read by name
              └─ _compute_child_histograms (sibling subtraction)
```

### Current Data Structures

```python
# Types (all immutable tuples):
FloatArray = tuple[float, ...]
IntArray = tuple[int, ...]
FloatMatrix = tuple[tuple[float, ...], ...]

# Histogram (NamedTuple with tuple fields):
class Histogram(NamedTuple):
    gradient_sums: tuple[float, ...]
    hessian_sums: tuple[float, ...]
    counts: tuple[int, ...]
```

---

## Priority Order

| # | Feature | Impact | Effort | Status |
|---|---------|--------|--------|--------|
| 1 | Mutable histogram buffers | High | Medium | Pending |
| 2 | Gradient quantization | Medium | Low | Pending |
| 3 | Leaf-wise tree growth | High | Medium | Pending |
| 4 | GOSS sampling | Medium | Low | Pending |
| 5 | Oblivious trees | Medium | Medium | Pending |

---

## Phase 1: Mutable Histogram Buffers

**Status**: Pending

**Goal**: Replace immutable `Histogram` NamedTuple with mutable `HistogramBuffer` class for in-place accumulation. Keep tuples for serialization but use mutable buffers during training.

### Problem

Current `build_histogram` in `histogram.py:215-248`:

```python
def build_histogram(...) -> Histogram:
    g_sums: list[float] = [0.0] * n_bins  # Allocate
    h_sums: list[float] = [0.0] * n_bins  # Allocate
    counts: list[int] = [0] * n_bins      # Allocate

    for idx in sample_indices:
        # Accumulate...

    # Convert to tuples (another allocation)
    return Histogram(
        gradient_sums=tuple(g_sums),
        hessian_sums=tuple(h_sums),
        counts=tuple(counts),
    )
```

Each histogram build allocates 6 objects (3 lists + 3 tuples). With 100 trees × 50 features × 10 nodes = 50,000 histograms per training run.

### Solution

Reusable histogram buffer:

```python
# New: src/cleargbm/buffers.py

class FloatBuffer:
    """Pre-allocated mutable float buffer.

    Provides O(1) element access and in-place mutation without
    allocation overhead. Used for gradients, hessians, and
    intermediate computations during tree building.

    Attributes:
        _data: Internal storage list.
        _size: Number of elements.
    """

    __slots__ = ('_data', '_size')

    def __init__(self, size: int) -> None:
        """Initialize buffer with given size.

        Args:
            size: Number of float elements to allocate.

        Raises:
            ValueError: If size is not positive.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self._data: list[float] = [0.0] * size
        self._size = size

    def __getitem__(self, index: int) -> float:
        """Get element at index.

        Args:
            index: Element index.

        Returns:
            Float value at index.

        Raises:
            IndexError: If index out of bounds.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of bounds for size {self._size}")
        return self._data[index]

    def __setitem__(self, index: int, value: float) -> None:
        """Set element at index.

        Args:
            index: Element index.
            value: Float value to set.

        Raises:
            IndexError: If index out of bounds.
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of bounds for size {self._size}")
        self._data[index] = value

    def __len__(self) -> int:
        """Return buffer size."""
        return self._size

    def fill(self, value: float) -> None:
        """Fill buffer with constant value.

        Args:
            value: Value to fill with.
        """
        for i in range(self._size):
            self._data[i] = value

    def to_tuple(self) -> tuple[float, ...]:
        """Convert to immutable tuple for serialization.

        Returns:
            Tuple copy of buffer contents.
        """
        return tuple(self._data)

    @staticmethod
    def from_tuple(data: tuple[float, ...]) -> 'FloatBuffer':
        """Create buffer from tuple.

        Args:
            data: Source tuple.

        Returns:
            New FloatBuffer with copied data.
        """
        buf = FloatBuffer(len(data))
        for i, v in enumerate(data):
            buf._data[i] = v
        return buf
```

### Types

```python
# src/cleargbm/types.py additions

class FloatBufferData(TypedDict):
    """Serialized FloatBuffer for JSON persistence."""
    values: tuple[float, ...]
    size: int


def encode_float_buffer_data(buf: FloatBuffer) -> FloatBufferData:
    """Encode FloatBuffer to serializable dict.

    Args:
        buf: Buffer to encode.

    Returns:
        Serializable TypedDict.
    """
    return FloatBufferData(
        values=buf.to_tuple(),
        size=len(buf),
    )


def decode_float_buffer_data(raw: dict[str, object]) -> FloatBuffer:
    """Decode FloatBuffer from dict.

    Args:
        raw: Raw dict from JSON.

    Returns:
        Reconstructed FloatBuffer.

    Raises:
        KeyError: If required key missing.
        TypeError: If value has wrong type.
        ValueError: If validation fails.
    """
    values = _require_float_tuple(raw, "values")
    size = require_positive_int(_require_int(raw, "size"), "size")
    if len(values) != size:
        raise ValueError(f"values length {len(values)} != size {size}")
    return FloatBuffer.from_tuple(values)
```

### Histogram Buffer

```python
# src/cleargbm/buffers.py

class HistogramBuffer:
    """Pre-allocated histogram accumulator.

    Stores gradient sums, hessian sums, and counts per bin.
    Supports in-place accumulation and subtraction.

    Attributes:
        _n_bins: Number of histogram bins.
        _grad_sums: Gradient sum per bin.
        _hess_sums: Hessian sum per bin.
        _counts: Sample count per bin.
    """

    __slots__ = ('_n_bins', '_grad_sums', '_hess_sums', '_counts')

    def __init__(self, n_bins: int) -> None:
        """Initialize histogram with given bin count.

        Args:
            n_bins: Number of bins (including NaN bin).

        Raises:
            ValueError: If n_bins is not positive.
        """
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        self._n_bins = n_bins
        self._grad_sums: list[float] = [0.0] * n_bins
        self._hess_sums: list[float] = [0.0] * n_bins
        self._counts: list[int] = [0] * n_bins

    def accumulate(
        self,
        bin_idx: int,
        gradient: float,
        hessian: float,
    ) -> None:
        """Add sample to bin.

        Args:
            bin_idx: Target bin index.
            gradient: Sample gradient.
            hessian: Sample hessian.

        Raises:
            IndexError: If bin_idx out of bounds.
        """
        if bin_idx < 0 or bin_idx >= self._n_bins:
            raise IndexError(f"bin_idx {bin_idx} out of bounds")
        self._grad_sums[bin_idx] += gradient
        self._hess_sums[bin_idx] += hessian
        self._counts[bin_idx] += 1

    def reset(self) -> None:
        """Reset all bins to zero."""
        for i in range(self._n_bins):
            self._grad_sums[i] = 0.0
            self._hess_sums[i] = 0.0
            self._counts[i] = 0

    def subtract_into(
        self,
        parent: 'HistogramBuffer',
        child: 'HistogramBuffer',
    ) -> None:
        """Compute self = parent - child (sibling subtraction).

        Args:
            parent: Parent histogram.
            child: Child histogram to subtract.

        Raises:
            ValueError: If bin counts don't match.
        """
        if parent._n_bins != self._n_bins or child._n_bins != self._n_bins:
            raise ValueError("Histogram bin counts must match")
        for i in range(self._n_bins):
            self._grad_sums[i] = parent._grad_sums[i] - child._grad_sums[i]
            self._hess_sums[i] = parent._hess_sums[i] - child._hess_sums[i]
            self._counts[i] = parent._counts[i] - child._counts[i]

    def to_histogram(self) -> 'Histogram':
        """Convert to immutable Histogram TypedDict.

        Returns:
            Immutable Histogram for serialization.
        """
        from cleargbm.histogram import Histogram
        return Histogram(
            gradient_sums=tuple(self._grad_sums),
            hessian_sums=tuple(self._hess_sums),
            counts=tuple(self._counts),
        )
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/buffers.py` | NEW: FloatBuffer, HistogramBuffer, IntBuffer |
| `src/cleargbm/types.py` | Add encode/decode for buffer serialization |
| `src/cleargbm/histogram.py` | Use HistogramBuffer for accumulation |
| `src/cleargbm/tree.py` | Use FloatBuffer for gradients/hessians |
| `src/cleargbm/parallel.py` | Update shared memory to work with buffers |
| `tests/test_buffers.py` | NEW: Full buffer test coverage |
| `tests/test_histogram.py` | Update for buffer-based histograms |
| `tests/test_tree.py` | Update for buffer-based tree building |

### Test Hooks

```python
# src/cleargbm/_test_hooks.py additions

class BufferFactoryProtocol(Protocol):
    """Protocol for buffer creation."""

    def create_float_buffer(self, size: int) -> FloatBuffer:
        """Create float buffer of given size."""
        ...

    def create_histogram_buffer(self, n_bins: int) -> HistogramBuffer:
        """Create histogram buffer with given bins."""
        ...


def _default_buffer_factory() -> BufferFactoryProtocol:
    """Production buffer factory."""
    ...


_buffer_factory: BufferFactoryProtocol = _default_buffer_factory()


def create_float_buffer(size: int) -> FloatBuffer:
    """Create float buffer via hook."""
    return _buffer_factory.create_float_buffer(size)
```

### Validation

- `require_positive_int` for buffer sizes
- Bounds checking on all index operations
- Size validation in encode/decode

### Determinism

- Buffer contents are deterministic given same inputs
- Serialization order is stable (tuple conversion)

---

## Phase 2: Gradient Quantization

**Status**: Pending

**Goal**: Discretize gradients into K levels (default 256 = uint8 range) before histogram building. Reduces memory, speeds up accumulation, provides regularization.

### Problem

Current: Float64 gradients (8 bytes each) accumulated with floating-point ops.

### Solution

Quantize gradients to integer levels, use integer accumulation:

```python
# src/cleargbm/quantize.py

class QuantizationParams(TypedDict):
    """Parameters for gradient quantization."""
    min_value: float
    max_value: float
    n_levels: int
    scale: float  # (max - min) / (n_levels - 1)


class QuantizedGradients(TypedDict):
    """Quantized gradient values."""
    levels: tuple[int, ...]  # 0 to n_levels-1
    params: QuantizationParams


def quantize_gradients(
    gradients: FloatBuffer,
    n_levels: int = 256,
) -> QuantizedGradients:
    """Quantize float gradients to integer levels.

    Discretizes gradient values into n_levels bins using linear scaling.
    This reduces memory usage and provides implicit regularization by
    smoothing the gradient landscape.

    Args:
        gradients: Float gradient buffer.
        n_levels: Number of quantization levels (default 256 for uint8).

    Returns:
        QuantizedGradients with integer levels and reconstruction params.

    Raises:
        ValueError: If n_levels < 2 or gradients empty.
    """
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels}")
    if len(gradients) == 0:
        raise ValueError("gradients must not be empty")

    # Find min/max
    g_min = gradients[0]
    g_max = gradients[0]
    for i in range(1, len(gradients)):
        g = gradients[i]
        if g < g_min:
            g_min = g
        if g > g_max:
            g_max = g

    # Handle constant gradients
    if g_max - g_min < 1e-10:
        mid_level = n_levels // 2
        levels = tuple(mid_level for _ in range(len(gradients)))
        return QuantizedGradients(
            levels=levels,
            params=QuantizationParams(
                min_value=g_min,
                max_value=g_max,
                n_levels=n_levels,
                scale=1.0,
            ),
        )

    scale = (g_max - g_min) / (n_levels - 1)

    levels: list[int] = []
    for i in range(len(gradients)):
        g = gradients[i]
        level = int((g - g_min) / scale + 0.5)  # Round to nearest
        level = max(0, min(n_levels - 1, level))  # Clamp
        levels.append(level)

    return QuantizedGradients(
        levels=tuple(levels),
        params=QuantizationParams(
            min_value=g_min,
            max_value=g_max,
            n_levels=n_levels,
            scale=scale,
        ),
    )


def dequantize_gradient(level: int, params: QuantizationParams) -> float:
    """Reconstruct float gradient from quantized level.

    Args:
        level: Quantized level (0 to n_levels-1).
        params: Quantization parameters.

    Returns:
        Reconstructed float gradient.
    """
    return params["min_value"] + level * params["scale"]
```

### Config

```python
# src/cleargbm/types.py additions to GradientBoostingConfig

class GradientBoostingConfig(TypedDict):
    # ... existing fields ...

    # Gradient quantization
    quantize_gradients: bool  # Enable gradient quantization
    gradient_levels: int  # Number of quantization levels (default 256)
```

### Validation

```python
def require_gradient_levels(value: int, name: str) -> int:
    """Validate gradient quantization levels.

    Args:
        value: Number of levels.
        name: Parameter name for error messages.

    Returns:
        Validated value.

    Raises:
        ValueError: If value not in [2, 65536].
    """
    if value < 2 or value > 65536:
        raise ValueError(f"{name} must be in [2, 65536], got {value}")
    return value
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/quantize.py` | NEW: Quantization functions and types |
| `src/cleargbm/types.py` | Add config fields, encode/decode, validation |
| `src/cleargbm/histogram.py` | Support quantized gradient accumulation |
| `src/cleargbm/tree.py` | Optional quantization before histogram building |
| `tests/test_quantize.py` | NEW: Quantization tests |
| `tests/test_types.py` | Config encode/decode tests |

### Determinism

- Quantization is deterministic: same inputs produce same levels
- Rounding uses round-half-up consistently

---

## Phase 3: Leaf-wise Tree Growth

**Status**: Pending

**Goal**: Replace depth-first tree building with leaf-wise growth. Always split the leaf with highest potential gain across the entire tree.

### Problem

Current depth-first approach:
1. Process nodes in stack order
2. May split low-gain nodes before high-gain nodes at different branches
3. Suboptimal tree structure for fixed number of leaves

### Solution

Priority queue based on split gain:

```python
# src/cleargbm/tree.py modifications

class LeafCandidate(TypedDict):
    """Candidate leaf for splitting."""
    sample_indices: tuple[int, ...]
    depth: int
    parent_id: int
    is_left_child: bool
    cached_histograms: tuple[Histogram, ...] | None
    best_split: SplitCandidate | None
    gain: float  # For priority ordering


class LeafPriorityQueue:
    """Priority queue for leaf-wise tree growth.

    Maintains leaves ordered by potential split gain (highest first).
    Uses a max-heap implementation.
    """

    __slots__ = ('_heap',)

    def __init__(self) -> None:
        """Initialize empty queue."""
        self._heap: list[tuple[float, int, LeafCandidate]] = []
        self._counter = 0  # For stable ordering of equal gains

    def push(self, leaf: LeafCandidate) -> None:
        """Add leaf to queue.

        Args:
            leaf: Leaf candidate with computed gain.
        """
        import heapq
        # Negate gain for max-heap behavior
        heapq.heappush(self._heap, (-leaf["gain"], self._counter, leaf))
        self._counter += 1

    def pop(self) -> LeafCandidate:
        """Remove and return highest-gain leaf.

        Returns:
            Leaf with highest potential gain.

        Raises:
            IndexError: If queue is empty.
        """
        import heapq
        if not self._heap:
            raise IndexError("pop from empty queue")
        _, _, leaf = heapq.heappop(self._heap)
        return leaf

    def __len__(self) -> int:
        """Return number of leaves in queue."""
        return len(self._heap)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._heap) == 0


def _build_tree_leaf_wise(
    x: FloatMatrix,
    gradients: FloatArray,
    hessians: FloatArray,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    feature_bins: FeatureBins,
    pool: WorkerPoolProtocol | None,
) -> DecisionTree:
    """Build tree using leaf-wise growth strategy.

    Always splits the leaf with highest potential gain, regardless of depth.
    This typically produces better trees than depth-first for a fixed
    number of leaves.

    Args:
        x: Feature matrix (n_samples, n_features).
        gradients: Gradient per sample.
        hessians: Hessian per sample.
        config: Training configuration.
        feature_names: Names for each feature.
        feature_bins: Precomputed feature bins.
        pool: Optional worker pool for parallel histogram building.

    Returns:
        Constructed decision tree.
    """
    max_depth = config["max_depth"]
    max_leaves = config.get("max_leaves")  # None = unlimited
    min_samples_split = config["min_samples_split"]
    min_samples_leaf = config["min_samples_leaf"]

    nodes: list[TreeNode] = []
    queue = LeafPriorityQueue()

    # Initialize with root
    all_indices = tuple(range(len(x)))
    root_split = _find_best_split(...)  # Compute root's best split
    root_gain = root_split["gain"] if root_split else 0.0

    queue.push(LeafCandidate(
        sample_indices=all_indices,
        depth=0,
        parent_id=-1,
        is_left_child=True,
        cached_histograms=None,
        best_split=root_split,
        gain=root_gain,
    ))

    n_leaves = 0

    while not queue.is_empty():
        # Check max_leaves constraint
        if max_leaves is not None and n_leaves >= max_leaves:
            break

        leaf = queue.pop()

        # Check if should create leaf node
        if (leaf["depth"] >= max_depth or
            leaf["best_split"] is None or
            len(leaf["sample_indices"]) < min_samples_split):
            # Create leaf node
            node = _create_leaf_node(...)
            nodes.append(node)
            n_leaves += 1
            continue

        # Split this node
        split = leaf["best_split"]
        node = _create_internal_node(...)
        nodes.append(node)

        # Add children to queue with their best splits computed
        for child_indices, is_left in [(split["left_indices"], True),
                                        (split["right_indices"], False)]:
            child_split = _find_best_split(...)
            child_gain = child_split["gain"] if child_split else 0.0
            queue.push(LeafCandidate(
                sample_indices=child_indices,
                depth=leaf["depth"] + 1,
                parent_id=node["node_id"],
                is_left_child=is_left,
                cached_histograms=...,
                best_split=child_split,
                gain=child_gain,
            ))

    # Convert remaining queue items to leaf nodes
    while not queue.is_empty():
        leaf = queue.pop()
        node = _create_leaf_node(...)
        nodes.append(node)

    return DecisionTree(
        nodes=tuple(nodes),
        max_depth=max_depth,
        n_leaves=n_leaves,
        feature_names=feature_names,
    )
```

### Config

```python
# src/cleargbm/types.py additions

class GradientBoostingConfig(TypedDict):
    # ... existing fields ...

    # Tree growth strategy
    tree_growth: Literal["depth_first", "leaf_wise"]  # default: "depth_first"
    max_leaves: int | None  # Max leaves for leaf-wise (None = unlimited)
```

### Validation

```python
def require_tree_growth(value: str, name: str) -> Literal["depth_first", "leaf_wise"]:
    """Validate tree growth strategy.

    Args:
        value: Strategy name.
        name: Parameter name for errors.

    Returns:
        Validated literal value.

    Raises:
        ValueError: If not a valid strategy.
    """
    if value not in ("depth_first", "leaf_wise"):
        raise ValueError(f"{name} must be 'depth_first' or 'leaf_wise', got {value!r}")
    if value == "depth_first":
        return "depth_first"
    return "leaf_wise"
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/types.py` | Add tree_growth, max_leaves to config |
| `src/cleargbm/tree.py` | Add LeafPriorityQueue, _build_tree_leaf_wise |
| `tests/test_tree.py` | Tests for leaf-wise growth |

### Determinism

- Ties in gain are broken by insertion order (counter in heap)
- Same config produces same tree structure

---

## Phase 4: GOSS Sampling

**Status**: Pending

**Goal**: Implement Gradient-based One-Side Sampling from LightGBM. Keep all large-gradient samples, randomly sample from small-gradient samples.

### Rationale

Large gradients = model is wrong = important samples.
Small gradients = model is confident = can subsample.

### Implementation

```python
# src/cleargbm/sampling.py

class GOSSResult(TypedDict):
    """Result of GOSS sampling."""
    indices: tuple[int, ...]  # Selected sample indices
    weights: tuple[float, ...]  # Sample weights for gradient scaling


def goss_sample(
    gradients: FloatBuffer,
    top_rate: float,
    other_rate: float,
    random_state: RandomStateProtocol,
) -> GOSSResult:
    """Gradient-based One-Side Sampling.

    Keeps all samples with large absolute gradients and randomly
    samples from the remaining samples. Small-gradient samples
    are upweighted to compensate for sampling.

    Args:
        gradients: Gradient values per sample.
        top_rate: Fraction of large-gradient samples to keep (0, 1).
        other_rate: Fraction of small-gradient samples to sample (0, 1).
        random_state: Random number generator.

    Returns:
        GOSSResult with selected indices and weights.

    Raises:
        ValueError: If rates are invalid.

    Example:
        With top_rate=0.2, other_rate=0.1 on 1000 samples:
        - Sort by |gradient|
        - Keep top 200 samples (weight=1.0)
        - Randomly sample 80 from remaining 800 (weight=10.0)
        - Total: 280 samples with weights
    """
    if not (0.0 < top_rate < 1.0):
        raise ValueError(f"top_rate must be in (0, 1), got {top_rate}")
    if not (0.0 < other_rate < 1.0):
        raise ValueError(f"other_rate must be in (0, 1), got {other_rate}")
    if top_rate + other_rate > 1.0:
        raise ValueError(f"top_rate + other_rate must be <= 1.0")

    n_samples = len(gradients)
    n_top = int(n_samples * top_rate)
    n_other = int((n_samples - n_top) * other_rate)

    # Sort indices by absolute gradient (descending)
    sorted_indices = sorted(
        range(n_samples),
        key=lambda i: abs(gradients[i]),
        reverse=True,
    )

    # Top samples (large gradients) - keep all
    top_indices = sorted_indices[:n_top]

    # Other samples - random subsample
    remaining_indices = sorted_indices[n_top:]
    other_indices = list(random_state.choice(
        len(remaining_indices),
        size=n_other,
        replace=False,
    ))
    sampled_other = [remaining_indices[i] for i in other_indices]

    # Compute weights
    # Small-gradient samples get upweighted to compensate for sampling
    weight_multiplier = (1.0 - top_rate) / other_rate if other_rate > 0 else 1.0

    indices: list[int] = []
    weights: list[float] = []

    for idx in top_indices:
        indices.append(idx)
        weights.append(1.0)

    for idx in sampled_other:
        indices.append(idx)
        weights.append(weight_multiplier)

    return GOSSResult(
        indices=tuple(indices),
        weights=tuple(weights),
    )
```

### Config

```python
class GradientBoostingConfig(TypedDict):
    # ... existing fields ...

    # GOSS sampling
    goss_enabled: bool  # Enable GOSS (default: False)
    goss_top_rate: float  # Fraction of large-gradient samples (default: 0.2)
    goss_other_rate: float  # Fraction to sample from rest (default: 0.1)
```

### Weighted Histogram Building

When GOSS is enabled, histogram accumulation uses weights:

```python
def build_weighted_histogram(
    sample_indices: tuple[int, ...],
    sample_weights: tuple[float, ...],
    gradients: FloatBuffer,
    hessians: FloatBuffer,
    sample_bins: tuple[int, ...],
    n_bins: int,
) -> HistogramBuffer:
    """Build histogram with sample weights.

    Args:
        sample_indices: Indices of samples to include.
        sample_weights: Weight per sample (same length as indices).
        gradients: All gradients.
        hessians: All hessians.
        sample_bins: Bin assignment per sample.
        n_bins: Number of histogram bins.

    Returns:
        Weighted histogram.
    """
    hist = HistogramBuffer(n_bins)
    for i, idx in enumerate(sample_indices):
        weight = sample_weights[i]
        bin_idx = sample_bins[idx]
        hist.accumulate_weighted(
            bin_idx,
            gradients[idx] * weight,
            hessians[idx] * weight,
            weight,
        )
    return hist
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/sampling.py` | NEW: GOSS implementation |
| `src/cleargbm/types.py` | Add GOSS config fields |
| `src/cleargbm/histogram.py` | Add weighted histogram building |
| `src/cleargbm/tree.py` | Integrate GOSS into tree building |
| `tests/test_sampling.py` | NEW: GOSS tests |

### Validation

```python
def require_goss_rate(value: float, name: str) -> float:
    """Validate GOSS rate parameter.

    Args:
        value: Rate value.
        name: Parameter name.

    Returns:
        Validated value.

    Raises:
        ValueError: If not in (0, 1).
    """
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be in (0, 1), got {value}")
    return value
```

---

## Phase 5: Oblivious Trees

**Status**: Pending

**Goal**: Implement symmetric/oblivious trees where all nodes at the same depth use the same split condition. This is CatBoost's key structural innovation.

### Problem

Standard trees: Each node chooses its own split independently.
- Complex structure
- Slower inference (must traverse path)
- More prone to overfitting

### Solution

Oblivious trees: Same split at each depth level.
- Tree is a lookup table of 2^depth entries
- Inference is O(1) with bitmask
- Acts as regularization

```python
# src/cleargbm/oblivious.py

class ObliviousSplit(TypedDict):
    """Single split condition used at one depth level."""
    feature_index: int
    feature_name: str
    threshold: float


class ObliviousTree(TypedDict):
    """Oblivious/symmetric decision tree.

    All nodes at same depth use identical split condition.
    Leaf values stored as flat array indexed by path bitmask.
    """
    splits: tuple[ObliviousSplit, ...]  # One per depth level
    leaf_values: tuple[float, ...]  # 2^depth values
    leaf_counts: tuple[int, ...]  # Samples per leaf
    depth: int
    feature_names: tuple[str, ...]


def predict_oblivious_single(
    tree: ObliviousTree,
    x: tuple[float, ...],
) -> float:
    """Predict single sample with oblivious tree.

    Uses bitmask for O(1) lookup instead of tree traversal.

    Args:
        tree: Oblivious tree structure.
        x: Single sample features.

    Returns:
        Leaf value prediction.
    """
    leaf_idx = 0
    for level, split in enumerate(tree["splits"]):
        # Bit is 1 if goes right, 0 if goes left
        if x[split["feature_index"]] > split["threshold"]:
            leaf_idx |= (1 << level)
    return tree["leaf_values"][leaf_idx]


def build_oblivious_tree(
    x: FloatMatrix,
    gradients: FloatArray,
    hessians: FloatArray,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    feature_bins: FeatureBins,
) -> ObliviousTree:
    """Build oblivious tree with same split per depth.

    At each depth, finds the single best split across ALL samples
    at that level (not per-node). This creates a symmetric tree
    structure that acts as regularization.

    Args:
        x: Feature matrix.
        gradients: Sample gradients.
        hessians: Sample hessians.
        config: Training configuration.
        feature_names: Feature names.
        feature_bins: Precomputed bins.

    Returns:
        Oblivious tree structure.
    """
    max_depth = config["max_depth"]
    reg_alpha = config["reg_alpha"]
    reg_lambda = config["reg_lambda"]

    n_samples = len(x)
    n_leaves = 2 ** max_depth

    # Track which leaf each sample belongs to
    sample_leaf: list[int] = [0] * n_samples  # All start in leaf 0

    splits: list[ObliviousSplit] = []

    for depth in range(max_depth):
        # Find best split across ALL current leaves combined
        best_split = _find_best_oblivious_split(
            x, gradients, hessians,
            sample_leaf, depth,
            feature_bins, config,
        )

        if best_split is None:
            # No valid split found, stop growing
            break

        splits.append(ObliviousSplit(
            feature_index=best_split["feature_index"],
            feature_name=feature_names[best_split["feature_index"]],
            threshold=best_split["threshold"],
        ))

        # Update sample leaf assignments
        for i in range(n_samples):
            if x[i][best_split["feature_index"]] > best_split["threshold"]:
                sample_leaf[i] |= (1 << depth)

    # Compute leaf values
    actual_depth = len(splits)
    actual_n_leaves = 2 ** actual_depth

    leaf_grad_sums: list[float] = [0.0] * actual_n_leaves
    leaf_hess_sums: list[float] = [0.0] * actual_n_leaves
    leaf_counts: list[int] = [0] * actual_n_leaves

    for i in range(n_samples):
        leaf_idx = sample_leaf[i] & ((1 << actual_depth) - 1)
        leaf_grad_sums[leaf_idx] += gradients[i]
        leaf_hess_sums[leaf_idx] += hessians[i]
        leaf_counts[leaf_idx] += 1

    leaf_values: list[float] = []
    for i in range(actual_n_leaves):
        value = _compute_leaf_value(
            leaf_grad_sums[i],
            leaf_hess_sums[i],
            reg_alpha,
            reg_lambda,
        )
        leaf_values.append(value)

    return ObliviousTree(
        splits=tuple(splits),
        leaf_values=tuple(leaf_values),
        leaf_counts=tuple(leaf_counts),
        depth=actual_depth,
        feature_names=feature_names,
    )
```

### Config

```python
class GradientBoostingConfig(TypedDict):
    # ... existing fields ...

    # Tree structure
    tree_structure: Literal["standard", "oblivious"]  # default: "standard"
```

### Model Type Updates

```python
class GradientBoostingModel(TypedDict):
    # ... existing fields ...

    # Support both tree types
    trees: tuple[DecisionTree, ...] | None  # Standard trees
    oblivious_trees: tuple[ObliviousTree, ...] | None  # Oblivious trees
    tree_structure: Literal["standard", "oblivious"]
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/oblivious.py` | NEW: Oblivious tree implementation |
| `src/cleargbm/types.py` | Add ObliviousTree, update model/config |
| `src/cleargbm/ensemble.py` | Support oblivious tree training |
| `tests/test_oblivious.py` | NEW: Oblivious tree tests |

---

## Phase 6: Ordered Boosting

**Status**: Pending

**Goal**: Implement CatBoost's ordered boosting to prevent target leakage in gradient estimation.

### Problem

Standard gradient boosting: Gradient for sample i is computed using a model trained on ALL data including sample i. This causes subtle overfitting.

### Solution

For each sample, use only samples that came "before" it (in a random permutation) to compute its gradient.

This is the most complex change and requires maintaining multiple model states during training.

### Implementation Sketch

```python
# src/cleargbm/ordered.py

def train_ordered_boosting(
    x_train: FloatMatrix,
    y_train: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
) -> GradientBoostingModel:
    """Train with ordered boosting (CatBoost-style).

    For each sample, gradients are computed using a model trained
    only on samples that appear before it in a random permutation.
    This prevents target leakage and improves generalization.

    Args:
        x_train: Training features.
        y_train: Training labels.
        config: Training configuration.
        feature_names: Feature names.

    Returns:
        Trained model.
    """
    n_samples = len(x_train)
    rng = get_random_state(config["random_state"])

    # Random permutation defines sample ordering
    permutation = rng.permutation(n_samples)

    # For each tree, we need to track predictions for each "prefix"
    # of the permutation

    # This is expensive - CatBoost uses approximations
    # We implement the exact version for correctness

    ...
```

### Note

Ordered boosting is the most complex feature. It requires:
- Maintaining O(n) model states during training
- Or using CatBoost's approximation schemes

This should be implemented last after the other features are stable.

---

## Testing Strategy

### Unit Tests

Each module gets dedicated test file:
- `tests/test_buffers.py` - Buffer operations, bounds checking
- `tests/test_quantize.py` - Quantization, dequantization, edge cases
- `tests/test_sampling.py` - GOSS sampling, weight computation
- `tests/test_oblivious.py` - Oblivious tree building, prediction

### Integration Tests

- `tests/test_hybrid_integration.py` - End-to-end training with all features
- Compare accuracy/speed with baseline
- Verify determinism with fixed seeds

### Regression Tests

- Ensure existing tests pass with new features disabled
- Verify backward compatibility of model serialization

### Coverage Requirements

- 100% statement coverage
- 100% branch coverage
- All error paths tested

---

## Benchmark Plan

### Metrics

1. **Training time** (seconds)
2. **Inference time** (ms per sample)
3. **Memory usage** (peak MB)
4. **AUC-ROC** on test set
5. **Trees per second**

### Datasets

- Taiwan bankruptcy (6,819 samples, 95 features)
- US bankruptcy (78,682 samples, 18 features)
- Polish bankruptcy (7,027 samples, 64 features)

### Configurations to Compare

| Config | Description |
|--------|-------------|
| Baseline | Current ClearGBM (depth-first, tuples) |
| +Buffers | Efficient data structures |
| +Quantize | Gradient quantization |
| +LeafWise | Leaf-wise growth |
| +GOSS | Gradient sampling |
| +Oblivious | Symmetric trees |
| Full Hybrid | All features combined |

### Script

```bash
poetry run python -m scripts.benchmark_hybrid \
    --dataset taiwan \
    --configs baseline,buffers,quantize,leafwise,goss,oblivious,hybrid \
    --n-runs 5
```

---

## Migration Path

### Phase 1: Non-Breaking

- Add new modules (buffers.py, quantize.py, sampling.py, oblivious.py)
- New config fields have defaults that preserve existing behavior
- Existing tests continue to pass

### Phase 2: Gradual Adoption

- Enable features via config flags
- Benchmark each feature independently
- Document performance characteristics

### Phase 3: Default Changes

- Once stable, consider changing defaults for better performance
- Maintain backward compatibility via explicit config

---

## Open Questions (Resolved)

### 1. Ordered boosting complexity
**Question**: Full implementation is O(n^2) per tree. Use approximation?

**Answer**: Yes, use CatBoost's approximation. Instead of maintaining n model states, use a fixed number of permutations (e.g., 4) and average predictions. This reduces complexity from O(n^2) to O(k*n) where k is the permutation count. Defer to Phase 7 after other features are stable.

### 2. Buffer pool
**Question**: Should we pool and reuse buffers across trees?

**Answer**: The current design already addresses this at the tree level - `HistogramBuffer.reset()` allows reuse within a tree. For cross-tree pooling, the gain is minimal because:
- Pool is only useful during tree building (training hot path)
- Each tree's histograms are independent
- GC handles cleanup efficiently for short-lived objects

**Recommendation**: Start with per-tree buffer reuse (Phase 1), benchmark, add cross-tree pooling only if profiling shows allocation as bottleneck.

### 3. Parallel oblivious trees
**Question**: How to parallelize oblivious tree building?

**Answer**: Based on code review of `parallel.py`:
- Oblivious trees find ONE split per depth level across ALL samples
- Use same shared memory pattern: broadcast gradients/hessians via SharedMemory
- Each worker computes partial histogram for subset of features
- Main process reduces to find global best split
- Same `_worker_initializer` pattern works - bins are feature-local

```python
# Parallel oblivious split finding (sketch)
def _find_best_oblivious_split_parallel(
    x: FloatMatrix,
    gradients: FloatArray,
    hessians: FloatArray,
    feature_bins: FeatureBins,
    config: GradientBoostingConfig,
    pool: WorkerPoolProtocol,
) -> ObliviousSplit | None:
    """Find best split across all samples for one depth level."""
    # Reuse existing parallel histogram infrastructure
    # Each worker handles subset of features
    # Reduce across workers to find global best
    ...
```

### 4. Mixed trees
**Question**: Allow mixing standard and oblivious trees in one model?

**Answer**: No. Keep model structure simple:
- `tree_structure: Literal["standard", "oblivious"]` is model-level config
- All trees in ensemble use same structure
- Simpler serialization, prediction, explanation
- If user wants both, train two models and ensemble externally

---

## Implementation Checklist

### Phase 1: Mutable Histogram Buffers
- [ ] Create `src/cleargbm/buffers.py` with `FloatBuffer`, `IntBuffer`, `HistogramBuffer`
- [ ] Add `encode_*` / `decode_*` functions to `types.py`
- [ ] Add `BufferFactoryProtocol` to `_test_hooks.py`
- [ ] Update `histogram.py` to use `HistogramBuffer` internally
- [ ] Update `tree.py` to pass buffers through stack
- [ ] Create `tests/test_buffers.py` with full coverage
- [ ] Update existing tests for buffer-based internals
- [ ] Run `make check` - all tests pass, 100% coverage

### Phase 2: Gradient Quantization
- [ ] Create `src/cleargbm/quantize.py`
- [ ] Add `quantize_gradients`, `gradient_levels` to `GradientBoostingConfig`
- [ ] Add validation: `require_gradient_levels`
- [ ] Update `ensemble.py` to optionally quantize before tree building
- [ ] Create `tests/test_quantize.py`
- [ ] Run `make check`

### Phase 3: Leaf-wise Tree Growth
- [ ] Add `tree_growth`, `max_leaves` to `GradientBoostingConfig`
- [ ] Add `LeafPriorityQueue` class to `tree.py`
- [ ] Implement `_build_tree_leaf_wise` function
- [ ] Route based on `config["tree_growth"]` in `build_tree`
- [ ] Add validation: `require_tree_growth`, `require_max_leaves`
- [ ] Update tests for both growth strategies
- [ ] Run `make check`

### Phase 4: GOSS Sampling
- [ ] Create `src/cleargbm/sampling.py` with `goss_sample`
- [ ] Add `goss_enabled`, `goss_top_rate`, `goss_other_rate` to config
- [ ] Add `accumulate_weighted` method to `HistogramBuffer`
- [ ] Integrate GOSS into `ensemble.py` training loop
- [ ] Create `tests/test_sampling.py`
- [ ] Run `make check`

### Phase 5: Oblivious Trees
- [ ] Create `src/cleargbm/oblivious.py`
- [ ] Add `ObliviousTree`, `ObliviousSplit` TypedDicts
- [ ] Add `tree_structure` to config
- [ ] Implement `build_oblivious_tree`, `predict_oblivious_single`
- [ ] Update `GradientBoostingModel` to support both tree types
- [ ] Create `tests/test_oblivious.py`
- [ ] Run `make check`

### Final Integration
- [ ] Create `tests/test_hybrid_integration.py` - end-to-end with all features
- [ ] Create `scripts/benchmark_hybrid.py` - performance comparison
- [ ] Update README.md with new features
- [ ] Run full benchmark suite on all datasets
- [ ] Document performance characteristics

---

## References

1. LightGBM paper: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (Ke et al., 2017)
2. CatBoost paper: "CatBoost: unbiased boosting with categorical features" (Prokhorenkova et al., 2018)
3. XGBoost paper: "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
