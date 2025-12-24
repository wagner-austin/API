# ClearGBM Refactor Roadmap

Performance optimizations and feature additions for ClearGBM. All changes maintain pure Python stdlib, strict typing (no Any/cast/ignore), and 100% test coverage.

## Priority Order

| # | Feature | Impact | Effort | Status |
|---|---------|--------|--------|--------|
| 1 | Sibling histogram subtraction | High | Low | Complete |
| 2 | L1/L2 regularization | High | Medium | Complete |
| 3 | Multiprocessing histograms | High | Medium | Complete |
| 4 | Missing value handling | High | Medium | Complete |
| 5 | Feature subsampling tests | Low | Low | Complete |
| 6 | Benchmark suite | Medium | Low | Complete |
| 7 | Pool optimization | Medium | Low | Complete |
| 8 | Autotune script | Medium | Medium | Complete |
| 9 | TreeSHAP explainability | Medium | High | Pending |

---

## 1. Sibling Histogram Subtraction ✓

**Status**: Complete

**Goal**: Use `subtract_histogram(parent, child)` to derive sibling instead of building both children from scratch. 2x speedup on histogram building per split.

**Implementation**:
- Parent histograms are stored per feature during split finding
- After partitioning, smaller child's histogram is built directly
- Larger child's histogram is derived via `sibling = parent - child`
- Histograms are passed through the stack to child nodes for reuse

**Key Functions**:
- `_find_best_histogram_split_with_cache` (parallel.py): Uses cached histograms when available
- `_compute_child_histograms` (tree.py): Computes both child histograms using subtraction trick

**Files**:
- `src/cleargbm/parallel.py` - Histogram caching in split finding
- `src/cleargbm/tree.py` - Node expansion logic in `_build_tree_with_histograms`
- `tests/test_tree.py` - Tests for histogram caching and subtraction

**Determinism**:
- If both children have equal size, left child's histogram is built (consistent tie-breaking)

---

## 2. L1/L2 Regularization ✓

**Status**: Complete

**Goal**: Add `reg_alpha` (L1) and `reg_lambda` (L2) parameters for regularization.

**Implementation**:
- Added `reg_alpha` and `reg_lambda` fields to `GradientBoostingConfig` TypedDict
- Updated `encode_gradient_boosting_config` and `decode_gradient_boosting_config` with validation
- L1 regularization: Soft thresholding - `leaf = -sign(G) * max(|G| - alpha, 0) / (H + lambda)`
- L2 regularization: Adds lambda to hessian denominator - `leaf = -G / (H + lambda)`
- Split gain with L2: `gain = G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - G^2/(H + lambda)`

**Key Functions** (split.py):
- `_compute_leaf_value`: Accepts `reg_alpha` and `reg_lambda` parameters
- `_compute_split_gain`: Accepts `reg_lambda` parameter
- `_find_split_for_feature`, `find_best_split`: Pass regularization params through
- `_create_leaf_node`, `_create_internal_node`: Accept and use regularization params

**Files**:
- `src/cleargbm/types.py` - Added fields to TypedDict, encode/decode with `require_non_negative_float`
- `src/cleargbm/split.py` - Leaf value and split gain computations with regularization
- `tests/test_split.py` - L1/L2 regularization tests (soft threshold, shrink to zero, combined)
- `tests/test_types.py` - Encode/decode tests

**Validation**:
- `reg_alpha`: `require_non_negative_float` (default: 0.0)
- `reg_lambda`: `require_non_negative_float` (default: 0.0)

---

## 3. Multiprocessing Histograms ✓

**Status**: Complete

**Goal**: Parallelize per-feature histogram building using `multiprocessing.Pool`.

**Implementation**:
- Added `n_jobs` field to `GradientBoostingConfig` TypedDict
- Added `_resolve_n_jobs` to convert -1 to cpu_count
- Top-level worker functions for Windows pickle compatibility:
  - `_build_histogram_worker`: Builds histogram for a single feature
  - `_find_split_worker`: Finds split from histogram for a feature
- Parallel path activated when `n_jobs > 1` and `len(features) >= 2`
- Sequential path unchanged for `n_jobs=1`

**Key Functions** (parallel.py):
- `_resolve_n_jobs`: Resolves n_jobs (-1 = all cores, n = n workers)
- `_build_histogram_worker`: Top-level worker for parallel histogram building
- `_find_split_worker`: Top-level worker for parallel split finding
- `_find_best_histogram_split_sequential`: Sequential split finding (n_jobs=1)
- `_find_best_histogram_split_parallel`: Parallel split finding (n_jobs>1)

**Determinism**:
- Histograms are built in parallel, but split selection is sequential
- Features are sorted by index for deterministic tie-breaking
- Prefer higher gain; on equal gain, prefer lower feature_index
- Tests verify identical results with n_jobs=1 and n_jobs=2

**Files**:
- `src/cleargbm/types.py` - Added `n_jobs` field, `require_n_jobs` validator
- `src/cleargbm/parallel.py` - Parallel histogram building, deterministic split selection
- `tests/test_parallel.py` - Tests for parallel equivalence, worker functions
- `tests/test_types.py` - Tests for `require_n_jobs` validation

**Validation**:
- `n_jobs`: `require_n_jobs` (must be -1 or positive, default: 1)

---

## 4. Missing Value Handling ✓

**Status**: Complete

**Goal**: Handle NaN values in features during training and prediction.

**Implementation**:
- NaN values get a dedicated bin in histograms (`NAN_BIN_OFFSET = 1`)
- At each split, both NaN-goes-left and NaN-goes-right are evaluated
- The direction that yields higher gain (subject to monotonic constraints) is chosen
- `nan_direction` is stored in TreeNode for consistent prediction routing

**Key Functions**:
- `_assign_bin` (histogram.py): Returns `nan_bin` for NaN values via `math.isnan()`
- `_evaluate_nan_direction` (histogram.py): Helper for gain comparison
- `find_best_split_from_histogram` (histogram.py): Evaluates both NaN directions
- `partition_by_bin` (histogram.py): Routes NaN samples based on `nan_direction`
- `_predict_single` (tree.py): Routes NaN inputs using stored `nan_direction`
- `explain_tree_prediction` (tree.py): Handles NaN in path explanation

**Type Changes**:
- `TreeNode`: Added `nan_direction: Literal["left", "right"] | None`
- `SplitCandidate`: Added `nan_direction: Literal["left", "right"]`
- `HistogramSplit`: Added `nan_direction: Literal["left", "right"]`

**Files**:
- `src/cleargbm/types.py` - Added `nan_direction` to TreeNode, encode/decode with validation
- `src/cleargbm/histogram.py` - NaN bin handling, `NAN_BIN_OFFSET`, direction evaluation
- `src/cleargbm/parallel.py` - NaN bin handling in split evaluation
- `src/cleargbm/split.py` - `nan_direction` in SplitCandidate and node creation
- `src/cleargbm/tree.py` - Prediction and explanation routing for NaN
- `tests/test_histogram.py` - NaN bin tests, partition tests
- `tests/test_tree.py` - NaN prediction and explanation tests
- `tests/test_types.py` - NaN direction validation tests

---

## 5. Feature Subsampling Tests ✓

**Status**: Complete

**Goal**: Tighten test coverage for existing `max_features` support.

**Tests Added**:
- `test_select_features_single`: `max_features=1` returns exactly one feature
- `test_select_features_equals_n_features`: `max_features == n_features` returns all features
- `test_select_features_exceeds_n_features`: `max_features > n_features` returns all features
- `test_select_features_deterministic`: Same seed produces same selection
- `test_select_features_different_seeds`: Different seeds produce different selections
- `test_select_features_no_replacement`: Selected features are unique
- `test_with_max_features_equals_n_features`: End-to-end build_tree test
- `test_with_max_features_exceeds_n_features`: End-to-end build_tree test
- `test_with_max_features_deterministic`: Same random_state produces identical tree

**Files**:
- `tests/test_tree.py` - Unit tests and end-to-end tests

---

## 6. Benchmark Suite ✓

**Status**: Complete

**Goal**: Measure performance of optimizations on synthetic datasets.

**Benchmarks**:
- Various `max_bins` values (32, 64, 128, 256)
- Sequential vs multiprocessing (n_jobs=1, 2, 4, -1)
- Tree depth impact (max_depth=2, 4, 6, 8)

**Implementation**:
- Pure stdlib (no external benchmark libs)
- Synthetic datasets via LCG random generator
- Warm-up runs for stable measurements
- Fully typed with no Any or type:ignore
- 100% test coverage

**Files**:
- `scripts/benchmark.py` - Benchmark runner with CLI
- `tests/test_scripts_benchmark.py` - Full test coverage

**Usage**:
```bash
poetry run python -m scripts.benchmark
poetry run python -m scripts.benchmark --samples 10000 --features 20 --trees 50
```

**Example Output**:
```
============================================================
ClearGBM Benchmark Suite
============================================================
Name                            Samples  Feats  Trees  Bins  Jobs     Time      T/s
-----------------------------------------------------------------------------------
max_bins=32                        5000     10     20    32     1    0.45s    44.4
max_bins=64 (default)              5000     10     20    64     1    0.82s    24.4
n_jobs=1 (sequential)              5000     10     20    64     1    0.81s    24.7
n_jobs=4                           5000     10     20    64     4    0.95s    21.1
```

---

## 7. Pool Optimization ✓

**Status**: Complete

**Goal**: Optimize multiprocessing pool usage to reduce overhead.

**Problem**: Creating a new `multiprocessing.Pool` for each tree was extremely slow due to Windows `spawn` overhead. Each pool creation involves:
- Starting new Python processes
- Serializing all data via pickle
- Incurring IPC overhead

**Solution**: Four-part optimization:

1. **Pool Reuse**: Single pool across all trees
   - Pool created once in `train_gradient_boosting` via `_test_hooks.create_worker_pool`
   - Same pool passed to all tree builds
   - Pool closed only at training completion

2. **Batched Workers**: Features grouped per IPC call
   - Workers process multiple features per call (one batch per worker)
   - Reduces IPC calls from O(n_features) to O(n_jobs)

3. **Pool Initializer for feature_bins**: Broadcast bin data once
   - `feature_bins` (bin_edges + sample_bins) set via pool initializer at creation
   - Workers access `_WORKER_FEATURE_BINS` global instead of receiving via IPC
   - Eliminates sending sample_bins (16KB-64KB per feature) with every batch
   - Reduces IPC data volume by 16-64x depending on tree depth

4. **Shared Memory for Gradients/Hessians**: Avoid pickle per-batch
   - `multiprocessing.shared_memory.SharedMemory` created in main process
   - Gradients/hessians written via `struct.pack_into` (8 bytes per float)
   - Workers receive shared memory NAMES (strings) in batched args
   - Workers open shared memory by name via `_read_floats_from_shm`
   - Main process cleans up shared memory in finally block
   - **Result**: Parallel speedup improved from 0.83x (slower) to 1.02-1.03x for large datasets

**Implementation**:
- `_worker_initializer`: Sets `_WORKER_FEATURE_BINS` global in worker processes
- `_WORKER_FEATURE_BINS`: Module-level global storing FeatureBins in workers
- `WorkerPoolProtocol.map_batched`: Batched worker method (receives shm names, not data)
- `_build_histogram_worker_batched`: Uses global for feature_bins, shared memory for gradients/hessians
- `_build_batched_args`: Constructs args with shared memory names (not data)
- `_read_floats_from_shm`: Opens shared memory by name and reads floats via struct
- `_unpack_double`: Helper to read single float from memoryview (mypy-safe)
- `_select_best_split`: Extracted for complexity reduction

**Files**:
- `src/cleargbm/_test_hooks.py` - `WorkerPoolProtocol` with initializer support, `create_worker_pool` accepts bin data
- `src/cleargbm/parallel.py` - Worker initializer, shared memory, global state, batched workers
- `src/cleargbm/ensemble.py` - Extracts bin_edges/sample_bins for pool creation
- `tests/test_parallel.py` - `_FakeSequentialPool` with shared memory simulation, `TestWorkerInitializer`

**Key Insight**: Shared memory eliminates pickle overhead for gradients/hessians. The autotune script helps users find when multiprocessing actually helps, since pool overhead exceeds benefit for smaller datasets.

---

## 8. Autotune Script ✓

**Status**: Complete

**Goal**: Provide empirical tuning to find optimal `n_jobs` and `max_bins` for user's data.

**Problem**: Hardcoded thresholds for when to use multiprocessing are unreliable because optimal configuration depends on:
- Dataset size (samples × features)
- Hardware (CPU cores, memory bandwidth)
- Python version and OS (Windows spawn vs Linux fork)

**Solution**: Autotune script that:
1. Generates synthetic data matching user's dimensions
2. Runs grid search over `{n_jobs, max_bins}` configurations
3. Times each configuration with warm-up runs
4. Emits `TuningReport` with empirically-derived recommendations

**Types** (`types.py`):
```python
class TimingResult(TypedDict):
    n_jobs: int
    max_bins: int
    max_depth: int
    learning_rate: float
    elapsed_seconds: float
    trees_per_second: float

class TuningReport(TypedDict):
    best_config: GradientBoostingConfig
    timing_results: tuple[TimingResult, ...]
    sample_size: int
    n_features: int
    recommended_n_jobs: int
    recommended_max_bins: int
    parallel_speedup: float
    total_tune_time_seconds: float
```

**Files**:
- `src/cleargbm/types.py` - `TimingResult`, `TuningReport` with encode/decode
- `scripts/autotune.py` - Grid search tuning script
- `tests/test_scripts_autotune.py` - Full test coverage

**Usage**:
```bash
poetry run python -m scripts.autotune
poetry run python -m scripts.autotune --samples 10000 --features 50 --trees 5
poetry run python -m scripts.autotune --quiet  # Suppress progress output
```

**Example Output**:
```
============================================================
ClearGBM Autotune Report
============================================================

Dataset: 10000 samples, 50 features
Tune time: 23.0s

Recommendations:
  n_jobs: 4
  max_bins: 32
  Speedup vs sequential: 1.27x

Timing Results:
--------------------------------------------------
  n_jobs   max_bins     time (s)    trees/s
--------------------------------------------------
       1         32         1.08        4.6
       4         32         1.03        4.9
--------------------------------------------------
```

**Key Findings**:
- `max_bins=32` is fastest across all tested sizes
- `n_jobs=1` wins for datasets < 10K samples (pool overhead dominates)
- `n_jobs=4` helps for datasets ≥ 10K samples with 50+ features
- Speedup from multiprocessing is modest (1.1-1.3x) due to histogram building being only part of workload

---

## 9. TreeSHAP Explainability

**Goal**: Implement exact TreeSHAP for per-feature contribution explanations.

**Existing**:
- `explain_tree_prediction` - Path-based explanation (which features were used)
- `extract_rules` - Human-readable rules

**New**:
- Exact TreeSHAP for single tree
- Sum contributions across ensemble
- Per-sample, per-feature Shapley values

**Types** (`types.py`):
```python
class TreeShapExplanation(TypedDict):
    base_value: float                    # E[f(x)]
    feature_contributions: tuple[float, ...]  # Per-feature Shapley values
    feature_names: tuple[str, ...]
    prediction: float                    # base_value + sum(contributions)
```

**Files**:
- `src/cleargbm/types.py` - `TreeShapExplanation` TypedDict
- `src/cleargbm/explain.py` - `compute_shap_values` function
- `tests/test_explain.py` - SHAP correctness tests

**Algorithm**: Recursive tree traversal tracking feature contributions via conditional expectations.

---

## Design Principles

1. **Strict Typing**: Update `GradientBoostingConfig` encode/decode for new fields. Validate with `require_*` helpers. No Any, cast, or ignore.

2. **Determinism**: Respect `RandomStateProtocol` where randomness is used. Parallel histogram building is deterministic (no RNG).

3. **DRY**: Centralize new logic (e.g., NaN bin handling) in appropriate modules. Exact path delegates when feasible.

4. **Tests**: For every new branch, add unit tests including error branches. Maintain 100% coverage.

5. **Pure Stdlib**: No external dependencies. Use only Python standard library.

---

## Completed Optimizations

- [x] Histogram binning for O(K) split finding
- [x] `max_bins` parameter (default: 64)
- [x] Typed comprehensions for tuple creation (no Any, no casts)
- [x] Delayed index allocation (only after best split found)
- [x] `subtract_histogram` function for sibling histogram computation
- [x] Sibling histogram subtraction wired into tree building (2x speedup on histogram building)
- [x] L1/L2 regularization (`reg_alpha`, `reg_lambda`) for leaf value shrinkage and split gain dampening
- [x] Multiprocessing histograms (`n_jobs`) for parallel feature histogram building
- [x] Modularized tree.py into focused modules (tree, split, parallel)
- [x] Missing value handling with dedicated NaN bin and optimal direction selection
- [x] Pool optimization (reuse pool across trees, batched workers for reduced IPC)
- [x] Pool initializer for feature_bins (broadcast bin data once, 16-64x IPC reduction)
- [x] Shared memory for gradients/hessians (parallel speedup from 0.83x to 1.02-1.03x)
- [x] Autotune script for empirical `n_jobs`/`max_bins` configuration

---

## Code Modularization ✓

**Status**: Complete

**Goal**: Split large files into focused, maintainable modules.

**Source Modules**:

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `split.py` | Split computation | `_compute_leaf_value`, `_compute_split_gain`, `find_best_split`, `_create_leaf_node` |
| `parallel.py` | Parallel histograms | `_resolve_n_jobs`, `_build_histogram_worker`, `_find_best_histogram_split_*` |
| `tree.py` | Core tree building | `build_tree`, `predict_tree`, `explain_tree_prediction`, `_compute_child_histograms` |

**Test Modules**:

| Module | Covers |
|--------|--------|
| `test_split.py` | Split computation, leaf/gain calculation, monotonicity |
| `test_parallel.py` | Parallel histogram building, n_jobs, worker functions |
| `test_tree.py` | Tree building, prediction, explanation, helper functions |
| `conftest.py` | Shared `make_config` helper |

**Result**: 352 tests, 100% coverage
