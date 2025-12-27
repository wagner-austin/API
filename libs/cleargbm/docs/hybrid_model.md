# ClearGBM v2: Future Roadmap

Planned features inspired by XGBoost, LightGBM, and CatBoost. For current architecture, see [architecture.md](architecture.md).

---

## Completed

| Feature | Status | Notes |
|---------|--------|-------|
| Numpy refactor | ✅ Complete | `NDArray` types, vectorized ops, strict mypy |
| Mutable histogram buffers | ✅ Complete | `buffers.py` with `np.add.at()` |
| SHAP TreeExplainer adapter | ✅ Complete | In `covenant_ml/explainers/cleargbm_shap.py` |

---

## Priority Order

| # | Feature | Impact | Effort | Status |
|---|---------|--------|--------|--------|
| 2 | Early stopping | High | Low | Pending |
| 3 | Gradient quantization | Medium | Low | Pending |
| 4 | Leaf-wise tree growth | High | Medium | Pending |
| 5 | GOSS sampling | Medium | Low | Pending |
| 6 | Oblivious trees | Medium | Medium | Pending |
| 7 | Ordered boosting | Medium | High | Pending |

---

## Phase 2: Early Stopping

**Status**: Pending

**Goal**: Stop training when validation metric stops improving, preventing overfitting and reducing training time.

### Config Additions

```python
class GradientBoostingConfig(TypedDict):
    # ... existing fields ...
    early_stopping_rounds: int | None  # None = disabled, int = patience
```

### Implementation

Monitor validation AUC after each tree. Stop if no improvement for `early_stopping_rounds` consecutive trees. Return the best model, not the last.

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/types.py` | Add `early_stopping_rounds` to config, encode/decode |
| `src/cleargbm/ensemble.py` | Implement early stopping logic |
| `tests/test_ensemble.py` | Early stopping tests |

---

## Phase 3: Gradient Quantization

**Status**: Pending

**Goal**: Discretize gradients into K levels (default 256 = uint8 range) before histogram building. Reduces memory, speeds up accumulation, provides regularization.

### Config

```python
class GradientBoostingConfig(TypedDict):
    # ... existing fields ...
    quantize_gradients: bool  # Enable gradient quantization
    gradient_levels: int  # Number of quantization levels (default 256)
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/quantize.py` | NEW: Quantization functions and types |
| `src/cleargbm/types.py` | Add config fields, encode/decode, validation |
| `src/cleargbm/histogram.py` | Support quantized gradient accumulation |
| `tests/test_quantize.py` | NEW: Quantization tests |

---

## Phase 4: Leaf-wise Tree Growth

**Status**: Pending

**Goal**: Replace depth-first tree building with leaf-wise growth. Always split the leaf with highest potential gain across the entire tree.

### Config

```python
class GradientBoostingConfig(TypedDict):
    # ... existing fields ...
    tree_growth: Literal["depth_first", "leaf_wise"]  # default: "depth_first"
    max_leaves: int | None  # Max leaves for leaf-wise (None = unlimited)
```

### Implementation

Priority queue based on split gain. `LeafPriorityQueue` class maintains leaves ordered by potential gain (highest first).

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/types.py` | Add tree_growth, max_leaves to config |
| `src/cleargbm/tree.py` | Add LeafPriorityQueue, _build_tree_leaf_wise |
| `tests/test_tree.py` | Tests for leaf-wise growth |

---

## Phase 5: GOSS Sampling

**Status**: Pending

**Goal**: Implement Gradient-based One-Side Sampling from LightGBM. Keep all large-gradient samples, randomly sample from small-gradient samples.

### Rationale

Large gradients = model is wrong = important samples.
Small gradients = model is confident = can subsample.

### Config

```python
class GradientBoostingConfig(TypedDict):
    # ... existing fields ...
    goss_enabled: bool  # Enable GOSS (default: False)
    goss_top_rate: float  # Fraction of large-gradient samples (default: 0.2)
    goss_other_rate: float  # Fraction to sample from rest (default: 0.1)
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/sampling.py` | NEW: GOSS implementation |
| `src/cleargbm/types.py` | Add GOSS config fields |
| `src/cleargbm/buffers.py` | Add `accumulate_weighted` method |
| `tests/test_sampling.py` | NEW: GOSS tests |

---

## Phase 6: Oblivious Trees

**Status**: Pending

**Goal**: Implement symmetric/oblivious trees where all nodes at the same depth use the same split condition (CatBoost's key structural innovation).

### Benefits

- Tree is a lookup table of 2^depth entries
- Inference is O(1) with bitmask
- Acts as regularization

### Config

```python
class GradientBoostingConfig(TypedDict):
    # ... existing fields ...
    tree_structure: Literal["standard", "oblivious"]  # default: "standard"
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/cleargbm/oblivious.py` | NEW: Oblivious tree implementation |
| `src/cleargbm/types.py` | Add ObliviousTree, update model/config |
| `src/cleargbm/ensemble.py` | Support oblivious tree training |
| `tests/test_oblivious.py` | NEW: Oblivious tree tests |

---

## Phase 7: Ordered Boosting

**Status**: Pending

**Goal**: Implement CatBoost's ordered boosting to prevent target leakage in gradient estimation.

### Problem

Standard gradient boosting: Gradient for sample i is computed using a model trained on ALL data including sample i. This causes subtle overfitting.

### Solution

For each sample, use only samples that came "before" it (in a random permutation) to compute its gradient.

### Note

This is the most complex change. Use CatBoost's approximation: instead of maintaining n model states, use a fixed number of permutations (e.g., 4) and average predictions. Defer until other features are stable.

---

## Implementation Checklist

### Pending

- [ ] **Phase 2**: Early stopping
- [ ] **Phase 3**: Gradient quantization (`quantize.py`)
- [ ] **Phase 4**: Leaf-wise tree growth (`LeafPriorityQueue`)
- [ ] **Phase 5**: GOSS sampling (`sampling.py`)
- [ ] **Phase 6**: Oblivious trees (`oblivious.py`)
- [ ] **Phase 7**: Ordered boosting

### Final Integration

- [ ] Create `tests/test_hybrid_integration.py` - end-to-end with all features
- [ ] Create `scripts/benchmark_hybrid.py` - performance comparison
- [ ] Update README.md with new features
- [ ] Document performance characteristics

---

## References

1. LightGBM paper: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (Ke et al., 2017)
2. CatBoost paper: "CatBoost: unbiased boosting with categorical features" (Prokhorenkova et al., 2018)
3. XGBoost paper: "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
