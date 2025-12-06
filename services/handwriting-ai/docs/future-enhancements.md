# Future Enhancements

Ideas for future accuracy and performance improvements. These are **not implemented**.

---

## Ensemble Inference

**Goal:** Average predictions across multiple models to improve accuracy.

**Current state:** Single model loaded via `DIGITS__ACTIVE_MODEL`.

**Proposed changes:**

### Configuration

```bash
# Current: single model
DIGITS__ACTIVE_MODEL=mnist_resnet18_v1

# Proposed: comma-separated list
DIGITS__ACTIVE_MODEL=mnist_resnet18_v1,mnist_resnet18_v2,mnist_wideresnet_v1
```

### Engine Changes

```python
# Current: single model
self._model: TorchModel | None = None
self._manifest: ModelManifest | None = None

# Proposed: list of loaded models
class _LoadedModel(NamedTuple):
    model: TorchModel
    manifest: ModelManifest

self._models: list[_LoadedModel] = []
```

### Prediction Changes

```python
# Current: single model prediction
logits = model(batch)
probs = _softmax_avg(logits, temperature)

# Proposed: average across ensemble
all_probs: list[list[float]] = []
for loaded in self._models:
    logits = loaded.model(batch)
    probs = _softmax_avg(logits, loaded.manifest["temperature"])
    all_probs.append(probs)

# Average probability distributions
ensemble_probs = [sum(p[i] for p in all_probs) / len(all_probs) for i in range(10)]
```

### Constraints

- All models must have same `preprocess_hash` (validated at load time)
- Minimum 1 model required, 3-5 recommended
- Each model uses its own manifest temperature

**Expected gain:** +0.05-0.15% accuracy depending on model diversity.

---

## Advanced TTA

**Goal:** Expand test-time augmentation from 9 to 17 variants.

**Current state:** 9 variants (identity + 4 shifts + 4 rotations).

**Proposed additions:**
- 4 diagonal shifts: `(1,1), (1,-1), (-1,1), (-1,-1)`
- 2 scale factors: `0.95x, 1.05x`
- 2 additional rotation angles: `±1.5°`

**Location:** `src/handwriting_ai/inference/engine.py:_augment_for_tta()`

**Expected gain:** +0.05-0.10% accuracy.

---

## Memory-Aware Calibration Samples

**Goal:** Dynamic `calibration_samples` based on available memory.

**Current state:** Static value of 100 samples.

**Proposed tier-based configuration:**

| Memory | Samples | Rationale |
|--------|---------|-----------|
| <1GB | 8 | Fast calibration, tight memory |
| 1-2GB | 16 | Moderate exploration |
| 2-4GB | 32 | Standard exploration |
| ≥4GB | 64 | Thorough exploration |

**Location:** `src/handwriting_ai/training/calibration/measure.py`

See `docs/calibration-improvements.md` for full design.

---

## Implementation Notes

All enhancements must:
- Pass `make check` (mypy strict, 100% coverage)
- Use `TypedDict` for data structures (no Pydantic, no dataclasses)
- Follow existing patterns in the codebase
- Maintain backward compatibility where possible
