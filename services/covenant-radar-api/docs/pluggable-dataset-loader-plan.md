# Pluggable Dataset Loader and Cross-Validation Plan

## Overview

Pluggable dataset loading system and cross-validation support for bankruptcy prediction.

**Goals:**
1. Support 38+ external datasets with minimal per-dataset code
2. Auto-detect target columns, feature types, and encodings
3. Add K-fold cross-validation with proper preprocessing isolation
4. Generate verified dataset configs from actual data inspection

---

## Progress Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset Types and Registry | ✅ COMPLETE |
| 2 | Generic Loaders (CSV, ARFF) | ✅ COMPLETE |
| 3 | Preprocessing Module | ✅ COMPLETE |
| 4 | Cross-Validation | ❌ NOT STARTED |
| 5 | Dataset Discovery Script | ❌ NOT STARTED |
| 6 | Service Integration | ✅ COMPLETE |

---

## What's Implemented

### Dataset Loading (`covenant_ml.datasets`)

| Component | Location |
|-----------|----------|
| `DatasetConfig` | `datasets/types.py` |
| `LoadedDataset` | `datasets/types.py` |
| `DatasetRegistry` | `datasets/registry.py` |
| `DatasetLoader` | `datasets/loader.py` |
| `CSVLoader` | `datasets/loaders/csv_loader.py` |
| `ARFFLoader` | `datasets/loaders/arff_loader.py` |

**Registered datasets:** taiwan, us, polish (3 of 38+)

### Preprocessing (`covenant_ml.preprocessing`)

| Component | Location |
|-----------|----------|
| `AutoPreprocessor` | `preprocessing/pipeline.py` |
| `PreprocessingState` | `preprocessing/types.py` |
| `preprocess_data_splits()` | `trainer.py` |

**Pipeline steps:**
1. Detect special codes (96, 98, 999, -1, -9, -999) → replace with NaN
2. Cap outliers (1st/99th percentile bounds)
3. Impute missing values (median per feature)
4. Z-score normalization (mean=0, std=1)

All backends (XGBoost, LightGBM, MLP, LSTM) use `preprocess_data_splits()`.

### Service Integration

| Component | Location |
|-----------|----------|
| Test hooks | `scripts/_test_hooks.py` |
| Optimize jobs | `worker/optimize_*_job.py` |
| Common loader | `worker/_optimize_common.py` |

---

## What Remains

### Cross-Validation Module

**Location:** `libs/covenant_ml/src/covenant_ml/validation/`

```
validation/
├── __init__.py
├── types.py      # CVResult, FoldResult, OOFPredictions
├── splitter.py   # StratifiedKFoldSplitter
├── runner.py     # CrossValidationRunner
└── oof.py        # OOF utilities for stacking
```

**Key requirements:**
- Preprocessing fits on training fold ONLY (use existing `AutoPreprocessor`)
- Collect OOF predictions for stacking
- Report mean ± std across folds (std shows sensitivity to split)

**Types needed:**

```python
class FoldResult(TypedDict, total=True):
    fold_number: int
    train_auc: float
    val_auc: float
    val_indices: NDArray[np.int64]
    val_predictions: NDArray[np.float64]

class CVResult(TypedDict, total=True):
    n_folds: int
    fold_results: tuple[FoldResult, ...]
    mean_val_auc: float
    std_val_auc: float
    oof_predictions: NDArray[np.float64]
```

### Dataset Discovery Script

**Location:** `scripts/discover_datasets/`

Scan `data/external/` and auto-generate `DatasetConfig` entries:
- Detect file format (CSV, ARFF, Excel)
- Auto-detect target column from common names
- Count samples and features
- Output verified configs for registry

**Target column candidates:** `target`, `class`, `label`, `bankrupt?`, `status_label`, `default`, `y`

---

## External Datasets Available

38 datasets in `data/external/`:

| Size | Examples |
|------|----------|
| Large (100K+) | AMEX (5.5M), Lending Club (2.2M), SBA (900K), Home Credit (307K) |
| Medium (10K-100K) | US Bankruptcy (78K), Loan Default (67K), Vehicle Loan (233K) |
| Small (<10K) | Taiwan (6.8K), Polish (13K), German Credit (1K) |

Formats: CSV (most), ARFF (Polish), Excel (some)

---

## Validation Checklist

### Code Quality
- [ ] `make check` passes
- [ ] 100% test coverage
- [ ] No `Any`, `cast()`, or `type: ignore`
- [ ] All TypedDicts use `total=True`

### Cross-Validation
- [x] Preprocessing fits on training data only ✅
- [ ] OOF predictions cover all samples
- [ ] Mean ± std reported for CV runs

### Dataset Loading
- [x] Registry-based loading works ✅
- [ ] Discovery script scans all 38 datasets
- [ ] At least 10 datasets have verified configs

---

## Data Leakage Prevention

**Problem:** Computing preprocessing stats on full data leaks validation info into training.

**Solution:** Use `AutoPreprocessor` which fits only on training data:

```python
from covenant_ml.trainer import preprocess_data_splits, stratified_split

splits = stratified_split(X, y, 0.7, 0.15, 0.15, random_state=42)
preprocessed = preprocess_data_splits(splits)  # Fits on train only
```

For cross-validation, fit preprocessing inside each fold:

```python
for fold in cv_split["folds"]:
    state = preprocessor.fit(x_train, y_train)  # Train fold only
    x_train = preprocessor.transform(x_train, state)
    x_val = preprocessor.transform(x_val, state)
```

---

*Last updated: December 2025*
