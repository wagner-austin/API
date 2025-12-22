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
| 4 | Cross-Validation | ✅ COMPLETE |
| 5 | Dataset Discovery Script | ✅ COMPLETE |
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

### Cross-Validation (`covenant_ml.validation`)

| Component | Location |
|-----------|----------|
| `FoldResult` | `validation/types.py` |
| `CVResult` | `validation/types.py` |
| `StratifiedKFoldSplitter` | `validation/splitter.py` |
| `CrossValidationRunner` | `validation/runner.py` |
| `compute_oof_predictions()` | `validation/oof.py` |

**Features:**
- Preprocessing fits on training fold ONLY
- Collects OOF predictions for stacking
- Reports mean ± std across folds

### Dataset Discovery Script (`scripts/discover_datasets/`)

```
scripts/discover_datasets/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── _test_hooks.py       # Dependency injection for testing
├── main.py              # CLI argument parsing and orchestration
├── types.py             # TypedDicts for discovery results
├── scanner.py           # Directory scanning and dataset discovery
├── detection.py         # Target column and value detection
├── encoding.py          # File encoding detection
└── parsers/
    ├── __init__.py      # Parser exports
    ├── csv.py           # CSV and .data file parsing
    ├── arff.py          # ARFF file parsing
    └── excel.py         # Excel (.xlsx, .xls) parsing
```

| Module | Responsibility |
|--------|----------------|
| `main.py` | CLI args, output formatting, config code generation |
| `scanner.py` | Directory scanning, file discovery, result aggregation |
| `detection.py` | Target column detection, positive/negative value classification |
| `encoding.py` | File encoding detection (UTF-8, Latin-1, CP1252) |
| `parsers/csv.py` | CSV parsing with auto delimiter detection, .data file streaming |
| `parsers/arff.py` | ARFF parsing (@RELATION, @ATTRIBUTE, @DATA sections) |
| `parsers/excel.py` | Excel parsing via openpyxl (.xlsx) and xlrd (.xls) |

**Supported formats:**
- CSV (comma, semicolon, tab delimited)
- Space-delimited .data files (no header)
- ARFF (Weka format)
- Excel (.xlsx, .xls)

**Features:**
- Auto-detect delimiter and encoding
- Memory-efficient streaming for large files
- Strip quotes from column names
- Identify target columns from 60+ patterns
- Detect positive/negative class values
- Calculate class ratio from sample data
- Exclude ID/date columns automatically
- Generate DatasetConfig code
- Validation mode for config verification

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
- [x] `make check` passes ✅
- [x] 100% test coverage ✅
- [x] No `Any`, `cast()`, or `type: ignore` ✅
- [x] All TypedDicts use `total=True` ✅

### Cross-Validation
- [x] Preprocessing fits on training data only ✅
- [x] OOF predictions cover all samples ✅
- [x] Mean ± std reported for CV runs ✅

### Dataset Loading
- [x] Registry-based loading works ✅
- [x] Discovery script scans all datasets ✅
- [x] Supports CSV (comma, semicolon, tab delimited), ARFF, and Excel formats ✅

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
