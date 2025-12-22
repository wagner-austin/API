# AMEX Competition Refactor Plan

## Status: 🟢 COMPLETE (All 6 Phases Complete)

## Goal

Beat 1st place score (0.80977) on Kaggle AMEX Default Prediction.

---

## Current State vs Target

| Aspect | Current | Target | Status |
|--------|---------|--------|--------|
| CV strategy | GroupKFold by customer_ID | GroupKFold by customer_ID | ✅ Done |
| Aggregations | mean, std, min, max, rank, diff, windows | + rank, diff, time windows | ✅ Done |
| Metric | AMEX metric implemented | AMEX metric (0.5 * Gini + 0.5 * D@4%) | ✅ Done |
| Ensemble | Weighted ensemble with OOF optimization | Weighted ensemble with OOF optimization | ✅ Done |

---

## Implementation Phases

### Phase 1: AMEX Competition Metric ✅ COMPLETE

**Location:** `libs/covenant_ml/src/covenant_ml/metrics.py`

Implemented `compute_amex_metric()`:
- Normalized Gini coefficient with 20x weight for subsampled negatives
- Default rate at 4% threshold (D@4%)
- Combined score: 0.5 * normalized_gini + 0.5 * D@4%
- Returns `AMEXMetricResult` TypedDict with score, normalized_gini, default_rate_at_4_percent
- Helper function `_compute_weighted_gini()` for Lorentz curve calculation

### Phase 2: GroupKFold Cross-Validation ✅ COMPLETE

**Location:** `libs/covenant_ml/src/covenant_ml/validation/`

Implemented group-stratified k-fold cross-validation:

**splitter.py:**
- `group_stratified_kfold_split()` - Groups samples by entity ID, stratifies by group label
- `_get_group_labels()` - Labels group positive if any sample is positive
- `_get_sample_indices_for_groups()` - Maps group IDs to sample indices

**runner.py:**
- `run_group_cross_validation()` - Full CV pipeline with preprocessing isolation
- Ensures no customer appears in both train and validation
- Per-fold preprocessing fit on training data only (no leakage)

**Exports via `__init__.py`:**
- `group_stratified_kfold_split`, `run_group_cross_validation`

All code uses strict typing (NDArray, TypedDicts), no Any/cast/ignore.

### Phase 3: Competition Feature Engineering ✅ COMPLETE

**Location:** `libs/covenant_ml/src/covenant_ml/datasets/loaders/`

Implemented three types of competition features:

**_polars_ranking.py:**
- `compute_entity_rank_features()` - Per-entity percentile rankings (0 to 1)
- `compute_diff_features()` - Row-to-row differences with mean/std/min/max/last aggregations

**_polars_window.py (NEW):**
- `compute_window_features()` - Stats over last N observations per entity
- `compute_multi_window_features()` - Multiple window sizes (e.g., last3, last6)
- Returns mean, std, min, max for each feature within the window

**TimeSeriesSpec updates:**
- Added `include_window_features: bool` flag
- Added `window_sizes: tuple[int, ...]` for configurable windows

**timeseries_csv_loader.py integration:**
- Computes window features when `include_window_features=True`
- Window features added after rank and diff features
- Cache hash includes window config for proper invalidation

| Feature Type | Output per Feature |
|--------------|-------------------|
| Rank | 1 feature (percentile) |
| Diff | 5 features (mean, std, min, max, last) |
| Window (per size) | 4 features (mean, std, min, max) |

All code uses strict typing (NDArray, TypedDicts, Protocols), no Any/cast/ignore.
100% test coverage on all new code.

### Phase 4: Ensemble Pipeline ✅ COMPLETE

**Location:** `libs/covenant_ml/src/covenant_ml/ensemble/`

Implemented weighted ensemble optimization:

**types.py:**
- `ModelOOFPredictions` - Per-model OOF predictions with fold indices
- `EnsembleOOFData` - Combined OOF data from all models
- `EnsembleWeights` - Optimized weights with model names
- `OptimizationConfig` - Metric, method, max_iterations, tolerance
- `OptimizationResult` - Weights, best_score, n_iterations, converged
- `EnsemblePrediction` - Final predictions with per-model contributions
- `make_default_optimization_config()` - Factory for default config

**weighted.py:**
- `validate_oof_data()` - Validates OOF data structure
- `validate_weights()` - Validates weights sum to 1, non-negative
- `create_equal_weights()` - Creates uniform weights for N models
- `create_oof_data()` - Factory for validated OOF data
- `compute_weighted_predictions()` - Applies weights to get ensemble predictions
- `extract_prediction_matrix()` - Extracts (n_models, n_samples) matrix

**optimizer.py:**
- `optimize_ensemble_weights()` - Scipy SLSQP optimization to maximize AMEX metric
- `set_minimize_hook()` - Hook for dependency injection (testing)
- `use_real_scipy()` - Sets real scipy.optimize.minimize at startup

**testing.py:**
- `FakeOptimizeResult` - Fake scipy result for testing
- `fake_minimize()` - Simple random search for testing without real scipy

All code uses strict typing, hook pattern for scipy, 100% test coverage.

### Phase 5: Competition Pipeline Script ✅ COMPLETE

**Location:** `services/covenant-radar-api/scripts/amex/`

Full CLI pipeline implemented with 100% test coverage:

**__main__.py:**
- CLI entry point with argparse
- `parse_args()` - Parses all configuration options
- `main()` - Orchestrates full pipeline
- Supports `--backends`, `--n-folds`, `--n-estimators`, `--learning-rate`, etc.

**pipeline.py:**
- `load_training_data()` - Loads dataset via hooks
- `load_test_data()` - Loads test dataset without labels
- `train_single_model()` - GroupKFold CV for one backend
- `train_all_models()` - Trains all specified backends
- `optimize_ensemble()` - Weight optimization via AMEX metric
- `generate_ensemble_predictions()` - Weighted predictions on test data
- `write_submission()` - Creates Kaggle submission.csv
- `run_pipeline()` - Full end-to-end pipeline

**_hooks.py:**
- `ConsoleProtocol` / `_RichConsoleAdapter` - Console output abstraction
- `RegistryProtocol` / registry hooks - Backend registry injection
- `TimeSeriesLoaderCallable` / loader hooks - Dataset loader injection
- `EnsembleOptimizerCallable` / optimizer hooks - Ensemble optimization injection
- `configure_real_scipy()` - Production scipy setup

**_test_hooks.py:**
- `FakeConsole` - Captures output for testing
- `FakeTimeseriesLoader` - Returns synthetic datasets
- `FakePreparedClassifier` / `FakeBackend` / `FakeRegistry` - Full ML backend fakes
- `configure_all_fakes()` - Sets up all hooks for testing

**types.py:**
- `AMEXPipelineConfig` - Full pipeline configuration TypedDict
- `ModelOOFResult` - Per-model OOF predictions
- `EnsembleResult` - Ensemble weights and scores
- `PipelineResult` - Full pipeline output

Full pipeline execution:
1. Load training data with competition features (rank, diff, window)
2. Train each backend (lightgbm, xgboost) with GroupKFold CV
3. Collect OOF predictions from all models
4. Optimize ensemble weights to maximize AMEX metric
5. Load test data with same features
6. Generate weighted ensemble predictions
7. Write submission.csv

All code uses strict typing, hook pattern for DI, 100% test coverage, no mocks.

### Phase 6: LightGBM DART Configuration ✅ COMPLETE

**Location:** `libs/covenant_ml/src/covenant_ml/optimizer/`

Added DART-specific search space matching 1st place solution:

**search_spaces.py:**
- Added `feature_fraction` (0.02-0.1) for aggressive feature subsampling
- Increased `reg_lambda` range to (0.1-50.0) for higher L2 regularization
- `drop_rate` (0.0-0.5) and `skip_drop` (0.0-0.5) already present

**optuna_backend.py:**
- `_sample_lightgbm_dart_params()` conditionally samples `feature_fraction` when DART selected
- `_extract_lightgbm_dart_best_params()` extracts `feature_fraction` from best trial

**objectives/lightgbm_objective.py:**
- Passes `feature_fraction` to LightGBM when present in float_params
- Early stopping automatically disabled for DART mode

**types.py:**
- Added `feature_fraction` to `SampledFloatParams` and `LightGBMSearchSpace` TypedDicts

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `boosting_type` | categorical | `gbdt`, `dart` | Enables DART when "dart" |
| `drop_rate` | float | 0.0-0.5 | Tree dropout rate (DART only) |
| `skip_drop` | float | 0.0-0.5 | Skip dropout probability (DART only) |
| `feature_fraction` | float | 0.02-0.1 | Aggressive feature subsampling (DART only) |
| `reg_lambda` | float | 0.1-50.0 | Higher L2 regularization (log scale) |

All code uses strict typing, 100% test coverage, no Any/cast/ignore.

---

## Expected Results

| Stage | Expected CV Score | Status |
|-------|-------------------|--------|
| Baseline (random split) | ~0.95 (inflated - leakage) | Baseline |
| After GroupKFold (Phase 2) | ~0.82-0.85 (realistic) | ✅ Complete |
| After rank/diff/window features (Phase 3) | ~0.80-0.82 | ✅ Complete |
| After ensemble (Phase 4) | ~0.80-0.81 | ✅ Complete |
| Full pipeline (Phase 5) | Ready to run | ✅ Complete |
| DART optimization (Phase 6) | Target: >0.80977 | ✅ Complete |

**Target:** > 0.80977

**All phases complete.** Ready to run full pipeline with DART optimization.

**Usage:**
```bash
python -m scripts.amex --backends lightgbm,xgboost --n-folds 5 --n-estimators 1000
```

---

## Usage Examples

### GroupKFold Cross-Validation (Phase 2)

```python
from covenant_ml.validation import (
    group_stratified_kfold_split,
    run_group_cross_validation,
)

# Option 1: Just get the splits
splits = group_stratified_kfold_split(
    y=labels,
    groups=customer_ids,  # NDArray[np.int64]
    n_folds=5,
    random_state=42,
)

# Option 2: Run full CV with trainer function
cv_result = run_group_cross_validation(
    x=features,
    y=labels,
    groups=customer_ids,
    n_folds=5,
    random_state=42,
    trainer=my_trainer_fn,
)
```

### Competition Features (Phase 3)

```python
from covenant_ml.datasets.types import TimeSeriesSpec, TimeSeriesDatasetConfig

config = TimeSeriesDatasetConfig(
    # ... other config ...
    time_series=TimeSeriesSpec(
        entity_column="customer_ID",
        time_column="S_2",
        aggregation="statistics",  # mean, std, min, max
        labels_file="train_labels.csv",
        labels_entity_column="customer_ID",
        include_rank_features=True,      # Per-entity percentile ranks
        include_diff_features=True,      # Row-to-row differences
        include_window_features=True,    # Last N observation stats
        window_sizes=(3, 6),             # last3 and last6
    ),
)
```

**Feature count formula:**
- Base features: N
- With `aggregation="statistics"`: N * 4 (mean, std, min, max)
- With `include_rank_features=True`: + N
- With `include_diff_features=True`: + N * 5
- With `include_window_features=True, window_sizes=(3, 6)`: + N * 4 * 2

### Ensemble Weight Optimization (Phase 4)

```python
from covenant_ml.ensemble import (
    create_oof_data,
    optimize_ensemble_weights,
    compute_weighted_predictions,
    make_default_optimization_config,
    use_real_scipy,
    ModelOOFPredictions,
)

# Set up scipy at application startup
use_real_scipy()

# Create OOF predictions from each model (after GroupKFold CV)
model1_oof = ModelOOFPredictions(
    model_name="xgboost",
    predictions=xgb_oof_preds,  # NDArray[np.float64]
    fold_indices=fold_indices,   # NDArray[np.int64]
)
model2_oof = ModelOOFPredictions(
    model_name="lightgbm",
    predictions=lgb_oof_preds,
    fold_indices=fold_indices,
)

# Create OOF data container
oof_data = create_oof_data(
    model_predictions=(model1_oof, model2_oof),
    labels=labels,  # NDArray[np.int64]
)

# Optimize weights to maximize AMEX metric
config = make_default_optimization_config()
result = optimize_ensemble_weights(oof_data, config)

# result["weights"] contains optimized weights
# result["best_score"] is the AMEX score with optimized weights
# result["initial_score"] is the AMEX score with equal weights

# Apply weights to get final predictions
ensemble_pred = compute_weighted_predictions(oof_data, result["weights"])
final_predictions = ensemble_pred["predictions"]
```

### Competition Pipeline CLI (Phase 5)

```bash
# Full pipeline with defaults
python -m scripts.amex

# Custom configuration
python -m scripts.amex \
    --backends lightgbm,xgboost \
    --n-folds 5 \
    --n-estimators 1000 \
    --learning-rate 0.05 \
    --aggregation statistics \
    --window-sizes 3,6 \
    --output submission.csv

# Available flags:
#   -b, --backends        Comma-separated backends (lightgbm,xgboost)
#   -k, --n-folds         Number of CV folds (default: 5)
#   -n, --n-estimators    Number of boosting rounds (default: 1000)
#   -l, --learning-rate   Learning rate (default: 0.05)
#   -a, --aggregation     Feature aggregation (last,first,mean,statistics)
#   -w, --window-sizes    Comma-separated window sizes for features
#   --no-rank-features    Disable rank features
#   --no-diff-features    Disable diff features
#   --no-window-features  Disable window features
#   --train-dir           Custom training data directory
#   --test-dir            Custom test data directory
#   --output              Custom output path for submission.csv
#   -s, --random-state    Random seed (default: 42)
```

---

## Reference

1st place solution: `amex_1st_place/` (at API root)
- `S2_manual_feature.py` - Feature engineering
- `S5_LGB_main.py` - LightGBM DART config
- `utils.py:61-80` - AMEX metric

---

## Coding Standards

Same as all covenant_ml code:
- No `Any`, `cast()`, `type: ignore`
- TypedDicts for all structured data
- Protocols for dynamic imports
- 100% test coverage
- No mocks

---

*Last updated: December 21, 2025 (Phase 6 Complete)*
