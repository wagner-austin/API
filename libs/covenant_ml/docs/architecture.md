# Architecture: covenant_ml Library

## Overview

The `covenant_ml` library provides pluggable ML backends for covenant breach risk prediction. It supports both **classification** (binary bankruptcy prediction) and **regression** (continuous targets like delinquency rates).

Classification backends (5): XGBoost, LightGBM, ClearGBM, LogReg, Random Forest.
Regression backends (2 tree-based): XGBoost, LightGBM.

PyTorch neural network backends (MLP, LSTM) for both classification and regression live in the separate [`covenant_nn`](../../covenant_nn/README.md) library.

## Two-Library Architecture

```
covenant_ml (this library)              covenant_nn (PyTorch neural)
├── Tree-based classifiers              ├── MLP classifier backend
│   ├── XGBoost                         ├── MLP regressor backend
│   ├── LightGBM                        ├── LSTM classifier backend
│   ├── ClearGBM                        ├── LSTM regressor backend
│   ├── LogReg                          ├── MLP classifier objective
│   └── Random Forest                   ├── MLP regressor objective
├── Tree-based regressors               ├── LSTM classifier objective
│   ├── XGBoost regressor               └── LSTM regressor objective
│   └── LightGBM regressor
├── Types, metrics, protocols           covenant_nn depends on covenant_ml
├── Preprocessing, datasets             for types, metrics, and protocols.
├── Validation (classification + regression)
├── Ensemble (classification + regression)
├── Optimizer framework + tree objectives
├── Calibration, explainers, finetuning
└── Testing utilities
```

The split isolates PyTorch as a dependency — `covenant_ml` does not require `torch`. This makes `covenant_ml` tests fast (no PyTorch training) and lets `covenant_nn` pin CUDA PyTorch independently.

## Dependencies

- `covenant-domain` — Domain types for loan features
- `xgboost` — XGBoost gradient boosting
- `lightgbm` — LightGBM gradient boosting
- `cleargbm` — Pure Python gradient boosting (local lib)
- `scikit-learn` — Metrics, preprocessing, cross-validation
- `numpy` — Array operations
- `optuna` — Hyperparameter optimization
- `shap` — SHAP TreeExplainer for feature importance

Note: `torch` is **not** a dependency of `covenant_ml`. PyTorch backends live in `covenant_nn`.

## Directory Structure

```
libs/covenant_ml/
├── pyproject.toml
├── README.md
├── Makefile
├── docs/
│   ├── architecture.md
│   └── api-reference.md
├── scripts/
│   ├── __init__.py
│   └── guard.py
├── src/covenant_ml/
│   ├── __init__.py              # Public exports
│   ├── py.typed                 # PEP 561 marker
│   ├── types.py                 # TypedDicts (classification + regression)
│   ├── trainer.py               # Training functions (classification + regression)
│   ├── metrics.py               # Evaluation metrics (classification + regression)
│   ├── testing.py               # Test utilities and config factories
│   ├── backends/
│   │   ├── __init__.py          # Backend exports
│   │   ├── protocol.py          # ClassifierBackend protocol
│   │   ├── registry.py          # Classifier backend registry
│   │   ├── regressor_protocol.py # RegressorBackend protocol
│   │   ├── regressor_registry.py # Regressor backend registry
│   │   ├── xgboost/
│   │   │   ├── backend.py       # XGBoost classifier
│   │   │   └── regressor.py     # XGBoost regressor
│   │   ├── lightgbm/
│   │   │   ├── backend.py       # LightGBM classifier
│   │   │   └── regressor.py     # LightGBM regressor
│   │   ├── cleargbm/            # ClearGBM classifier
│   │   ├── logreg/              # Logistic regression classifier
│   │   └── random_forest/       # Random forest classifier
│   ├── calibration/             # Probability calibration (isotonic, Platt)
│   ├── preprocessing/           # AutoPreprocessor pipeline
│   ├── datasets/
│   │   ├── types.py             # All dataset + temporal TypedDicts
│   │   ├── testing.py           # Fake loaders + synthetic temporal factories
│   │   └── loaders/
│   │       ├── _netcdf_temporal.py      # Fourier seasonal cycle + heat metrics
│   │       └── _netcdf_trend_testing.py # Rank-trend hypothesis testing
│   ├── explainers/              # Feature importance (SHAP, permutation)
│   ├── optimizer/
│   │   ├── objectives/
│   │   │   ├── xgboost_objective.py           # Classification
│   │   │   ├── lightgbm_objective.py          # Classification
│   │   │   ├── cleargbm_objective.py          # Classification
│   │   │   ├── xgboost_regressor_objective.py # Regression
│   │   │   └── lightgbm_regressor_objective.py # Regression
│   │   └── ...
│   ├── validation/
│   │   ├── types.py             # Classification CV types
│   │   ├── runner.py            # Classification CV runner
│   │   ├── regression_types.py  # Regression CV types
│   │   ├── regression_runner.py # Regression CV runner
│   │   └── ...
│   ├── ensemble/
│   │   ├── types.py             # Classification ensemble types
│   │   ├── optimizer.py         # Classification ensemble optimizer
│   │   ├── regression_types.py  # Regression ensemble types
│   │   ├── regression_optimizer.py # Regression ensemble optimizer
│   │   └── ...
│   └── finetuning/              # Transfer learning utilities
└── tests/
    ├── conftest.py
    ├── backends/
    │   ├── xgboost/
    │   ├── lightgbm/
    │   ├── cleargbm/
    │   ├── logreg/
    │   └── random_forest/
    ├── calibration/
    ├── preprocessing/
    ├── datasets/
    ├── explainers/
    ├── optimizer/
    │   └── objectives/
    ├── validation/
    └── ensemble/
```

## Core Abstractions

### ClassifierBackend Protocol

All classification backends implement this protocol:

```python
class ClassifierBackend(Protocol):
    def backend_name(self) -> BackendName: ...
    def capabilities(self) -> BackendCapabilities: ...

    def prepare(
        self, *, n_features: int, feature_names: list[str] | None,
    ) -> PreparedClassifier: ...

    def train(
        self, *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
    ) -> TrainOutcome: ...

    def evaluate(
        self, *, model: PreparedClassifier,
        x: NDArray[np.float64], y: NDArray[np.int64],
    ) -> EvalMetrics: ...

    def save(self, *, model: PreparedClassifier, path: str) -> None: ...
    def load(self, *, path: str) -> PreparedClassifier: ...

    def get_feature_importances(
        self, *, model: PreparedClassifier, feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None: ...
```

### RegressorBackend Protocol

All regression backends implement this protocol:

```python
class RegressorBackend(Protocol):
    def backend_name(self) -> RegressorBackendName: ...
    def capabilities(self) -> BackendCapabilities: ...

    def prepare(
        self, *, n_features: int, feature_names: list[str] | None,
    ) -> PreparedRegressor: ...

    def train(
        self, *,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],    # float64 continuous targets
        feature_names: list[str] | None,
        config: RegressorTrainConfig,
        output_dir: Path,
        progress: RegressorProgressCallback | None,
    ) -> RegressionTrainOutcome: ...

    def evaluate(
        self, *, model: PreparedRegressor,
        x: NDArray[np.float64], y: NDArray[np.float64],
    ) -> RegressionMetrics: ...

    def save(self, *, model: PreparedRegressor, path: str) -> None: ...
    def load(self, *, path: str) -> PreparedRegressor: ...

    def get_feature_importances(
        self, *, model: PreparedRegressor, feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None: ...
```

### Key Differences: Classification vs Regression

| Aspect | Classification | Regression |
|--------|---------------|------------|
| Target type | `NDArray[np.int64]` (0/1) | `NDArray[np.float64]` (continuous) |
| Prediction | `predict_proba()` → shape `(n, 2)` | `predict()` → shape `(n,)` |
| Metrics | `EvalMetrics` (AUC, F1, ...) | `RegressionMetrics` (RMSE, R², ...) |
| Training outcome | `TrainOutcome` with `best_val_auc` | `RegressionTrainOutcome` with `best_val_rmse` |
| Early stopping | Maximize val AUC | Minimize val RMSE |
| Class weighting | `scale_pos_weight` | Not applicable |
| CV strategy | `StratifiedKFold` | `KFold` |

### PreparedClassifier and PreparedRegressor

```python
class PreparedClassifier(Protocol):
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

class PreparedRegressor(Protocol):
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

## Backend Architecture

### Classification Backend Structure

```
backends/<name>/
├── __init__.py          # Public exports
├── backend.py           # ClassifierBackend implementation
└── regressor.py         # RegressorBackend implementation (xgboost, lightgbm only)
```

### Classification Backend Registry

`default_registry()` registers 5 classification backends:

| Name | Backend | Model Format |
|------|---------|-------------|
| `"xgboost"` | XGBoost classifier | `.ubj` |
| `"lightgbm"` | LightGBM classifier | `.txt` |
| `"cleargbm"` | ClearGBM classifier | `.json` |
| `"logreg"` | Logistic regression | `.joblib` |
| `"random_forest"` | Random forest | `.joblib` |

MLP and LSTM classifiers are **not** in this registry — they live in `covenant_nn`.

### Regressor Backend Registry

`default_regressor_registry()` registers 2 tree-based regression backends:

| Name | Backend | Model Format |
|------|---------|-------------|
| `"xgboost_reg"` | XGBoost regressor | `.ubj` |
| `"lightgbm_reg"` | LightGBM regressor | `.txt` |

MLP and LSTM regressors are in `covenant_nn` and not in this registry.

## Regression Architecture

### Regression Trainer

`train_regression_model_with_validation()` in `trainer.py` — parallel to the classification `train_model_with_validation()`:

- Uses `objective="reg:squarederror"` (XGBoost) or `objective="regression"` (LightGBM)
- Early stops on val RMSE (lower is better), not val AUC
- Uses `regression_split()` — random split, not stratified
- Returns `RegressionTrainOutcome` with `RegressionMetrics`

### Regression Cross-Validation

`run_regression_cross_validation()` in `validation/regression_runner.py`:

- Uses `KFold` (not `StratifiedKFold` — no classes to stratify)
- Returns `RegressionCVResult` with per-fold RMSE and OOF predictions
- Accepts `FoldRegressorTrainer` protocol for fold training

### Regression Ensemble

`optimize_regression_ensemble_weights()` in `ensemble/regression_optimizer.py`:

- Optimizes model weights on out-of-fold predictions
- Three objective metrics: `neg_rmse`, `neg_mae`, `r_squared`
- Uses same scipy minimize hook pattern as classification

### Regression Optuna Objectives

Tree-based objectives in `optimizer/objectives/`:
- `XGBoostRegressorObjective` — returns negative RMSE
- `LightGBMRegressorObjective` — returns negative RMSE

Neural network objectives in `covenant_nn/objectives/`:
- `MLPRegressorObjective` — returns negative RMSE
- `LSTMRegressorObjective` — returns negative RMSE

## Explainers Architecture

Registry-based explainer system with backend compatibility:

```python
class ExplainerRegistry:
    def get(self, name: SupportedExplainer) -> ExplainerAdapter: ...
    def list_explainers(self) -> tuple[SupportedExplainer, ...]: ...
    def list_compatible_explainers(self, backend: BackendName) -> tuple[SupportedExplainer, ...]: ...
    def is_compatible(self, explainer: SupportedExplainer, backend: BackendName) -> bool: ...
```

### Explainer Adapters

| Adapter | Backends | Implementation |
|---------|----------|----------------|
| `_ShapTreeAdapter` | xgboost, lightgbm, cleargbm | SHAP TreeExplainer |
| `_PermutationAdapter` | all | sklearn permutation_importance |
| `_GradientAdapter` | mlp, lstm (covenant_nn) | PyTorch gradients |
| `_IntegratedGradientsAdapter` | mlp, lstm (covenant_nn) | Integrated gradients |

## Optimizer Architecture

Optuna-based hyperparameter optimization with hook system for testing:

```python
# Production setup
use_real_optuna()

# Test setup
set_optuna_module_hook(lambda: FakeOptuna())
```

### Classification Optimizers

| Class | Backend | Search Space |
|-------|---------|--------------|
| `OptunaXGBoostOptimizer` | XGBoost | `XGBoostSearchSpace` |
| `OptunaLightGBMOptimizer` | LightGBM | `LightGBMSearchSpace` |
| `OptunaClearGBMOptimizer` | ClearGBM | `ClearGBMSearchSpace` |

MLP and LSTM classification optimizers live in `covenant_nn`.

### Regression Objectives

| Class | Backend | Library |
|-------|---------|---------|
| `XGBoostRegressorObjective` | XGBoost | covenant_ml |
| `LightGBMRegressorObjective` | LightGBM | covenant_ml |
| `MLPRegressorObjective` | MLP | covenant_nn |
| `LSTMRegressorObjective` | LSTM | covenant_nn |

All regression objectives return negative RMSE for Optuna maximization.

## Calibration Architecture

Post-hoc probability calibration for classification models:

```python
class Calibrator:
    def __init__(self, config: CalibratorConfig) -> None: ...
    def fit(self, y_true: NDArray[np.int64], y_prob: NDArray[np.float64]) -> CalibrationResult: ...
    def transform(self, y_prob: NDArray[np.float64]) -> CalibratedPredictions: ...
    def get_state(self) -> CalibratorState: ...
    @classmethod
    def load_state(cls, state: CalibratorState) -> Calibrator: ...
```

| Method | Implementation | Use Case |
|--------|----------------|----------|
| Isotonic | `_IsotonicWrapper` wrapping sklearn | Non-parametric, flexible |
| Platt | Logistic regression via sklearn | Parametric sigmoid fit |

## Temporal Feature Extraction (McKinnon PNAS 2024)

### Overview

Two internal modules implement the McKinnon (PNAS 2024) methodology for atmospheric extremes analysis:

- `_netcdf_temporal.py` — Steps 1-3: Fourier deseasonalization, residual computation, tail-excess heat metrics
- `_netcdf_trend_testing.py` — Steps 4-7: Rank conversion, spatial DOF estimation, Monte Carlo null distributions, p-values

Both are pure numpy (no scipy, no xarray) with Protocol wrappers for numpy functions with insufficiently typed stubs.

### Temporal Feature Pipeline

```
Raw daily temperatures
  → fit_seasonal_cycle()         # Fourier coefficients (step 1)
  → remove_seasonal_cycle()      # Anomalies (step 2)
  → compute_within_season_medians()
  → compute_residuals()          # Detrended residuals
  → fit_tail_thresholds()        # Hot/cold percentile thresholds (step 3)
  → compute_heat_metrics()       # 9 metrics per location per year
  → rank_heat_metrics()          # Rank conversion (step 4)
  → estimate_spatial_dof()       # Bretherton et al. 1999 (step 5-6)
  → generate_null_trend_slopes() # Monte Carlo null distribution (step 7)
  → compute_trend_pvalue()       # Two-sided p-values
```

### Rank-Trend Sign Conventions

| Category | Metrics | Ranking |
|----------|---------|---------|
| HOT (negate before rank) | seasonal_max, cum_excess_hot, avg_excess_hot, ndays_excess_hot, ndays_excess_cold, ar1 | Rank 1 = most extreme (largest value) |
| COLD (rank directly) | seasonal_min, cum_excess_cold, avg_excess_cold | Rank 1 = most extreme (smallest value) |

### Temporal Types

All temporal types are immutable TypedDicts with encode/decode/require_* validation:

| Type | Description |
|------|-------------|
| `TemporalFeatureConfig` | Fourier harmonics, percentile thresholds, season definition |
| `SeasonalCycleCoefficients` | Fitted cosine/sine coefficients per location |
| `TailThresholds` | Hot/cold percentile thresholds per location |
| `TemporalFeatureState` | Complete fitted state (config + cycle + thresholds) |
| `HeatMetricResult` | Per-entity heat metrics across years |
| `RankTrendConfig` | Monte Carlo sample count and random seed |
| `MetricTrendResult` | Per-metric OLS slope, p-value, significance |
| `RankTrendResult` | Complete rank-trend results across all metrics |

### Testing Utilities

| Factory | Description |
|---------|-------------|
| `create_synthetic_daily_timeseries()` | Generates multi-location daily data with known Fourier seasonal cycle + noise |
| `create_synthetic_trending_metrics()` | Generates multi-location metric data with known linear trends for rank-trend testing |

## Dataset Loading

### Tabular Datasets

```python
registry = make_default_registry()
loader = create_dataset_loader()
dataset = loader.load(registry.get("taiwan"), data_path)
```

### Time-Series Datasets

```python
registry = make_default_timeseries_registry()
loader = TimeSeriesCSVLoader()
dataset = loader.load(registry.get("kaggle_amex_default"), data_path)
```

### Regression Datasets (Testing)

```python
from covenant_ml.datasets.testing import create_fake_regression_dataset_loader

loader = create_fake_regression_dataset_loader(n_samples=200, n_features=10)
dataset = loader.load(config, Path("/fake"))
# dataset["y"] is NDArray[np.float64] (continuous targets)
```

## Testing Strategy

### Hooks Pattern

Dependency injection via hooks for testability. Production code sets hooks to real implementations at startup; tests set them to fakes.

```python
# In testing.py (public for consumers)
set_cuda_hook(lambda: False)  # Force CPU in tests

# In optimizer
set_optuna_module_hook(lambda: FakeOptuna())
```

### Config Factories

`testing.py` provides factories for both classification and regression configs:

```python
# Classification
make_xgboost_config(...)
make_lightgbm_config(...)
make_mlp_config(...)
make_lstm_config(...)

# Regression
make_xgboost_regressor_config(...)
make_lightgbm_regressor_config(...)
make_mlp_regressor_config(...)
make_lstm_regressor_config(...)
```

### Test Organization

```
tests/
├── conftest.py
├── backends/
│   ├── xgboost/          # Classifier + regressor tests
│   ├── lightgbm/         # Classifier + regressor tests
│   ├── cleargbm/         # Classifier tests
│   ├── logreg/           # Classifier tests
│   └── random_forest/    # Classifier tests
├── calibration/
├── preprocessing/
├── datasets/
├── explainers/
├── optimizer/
│   └── objectives/       # Classification + regression objective tests
├── validation/           # Classification + regression CV tests
└── ensemble/             # Classification + regression ensemble tests
```

## Type Safety

- 100% typed with TypedDicts and Protocols
- No `Any`, no casts, no `type: ignore`
- No mocks, no stubs, no `.pyi` files
- Encode/decode functions for all TypedDicts with `require_*` validation
- Google-style docstrings

## Coverage Requirements

- 100% statement coverage
- 100% branch coverage
- No weak assertions (`isinstance`, `in`, `is not None`, `hasattr`)
