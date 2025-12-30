# Architecture: covenant_ml Library

## Overview

The `covenant_ml` library provides pluggable ML backends for covenant breach risk prediction. It supports seven backends (XGBoost, LightGBM, ClearGBM, MLP, LSTM, LogReg, Random Forest) with a unified training/inference interface, hyperparameter optimization, feature importance explainers, probability calibration, and dataset loading utilities.

## Dependencies

- `covenant-domain` - Domain types for loan features
- `xgboost` - XGBoost gradient boosting
- `lightgbm` - LightGBM gradient boosting
- `cleargbm` - Pure Python gradient boosting (local lib)
- `torch` - PyTorch for MLP and LSTM backends
- `scikit-learn` - Metrics, preprocessing, cross-validation
- `numpy` - Array operations
- `optuna` - Hyperparameter optimization
- `shap` - SHAP TreeExplainer for feature importance

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
│   ├── types.py                 # TypedDicts and Protocols
│   ├── trainer.py               # BaseTabularTrainer
│   ├── metrics.py               # Evaluation metrics
│   ├── testing.py               # Test utilities
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── xgboost/             # XGBoost backend
│   │   ├── lightgbm/            # LightGBM backend
│   │   ├── cleargbm/            # ClearGBM backend
│   │   ├── mlp/                 # MLP neural network backend
│   │   ├── lstm/                # LSTM sequence backend
│   │   ├── logreg/              # Logistic regression backend
│   │   └── random_forest/       # Random forest backend
│   ├── calibration/             # Probability calibration (isotonic, Platt)
│   ├── preprocessing/           # AutoPreprocessor pipeline
│   ├── datasets/                # Dataset loading (tabular + time-series)
│   ├── explainers/              # Feature importance (SHAP, permutation, gradient)
│   ├── optimizer/               # Optuna hyperparameter optimization
│   ├── validation/              # Cross-validation utilities
│   ├── ensemble/                # Model ensembling
│   └── finetuning/              # Transfer learning utilities
└── tests/
    ├── conftest.py
    ├── backends/
    │   ├── xgboost/
    │   ├── lightgbm/
    │   ├── cleargbm/
    │   ├── mlp/
    │   ├── lstm/
    │   ├── logreg/
    │   └── random_forest/
    ├── calibration/
    ├── preprocessing/
    ├── datasets/
    ├── explainers/
    ├── optimizer/
    └── validation/
```

## Core Abstractions

### ClassifierBackend Protocol

All backends implement this protocol:

```python
class ClassifierBackend(Protocol):
    def prepare(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        config: ClassifierTrainConfig,
        feature_names: tuple[str, ...],
    ) -> PreparedClassifier: ...

    def train(
        self,
        prepared: PreparedClassifier,
        output_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> TrainOutcome: ...

    def load(self, path: str) -> PreparedClassifier: ...

    def predict(
        self,
        prepared: PreparedClassifier,
        x_features: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
```

### PreparedClassifier Protocol

Prepared state ready for training:

```python
class PreparedClassifier(Protocol):
    @property
    def model(self) -> object: ...

    @property
    def config(self) -> ClassifierTrainConfig: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...
```

## Backend Architecture

Each backend follows the same structure:

```
backends/<name>/
├── __init__.py          # Public exports
├── backend.py           # ClassifierBackend implementation
├── types.py             # Backend-specific types (optional)
└── _internal.py         # Private helpers (optional)
```

### Backend Implementations

| Backend | Model Class | Serialization | GPU Support |
|---------|-------------|---------------|-------------|
| XGBoost | `XGBClassifier` | `.ubj` (binary) | CUDA |
| LightGBM | `LGBMClassifier` | `.txt` (text) | CUDA |
| ClearGBM | `GradientBoostingModel` | `.json` | CPU only |
| MLP | `nn.Module` | `.pt` (state_dict) | CUDA |
| LSTM | `nn.Module` | `.pt` (state_dict) | CUDA |
| LogReg | `LogisticRegression` | `.joblib` | CPU only |
| Random Forest | `RandomForestClassifier` | `.joblib` | CPU only |

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
| `_GradientAdapter` | mlp, lstm | PyTorch gradients |
| `_IntegratedGradientsAdapter` | mlp, lstm | Integrated gradients |

### ClearGBM SHAP Integration

ClearGBM models are converted to SHAP-compatible format via `ClearGBMShapWrapper`:

```python
def try_extract_cleargbm_model(prepared: PreparedClassifier) -> GradientBoostingModel | None:
    """Extract ClearGBM model if backend is cleargbm."""

class ClearGBMShapWrapper:
    """Wraps GradientBoostingModel for SHAP TreeExplainer."""
    def __init__(self, model: GradientBoostingModel, feature_names: tuple[str, ...]) -> None: ...
```

## Optimizer Architecture

Optuna-based hyperparameter optimization with hook system for testing:

```python
# Production setup
use_real_optuna()

# Test setup
set_optuna_module_hook(lambda: FakeOptuna())
```

### Optimizer Classes

| Class | Backend | Search Space |
|-------|---------|--------------|
| `OptunaXGBoostOptimizer` | XGBoost | `XGBoostSearchSpace` |
| `OptunaLightGBMOptimizer` | LightGBM | `LightGBMSearchSpace` |
| `OptunaClearGBMOptimizer` | ClearGBM | `ClearGBMSearchSpace` |
| `OptunaMLPOptimizer` | MLP | `MLPSearchSpace` |
| `OptunaLSTMOptimizer` | LSTM | `LSTMSearchSpace` |

## Calibration Architecture

Post-hoc probability calibration to improve model predictions:

```python
class Calibrator:
    def __init__(self, config: CalibratorConfig) -> None: ...
    def fit(self, y_true: NDArray[np.int64], y_prob: NDArray[np.float64]) -> CalibrationResult: ...
    def transform(self, y_prob: NDArray[np.float64]) -> CalibratedPredictions: ...
    def get_state(self) -> CalibratorState: ...
    @classmethod
    def load_state(cls, state: CalibratorState) -> Calibrator: ...
```

### Calibration Methods

| Method | Implementation | Use Case |
|--------|----------------|----------|
| Isotonic | `_IsotonicWrapper` wrapping sklearn | Non-parametric, flexible |
| Platt | Logistic regression via sklearn | Parametric sigmoid fit |

### State Serialization

CalibratorState TypedDict with encode/decode functions for JSON persistence:

```python
encode_calibrator_state(state: CalibratorState) -> dict[str, JSONValue]
decode_calibrator_state(data: dict[str, JSONValue]) -> CalibratorState
```

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

Memory-efficient implementation using Polars-native groupby operations.

## Testing Strategy

### Hooks Pattern

Dependency injection via hooks for testability:

```python
# optimizer/optuna_backend.py
_optuna_module_hook: Callable[[], OptunaModule] | None = None

def set_optuna_module_hook(hook: Callable[[], OptunaModule] | None) -> None:
    global _optuna_module_hook
    _optuna_module_hook = hook
```

### Fake Implementations

- `MockXGBModel` - Fake XGBoost model for unit tests
- `_FakeTrial` / `_FakeStudy` - Fake Optuna for optimizer tests
- `FakeKaggleModule` - Fake Kaggle API (in platform_kaggle)

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── backends/<name>/
│   ├── test_backend.py      # Backend protocol tests
│   └── test_<name>_integration.py  # Integration tests
├── explainers/
│   ├── test_registry.py     # Registry tests
│   └── test_<explainer>.py  # Per-explainer tests
└── optimizer/
    ├── test_optuna_backend.py    # Optimizer tests
    ├── test_search_spaces.py     # Search space tests
    └── test_<backend>_objective.py  # Objective tests
```

## Type Safety

- 100% typed with TypedDicts and Protocols
- No `Any`, no casts, no `type: ignore`
- No mocks, no stubs, no TypeGuard
- Encode/decode functions for all TypedDicts

## Coverage Requirements

- 100% statement coverage
- 100% branch coverage
- No weak assertions (`isinstance`, `in`, `is not None`, `hasattr`)
