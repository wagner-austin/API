# covenant-nn

PyTorch neural network backends (MLP, LSTM) for [`covenant_ml`](../covenant_ml/README.md). Provides both **classification** and **regression** backends that implement the `ClassifierBackend` and `RegressorBackend` protocols, plus Optuna objective functions for hyperparameter optimization.

## Why a Separate Library?

`covenant_ml` handles tree-based backends (XGBoost, LightGBM, ClearGBM, LogReg, Random Forest) and does not depend on PyTorch. Isolating the PyTorch backends into `covenant_nn`:

- Makes `covenant_ml` tests fast (no PyTorch training)
- Isolates the CUDA dependency to one library
- Lets `covenant_nn` pin CUDA PyTorch independently
- Keeps tree-based and neural concerns separate

## Installation

```bash
poetry add covenant-nn
```

Requires `covenant-ml`, `platform-ml`, `platform-core`, `torch` (CUDA build), and `numpy`.

## Backends

### MLP Classifier

Multi-layer perceptron for binary classification with configurable hidden layers, dropout, and mixed precision:

```python
from covenant_nn.backends import create_mlp_backend
from covenant_ml.testing import make_mlp_config

config = make_mlp_config(hidden_sizes=(64, 32), n_epochs=50, dropout=0.2)
backend = create_mlp_backend()

outcome = backend.train(
    x_features=X,
    y_labels=y,
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)

loaded = backend.load(path=outcome["model_path"])
probabilities = loaded.predict_proba(X_new)  # shape (n, 2)
```

### LSTM Classifier

Long Short-Term Memory for temporal sequence classification:

```python
from covenant_nn.backends import create_lstm_backend
from covenant_ml.testing import make_lstm_config

config = make_lstm_config(
    hidden_size=64,
    num_layers=2,
    sequence_length=5,
    n_epochs=50,
    bidirectional=True,
)
backend = create_lstm_backend()

outcome = backend.train(
    x_features=X,
    y_labels=y,
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)

loaded = backend.load(path=outcome["model_path"])
probabilities = loaded.predict_proba(X_new)  # shape (n, 2)
```

### MLP Regressor

MLP for continuous target prediction:

```python
from covenant_nn.backends import create_mlp_regressor_backend
from covenant_ml.testing import make_mlp_regressor_config

config = make_mlp_regressor_config(hidden_sizes=(64, 32), n_epochs=50, dropout=0.2)
backend = create_mlp_regressor_backend()

outcome = backend.train(
    x_features=X,
    y_targets=y,  # float64 continuous targets
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)

loaded = backend.load(path=outcome["model_path"])
predictions = loaded.predict(X_new)  # 1D array of continuous values
```

### LSTM Regressor

LSTM for continuous temporal target prediction:

```python
from covenant_nn.backends import create_lstm_regressor_backend
from covenant_ml.testing import make_lstm_regressor_config

config = make_lstm_regressor_config(
    hidden_size=64,
    num_layers=2,
    sequence_length=5,
    n_epochs=50,
    bidirectional=True,
)
backend = create_lstm_regressor_backend()

outcome = backend.train(
    x_features=X,
    y_targets=y,  # float64 continuous targets
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)

loaded = backend.load(path=outcome["model_path"])
predictions = loaded.predict(X_new)  # 1D array of continuous values
```

## Backend Comparison

| Aspect | MLP Classifier | MLP Regressor | LSTM Classifier | LSTM Regressor |
|--------|---------------|---------------|-----------------|----------------|
| Output | `predict_proba()` (n, 2) | `predict()` (n,) | `predict_proba()` (n, 2) | `predict()` (n,) |
| Loss | CrossEntropyLoss | MSELoss | CrossEntropyLoss | MSELoss |
| Early stopping | Max val AUC | Min val RMSE | Max val AUC | Min val RMSE |
| Model format | `.pt` + `.json` | `.pt` + `.json` | `.pt` + `.json` | `.pt` + `.json` |
| GPU | CUDA (fp16/bf16) | CUDA (fp16/bf16) | CUDA (fp16/bf16) | CUDA (fp16/bf16) |
| Class weighting | Yes | N/A | Yes | N/A |

## Hyperparameter Optimization (Optuna)

Each backend has a corresponding Optuna objective that returns the optimization score (AUC for classification, negative RMSE for regression):

### Classification Objectives

```python
from covenant_nn.objectives.mlp_objective import create_mlp_objective
from covenant_nn.objectives.lstm_objective import create_lstm_objective
```

### Regression Objectives

```python
from covenant_nn.objectives.mlp_regressor_objective import create_mlp_regressor_objective
from covenant_nn.objectives.lstm_regressor_objective import create_lstm_regressor_objective
```

All regression objectives return negative RMSE (higher = better for Optuna maximization).

## Public API

### Backends

| Export | Description |
|--------|-------------|
| `create_mlp_backend()` | MLP classifier backend factory |
| `create_mlp_regressor_backend()` | MLP regressor backend factory |
| `create_lstm_backend()` | LSTM classifier backend factory |
| `create_lstm_regressor_backend()` | LSTM regressor backend factory |
| `MLP_CAPABILITIES` | MLP classifier backend capabilities |
| `MLP_REGRESSOR_CAPABILITIES` | MLP regressor backend capabilities |
| `LSTM_CAPABILITIES` | LSTM classifier backend capabilities |
| `LSTM_REGRESSOR_CAPABILITIES` | LSTM regressor backend capabilities |

### Objectives

| Export | Description |
|--------|-------------|
| `MLPObjective` | MLP classifier Optuna objective class |
| `create_mlp_objective(...)` | MLP classifier objective factory |
| `MLPRegressorObjective` | MLP regressor Optuna objective class |
| `create_mlp_regressor_objective(...)` | MLP regressor objective factory |
| `LSTMObjective` | LSTM classifier Optuna objective class |
| `create_lstm_objective(...)` | LSTM classifier objective factory |
| `LSTMRegressorObjective` | LSTM regressor Optuna objective class |
| `create_lstm_regressor_objective(...)` | LSTM regressor objective factory |

## Architecture

Types, metrics, and protocols live in `covenant_ml` (shared by all backends). `covenant_nn` imports from `covenant_ml`:

- `ClassifierBackend` / `RegressorBackend` protocols
- `TrainOutcome` / `RegressionTrainOutcome` result types
- `EvalMetrics` / `RegressionMetrics` metric types
- `MLPConfig` / `LSTMConfig` configuration types
- `compute_all_metrics()` / `compute_all_regression_metrics()` metric functions
- `AutoPreprocessor` for feature normalization
- `stratified_split()` / `regression_split()` for data splitting

### Model Serialization

Each backend saves two files:
- `.pt` — PyTorch state dict (model weights)
- `.json` — Model metadata (architecture params needed to reconstruct the model before loading weights)

The metadata JSON uses `platform_core.json_utils` for encoding/decoding with strict `require_*` validation.

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- covenant-ml (types, metrics, protocols)
- platform-ml (device selection, torch type protocols)
- platform-core (logging, JSON utilities)
- torch 2.5+ (CUDA build from pytorch-cuda source)
- numpy 2.3+
- 100% test coverage enforced (statements + branches)
- Strict mypy (no Any, no casts, no type: ignore)

## Directory Structure

```
libs/covenant_nn/
├── pyproject.toml
├── README.md
├── Makefile
├── smoke_test.py                    # End-to-end smoke test (all 4 backends)
├── scripts/
│   ├── __init__.py
│   └── guard.py
├── src/covenant_nn/
│   ├── __init__.py                  # Public exports
│   ├── py.typed                     # PEP 561 marker
│   ├── backends/
│   │   ├── __init__.py              # Backend exports
│   │   ├── mlp/
│   │   │   ├── __init__.py
│   │   │   ├── backend.py           # MLP classifier
│   │   │   ├── model.py             # MLP model definition
│   │   │   └── regressor.py         # MLP regressor
│   │   └── lstm/
│   │       ├── __init__.py
│   │       ├── backend.py           # LSTM classifier
│   │       ├── sequences.py         # Sequence reshaping utilities
│   │       └── regressor.py         # LSTM regressor
│   └── objectives/
│       ├── __init__.py
│       ├── mlp_objective.py         # MLP classifier Optuna objective
│       ├── mlp_regressor_objective.py # MLP regressor Optuna objective
│       ├── lstm_objective.py        # LSTM classifier Optuna objective
│       └── lstm_regressor_objective.py # LSTM regressor Optuna objective
└── tests/
    ├── conftest.py
    ├── data/
    │   └── american_bankruptcy.csv
    ├── backends/
    │   ├── mlp/
    │   │   ├── test_mlp_integration.py  # MLP classifier tests
    │   │   └── test_mlp_regressor.py    # MLP regressor tests
    │   └── lstm/
    │       ├── test_lstm_integration.py # LSTM classifier tests
    │       └── test_lstm_regressor.py   # LSTM regressor tests
    ├── objectives/
    │   ├── test_mlp_objective.py
    │   ├── test_mlp_regressor_objective.py
    │   ├── test_lstm_objective.py
    │   └── test_lstm_regressor_objective.py
    └── test_base_trainer_mlp.py
```
