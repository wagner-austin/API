# covenant-ml

Pluggable ML backends for covenant breach risk prediction: training, validation, and inference. Supports seven backends: XGBoost (gradient boosting), MLP (neural networks), LSTM (temporal sequences), LightGBM (large-scale datasets), ClearGBM (interpretable pure-Python gradient boosting), LogReg (logistic regression baseline), and Random Forest (bagging ensemble). Includes probability calibration with isotonic regression and Platt scaling.

## Installation

```bash
poetry add covenant-ml
```

Requires `covenant-domain`, `xgboost`, `torch`, `scikit-learn`, and `numpy` for runtime.

## Quick Start

```python
from pathlib import Path
from covenant_ml import train_model_with_validation, load_model, predict_probabilities
from covenant_ml.types import TrainConfig

# Train XGBoost model
config: TrainConfig = {
    "device": "auto",
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "early_stopping_rounds": 10,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

outcome = train_model_with_validation(
    x_features=X,
    y_labels=y,
    config=config,
    output_dir=Path("/models"),
    feature_names=["debt_ratio", "interest_cover", ...],
)

# Load and predict
model = load_model(Path("/models/active.ubj"))
probabilities = predict_probabilities(model, features)
```

## Backends

### XGBoost (Default)

```python
from covenant_ml import train_model_with_validation
from covenant_ml.types import TrainConfig

config: TrainConfig = {"device": "auto", "learning_rate": 0.1, ...}
outcome = train_model_with_validation(X, y, config, output_dir, feature_names)
```

### LightGBM

```python
from covenant_ml.backends.lightgbm import create_lightgbm_backend
from covenant_ml.types import LightGBMConfig

config: LightGBMConfig = {"device": "auto", "num_leaves": 31, ...}
backend = create_lightgbm_backend()
outcome = backend.train(X, y, feature_names, config, output_dir)
```

### ClearGBM

Pure Python gradient boosting with built-in interpretability:

```python
from covenant_ml.backends.cleargbm import create_cleargbm_backend
from covenant_ml.types import ClearGBMConfig

config: ClearGBMConfig = {"n_estimators": 100, "max_depth": 4, ...}
backend = create_cleargbm_backend()
outcome = backend.train(X, y, feature_names, config, output_dir)
```

### MLP Neural Network

```python
from covenant_ml.backends.mlp import MLPBackend
from covenant_ml.types import MLPConfig

config: MLPConfig = {"hidden_sizes": (64, 32), "dropout": 0.2, ...}
backend = MLPBackend()
prepared = backend.prepare(X, y, config, feature_names)
outcome = backend.train(prepared, output_dir)
```

### LSTM

For temporal bankruptcy sequences:

```python
from covenant_ml.backends.lstm import create_lstm_backend
from covenant_ml.types import LSTMConfig

config: LSTMConfig = {"hidden_size": 64, "num_layers": 2, "sequence_length": 5, ...}
backend = create_lstm_backend()
outcome = backend.train(X, y, feature_names, config, output_dir)
```

### Logistic Regression

Interpretable linear baseline with L1/L2/ElasticNet regularization:

```python
from covenant_ml.backends.logreg import create_logreg_backend
from covenant_ml.types import LogRegConfig

config: LogRegConfig = {
    "solver": "lbfgs",
    "penalty": "l2",
    "C": 1.0,
    "max_iter": 100,
    "tol": 1e-4,
    "class_weight_balanced": True,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "l1_ratio": 0.5,
}
backend = create_logreg_backend()
outcome = backend.train(x_features=X, y_labels=y, feature_names=names, config=config, output_dir=output_dir, progress=None)
```

### Random Forest

Bagging ensemble with Gini-based feature importance:

```python
from covenant_ml.backends.random_forest import create_random_forest_backend
from covenant_ml.types import RandomForestConfig

config: RandomForestConfig = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight_balanced": True,
    "n_jobs": -1,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "oob_score": False,
}
backend = create_random_forest_backend()
outcome = backend.train(x_features=X, y_labels=y, feature_names=names, config=config, output_dir=output_dir, progress=None)
```

## Backend Comparison

| Aspect | XGBoost | MLP | LSTM | LightGBM | ClearGBM | LogReg | Random Forest |
|--------|---------|-----|------|----------|----------|--------|---------------|
| Model format | `.ubj` | `.pt` | `.pt` | `.txt` | `.json` | `.joblib` | `.joblib` |
| Feature importances | Yes | No | No | Yes | Yes | Yes (coef) | Yes (Gini) |
| GPU support | CUDA | CUDA (fp16/bf16) | CUDA (fp16/bf16) | CUDA | CPU only | CPU only | CPU only |
| Best for | Tabular data | Non-linear patterns | Temporal sequences | Large datasets | Interpretability | Baseline/calibration | Robust ensemble |
| Training speed | Fast | Moderate | Slow | Very fast | Moderate | Very fast | Fast |
| Interpretability | High | Low | Low | High | Very high | Very high | Medium |
| Dependencies | C++ lib | PyTorch | PyTorch | C++ lib | Python stdlib | sklearn | sklearn |

## Inference

```python
from covenant_ml import load_model, predict_probabilities

model = load_model(Path("/models/active.ubj"))
probabilities = predict_probabilities(model, features)
```

## Metrics

```python
from covenant_ml import compute_all_metrics, format_metrics_str

metrics = compute_all_metrics(y_true, y_pred, y_proba)
print(format_metrics_str(metrics))
# "loss=0.32 auc=0.89 acc=0.85 prec=0.82 rec=0.78 f1=0.80"
```

## Feature Importance Explainers

```python
from covenant_ml.explainers import default_explainer_registry

registry = default_explainer_registry()
explainer = registry.get("shap_tree")  # or "permutation", "gradient", "integrated_gradients"
importance = explainer.compute_importance(model, x_data, feature_names, target_class=1)
```

| Explainer | Compatible Backends | Speed |
|-----------|---------------------|-------|
| `permutation` | All backends | Medium |
| `gradient` | MLP, LSTM | Fast |
| `integrated_gradients` | MLP, LSTM | Slow |
| `shap_tree` | XGBoost, LightGBM, ClearGBM | Medium |

## Probability Calibration

Post-hoc calibration to improve probability estimates:

```python
from covenant_ml.calibration import (
    create_isotonic_calibrator,
    create_platt_calibrator,
    Calibrator,
)

# Isotonic regression (non-parametric, monotonic)
calibrator = create_isotonic_calibrator(clip_proba=True, eps=1e-10)
result = calibrator.fit(y_true=y_val, y_prob=model_probs)
calibrated = calibrator.transform(y_prob=test_probs)

# Platt scaling (parametric, sigmoid)
calibrator = create_platt_calibrator()
result = calibrator.fit(y_true=y_val, y_prob=model_probs)
calibrated = calibrator.transform(y_prob=test_probs)

# Serialize and restore calibrator state
state = calibrator.get_state()
restored = Calibrator.load_state(state)
```

| Method | Type | Best for |
|--------|------|----------|
| Isotonic | Non-parametric | Flexible, any distribution |
| Platt | Parametric (sigmoid) | Well-calibrated models needing minor adjustments |

CalibrationResult includes Brier score and ECE (Expected Calibration Error) before/after calibration.

## Hyperparameter Optimization

```python
from covenant_ml.optimizer import (
    create_xgboost_optimizer,
    create_xgboost_objective,
    make_xgboost_default_space,
    make_default_optimization_config,
    use_real_optuna,
)

use_real_optuna()
optimizer = create_xgboost_optimizer()
objective = create_xgboost_objective(output_dir=Path("/models"))
space = make_xgboost_default_space()
config = make_default_optimization_config(n_trials=100)

summary = optimizer.optimize(X, y, feature_names, space, config, objective)
print(f"Best AUC: {summary['best_value']:.4f}")
```

Optimizers available: `create_xgboost_optimizer`, `create_lightgbm_optimizer`, `create_cleargbm_optimizer`, `create_mlp_optimizer`, `create_lstm_optimizer`.

## Testing

```python
from covenant_ml.testing import MockXGBModel

model = MockXGBModel(default_proba=0.5)
proba = model.predict_proba([[1, 2, 3]])  # Returns [[0.5, 0.5]]
```

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- covenant-domain
- xgboost 2.0.0+ (XGBoost backend)
- torch 2.0.0+ (MLP and LSTM backends)
- lightgbm 4.0.0+ (LightGBM backend)
- scikit-learn 1.5.0+
- numpy 1.26.0+
- optuna 3.0.0+ (hyperparameter optimization)
- 100% test coverage enforced

## Documentation

See [docs/api-reference.md](docs/api-reference.md) for detailed API documentation including:
- Config field definitions for all backends
- TypedDict and Protocol specifications
- Preprocessing pipeline details
- Dataset loading (tabular and time-series)
- Cross-validation utilities
- Feature engineering transforms
