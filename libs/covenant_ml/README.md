# covenant-ml

Pluggable ML backends for covenant breach risk prediction: training, validation, inference, and hyperparameter optimization. Supports both **classification** (bankruptcy yes/no) and **regression** (continuous targets like delinquency rates).

Classification backends: XGBoost, LightGBM, ClearGBM, LogReg, Random Forest (5 backends).
Regression backends: XGBoost, LightGBM (2 tree-based backends in this library).

PyTorch neural network backends (MLP, LSTM) for both classification and regression live in [`covenant_nn`](../covenant_nn/README.md).

## Installation

```bash
poetry add covenant-ml
```

Requires `covenant-domain`, `xgboost`, `lightgbm`, `scikit-learn`, and `numpy` for runtime. Does **not** require PyTorch (that dependency moved to `covenant_nn`).

## Quick Start — Classification

```python
from pathlib import Path
from covenant_ml import train_model_with_validation, load_model, predict_probabilities
from covenant_ml.types import TrainConfig

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

model = load_model(Path("/models/active.ubj"))
probabilities = predict_probabilities(model, features)
```

## Quick Start — Regression

```python
from pathlib import Path
from covenant_ml.backends import create_xgboost_regressor_backend
from covenant_ml.testing import make_xgboost_regressor_config

config = make_xgboost_regressor_config(n_estimators=100, learning_rate=0.1)
backend = create_xgboost_regressor_backend()

outcome = backend.train(
    x_features=X,
    y_targets=y,  # float64 continuous targets
    feature_names=names,
    config=config,
    output_dir=Path("/models"),
    progress=None,
)

loaded = backend.load(path=outcome["model_path"])
predictions = loaded.predict(X_new)  # 1D array of continuous values
```

## Classification Backends

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
outcome = backend.train(
    x_features=X, y_labels=y, feature_names=names,
    config=config, output_dir=output_dir, progress=None,
)
```

### ClearGBM

Numpy-based gradient boosting with built-in interpretability (no C++ dependencies):

```python
from covenant_ml.backends.cleargbm import create_cleargbm_backend
from covenant_ml.types import ClearGBMConfig

config: ClearGBMConfig = {"n_estimators": 100, "max_depth": 4, ...}
backend = create_cleargbm_backend()
outcome = backend.train(
    x_features=X, y_labels=y, feature_names=names,
    config=config, output_dir=output_dir, progress=None,
)
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
outcome = backend.train(
    x_features=X,
    y_labels=y,
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)
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
outcome = backend.train(
    x_features=X,
    y_labels=y,
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)
```

## Regression Backends

### XGBoost Regressor

```python
from covenant_ml.backends import create_xgboost_regressor_backend
from covenant_ml.testing import make_xgboost_regressor_config

config = make_xgboost_regressor_config(n_estimators=100, learning_rate=0.1)
backend = create_xgboost_regressor_backend()
outcome = backend.train(
    x_features=X,
    y_targets=y,
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)
```

### LightGBM Regressor

```python
from covenant_ml.backends import create_lightgbm_regressor_backend
from covenant_ml.testing import make_lightgbm_regressor_config

config = make_lightgbm_regressor_config(n_estimators=100, learning_rate=0.1)
backend = create_lightgbm_regressor_backend()
outcome = backend.train(
    x_features=X,
    y_targets=y,
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)
```

### MLP Regressor (covenant_nn)

```python
from covenant_nn.backends import create_mlp_regressor_backend
from covenant_ml.testing import make_mlp_regressor_config

config = make_mlp_regressor_config(hidden_sizes=(64, 32), n_epochs=50)
backend = create_mlp_regressor_backend()
outcome = backend.train(
    x_features=X,
    y_targets=y,
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)
```

### LSTM Regressor (covenant_nn)

```python
from covenant_nn.backends import create_lstm_regressor_backend
from covenant_ml.testing import make_lstm_regressor_config

config = make_lstm_regressor_config(hidden_size=64, num_layers=2, sequence_length=5)
backend = create_lstm_regressor_backend()
outcome = backend.train(
    x_features=X,
    y_targets=y,
    feature_names=names,
    config=config,
    output_dir=output_dir,
    progress=None,
)
```

## Backend Comparison

### Classification Backends

| Aspect | XGBoost | LightGBM | ClearGBM | LogReg | Random Forest |
|--------|---------|----------|----------|--------|---------------|
| Model format | `.ubj` | `.txt` | `.json` | `.joblib` | `.joblib` |
| Feature importances | Yes | Yes | Yes | Yes (coef) | Yes (Gini) |
| GPU support | CUDA | CUDA | CPU only | CPU only | CPU only |
| Best for | Tabular data | Large datasets | Interpretability | Baseline | Robust ensemble |
| Dependencies | C++ lib | C++ lib | numpy only | sklearn | sklearn |

### Neural Network Backends (covenant_nn)

| Aspect | MLP | LSTM |
|--------|-----|------|
| Model format | `.pt` | `.pt` |
| Feature importances | No | No |
| GPU support | CUDA (fp16/bf16) | CUDA (fp16/bf16) |
| Best for | Non-linear patterns | Temporal sequences |
| Dependencies | PyTorch | PyTorch |
| Classification | Yes | Yes |
| Regression | Yes | Yes |

### Regression Backends

| Aspect | XGBoost | LightGBM | MLP | LSTM |
|--------|---------|----------|-----|------|
| Library | covenant_ml | covenant_ml | covenant_nn | covenant_nn |
| Model format | `.ubj` | `.txt` | `.pt` | `.pt` |
| Feature importances | Yes | Yes | No | No |
| GPU support | CUDA | CUDA | CUDA | CUDA |

## Classification Metrics

```python
from covenant_ml import compute_all_metrics, format_metrics_str

metrics = compute_all_metrics(y_true, y_pred, y_proba)
print(format_metrics_str(metrics))
# "loss=0.32 auc=0.89 acc=0.85 prec=0.82 rec=0.78 f1=0.80"
```

## Regression Metrics

```python
from covenant_ml import compute_all_regression_metrics, format_regression_metrics_str

metrics = compute_all_regression_metrics(y_true, y_pred)
print(format_regression_metrics_str(metrics))
# "mse=0.0123 rmse=0.1109 mae=0.0876 r2=0.9541 mape=5.23%"
```

Individual metric functions: `compute_mse`, `compute_rmse`, `compute_mae`, `compute_r_squared`, `compute_mape`.

## Feature Importance Explainers

```python
from covenant_ml.explainers import default_explainer_registry

registry = default_explainer_registry()
explainer = registry.get("shap_tree")  # or "permutation"
importance = explainer.compute_importance(model, x_data, feature_names, target_class=1)
```

| Explainer | Compatible Backends | Speed |
|-----------|---------------------|-------|
| `permutation` | All backends | Medium |
| `shap_tree` | XGBoost, LightGBM, ClearGBM | Medium |
| `gradient` | MLP, LSTM (covenant_nn) | Fast |
| `integrated_gradients` | MLP, LSTM (covenant_nn) | Slow |

## Probability Calibration

Post-hoc calibration to improve probability estimates:

```python
from covenant_ml.calibration import (
    create_isotonic_calibrator,
    create_platt_calibrator,
    Calibrator,
)

calibrator = create_isotonic_calibrator(clip_proba=True, eps=1e-10)
result = calibrator.fit(y_true=y_val, y_prob=model_probs)
calibrated = calibrator.transform(y_prob=test_probs)

state = calibrator.get_state()
restored = Calibrator.load_state(state)
```

| Method | Type | Best for |
|--------|------|----------|
| Isotonic | Non-parametric | Flexible, any distribution |
| Platt | Parametric (sigmoid) | Well-calibrated models needing minor adjustments |

## Hyperparameter Optimization

### Classification

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

Classification optimizers: `create_xgboost_optimizer`, `create_lightgbm_optimizer`, `create_cleargbm_optimizer`.

Neural network classification optimizers live in `covenant_nn`: `create_mlp_optimizer`, `create_lstm_optimizer`.

### Regression

Regression Optuna objectives return negative RMSE (higher = better for Optuna maximization):

- Tree-based (covenant_ml): `create_xgboost_regressor_objective`, `create_lightgbm_regressor_objective`
- Neural (covenant_nn): `create_mlp_regressor_objective`, `create_lstm_regressor_objective`

## Cross-Validation

### Classification

```python
from covenant_ml.validation import run_cross_validation
```

Uses `StratifiedKFold` to preserve class balance across folds.

### Regression

```python
from covenant_ml.validation import run_regression_cross_validation
```

Uses `KFold` (no stratification — regression has no classes).

## Ensemble Optimization

### Classification

```python
from covenant_ml.ensemble import optimize_ensemble_weights
```

Optimizes model weights to maximize AUC on out-of-fold predictions.

### Regression

```python
from covenant_ml.ensemble import optimize_regression_ensemble_weights
```

Optimizes model weights on out-of-fold predictions. Supports three metrics: `neg_rmse`, `neg_mae`, `r_squared`.

## Testing

```python
from covenant_ml.testing import (
    # Classification config factories
    make_xgboost_config,
    make_lightgbm_config,
    make_mlp_config,
    make_lstm_config,
    # Regression config factories
    make_xgboost_regressor_config,
    make_lightgbm_regressor_config,
    make_mlp_regressor_config,
    make_lstm_regressor_config,
    # CUDA hook
    set_cuda_hook,
)
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
- xgboost 3.1+
- lightgbm 4.6+
- scikit-learn 1.7+
- numpy 2.3+
- optuna 4.6+
- 100% test coverage enforced

For PyTorch neural network backends (MLP, LSTM), see [`covenant_nn`](../covenant_nn/README.md).

## Documentation

See [docs/api-reference.md](docs/api-reference.md) for detailed API documentation including:
- Config field definitions for all classification and regression backends
- TypedDict and Protocol specifications
- Preprocessing pipeline details
- Dataset loading (tabular and time-series)
- Cross-validation utilities (classification and regression)
- Ensemble optimization (classification and regression)
- Feature engineering transforms
