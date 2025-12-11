# covenant-ml

XGBoost wrapper for covenant breach risk prediction: training, validation, and inference.

## Installation

```bash
poetry add covenant-ml
```

Requires `platform-core`, `xgboost`, `scikit-learn`, and `numpy` for runtime.

## Training

Train XGBoost classifier with train/validation/test splits and early stopping:

```python
from covenant_ml import train_model_with_validation, stratified_split
from covenant_ml.types import TrainConfig
import numpy as np

# Prepare data
X = np.array([[...], [...]])  # Features
y = np.array([0, 1, 0, 1])    # Labels (0=healthy, 1=breach)

# Configure training
config: TrainConfig = {
    "device": "auto",           # "auto" picks CUDA when available
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
    "reg_alpha": 0.0,           # L1 regularization
    "reg_lambda": 1.0,          # L2 regularization
}

# Train with validation
from pathlib import Path

outcome = train_model_with_validation(
    x_features=X,
    y_labels=y,
    config=config,
    output_dir=Path("/models"),
    feature_names=["debt_ratio", "interest_cover", ...],
)

# Result includes metrics and feature importances
print(f"Test AUC: {outcome['test_metrics']['auc']}")
print(f"Top feature: {outcome['feature_importances'][0]['name']}")
```

### Training Outcome

The `TrainOutcome` TypedDict contains:

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | str | Unique model identifier |
| `model_path` | str | Path to saved .ubj file |
| `samples_total` | int | Total samples |
| `samples_train` | int | Training samples |
| `samples_val` | int | Validation samples |
| `samples_test` | int | Test samples |
| `best_val_auc` | float | Best validation AUC |
| `best_round` | int | Round with best AUC |
| `early_stopped` | bool | Whether training stopped early |
| `train_metrics` | EvalMetrics | Training set metrics |
| `val_metrics` | EvalMetrics | Validation set metrics |
| `test_metrics` | EvalMetrics | Test set metrics |
| `feature_importances` | list | Ranked feature importances |

## Inference

Load a trained model and predict breach probabilities:

```python
from covenant_ml import load_model, predict_probabilities

# Load model
model = load_model(Path("/models/active.ubj"))

# Predict probabilities
features = [...]  # List of LoanFeatures dicts
probabilities = predict_probabilities(model, features)
# Returns [0.23, 0.87, ...] - breach probabilities
```

## Metrics

Compute evaluation metrics:

```python
from covenant_ml import (
    compute_all_metrics,
    compute_auc,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1_score,
    compute_log_loss,
    format_metrics_str,
)

# Compute all metrics at once
metrics = compute_all_metrics(y_true, y_pred, y_proba)
# Returns EvalMetrics with loss, auc, accuracy, precision, recall, f1_score

# Or compute individually
auc = compute_auc(y_true, y_proba)
accuracy = compute_accuracy(y_true, y_pred)

# Format for logging
print(format_metrics_str(metrics))
# "loss=0.32 auc=0.89 acc=0.85 prec=0.82 rec=0.78 f1=0.80"
```

## Data Splitting

Stratified train/val/test splitting:

```python
from covenant_ml import stratified_split, DataSplits

splits: DataSplits = stratified_split(
    X, y,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
)

print(f"Train: {splits.n_train}, Val: {splits.n_val}, Test: {splits.n_test}")
```

## Type Protocols

Typed protocols for XGBoost compatibility:

```python
from covenant_ml.types import (
    XGBModelProtocol,      # Model with predict_proba
    XGBBoosterProtocol,    # Low-level booster
    XGBClassifierFactory,  # Classifier constructor
    XGBClassifierLoader,   # Model loader
    TrainConfig,           # Training configuration
    TrainOutcome,          # Training result
    EvalMetrics,           # Evaluation metrics
    FeatureImportance,     # Feature importance entry
)
```

## Testing

Mock model for unit tests:

```python
from covenant_ml.testing import MockXGBModel

model = MockXGBModel(default_proba=0.5)
proba = model.predict_proba([[1, 2, 3]])
# Returns [[0.5, 0.5]]
```

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- platform-core
- xgboost 2.0.0+
- scikit-learn 1.5.0+
- numpy 1.26.0+
- 100% test coverage enforced
