# covenant-ml

Pluggable ML backends for covenant breach risk prediction: training, validation, and inference. Supports XGBoost (gradient boosting) and MLP (neural networks).

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
import numpy as np

# Train a model
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

## Training

Train XGBoost classifier with train/validation/test splits and early stopping:

```python
from covenant_ml import train_model_with_validation, train_model, save_model, stratified_split
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
    "scale_pos_weight": 2.0,    # Optional: handle class imbalance
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

### Training with Progress Callback

```python
from covenant_ml import train_model_with_validation, ProgressCallback
from covenant_ml.types import TrainProgress

def on_progress(progress: TrainProgress) -> None:
    print(f"Round {progress['round']}/{progress['total_rounds']}")
    print(f"Train AUC: {progress['train_auc']:.4f}")

outcome = train_model_with_validation(
    x_features=X,
    y_labels=y,
    config=config,
    output_dir=Path("/models"),
    feature_names=feature_names,
    progress_callback=on_progress,
)
```

## MLP Neural Network Backend

Train an MLP classifier with configurable architecture and deterministic setup:

```python
from pathlib import Path
from covenant_ml.types import MLPConfig
from covenant_ml.backends.mlp import MLPBackend

# Configure MLP training
config: MLPConfig = {
    "device": "auto",
    "precision": "fp32",
    "optimizer": "adamw",
    "hidden_sizes": (64, 32),
    "learning_rate": 0.001,
    "batch_size": 32,
    "n_epochs": 100,
    "dropout": 0.2,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "early_stopping_patience": 10,
}

# Create backend and train
backend = MLPBackend()
prepared = backend.prepare(X, y, config, feature_names)
outcome = backend.train(prepared, Path("/models"))

# Result includes metrics but no feature importances
print(f"Test AUC: {outcome['test_metrics']['auc']}")
print(f"Model format: {outcome['model_format']}")  # "pt"
```

### Determinism

- Seeds are applied from `config["random_state"]` at component preparation.
- CUDA deterministic algorithms are enabled when feasible and safe.
- A tiny learning-rate warmup is used at the start of training to stabilize early updates on small datasets.

### MLPConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `device` | str | `"cpu"`, `"cuda"`, or `"auto"` |
| `precision` | str | `"fp32"`, `"fp16"`, `"bf16"`, or `"auto"` |
| `optimizer` | str | `"adamw"`, `"adam"`, or `"sgd"` |
| `hidden_sizes` | tuple[int, ...] | Hidden layer sizes (e.g., `(64, 32)`) |
| `learning_rate` | float | Learning rate |
| `batch_size` | int | Training batch size |
| `n_epochs` | int | Maximum training epochs |
| `dropout` | float | Dropout rate (0.0-1.0) |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_patience` | int | Epochs without improvement before stopping |

### TrainOutcome Fields

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
| `total_rounds` | int | Total training rounds |
| `early_stopped` | bool | Whether training stopped early |
| `train_metrics` | EvalMetrics | Training set metrics |
| `val_metrics` | EvalMetrics | Validation set metrics |
| `test_metrics` | EvalMetrics | Test set metrics |
| `feature_importances` | list[FeatureImportance] | Ranked feature importances |
| `config` | TrainConfig | Training configuration used |

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

## API Reference

### Training Functions

| Function | Description |
|----------|-------------|
| `train_model_with_validation` | Train with validation and early stopping |
| `train_model` | Train without validation (simpler API) |
| `save_model` | Save trained model to file |
| `stratified_split` | Split data with stratification |

### Inference Functions

| Function | Description |
|----------|-------------|
| `load_model` | Load model from file |
| `predict_probabilities` | Get breach probabilities |

### Metrics Functions

| Function | Description |
|----------|-------------|
| `compute_all_metrics` | Compute all evaluation metrics |
| `compute_auc` | Area under ROC curve |
| `compute_accuracy` | Classification accuracy |
| `compute_precision` | Precision for breach class |
| `compute_recall` | Recall for breach class |
| `compute_f1_score` | F1 score |
| `compute_log_loss` | Log loss (cross-entropy) |
| `format_metrics_str` | Format metrics for logging |

### Types

| Type | Description |
|------|-------------|
| `TrainConfig` | XGBoost training configuration |
| `MLPConfig` | MLP neural network configuration |
| `TrainOutcome` | Complete training result |
| `TrainProgress` | Progress update during training |
| `EvalMetrics` | Evaluation metrics for a split |
| `FeatureImportance` | Feature importance entry (name, importance, rank) |
| `DataSplits` | Train/val/test data splits |
| `ProgressCallback` | Callback type for progress updates |

### Manifest Types

TypedDicts for model manifest serialization:

| Type | Description |
|------|-------------|
| `ClassifierManifest` | Complete model manifest with all metadata |
| `ManifestVersions` | Library versions (covenant_ml, python, xgboost, torch, etc.) |
| `ManifestSystem` | System info (platform, device_used, cuda_version, gpu_name) |
| `ManifestDataset` | Dataset info (samples, features, class distribution) |
| `ManifestTraining` | Training info (backend, config, rounds, duration) |
| `ManifestMetrics` | Train/val/test metrics and best_val_auc |

### Protocols

| Protocol | Description |
|----------|-------------|
| `ClassifierBackend` | Backend interface (prepare, train, load, predict) |
| `PreparedClassifier` | Prepared classifier ready for training |
| `ClassifierRegistry` | Backend registry used by `BaseTabularTrainer` |
| `XGBModelProtocol` | XGBoost model with predict_proba |
| `XGBBoosterProtocol` | Low-level XGBoost booster |
| `XGBClassifierFactory` | XGBoost classifier constructor |
| `XGBClassifierLoader` | XGBoost model loader |
| `Proba2DProtocol` | 2D probability array |

## Testing

Mock model for unit tests:

```python
from covenant_ml.testing import MockXGBModel

model = MockXGBModel(default_proba=0.5)
proba = model.predict_proba([[1, 2, 3]])
# Returns [[0.5, 0.5]]
```

End-to-end MLP tests verify loss progression, optimizer variants, dropout, CUDA device handling, and early stopping. All tests run with 100% statement and branch coverage and enforce strict typing (no Any/casts/ignores).

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- covenant-domain
- xgboost 2.0.0+ (gradient boosting backend)
- torch 2.0.0+ (MLP neural network backend)
- scikit-learn 1.5.0+
- numpy 1.26.0+
- 100% test coverage enforced

## Backend Comparison

| Aspect | XGBoost | MLP |
|--------|---------|-----|
| Model format | `.ubj` | `.pt` |
| Feature importances | Yes (ranked) | No |
| GPU support | CUDA | CUDA (fp16/bf16) |
| Best for | Tabular data | Non-linear patterns |
| Training speed | Faster | Slower |
| Interpretability | High | Low |
