# covenant-ml

Pluggable ML backends for covenant breach risk prediction: training, validation, and inference. Supports five backends: XGBoost (gradient boosting), MLP (neural networks), LSTM (temporal sequences), LightGBM (large-scale datasets), and ClearGBM (interpretable pure-Python gradient boosting).

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

## LSTM Backend

Train an LSTM classifier for temporal bankruptcy sequences:

```python
from pathlib import Path
from covenant_ml.types import LSTMConfig
from covenant_ml.backends.lstm import create_lstm_backend

# Configure LSTM training
config: LSTMConfig = {
    "device": "auto",
    "precision": "fp32",
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True,
    "sequence_length": 5,
    "learning_rate": 0.001,
    "batch_size": 32,
    "n_epochs": 100,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "early_stopping_patience": 10,
}

# Create backend and train
backend = create_lstm_backend()
outcome = backend.train(
    x_features=X,
    y_labels=y,
    feature_names=feature_names,
    config=config,
    output_dir=Path("/models"),
    progress=on_progress,
)

print(f"Test AUC: {outcome['test_metrics']['auc']}")
print(f"Model format: {outcome['model_format']}")  # "pt"
```

### LSTMConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `device` | str | `"cpu"`, `"cuda"`, or `"auto"` |
| `precision` | str | `"fp32"`, `"fp16"`, `"bf16"`, or `"auto"` |
| `hidden_size` | int | LSTM hidden state size |
| `num_layers` | int | Number of stacked LSTM layers |
| `dropout` | float | Dropout rate between layers (0.0-1.0) |
| `bidirectional` | bool | Use bidirectional LSTM |
| `sequence_length` | int | Number of time periods per sequence |
| `learning_rate` | float | Learning rate |
| `batch_size` | int | Training batch size |
| `n_epochs` | int | Maximum training epochs |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_patience` | int | Epochs without improvement before stopping |

## LightGBM Backend

Train a LightGBM classifier for large-scale tabular data:

```python
from pathlib import Path
from covenant_ml.types import LightGBMConfig
from covenant_ml.backends.lightgbm import create_lightgbm_backend

# Configure LightGBM training
config: LightGBMConfig = {
    "device": "auto",
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "early_stopping_rounds": 10,
}

# Create backend and train
backend = create_lightgbm_backend()
outcome = backend.train(
    x_features=X,
    y_labels=y,
    feature_names=feature_names,
    config=config,
    output_dir=Path("/models"),
    progress=on_progress,
)

print(f"Test AUC: {outcome['test_metrics']['auc']}")
print(f"Top feature: {outcome['feature_importances'][0]['name']}")
```

### LightGBMConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `device` | str | `"cpu"`, `"cuda"`, or `"auto"` |
| `learning_rate` | float | Learning rate (alias: eta) |
| `max_depth` | int | Maximum tree depth |
| `n_estimators` | int | Number of boosting rounds |
| `num_leaves` | int | Maximum leaves per tree (LightGBM-specific) |
| `min_child_samples` | int | Minimum samples per leaf (LightGBM-specific) |
| `subsample` | float | Row sampling ratio |
| `colsample_bytree` | float | Column sampling ratio |
| `reg_alpha` | float | L1 regularization |
| `reg_lambda` | float | L2 regularization |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_rounds` | int | Rounds without improvement before stopping |

## ClearGBM Backend

Train a ClearGBM classifier for interpretable gradient boosting with pure Python:

```python
from pathlib import Path
from covenant_ml.types import ClearGBMConfig
from covenant_ml.backends.cleargbm import create_cleargbm_backend

# Configure ClearGBM training
config: ClearGBMConfig = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_bins": 64,
    "subsample": 1.0,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "early_stopping_rounds": 10,
}

# Create backend and train
backend = create_cleargbm_backend()
outcome = backend.train(
    x_features=X,
    y_labels=y,
    feature_names=feature_names,
    config=config,
    output_dir=Path("/models"),
    progress=on_progress,
)

print(f"Test AUC: {outcome['test_metrics']['auc']}")
print(f"Top feature: {outcome['feature_importances'][0]['name']}")
```

### ClearGBMConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | int | Number of boosting rounds |
| `max_depth` | int | Maximum tree depth |
| `learning_rate` | float | Shrinkage factor for updates |
| `min_samples_split` | int | Minimum samples to split a node |
| `min_samples_leaf` | int | Minimum samples in a leaf |
| `max_bins` | int | Histogram bins for O(K) split finding (default: 64) |
| `subsample` | float | Row subsampling ratio (1.0 = no subsampling) |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_rounds` | int | Rounds without improvement before stopping |

### ClearGBM Features

- **Pure Python**: No C++ dependencies, runs anywhere Python runs
- **Interpretable**: Built-in rule extraction and feature contributions
- **Histogram-based**: LightGBM-style O(K) split finding for efficient training
- **Strict typing**: 100% typed with TypedDicts and Protocols

### TrainConfig Fields (XGBoost)

| Field | Type | Description |
|-------|------|-------------|
| `device` | str | `"cpu"`, `"cuda"`, or `"auto"` |
| `learning_rate` | float | Learning rate (alias: eta) |
| `max_depth` | int | Maximum tree depth |
| `n_estimators` | int | Number of boosting rounds |
| `subsample` | float | Row sampling ratio |
| `colsample_bytree` | float | Column sampling ratio |
| `reg_alpha` | float | L1 regularization |
| `reg_lambda` | float | L2 regularization |
| `train_ratio` | float | Training set ratio |
| `val_ratio` | float | Validation set ratio |
| `test_ratio` | float | Test set ratio |
| `random_state` | int | Random seed |
| `early_stopping_rounds` | int | Rounds without improvement before stopping |
| `scale_pos_weight` | float | Optional: positive class weight for imbalanced data |

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
| `model_path` | str | Path to saved model file |
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
| `feature_importances` | list[FeatureImportance] | Ranked feature importances (XGBoost/LightGBM only) |
| `config` | ClassifierTrainConfig | Training configuration (union of all config types) |
| `scale_pos_weight_computed` | float | Auto-calculated class weight used for training |

### EvalMetrics Fields

| Field | Type | Description |
|-------|------|-------------|
| `loss` | float | Log loss (cross-entropy) |
| `ppl` | float | Perplexity (exp(loss)) |
| `auc` | float | Area under ROC curve |
| `accuracy` | float | Classification accuracy |
| `precision` | float | Precision for breach class |
| `recall` | float | Recall for breach class |
| `f1_score` | float | F1 score |

### FeatureImportance Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Feature name |
| `importance` | float | Importance score (gain-based) |
| `rank` | int | Rank (1 = most important) |

### TrainProgress Fields

| Field | Type | Description |
|-------|------|-------------|
| `round` | int | Current training round |
| `total_rounds` | int | Total training rounds |
| `train_loss` | float | Training loss |
| `train_auc` | float | Training AUC |
| `val_loss` | float \| None | Validation loss (None if no validation) |
| `val_auc` | float \| None | Validation AUC (None if no validation) |

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

## Preprocessing

Automatic preprocessing pipeline applied to all backends. Fits on training data only to prevent data leakage.

```python
from covenant_ml.trainer import preprocess_data_splits, stratified_split

# Split data
splits = stratified_split(X, y, 0.7, 0.15, 0.15, random_state=42)

# Preprocess (fits on train, transforms all splits)
preprocessed = preprocess_data_splits(splits)

# Access preprocessed data
print(f"Train shape: {preprocessed.x_train.shape}")
print(f"Preprocessing state: {preprocessed.state}")
```

### Preprocessing Pipeline

The `AutoPreprocessor` applies these transforms in order:

| Step | Description | Details |
|------|-------------|---------|
| 1. Special Code Detection | Replace sentinel values with NaN | 96, 98, 99, 999, -1, -9, -999 |
| 2. Outlier Capping | Cap extreme values | 1st/99th percentile bounds |
| 3. Missing Imputation | Fill NaN with median | Per-feature median from training data |
| 4. Z-Score Normalization | Standardize features | Mean=0, std=1 using training stats |

### Using AutoPreprocessor Directly

```python
from covenant_ml.preprocessing import AutoPreprocessor, PreprocessingState

# Create preprocessor
preprocessor = AutoPreprocessor()

# Fit on training data only
state: PreprocessingState = preprocessor.fit(x_train, y_train)

# Transform any split using fitted state
x_train_processed = preprocessor.transform(x_train, state)
x_val_processed = preprocessor.transform(x_val, state)
x_test_processed = preprocessor.transform(x_test, state)
```

### PreprocessingState Fields

| Field | Type | Description |
|-------|------|-------------|
| `n_features` | int | Number of features |
| `outlier_bounds` | tuple[OutlierBounds, ...] | Per-feature lower/upper bounds |
| `special_codes` | tuple[SpecialCodeSpec, ...] | Per-feature detected special codes |
| `imputation_values` | tuple[ImputationSpec, ...] | Per-feature imputation values |
| `feature_means` | NDArray[np.float64] | Per-feature means for z-score |
| `feature_stds` | NDArray[np.float64] | Per-feature stds for z-score |

### PreprocessedDataSplits

Container returned by `preprocess_data_splits()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `x_train` | NDArray[np.float64] | Preprocessed training features |
| `y_train` | NDArray[np.int64] | Training labels |
| `x_val` | NDArray[np.float64] | Preprocessed validation features |
| `y_val` | NDArray[np.int64] | Validation labels |
| `x_test` | NDArray[np.float64] | Preprocessed test features |
| `y_test` | NDArray[np.int64] | Test labels |
| `state` | PreprocessingState | Fitted preprocessing state |
| `n_train` | int | Number of training samples |
| `n_val` | int | Number of validation samples |
| `n_test` | int | Number of test samples |
| `n_total` | int | Total samples across all splits |

## API Reference

### Training Functions

| Function | Description |
|----------|-------------|
| `train_model_with_validation` | Train with validation and early stopping |
| `train_model` | Train without validation (simpler API) |
| `save_model` | Save trained model to file |
| `stratified_split` | Split data with stratification |
| `preprocess_data_splits` | Apply full preprocessing pipeline to splits |

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
| `LSTMConfig` | LSTM sequence model configuration |
| `LightGBMConfig` | LightGBM gradient boosting configuration |
| `ClearGBMConfig` | ClearGBM pure-Python gradient boosting configuration |
| `ClassifierTrainConfig` | Union of all backend config types |
| `TrainOutcome` | Complete training result |
| `TrainProgress` | Progress update during training |
| `EvalMetrics` | Evaluation metrics for a split |
| `FeatureImportance` | Feature importance entry (name, importance, rank) |
| `DataSplits` | Train/val/test data splits |
| `PreprocessedDataSplits` | Preprocessed train/val/test splits with state |
| `ProgressCallback` | Callback type for progress updates |
| `BackendName` | Literal type: `"xgboost" | "mlp" | "lstm" | "lightgbm" | "cleargbm"` |

### Preprocessing Types (covenant_ml.preprocessing)

| Type | Description |
|------|-------------|
| `AutoPreprocessor` | Preprocessor with fit/transform interface |
| `PreprocessingState` | Fitted state (outlier bounds, imputation values, z-score stats) |
| `OutlierBounds` | Per-feature outlier lower/upper bounds |
| `SpecialCodeSpec` | Per-feature special code definitions |
| `ImputationSpec` | Per-feature imputation values |

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
| `PredictorProtocol` | Any model with predict_proba |

### Feature Engineering

| Type/Function | Description |
|---------------|-------------|
| `FeatureEngineeringConfig` | TypedDict for transform configuration |
| `EngineeredFeatures` | TypedDict with transformed features and counts |
| `FeaturePreset` | Literal type: `"minimal" \| "standard" \| "full"` |
| `engineer_features()` | Apply all configured transforms |
| `default_feature_config()` | Get default configuration |
| `get_feature_config_for_preset()` | Get config for a preset |
| `compute_pairwise_ratios()` | Compute Xi/Xj ratio features |
| `compute_pairwise_products()` | Compute Xi*Xj product features |
| `compute_log_transforms()` | Compute log(1+\|x\|)*sign(x) features |

### Dataset Loading (covenant_ml.datasets)

| Type | Description |
|------|-------------|
| `DatasetConfig` | TypedDict for dataset configuration |
| `LoadedDataset` | TypedDict with loaded features, labels, metadata |
| `DatasetMeta` | TypedDict with dataset metadata (samples, features, encodings) |
| `CategoricalEncoding` | TypedDict for categorical column label encoding |
| `TargetColumnSpec` | TypedDict for target column configuration |
| `FileFormat` | Literal type: `"csv" \| "arff"` |
| `FileEncoding` | Literal type: `"utf-8" \| "latin-1" \| ...` |
| `DatasetRegistry` | Registry of dataset configurations |
| `DatasetLoader` | Loads datasets from files |
| `DatasetLoaderProtocol` | Protocol for dataset loaders |
| `make_default_registry()` | Create registry with pre-configured datasets |
| `create_dataset_loader()` | Create default dataset loader |
| `TimeSeriesDatasetConfig` | TypedDict for time-series dataset configuration |
| `TimeSeriesSpec` | TypedDict for time-series specification |
| `AggregationStrategy` | Literal type: `"last" \| "first" \| "mean" \| "statistics"` |
| `TimeSeriesDatasetRegistry` | Registry for time-series dataset configurations |
| `TimeSeriesCSVLoader` | Loader for time-series CSV datasets |
| `make_default_timeseries_registry()` | Create registry with pre-configured time-series datasets |
| `create_timeseries_csv_loader()` | Create time-series CSV loader |

### Explainers (covenant_ml.explainers)

| Type | Description |
|------|-------------|
| `ExplainerRegistry` | Registry of available explainers |
| `ExplainerRegistration` | Registration entry with factory and metadata |
| `FeatureImportanceScore` | TypedDict with name, importance, rank |
| `SupportedExplainer` | Literal type of explainer names |
| `ExplainerCapabilities` | TypedDict with requirements and cost |
| `PermutationConfig` | Config for permutation explainer |
| `GradientConfig` | Config for gradient explainer |
| `IntegratedGradientsConfig` | Config for integrated gradients |
| `default_explainer_registry()` | Create registry with all explainers |

### Hyperparameter Optimization (covenant_ml.optimizer)

| Type | Description |
|------|-------------|
| `OptimizationConfig` | TypedDict for optimization settings |
| `OptimizationSummary` | TypedDict with optimization results |
| `TrialResult` | TypedDict for single trial result |
| `XGBoostSearchSpace` | TypedDict for XGBoost hyperparameters |
| `LightGBMSearchSpace` | TypedDict for LightGBM hyperparameters |
| `ClearGBMSearchSpace` | TypedDict for ClearGBM hyperparameters |
| `MLPSearchSpace` | TypedDict for MLP hyperparameters |
| `LSTMSearchSpace` | TypedDict for LSTM hyperparameters |
| `FloatRangeSpec` | Float parameter range specification |
| `IntRangeSpec` | Integer parameter range specification |
| `OptunaXGBoostOptimizer` | XGBoost hyperparameter optimizer |
| `OptunaLightGBMOptimizer` | LightGBM hyperparameter optimizer |
| `OptunaClearGBMOptimizer` | ClearGBM hyperparameter optimizer |
| `OptunaMLPOptimizer` | MLP hyperparameter optimizer |
| `OptunaLSTMOptimizer` | LSTM hyperparameter optimizer |

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
- xgboost 2.0.0+ (XGBoost backend)
- torch 2.0.0+ (MLP and LSTM backends)
- lightgbm 4.0.0+ (LightGBM backend)
- scikit-learn 1.5.0+
- numpy 1.26.0+
- optuna 3.0.0+ (hyperparameter optimization)
- 100% test coverage enforced

## Backend Comparison

| Aspect | XGBoost | MLP | LSTM | LightGBM | ClearGBM |
|--------|---------|-----|------|----------|----------|
| Model format | `.ubj` | `.pt` | `.pt` | `.txt` | `.json` |
| Feature importances | Yes | No | No | Yes | Yes |
| GPU support | CUDA | CUDA (fp16/bf16) | CUDA (fp16/bf16) | CUDA | CPU only |
| Best for | Tabular data | Non-linear patterns | Temporal sequences | Large datasets | Interpretability |
| Training speed | Fast | Moderate | Slow | Very fast | Moderate |
| Interpretability | High | Low | Low | High | Very high |
| Early stopping | Yes | Yes | Yes | Yes | Yes |
| Rule extraction | Post-hoc | No | No | Post-hoc | Built-in |
| Dependencies | C++ lib | PyTorch | PyTorch | C++ lib | Python stdlib |

## Feature Engineering

Create derived features from raw financial ratios to improve model performance:

```python
from covenant_ml import (
    engineer_features,
    default_feature_config,
    get_feature_config_for_preset,
    FeatureEngineeringConfig,
    EngineeredFeatures,
)
import numpy as np

# Raw features
X = np.array([[0.5, 1.2, 3.0], [0.8, 0.9, 2.5]])
feature_names = ["debt_ratio", "interest_cover", "current_ratio"]

# Use default config
config = default_feature_config()
result: EngineeredFeatures = engineer_features(X, feature_names, config)

print(f"Original: {result['n_original']} features")
print(f"Ratios: {result['n_ratios']} features")
print(f"Products: {result['n_products']} features")
print(f"Log transforms: {result['n_log']} features")
print(f"Total: {len(result['feature_names'])} features")

# Use preset for different scenarios
config = get_feature_config_for_preset("minimal")  # Original + log only
config = get_feature_config_for_preset("standard")  # + ratios
config = get_feature_config_for_preset("full")      # + products
```

### Feature Transforms

| Transform | Description | Example |
|-----------|-------------|---------|
| Pairwise Ratios | Xi/Xj for relative relationships | debt_ratio/interest_cover |
| Pairwise Products | Xi*Xj for interaction effects | debt_ratio*current_ratio |
| Log Transforms | log(1 + \|x\|) * sign(x) for skewed data | log_debt_ratio |

### FeatureEngineeringConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `use_ratios` | bool | Include pairwise ratio features |
| `use_products` | bool | Include pairwise product features |
| `use_log_transforms` | bool | Include log-transformed features |
| `max_ratio_features` | int | Limit ratio features (0 = no limit) |
| `max_product_features` | int | Limit product features (0 = no limit) |

### FeaturePreset Options

| Preset | Ratios | Products | Log | Description |
|--------|--------|----------|-----|-------------|
| `"minimal"` | No | No | Yes | Original + log transforms only |
| `"standard"` | Yes | No | Yes | Default: ratios but no products |
| `"full"` | Yes | Yes | Yes | All transforms enabled |

## Dataset Loading

Pluggable dataset loading system with auto-detection of target columns:

```python
from pathlib import Path
from covenant_ml.datasets import (
    make_default_registry,
    create_dataset_loader,
    DatasetConfig,
    LoadedDataset,
)

# Get registry with pre-configured datasets
registry = make_default_registry()

# List available datasets
print(registry.list())  # ["taiwan", "polish", "us", ...]

# Get config for a specific dataset
config: DatasetConfig = registry.get("taiwan")
print(f"Format: {config['format']}")  # "csv"
print(f"Target: {config['target_column']}")

# Load dataset
loader = create_dataset_loader()
dataset: LoadedDataset = loader.load(config, Path("data/external"))

print(f"Features: {dataset['x'].shape}")  # (n_samples, n_features)
print(f"Labels: {dataset['y'].shape}")    # (n_samples,)
print(f"Feature names: {dataset['meta']['feature_names']}")
```

### Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| `"csv"` | `.csv` | Comma-separated values |
| `"arff"` | `.arff` | Weka ARFF format |

### DatasetConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Dataset identifier |
| `path` | str | Relative path within data directory |
| `format` | FileFormat | `"csv"` or `"arff"` |
| `target_column` | TargetColumnSpec | Target column config (name/index + positive value) |
| `encoding` | FileEncoding | `"utf-8"`, `"latin-1"`, etc. |
| `description` | str | Human-readable description |

### LoadedDataset Fields

| Field | Type | Description |
|-------|------|-------------|
| `meta` | DatasetMeta | Dataset metadata with statistics |
| `x` | NDArray[np.float64] | Feature matrix (n_samples, n_features) |
| `y` | NDArray[np.int64] | Labels (n_samples,) - 0=healthy, 1=breach |

### DatasetMeta Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Dataset identifier |
| `n_samples` | int | Total number of samples |
| `n_features` | int | Number of feature columns |
| `n_positive` | int | Number of positive class samples |
| `n_negative` | int | Number of negative class samples |
| `positive_ratio` | float | Fraction of positive samples |
| `feature_names` | tuple[str, ...] | Ordered tuple of feature column names |
| `categorical_encodings` | tuple[CategoricalEncoding, ...] | Encodings for categorical columns (empty if none) |

### CategoricalEncoding Fields

| Field | Type | Description |
|-------|------|-------------|
| `column_name` | str | Name of the encoded column |
| `mapping` | tuple[tuple[str, int], ...] | (value, code) pairs sorted alphabetically by value |
| `n_categories` | int | Number of unique categories including missing |

## Time-Series Dataset Loading

Load time-series datasets where each entity has multiple observations over time:

```python
from pathlib import Path
from covenant_ml.datasets import (
    make_default_timeseries_registry,
    DatasetLoader,
    create_dataset_loader,
)
from covenant_ml.datasets.loaders import TimeSeriesCSVLoader

# Get registry with pre-configured time-series datasets
registry = make_default_timeseries_registry()

# List available time-series datasets
print(registry.list_names())  # ("kaggle_amex_default",)

# Get config for AMEX dataset
config = registry.get("kaggle_amex_default")
print(f"Entity column: {config['time_series']['entity_column']}")  # "customer_ID"
print(f"Time column: {config['time_series']['time_column']}")      # "S_2"
print(f"Aggregation: {config['time_series']['aggregation']}")      # "last"

# Load time-series dataset
loader = TimeSeriesCSVLoader()
dataset = loader.load(config, Path("data/external"))

print(f"Entities: {dataset['meta']['n_samples']}")
print(f"Features: {dataset['meta']['n_features']}")
```

### Aggregation Strategies

Time-series data is aggregated per entity into a single feature vector:

| Strategy | Description | Output per Feature |
|----------|-------------|-------------------|
| `"last"` | Take most recent observation | 1 |
| `"first"` | Take oldest observation | 1 |
| `"mean"` | Average all observations | 1 |
| `"statistics"` | Compute mean, std, min, max | 4 |

### Competition Feature Engineering

Additional features for Kaggle-style competitions:

| Feature Type | Description | Output per Feature |
|--------------|-------------|-------------------|
| **Rank** | Per-entity percentile (0.0-1.0) of final value | 1 |
| **Diff** | Row-to-row differences: mean, std, min, max, last | 5 |
| **Window** | Stats over last N observations: mean, std, min, max | 4 per window size |

```python
from covenant_ml.datasets.types import TimeSeriesDatasetConfig, TimeSeriesSpec, TargetColumnSpec

# Configure time-series dataset with all competition features
config = TimeSeriesDatasetConfig(
    name="my_timeseries",
    display_name="My Time Series Dataset",
    folder="my_data",
    file_name="features.csv",
    file_format="csv",
    encoding="utf-8",
    target=TargetColumnSpec(
        column_name="target",
        label_type="binary_int",
        positive_values=(1,),
        negative_values=(0,),
    ),
    exclude_columns=(),
    n_samples_expected=1000,
    n_features_expected=50,
    positive_class_ratio_expected=0.05,
    time_series=TimeSeriesSpec(
        entity_column="customer_id",
        time_column="date",
        aggregation="statistics",         # 50 * 4 = 200 features
        labels_file="labels.csv",
        labels_entity_column="customer_id",
        include_rank_features=True,       # + 50 = 250 features
        include_diff_features=True,       # + 50 * 5 = 500 features
        include_window_features=True,     # Enable window aggregations
        window_sizes=(3, 6),              # + 50 * 4 * 2 = 900 features
    ),
)
# Total: 900 features from 50 base columns
```

### Feature Count Formula

For N base features:
- Base aggregation (`"statistics"`): N * 4
- With `include_rank_features=True`: + N
- With `include_diff_features=True`: + N * 5
- With `include_window_features=True, window_sizes=(3, 6)`: + N * 4 * len(window_sizes)

### TimeSeriesSpec Fields

| Field | Type | Description |
|-------|------|-------------|
| `entity_column` | str | Column identifying unique entities |
| `time_column` | str | Column for temporal ordering |
| `aggregation` | AggregationStrategy | `"last"`, `"first"`, `"mean"`, or `"statistics"` |
| `labels_file` | str | Separate CSV file containing entity labels |
| `labels_entity_column` | str | Entity column name in labels file |
| `include_rank_features` | bool | Add per-entity percentile rank features |
| `include_diff_features` | bool | Add row-to-row difference features |
| `include_window_features` | bool | Add window aggregation features |
| `window_sizes` | tuple[int, ...] | Window sizes for window features (e.g., `(3, 6)`) |

### Memory-Efficient Implementation

The time-series loader uses Polars-native groupby operations for memory efficiency on large datasets (16GB+):

```
CSV file → Polars DataFrame → Polars groupby → NumPy arrays
```

This avoids Python list conversions that would multiply memory usage 3-4x. Key optimizations:

- **Polars-native aggregation**: All groupby operations (last, first, mean, statistics) run in Polars without Python intermediaries
- **Categorical encoding**: Uses Polars when/then/otherwise instead of Python loops
- **Parquet caching**: Repeat loads are 10-100x faster via `.cache/<hash>/` directories
- **Progress callbacks**: Real-time progress reporting during loading phases

### Time-Series Types

| Type | Description |
|------|-------------|
| `TimeSeriesDatasetConfig` | TypedDict extending DatasetConfig with time_series field |
| `TimeSeriesSpec` | TypedDict for time-series configuration |
| `AggregationStrategy` | Literal type: `"last" \| "first" \| "mean" \| "statistics"` |
| `TimeSeriesDatasetRegistry` | Registry for time-series dataset configurations |
| `TimeSeriesCSVLoader` | Loader for time-series CSV datasets |

## Cross-Validation

Stratified k-fold cross-validation with optional group constraints:

```python
from covenant_ml.validation import (
    stratified_kfold_split,
    group_stratified_kfold_split,
    run_cross_validation,
    run_group_cross_validation,
    compute_oof_metrics,
)

# Standard stratified k-fold (maintains class proportions)
splits = stratified_kfold_split(y=labels, n_folds=5, random_state=42)

# Group-stratified k-fold (ensures no entity appears in both train and val)
# Critical for time-series data to prevent data leakage
splits = group_stratified_kfold_split(
    y=labels,
    groups=customer_ids,  # NDArray[np.int64] mapping samples to entities
    n_folds=5,
    random_state=42,
)

# Run full CV pipeline with trainer function
cv_result = run_group_cross_validation(
    x=features,
    y=labels,
    groups=customer_ids,
    n_folds=5,
    random_state=42,
    trainer=my_trainer_fn,
)

# Compute OOF metrics
oof_metrics = compute_oof_metrics(labels, cv_result)
print(f"OOF AUC: {oof_metrics['oof_auc']:.4f}")
```

### GroupKFold for Time-Series

When training on time-series data (e.g., AMEX competition), standard k-fold causes data leakage because:
- The same customer appears in both train and validation
- Model learns customer-specific patterns rather than generalizable features
- CV scores are inflated (~0.95) vs realistic test scores (~0.80)

`group_stratified_kfold_split` ensures:
- All samples from a customer stay in the same fold
- Groups are stratified by label (positive if any sample is positive)
- No customer appears in both train and validation

### Cross-Validation Types

| Type | Description |
|------|-------------|
| `CVSplit` | Single fold with train/val indices |
| `CVSplitInfo` | All folds with metadata |
| `CVResult` | Complete CV results with OOF predictions |
| `FoldResult` | Single fold training result |
| `FoldTrainer` | Protocol for fold training function |
| `OOFMetrics` | Out-of-fold evaluation metrics |

### Cross-Validation Functions

| Function | Description |
|----------|-------------|
| `stratified_kfold_split` | Create stratified train/val splits |
| `group_stratified_kfold_split` | Create group-aware splits (no entity leakage) |
| `run_cross_validation` | Execute k-fold CV with trainer |
| `run_group_cross_validation` | Execute group-aware CV |
| `compute_oof_metrics` | Compute metrics from OOF predictions |
| `get_fold_data` | Extract train/val data for a fold |

## Feature Importance Explainers

Registry-based feature importance explainers with backend compatibility:

```python
from covenant_ml.explainers import (
    default_explainer_registry,
    ExplainerRegistry,
    FeatureImportanceScore,
)

# Get default registry with all explainers
registry = default_explainer_registry()

# List all explainers
print(registry.list_explainers())
# ["gradient", "integrated_gradients", "permutation", "shap_tree"]

# List compatible explainers for a backend
compatible = registry.list_compatible_explainers("mlp")
# ["gradient", "integrated_gradients", "permutation"]

compatible = registry.list_compatible_explainers("xgboost")
# ["permutation", "shap_tree"]

# Check compatibility
if registry.is_compatible("gradient", "mlp"):
    explainer = registry.get("gradient")

    importance: list[FeatureImportanceScore] = explainer.compute_importance(
        model=model,
        x_data=x_test,
        feature_names=["debt_ratio", "interest_cover", "current_ratio"],
        target_class=1,
    )

    for score in importance:
        print(f"{score['rank']}. {score['name']}: {score['importance']:.4f}")
```

### Available Explainers

| Explainer | Compatible Backends | Requires Gradients | Speed |
|-----------|--------------------|--------------------|-------|
| `permutation` | All (xgboost, lightgbm, cleargbm, mlp, lstm) | No | Medium |
| `gradient` | Neural nets (mlp, lstm) | Yes | Fast |
| `integrated_gradients` | Neural nets (mlp, lstm) | Yes | Slow |
| `shap_tree` | Tree models (xgboost, lightgbm, cleargbm) | No | Medium |

### Explainer Types

| Type | Description |
|------|-------------|
| `ExplainerRegistry` | Registry of available explainers |
| `ExplainerRegistration` | Registration entry with factory and metadata |
| `FeatureImportanceScore` | Result with name, importance, rank |
| `SupportedExplainer` | Literal type of explainer names |
| `ExplainerCapabilities` | TypedDict with requirements and cost |

## Hyperparameter Optimization

Bayesian optimization using Optuna's TPE algorithm:

```python
from pathlib import Path
from covenant_ml.optimizer import (
    create_xgboost_optimizer,
    create_xgboost_objective,
    make_xgboost_default_space,
    make_default_optimization_config,
    use_real_optuna,
    OptimizationSummary,
)

# Set up Optuna hook at application startup
use_real_optuna()

# Create optimizer, objective, and search space
optimizer = create_xgboost_optimizer()
objective = create_xgboost_objective(output_dir=Path("/models"))
space = make_xgboost_default_space()
config = make_default_optimization_config(n_trials=100)

# Run optimization
summary: OptimizationSummary = optimizer.optimize(
    x_features=X,
    y_labels=y,
    feature_names=feature_names,
    search_space=space,
    config=config,
    objective=objective,
)

print(f"Best AUC: {summary['best_value']:.4f}")
print(f"Best params: {summary['best_params']}")
print(f"Trials: {summary['n_trials']}")
```

### Optimizer Factory Functions

| Function | Backend | Description |
|----------|---------|-------------|
| `create_xgboost_optimizer()` | XGBoost | Create XGBoost hyperparameter optimizer |
| `create_lightgbm_optimizer()` | LightGBM | Create LightGBM hyperparameter optimizer |
| `create_cleargbm_optimizer()` | ClearGBM | Create ClearGBM hyperparameter optimizer |
| `create_mlp_optimizer()` | MLP | Create MLP hyperparameter optimizer |
| `create_lstm_optimizer()` | LSTM | Create LSTM hyperparameter optimizer |

### Search Space Functions

| Function | Description |
|----------|-------------|
| `make_xgboost_default_space()` | Default XGBoost search space (includes DART booster) |
| `make_xgboost_focused_space()` | Narrower space for fine-tuning |
| `make_lightgbm_default_space()` | Default LightGBM search space (includes DART boosting) |
| `make_lightgbm_focused_space()` | Narrower LightGBM space for fine-tuning |
| `make_cleargbm_default_space()` | Default ClearGBM search space |
| `make_cleargbm_focused_space()` | Narrower ClearGBM space for fine-tuning |
| `make_mlp_default_space()` | Default MLP search space |
| `make_lstm_default_space()` | Default LSTM search space |
| `make_default_optimization_config(n_trials)` | Default optimization config |

### DART Boosting Support

Both XGBoost and LightGBM search spaces include DART (Dropouts meet Multiple Additive Regression Trees) as an optional boosting method. DART applies dropout regularization during boosting to reduce overfitting.

**XGBoost DART Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `booster` | categorical | `"gbtree"`, `"dart"` | Boosting algorithm (DART enables dropout) |
| `rate_drop` | float | 0.0-0.5 | Dropout rate for trees (only when booster="dart") |
| `skip_drop` | float | 0.0-0.5 | Probability of skipping dropout (only when booster="dart") |

**LightGBM DART Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `boosting_type` | categorical | `"gbdt"`, `"dart"` | Boosting type (DART enables dropout) |
| `drop_rate` | float | 0.0-0.5 | Dropout rate for trees (only when boosting_type="dart") |
| `skip_drop` | float | 0.0-0.5 | Probability of skipping dropout (only when boosting_type="dart") |
| `feature_fraction` | float | 0.02-0.1 | Aggressive feature subsampling for DART regularization (only when boosting_type="dart") |

DART parameters are conditionally sampled only when the DART booster/boosting_type is selected by Optuna during optimization. This allows Optuna to explore both standard gradient boosting and DART configurations to find the optimal approach for your dataset.

**Note:** Early stopping is automatically disabled for LightGBM DART mode. DART's random tree dropout causes validation metrics to fluctuate, making early stopping unreliable. When DART is selected, training runs for the full `n_estimators` rounds.

### OptimizationConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `n_trials` | int | Number of optimization trials |
| `timeout_seconds` | int \| None | Optional timeout |
| `n_jobs` | int | Parallel jobs (-1 = all cores) |
| `direction` | str | `"maximize"` or `"minimize"` |
| `sampler_seed` | int | Random seed for reproducibility |

### OptimizationSummary Fields

| Field | Type | Description |
|-------|------|-------------|
| `best_value` | float | Best objective value (e.g., AUC) |
| `best_params` | dict | Best hyperparameters found |
| `best_trial_number` | int | Trial number of best result |
| `n_trials` | int | Total trials completed |
| `n_failed` | int | Number of failed trials |
| `duration_seconds` | float | Total optimization time |
| `all_trials` | list[TrialResult] | All trial results |

### Testing with Fake Optuna

For tests, use the hook system instead of real Optuna:

```python
from covenant_ml.optimizer import set_optuna_module_hook

# Create a fake Optuna module for testing
class FakeStudy:
    def optimize(self, func, n_trials, **kwargs):
        # Minimal implementation for tests
        pass

class FakeOptuna:
    def create_study(self, **kwargs):
        return FakeStudy()

# Set hook before running tests
set_optuna_module_hook(lambda: FakeOptuna())
```
