# ML Endpoints

Machine learning endpoints for model training, optimization, prediction, and explanation.

---

## POST /ml/predict

Predict breach risk for a deal.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `deal_id` | string | Yes | Deal UUID |

**Request Example:**
```json
{
  "deal_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
}
```

**Response (200):**
```json
{
  "deal_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
  "probability": 0.23,
  "risk_tier": "LOW"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `deal_id` | string | Deal UUID |
| `probability` | float | Breach probability (0.0-1.0) |
| `risk_tier` | string | `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` |

**Risk Tier Thresholds:**

| Tier | Probability Range | Description |
|------|-------------------|-------------|
| `LOW` | < 0.25 | Normal risk, no action needed |
| `MEDIUM` | 0.25 - 0.50 | Elevated risk, monitor |
| `HIGH` | 0.50 - 0.80 | High risk, review required |
| `CRITICAL` | >= 0.80 | Critical risk, immediate action |

---

## POST /ml/train

Enqueue a model training job using internal deal/measurement data.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `learning_rate` | float | Yes | - | Learning rate |
| `max_depth` | int | Yes | - | XGBoost max tree depth |
| `n_estimators` | int | Yes | - | Number of trees |
| `subsample` | float | Yes | - | Row subsample ratio |
| `colsample_bytree` | float | Yes | - | Column subsample ratio |
| `random_state` | int | Yes | - | Random seed |
| `train_ratio` | float | Yes | - | Training set ratio (e.g., 0.7) |
| `val_ratio` | float | Yes | - | Validation set ratio (e.g., 0.15) |
| `test_ratio` | float | Yes | - | Test set ratio (e.g., 0.15) |
| `early_stopping_rounds` | int | Yes | - | Early stopping patience |
| `device` | string | No | `auto` | `cpu`, `cuda`, or `auto` |
| `reg_alpha` | float | No | `0.0` | L1 regularization strength |
| `reg_lambda` | float | No | `1.0` | L2 regularization strength |
| `scale_pos_weight` | float | No | auto | Class weight for positives. Auto-calculated as (n_negative / n_positive) if omitted |

**Device Options:**
- `"cpu"` - Force CPU training (slower but always available)
- `"cuda"` - Force GPU training (requires NVIDIA GPU with CUDA)
- `"auto"` - Auto-detect: uses GPU if available, falls back to CPU

**Class Imbalance:** If `scale_pos_weight` is omitted, it's automatically calculated as (n_negative / n_positive) from the training data.

**Request Example (GPU with auto-detect):**
```json
{
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
  "device": "auto"
}
```

**Response (202):**
```json
{
  "job_id": "train-job-uuid",
  "status": "queued"
}
```

**Job Result (when complete):**

Poll `/ml/jobs/{job_id}` to get the result:

```json
{
  "job_id": "train-job-uuid",
  "status": "finished",
  "result": {
    "status": "complete",
    "model_id": "model-2024-01-15-143052",
    "model_path": "/data/models/model-2024-01-15-143052.ubj",
    "active_model_path": "/data/models/active.ubj",
    "samples_total": 100,
    "samples_train": 70,
    "samples_val": 15,
    "samples_test": 15,
    "best_val_auc": 0.89,
    "best_round": 45,
    "total_rounds": 100,
    "early_stopped": true,
    "train_metrics": {
      "loss": 0.32,
      "auc": 0.95,
      "accuracy": 0.88,
      "precision": 0.85,
      "recall": 0.82,
      "f1_score": 0.83
    },
    "val_metrics": {
      "loss": 0.41,
      "auc": 0.89,
      "accuracy": 0.84,
      "precision": 0.81,
      "recall": 0.78,
      "f1_score": 0.79
    },
    "test_metrics": {
      "loss": 0.43,
      "auc": 0.87,
      "accuracy": 0.82,
      "precision": 0.79,
      "recall": 0.76,
      "f1_score": 0.77
    },
    "config": {
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
      "reg_lambda": 1.0
    }
  }
}
```

**Result Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Unique model identifier |
| `model_path` | string | Path to saved model file |
| `active_model_path` | string | Path to active model (copied for API use) |
| `samples_*` | int | Sample counts for train/val/test splits |
| `best_val_auc` | float | Best validation AUC achieved |
| `best_round` | int | Round with best validation AUC |
| `early_stopped` | bool | Whether training stopped early |
| `*_metrics` | object | Metrics for train/val/test sets |

**Metrics Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `loss` | float | Log loss |
| `auc` | float | Area under ROC curve |
| `accuracy` | float | Classification accuracy |
| `precision` | float | Precision score |
| `recall` | float | Recall score |
| `f1_score` | float | F1 score |

---

## Automatic Preprocessing

Every backend applies the same preprocessing pipeline before training. It is
fitted on the training split only, to prevent leakage, and the fitted state is
applied unchanged to validation and test. No configuration needed.

| Step | Description |
|------|-------------|
| Special Code Detection | Replaces sentinel values (96, 98, 999, -1, -9, -999) with NaN |
| Outlier Capping | Caps extreme values at 1st/99th percentile bounds |
| Missing Imputation | Fills NaN with per-feature median from training data |
| Z-Score Normalization | Standardizes features to mean=0, std=1 |

---

## POST /ml/train-external

Train on external CSV datasets with pluggable ML backends. Supports all seven classification backends: XGBoost, LightGBM, ClearGBM, LogReg, Random Forest, MLP, and LSTM.

**Common Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dataset` | string | Yes | - | Dataset to use — `taiwan`, `us`, `polish`, `kaggle_company_bankruptcy`, `kaggle_credit_default`, `kaggle_credit_risk`, `kaggle_heloc`, `kaggle_give_me_credit`, `kaggle_loan_default`, or the time-series `kaggle_amex_default` |
| `backend` | string | No | `xgboost` | Backend: `xgboost`, `lightgbm`, `cleargbm`, `logreg`, `random_forest`, `mlp`, or `lstm` |
| `learning_rate` | float | Yes | - | Learning rate |
| `random_state` | int | Yes | - | Random seed |
| `device` | string | No | `auto` | `cpu`, `cuda`, or `auto` |
| `train_ratio` | float | Yes | - | Training set ratio (e.g., 0.7) |
| `val_ratio` | float | Yes | - | Validation set ratio (e.g., 0.15) |
| `test_ratio` | float | Yes | - | Test set ratio (e.g., 0.15) |

**XGBoost-Specific Fields (`backend: "xgboost"`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_depth` | int | Yes | Max tree depth |
| `n_estimators` | int | Yes | Number of trees |
| `subsample` | float | Yes | Row subsample ratio |
| `colsample_bytree` | float | Yes | Column subsample ratio |
| `reg_alpha` | float | Yes | L1 regularization strength |
| `reg_lambda` | float | Yes | L2 regularization strength |
| `early_stopping_rounds` | int | Yes | Early stopping patience |
| `scale_pos_weight` | float | No | Class weight (auto-calculated if omitted) |

**MLP-Specific Fields (`backend: "mlp"`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `precision` | string | Yes | `fp32`, `fp16`, `bf16`, or `auto` |
| `optimizer` | string | Yes | `adamw`, `adam`, or `sgd` |
| `hidden_sizes` | array[int] | Yes | Hidden layer sizes (e.g., `[64, 32]`) |
| `batch_size` | int | Yes | Training batch size |
| `n_epochs` | int | Yes | Maximum training epochs |
| `dropout` | float | Yes | Dropout rate (0.0 to 1.0) |
| `early_stopping_patience` | int | Yes | Epochs without improvement before stopping |

**LightGBM-Specific Fields (`backend: "lightgbm"`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_depth` | int | Yes | Max tree depth |
| `n_estimators` | int | Yes | Number of trees |
| `num_leaves` | int | Yes | Maximum leaves per tree |
| `min_child_samples` | int | Yes | Minimum samples per leaf |
| `subsample` | float | Yes | Row subsample ratio |
| `colsample_bytree` | float | Yes | Column subsample ratio |
| `reg_alpha` | float | Yes | L1 regularization strength |
| `reg_lambda` | float | Yes | L2 regularization strength |
| `early_stopping_rounds` | int | Yes | Early stopping patience |

**LSTM-Specific Fields (`backend: "lstm"`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `precision` | string | Yes | `fp32`, `fp16`, `bf16`, or `auto` |
| `hidden_size` | int | Yes | LSTM hidden state dimension |
| `num_layers` | int | Yes | Number of stacked LSTM layers |
| `dropout` | float | Yes | Dropout rate between layers (0.0 to 1.0) |
| `bidirectional` | bool | Yes | Process sequences in both directions |
| `sequence_length` | int | Yes | Number of time periods per sequence |
| `batch_size` | int | Yes | Training batch size |
| `n_epochs` | int | Yes | Maximum training epochs |
| `early_stopping_patience` | int | Yes | Epochs without improvement before stopping |

**Available Datasets:**
- `taiwan`: 6,819 samples, 95 financial ratio features
- `us`: 78,682 samples, 18 features
- `polish`: 7,027 samples, 64 financial ratio features

**Device Options:**
- `"cpu"` - Force CPU training (slower but always available)
- `"cuda"` - Force GPU training (requires NVIDIA GPU with CUDA)
- `"auto"` - Auto-detect: uses GPU if available, falls back to CPU

**Precision Options (MLP/LSTM):**
- `"fp32"` - Full precision (default, most compatible)
- `"fp16"` - Half precision (faster on GPU, requires CUDA)
- `"bf16"` - BFloat16 (faster on Ampere+ GPUs)
- `"auto"` - Auto-detect based on device

**Class Imbalance:**
- XGBoost: `scale_pos_weight` auto-calculated as (n_negative / n_positive) if omitted
- MLP/LSTM: Weighted BCE loss based on class distribution
- LightGBM: Auto-computed class weights

**Request Example - XGBoost:**
```json
{
  "dataset": "taiwan",
  "backend": "xgboost",
  "device": "auto",
  "learning_rate": 0.1,
  "max_depth": 6,
  "n_estimators": 100,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.0,
  "reg_lambda": 1.0,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "random_state": 42,
  "early_stopping_rounds": 10
}
```

**Request Example - MLP Neural Network:**
```json
{
  "dataset": "taiwan",
  "backend": "mlp",
  "device": "auto",
  "precision": "fp32",
  "optimizer": "adamw",
  "hidden_sizes": [64, 32],
  "learning_rate": 0.001,
  "batch_size": 32,
  "n_epochs": 100,
  "dropout": 0.2,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "random_state": 42,
  "early_stopping_patience": 10
}
```

**Request Example - LightGBM:**
```json
{
  "dataset": "taiwan",
  "backend": "lightgbm",
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
  "early_stopping_rounds": 10
}
```

**Request Example - LSTM:**
```json
{
  "dataset": "taiwan",
  "backend": "lstm",
  "device": "auto",
  "precision": "fp32",
  "hidden_size": 64,
  "num_layers": 2,
  "dropout": 0.2,
  "bidirectional": true,
  "sequence_length": 5,
  "learning_rate": 0.001,
  "batch_size": 32,
  "n_epochs": 100,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "random_state": 42,
  "early_stopping_patience": 10
}
```

**Response (202):**
```json
{
  "job_id": "train-job-uuid",
  "status": "queued"
}
```

**Job Result - XGBoost (when complete):**

Poll `/ml/jobs/{job_id}` to get the result with automatic feature importance ranking:

```json
{
  "job_id": "train-job-uuid",
  "status": "finished",
  "result": {
    "status": "complete",
    "backend": "xgboost",
    "dataset": "taiwan",
    "model_id": "model-2024-01-15-143052",
    "model_path": "/data/models/model-2024-01-15-143052.ubj",
    "model_format": "ubj",
    "active_model_path": "/data/models/active.ubj",
    "samples_total": 6819,
    "samples_train": 4773,
    "samples_val": 1023,
    "samples_test": 1023,
    "n_features": 95,
    "scale_pos_weight": 29.99,
    "best_val_auc": 0.94,
    "best_round": 67,
    "total_rounds": 100,
    "early_stopped": true,
    "train_metrics": {
      "loss": 0.18,
      "ppl": 1.20,
      "auc": 0.98,
      "accuracy": 0.94,
      "precision": 0.91,
      "recall": 0.88,
      "f1_score": 0.89
    },
    "val_metrics": {
      "loss": 0.24,
      "ppl": 1.27,
      "auc": 0.94,
      "accuracy": 0.91,
      "precision": 0.87,
      "recall": 0.84,
      "f1_score": 0.85
    },
    "test_metrics": {
      "loss": 0.26,
      "ppl": 1.30,
      "auc": 0.93,
      "accuracy": 0.90,
      "precision": 0.86,
      "recall": 0.83,
      "f1_score": 0.84
    },
    "feature_importances": [
      {"name": "X6", "importance": 0.1842, "rank": 1},
      {"name": "X1", "importance": 0.0923, "rank": 2},
      {"name": "X5", "importance": 0.0856, "rank": 3},
      {"name": "X9", "importance": 0.0734, "rank": 4},
      {"name": "X3", "importance": 0.0612, "rank": 5}
    ]
  }
}
```

**Job Result - MLP (when complete):**

MLP neural network results use PyTorch format and don't include feature importances:

```json
{
  "job_id": "train-job-uuid",
  "status": "finished",
  "result": {
    "status": "complete",
    "backend": "mlp",
    "dataset": "taiwan",
    "model_id": "model-2024-01-15-143052",
    "model_path": "/data/models/model-2024-01-15-143052.pt",
    "model_format": "pt",
    "active_model_path": "/data/models/active.pt",
    "samples_total": 6819,
    "samples_train": 4773,
    "samples_val": 1023,
    "samples_test": 1023,
    "n_features": 95,
    "scale_pos_weight": 29.99,
    "best_val_auc": 0.92,
    "best_round": 45,
    "total_rounds": 100,
    "early_stopped": true,
    "train_metrics": {
      "loss": 0.22,
      "ppl": 1.25,
      "auc": 0.96,
      "accuracy": 0.92,
      "precision": 0.89,
      "recall": 0.86,
      "f1_score": 0.87
    },
    "val_metrics": {
      "loss": 0.28,
      "ppl": 1.32,
      "auc": 0.92,
      "accuracy": 0.89,
      "precision": 0.85,
      "recall": 0.82,
      "f1_score": 0.83
    },
    "test_metrics": {
      "loss": 0.30,
      "ppl": 1.35,
      "auc": 0.91,
      "accuracy": 0.88,
      "precision": 0.84,
      "recall": 0.81,
      "f1_score": 0.82
    },
    "feature_importances": []
  }
}
```

**Job Result - LightGBM (when complete):**

LightGBM results include feature importances similar to XGBoost:

```json
{
  "job_id": "train-job-uuid",
  "status": "finished",
  "result": {
    "status": "complete",
    "backend": "lightgbm",
    "dataset": "taiwan",
    "model_id": "model-2024-01-15-143052",
    "model_path": "/data/models/model-2024-01-15-143052.txt",
    "model_format": "txt",
    "active_model_path": "/data/models/active.txt",
    "samples_total": 6819,
    "samples_train": 4773,
    "samples_val": 1023,
    "samples_test": 1023,
    "n_features": 95,
    "scale_pos_weight": 29.99,
    "best_val_auc": 0.93,
    "best_round": 58,
    "total_rounds": 100,
    "early_stopped": true,
    "train_metrics": {
      "loss": 0.20,
      "ppl": 1.22,
      "auc": 0.97,
      "accuracy": 0.93,
      "precision": 0.90,
      "recall": 0.87,
      "f1_score": 0.88
    },
    "val_metrics": {
      "loss": 0.26,
      "ppl": 1.30,
      "auc": 0.93,
      "accuracy": 0.90,
      "precision": 0.86,
      "recall": 0.83,
      "f1_score": 0.84
    },
    "test_metrics": {
      "loss": 0.28,
      "ppl": 1.32,
      "auc": 0.92,
      "accuracy": 0.89,
      "precision": 0.85,
      "recall": 0.82,
      "f1_score": 0.83
    },
    "feature_importances": [
      {"name": "X6", "importance": 0.1523, "rank": 1},
      {"name": "X1", "importance": 0.0891, "rank": 2},
      {"name": "X5", "importance": 0.0812, "rank": 3}
    ]
  }
}
```

**Job Result - LSTM (when complete):**

LSTM results use PyTorch format and don't include feature importances:

```json
{
  "job_id": "train-job-uuid",
  "status": "finished",
  "result": {
    "status": "complete",
    "backend": "lstm",
    "dataset": "taiwan",
    "model_id": "model-2024-01-15-143052",
    "model_path": "/data/models/model-2024-01-15-143052.pt",
    "model_format": "pt",
    "active_model_path": "/data/models/active.pt",
    "samples_total": 6819,
    "samples_train": 4773,
    "samples_val": 1023,
    "samples_test": 1023,
    "n_features": 95,
    "scale_pos_weight": 29.99,
    "best_val_auc": 0.90,
    "best_round": 52,
    "total_rounds": 100,
    "early_stopped": true,
    "train_metrics": {
      "loss": 0.25,
      "ppl": 1.28,
      "auc": 0.94,
      "accuracy": 0.90,
      "precision": 0.87,
      "recall": 0.84,
      "f1_score": 0.85
    },
    "val_metrics": {
      "loss": 0.32,
      "ppl": 1.38,
      "auc": 0.90,
      "accuracy": 0.87,
      "precision": 0.83,
      "recall": 0.80,
      "f1_score": 0.81
    },
    "test_metrics": {
      "loss": 0.34,
      "ppl": 1.40,
      "auc": 0.89,
      "accuracy": 0.86,
      "precision": 0.82,
      "recall": 0.79,
      "f1_score": 0.80
    },
    "feature_importances": []
  }
}
```

**Result Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `backend` | string | Backend used (`xgboost`, `mlp`, `lightgbm`, or `lstm`) |
| `dataset` | string | Dataset used (`taiwan`, `us`, or `polish`) |
| `model_format` | string | Model file format (`ubj`, `pt`, or `txt`) |
| `n_features` | int | Number of features in the dataset |
| `scale_pos_weight` | float | Class weight used (auto-calculated if not provided) |
| `feature_importances` | array | Ranked list (XGBoost/LightGBM only, empty for MLP/LSTM) |

**Feature Importance Fields (XGBoost/LightGBM only):**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Feature/column name from the dataset |
| `importance` | float | Feature importance score (0.0-1.0) |
| `rank` | int | Rank by importance (1 = most important) |

**Backend Differences:**

| Aspect | XGBoost | MLP | LightGBM | LSTM |
|--------|---------|-----|----------|------|
| Model format | `.ubj` | `.pt` | `.txt` | `.pt` |
| Feature importances | Yes (ranked) | No | Yes (ranked) | No |
| GPU support | CUDA | CUDA (fp16/bf16) | CUDA | CUDA (fp16/bf16) |
| Training speed | Fast | Moderate | Very fast | Slow |
| Best for | Tabular data | Non-linear patterns | Large datasets | Temporal sequences |
| Interpretability | High | Low | High | Low |

The `feature_importances` array for XGBoost and LightGBM contains ALL features ranked by importance, allowing you to identify which financial ratios are most predictive of bankruptcy/default. For MLP/LSTM, use `platform_ml.ShapTreeWrapper` for type-safe SHAP-based feature importance analysis.

---

## POST /ml/optimize

Run Bayesian hyperparameter optimization using Optuna's Tree-structured Parzen Estimator (TPE) on external bankruptcy datasets. Supports all four ML backends: XGBoost, MLP, LightGBM, and LSTM.

**Common Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dataset` | string | Yes | - | Dataset to optimize on: `taiwan`, `us`, or `polish` |
| `backend` | string | No | `xgboost` | Backend: `xgboost`, `mlp`, `lightgbm`, or `lstm` |
| `n_trials` | int | Yes | - | Number of optimization trials to run |
| `timeout_seconds` | int | No | null | Maximum time in seconds (null = no timeout) |
| `device` | string | No | `auto` | `cpu`, `cuda`, or `auto` |
| `feature_preset` | string | No | `none` | Feature engineering: `none`, `log_only`, `ratios_only`, `full` |
| `random_state` | int | No | `42` | Random seed for reproducibility |

**Backend-Specific Fields:**

| Field | Backends | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `space_profile` | xgboost | string | `default` | `default` or `categorical` |
| `precision` | mlp, lstm | string | `fp32` | `fp32`, `fp16`, `bf16`, or `auto` |
| `optimizer` | mlp | string | `adamw` | `adamw`, `adam`, or `sgd` |
| `n_epochs` | mlp, lstm | int | `50` | Training epochs per trial |
| `early_stopping_patience` | mlp, lstm | int | `10` | Early stopping patience |
| `early_stopping_rounds` | lightgbm | int | `10` | Early stopping rounds |
| `sequence_length` | lstm | int | `5` | LSTM sequence length |
| `bidirectional` | lstm | bool | `false` | Use bidirectional LSTM |

**DART Boosting Support:**

XGBoost and LightGBM search spaces include DART (Dropouts meet Multiple Additive Regression Trees) as an optional boosting method. DART parameters are conditionally sampled only when DART boosting is selected.

| Backend | Parameter | Type | Range | Description |
|---------|-----------|------|-------|-------------|
| XGBoost | `booster` | categorical | `gbtree`, `dart` | Enables DART when "dart" |
| XGBoost | `rate_drop` | float | 0.0-0.5 | Tree dropout rate (DART only) |
| XGBoost | `skip_drop` | float | 0.0-0.5 | Skip dropout probability (DART only) |
| LightGBM | `boosting_type` | categorical | `gbdt`, `dart` | Enables DART when "dart" |
| LightGBM | `drop_rate` | float | 0.0-0.5 | Tree dropout rate (DART only) |
| LightGBM | `skip_drop` | float | 0.0-0.5 | Skip dropout probability (DART only) |
| LightGBM | `feature_fraction` | float | 0.02-0.1 | Aggressive feature subsampling (DART only) |

**Note:** Early stopping is automatically disabled for LightGBM DART mode, as DART's random tree dropout makes early stopping unreliable.

**XGBoost Full Config:**
```json
{
  "dataset": "taiwan",
  "backend": "xgboost",
  "n_trials": 50,
  "timeout_seconds": 3600,
  "device": "auto",
  "space_profile": "default",
  "feature_preset": "full",
  "random_state": 42
}
```

| Field | Type | Required | Default | Options |
|-------|------|----------|---------|---------|
| `dataset` | string | Yes | - | `taiwan`, `us`, `polish` |
| `backend` | string | No | `xgboost` | `xgboost` |
| `n_trials` | int | Yes | - | Any positive integer |
| `timeout_seconds` | int | No | `null` | Seconds or `null` |
| `device` | string | No | `auto` | `cpu`, `cuda`, `auto` |
| `space_profile` | string | No | `default` | `default`, `categorical` |
| `feature_preset` | string | No | `none` | `none`, `log_only`, `ratios_only`, `full` |
| `random_state` | int | No | `42` | Any integer |

**MLP Full Config:**
```json
{
  "dataset": "taiwan",
  "backend": "mlp",
  "n_trials": 50,
  "timeout_seconds": 3600,
  "device": "cuda",
  "feature_preset": "full",
  "random_state": 42,
  "precision": "fp16",
  "optimizer": "adamw",
  "n_epochs": 100,
  "early_stopping_patience": 15
}
```

| Field | Type | Required | Default | Options |
|-------|------|----------|---------|---------|
| `dataset` | string | Yes | - | `taiwan`, `us`, `polish` |
| `backend` | string | Yes | - | `mlp` |
| `n_trials` | int | Yes | - | Any positive integer |
| `timeout_seconds` | int | No | `null` | Seconds or `null` |
| `device` | string | No | `auto` | `cpu`, `cuda`, `auto` |
| `feature_preset` | string | No | `none` | `none`, `log_only`, `ratios_only`, `full` |
| `random_state` | int | No | `42` | Any integer |
| `precision` | string | No | `fp32` | `fp32`, `fp16`, `bf16`, `auto` |
| `optimizer` | string | No | `adamw` | `adamw`, `adam`, `sgd` |
| `n_epochs` | int | No | `50` | Any positive integer |
| `early_stopping_patience` | int | No | `10` | Any positive integer |

**LightGBM Full Config:**
```json
{
  "dataset": "polish",
  "backend": "lightgbm",
  "n_trials": 30,
  "timeout_seconds": 1800,
  "device": "auto",
  "feature_preset": "log_only",
  "random_state": 42,
  "early_stopping_rounds": 20
}
```

| Field | Type | Required | Default | Options |
|-------|------|----------|---------|---------|
| `dataset` | string | Yes | - | `taiwan`, `us`, `polish` |
| `backend` | string | Yes | - | `lightgbm` |
| `n_trials` | int | Yes | - | Any positive integer |
| `timeout_seconds` | int | No | `null` | Seconds or `null` |
| `device` | string | No | `auto` | `cpu`, `cuda`, `auto` |
| `feature_preset` | string | No | `none` | `none`, `log_only`, `ratios_only`, `full` |
| `random_state` | int | No | `42` | Any integer |
| `early_stopping_rounds` | int | No | `10` | Any positive integer |

**LSTM Full Config:**
```json
{
  "dataset": "us",
  "backend": "lstm",
  "n_trials": 25,
  "timeout_seconds": 7200,
  "device": "cuda",
  "feature_preset": "ratios_only",
  "random_state": 42,
  "precision": "bf16",
  "n_epochs": 75,
  "early_stopping_patience": 8,
  "sequence_length": 10,
  "bidirectional": true
}
```

| Field | Type | Required | Default | Options |
|-------|------|----------|---------|---------|
| `dataset` | string | Yes | - | `taiwan`, `us`, `polish` |
| `backend` | string | Yes | - | `lstm` |
| `n_trials` | int | Yes | - | Any positive integer |
| `timeout_seconds` | int | No | `null` | Seconds or `null` |
| `device` | string | No | `auto` | `cpu`, `cuda`, `auto` |
| `feature_preset` | string | No | `none` | `none`, `log_only`, `ratios_only`, `full` |
| `random_state` | int | No | `42` | Any integer |
| `precision` | string | No | `fp32` | `fp32`, `fp16`, `bf16`, `auto` |
| `n_epochs` | int | No | `50` | Any positive integer |
| `early_stopping_patience` | int | No | `10` | Any positive integer |
| `sequence_length` | int | No | `5` | Any positive integer |
| `bidirectional` | bool | No | `false` | `true`, `false` |

**Response (202):**
```json
{
  "job_id": "optimize-job-uuid",
  "status": "queued"
}
```

**Job Result (when complete):**

Poll `/ml/jobs/{job_id}` to get the result. Result structure varies by backend.

**XGBoost Result Example:**
```json
{
  "job_id": "optimize-job-uuid",
  "status": "finished",
  "result": {
    "backend": "xgboost",
    "status": "complete",
    "dataset": "taiwan",
    "n_samples": 6819,
    "n_features": 95,
    "feature_preset": "full",
    "n_trials_complete": 50,
    "n_trials_pruned": 0,
    "n_trials_failed": 0,
    "best_trial_number": 35,
    "best_val_auc": 0.94,
    "best_max_depth": 5,
    "best_n_estimators": 150,
    "best_learning_rate": 0.08,
    "best_reg_alpha": 0.01,
    "best_reg_lambda": 1.5,
    "best_subsample": 0.85,
    "best_colsample_bytree": 0.9,
    "duration_seconds": 542.3,
    "recommended_config": {...}
  }
}
```

**Result Fields (Common):**

| Field | Type | Description |
|-------|------|-------------|
| `backend` | string | Backend used for optimization |
| `dataset` | string | Dataset used for optimization |
| `n_samples` | int | Number of samples in dataset |
| `n_features` | int | Number of features in dataset |
| `feature_preset` | string | Feature engineering preset used |
| `n_trials_complete` | int | Number of trials completed |
| `n_trials_pruned` | int | Number of trials pruned early |
| `n_trials_failed` | int | Number of trials that failed |
| `best_trial_number` | int | Trial number that found best result |
| `best_val_auc` | float | Best validation AUC achieved |
| `duration_seconds` | float | Total optimization time |
| `recommended_config` | object | Ready-to-use config for `/train-external` |

**Workflow Example:**

1. Run optimization to find best hyperparameters:
   ```bash
   curl -X POST http://localhost:8007/ml/optimize \
     -H "Content-Type: application/json" \
     -d '{"dataset": "taiwan", "backend": "lightgbm", "n_trials": 50}'
   ```

2. Poll for completion and get `recommended_config`

3. Use `recommended_config` with `/train-external` for final model training:
   ```bash
   curl -X POST http://localhost:8007/ml/train-external \
     -H "Content-Type: application/json" \
     -d '{"dataset": "taiwan", "backend": "lightgbm", ...recommended_config...}'
   ```

---

## POST /ml/explain

Run feature importance explanation on a trained model using various explainer methods.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dataset` | string | Yes | - | Dataset name (see `/ml/train-external` for the full list) |
| `backend` | string | Yes | - | Backend: `xgboost`, `lightgbm`, `cleargbm`, `logreg`, `random_forest`, `mlp`, or `lstm` |
| `model_path` | string | Yes | - | Path to trained model file; must resolve inside `APP__MODELS_ROOT` |
| `explainer` | string | Yes | - | Explainer method (see compatibility below) |
| `target_class` | int | No | `1` | Target class for importance computation |
| `n_samples` | int | No | `1000` | Number of samples for explanation |
| `random_state` | int | No | `42` | Random seed for reproducibility |

**Supported Explainers:**

| Explainer | Description | Compatible Backends |
|-----------|-------------|---------------------|
| `permutation` | Permutation feature importance | All seven backends |
| `gradient` | Gradient-based importance | Neural networks only (mlp, lstm) |
| `integrated_gradients` | Integrated gradients | Neural networks only (mlp, lstm) |
| `shap_tree` | SHAP TreeExplainer | Tree models only (xgboost, lightgbm, cleargbm, random_forest) — not logreg |

`model_path` must resolve inside the configured models root (`APP__MODELS_ROOT`).
A path outside it is rejected with 400 and no file is opened.

**Request Example:**
```json
{
  "dataset": "taiwan",
  "backend": "xgboost",
  "model_path": "/data/models/xgboost/taiwan_xgboost_best.ubj",
  "explainer": "permutation",
  "target_class": 1,
  "n_samples": 1000,
  "random_state": 42
}
```

**Response (202):**
```json
{
  "job_id": "explain-job-uuid",
  "status": "queued"
}
```

**Job Result (when complete):**

Poll `/ml/jobs/{job_id}` to get the result:

```json
{
  "job_id": "explain-job-uuid",
  "status": "finished",
  "result": {
    "status": "complete",
    "backend": "xgboost",
    "explainer": "permutation",
    "n_samples_used": 1000,
    "n_features": 95,
    "target_class": 1,
    "feature_importances": [
      {"name": "X6", "importance": 0.1842},
      {"name": "X1", "importance": 0.0923},
      {"name": "X5", "importance": 0.0856}
    ],
    "duration_seconds": 12.5
  }
}
```

**Result Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `backend` | string | Backend used for the model |
| `explainer` | string | Explainer method used |
| `n_samples_used` | int | Actual number of samples used |
| `n_features` | int | Number of features in the dataset |
| `target_class` | int | Target class for importance |
| `feature_importances` | array | Ranked feature importances |
| `duration_seconds` | float | Computation time |

---

## GET /ml/jobs/{job_id}

Get training job status.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Training job UUID |

**Response (200):**
```json
{
  "job_id": "train-job-uuid",
  "status": "finished",
  "result": {
    "model_path": "/data/models/model-2024-01-01.ubj",
    "accuracy": 0.92
  }
}
```

**Status Values:**

| Status | Description |
|--------|-------------|
| `queued` | Job is waiting to be processed |
| `started` | Job is currently running |
| `finished` | Job completed successfully |
| `failed` | Job failed with error |
| `not_found` | Job ID not found in queue |

---

## GET /ml/models/active

Get information about the currently active model.

**Response (200):**
```json
{
  "model_id": "default",
  "model_path": "/data/models/active_xgb.ubj",
  "is_loaded": true
}
```

---

## Regression Endpoints

Continuous-target counterparts of the classification routes. Backends are
`xgboost_reg`, `lightgbm_reg`, `mlp_reg`, and `lstm_reg`.

| Endpoint | Method | Backends accepted |
|----------|--------|-------------------|
| `/ml/optimize-regression` | POST | all four |
| `/ml/train-external-regression` | POST | `xgboost_reg`, `lightgbm_reg` only |
| `/ml/explain-regression` | POST | all four |
| `/ml/predict-regression` | POST | all four |

The regression routes use their own dataset registry — currently
`financial_distress`. The neural-net optimizer is selected with `optimizer`
(`adamw` | `adam` | `sgd`), the same wire key the classification routes use;
`nn_optimizer` is the internal field name and is not accepted on the wire.

**Optimize hyperparameters:**
```bash
curl -X POST http://localhost:8007/ml/optimize-regression \
  -H "Content-Type: application/json" \
  -d '{"dataset": "financial_distress", "backend": "xgboost_reg", "n_trials": 50}'
```

**Train a regressor:**
```bash
curl -X POST http://localhost:8007/ml/train-external-regression \
  -H "Content-Type: application/json" \
  -d '{"dataset": "financial_distress", "backend": "xgboost_reg"}'
```

**Explain a trained regressor:**
```bash
curl -X POST http://localhost:8007/ml/explain-regression \
  -H "Content-Type: application/json" \
  -d '{"dataset": "financial_distress", "backend": "xgboost_reg",
       "model_path": "/data/models/model.ubj", "explainer": "permutation"}'
```

**Predict continuous values from a feature matrix:**
```bash
curl -X POST http://localhost:8007/ml/predict-regression \
  -H "Content-Type: application/json" \
  -d '{"backend": "xgboost_reg", "model_path": "/data/models/model.ubj",
       "features": [[1.0, 2.0, 3.0]]}'
```
