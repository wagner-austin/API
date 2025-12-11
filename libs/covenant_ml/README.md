# covenant-ml

XGBoost wrapper for covenant breach risk prediction: training and inference.

## Features

- `train_model`: Train XGBoost classifier with configurable hyperparameters
- `predict_probabilities`: Run inference on loan features
- `load_model` / `save_model`: Model persistence

## Usage

```python
from covenant_ml import predict_probabilities, load_model, train_model
from covenant_ml.types import TrainConfig

# Training
config: TrainConfig = {
    "device": "auto",  # "auto" picks CUDA when available, else CPU
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
model = train_model(X_train, y_train, config)

# Inference
probabilities = predict_probabilities(model, features)
```
