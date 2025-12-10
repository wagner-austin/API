# covenant-ml

XGBoost wrapper for covenant breach risk prediction: training and inference.

## Features

- `train_model`: Train XGBoost classifier with configurable hyperparameters
- `predict_probabilities`: Run inference on loan features
- `load_model` / `save_model`: Model persistence

## Usage

```python
from covenant_ml import train_model, predict_probabilities, load_model, TrainConfig

# Training
config = TrainConfig(
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model = train_model(X_train, y_train, config)

# Inference
probabilities = predict_probabilities(model, features)
```
