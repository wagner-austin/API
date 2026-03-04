# AI Instructions for Covenant Radar API

## When to Use This

Use these instructions when the user asks about:
- Covenant breach risk prediction
- ML model training for bankruptcy/default prediction
- Loan deal management and covenant compliance
- Financial ratio analysis

## Production API

**Base URL:** `https://covenant-radar-api-production.up.railway.app`

## Exactly What To Do

### 1. Predict Breach Risk for a Deal

```bash
curl -X POST "https://covenant-radar-api-production.up.railway.app/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{"deal_id": "uuid-here"}'
```

Response includes `probability` (0.0-1.0) and `risk_tier` (LOW/MEDIUM/HIGH/CRITICAL).

### 2. Train a Model on Internal Data

```bash
curl -X POST "https://covenant-radar-api-production.up.railway.app/ml/train" \
  -H "Content-Type: application/json" \
  -d '{
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
  }'
```

Trains XGBoost on internal deal/measurement data. Optional: `device`, `train_ratio`, `val_ratio`, `test_ratio`, `early_stopping_rounds`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`.

### 3. Train a Model on External Data

```bash
curl -X POST "https://covenant-radar-api-production.up.railway.app/ml/train-external" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "taiwan",
    "backend": "xgboost",
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
  }'
```

Returns `job_id` - poll `/ml/jobs/{job_id}` for status.

### 4. Optimize Hyperparameters

```bash
curl -X POST "https://covenant-radar-api-production.up.railway.app/ml/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "taiwan",
    "backend": "xgboost",
    "n_trials": 50,
    "device": "auto"
  }'
```

### 5. Explain Feature Importance

```bash
curl -X POST "https://covenant-radar-api-production.up.railway.app/ml/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "taiwan",
    "backend": "xgboost",
    "model_path": "/path/to/model.ubj",
    "explainer": "shap_tree"
  }'
```

Explainers by backend:
- **XGBoost/LightGBM**: `permutation`, `shap_tree`
- **MLP/LSTM**: `gradient`, `integrated_gradients`, `permutation`

Optional: `target_class` (default 1), `n_samples` (default 1000), `random_state` (default 42).

### 6. Check Job Status

```bash
curl "https://covenant-radar-api-production.up.railway.app/ml/jobs/{job_id}"
```

Status values: `queued`, `started`, `finished`, `failed`, `not_found`.

### 7. Get Active Model Info

```bash
curl "https://covenant-radar-api-production.up.railway.app/ml/models/active"
```

### 8. View Dashboard

Open in browser: `https://covenant-radar-api-production.up.railway.app/dashboard`

## ML Backends (Classification)

Tree/linear backends live in `covenant_ml`. PyTorch backends (`mlp`, `lstm`) live in `covenant_nn`.

| Backend | Package | Format | GPU | Feature Importance | Best For |
|---------|---------|--------|-----|-------------------|----------|
| `xgboost` | covenant_ml | `.ubj` | CUDA | Yes (ranked, shap_tree, permutation) | Tabular data |
| `lightgbm` | covenant_ml | `.txt` | CUDA | Yes (ranked, shap_tree, permutation) | Large datasets |
| `cleargbm` | covenant_ml | `.pkl` | No | Yes (shap, permutation) | Interpretable boosting |
| `logreg` | covenant_ml | `.joblib` | No | Yes (permutation) | Linear baselines |
| `random_forest` | covenant_ml | `.joblib` | No | Yes (permutation) | Ensemble baselines |
| `mlp` | covenant_nn | `.pt` | CUDA (fp16/bf16) | Yes (gradient, integrated_gradients, permutation) | Non-linear patterns |
| `lstm` | covenant_nn | `.pt` | CUDA (fp16/bf16) | Yes (gradient, integrated_gradients, permutation) | Temporal sequences |

## Datasets

| Dataset | Samples | Features | Description |
|---------|---------|----------|-------------|
| `taiwan` | 6,819 | 95 | Taiwan Economic Journal bankruptcy |
| `us` | 78,682 | 18 | American bankruptcy |
| `polish` | 7,027 | 64 | Polish companies |

## Feature Engineering Presets

Use `feature_preset` in optimize/train requests:
- `none` - Original features only (default)
- `log_only` - Original + signed log transforms
- `ratios_only` - Original + pairwise ratios
- `full` - All of the above

## Risk Tiers

| Tier | Probability | Action |
|------|-------------|--------|
| `LOW` | < 0.25 | No action |
| `MEDIUM` | 0.25-0.50 | Monitor |
| `HIGH` | 0.50-0.80 | Review required |
| `CRITICAL` | >= 0.80 | Immediate action |

## CRUD Endpoints

```bash
# Deals
curl "https://covenant-radar-api-production.up.railway.app/deals"
curl -X POST "https://covenant-radar-api-production.up.railway.app/deals" -d '{...}'

# Covenants
curl "https://covenant-radar-api-production.up.railway.app/covenants/by-deal/{deal_id}"

# Measurements
curl "https://covenant-radar-api-production.up.railway.app/measurements/by-deal/{deal_id}"

# Evaluate compliance
curl -X POST "https://covenant-radar-api-production.up.railway.app/evaluate" \
  -d '{"deal_id": "...", "period_start_iso": "2024-01-01", "period_end_iso": "2024-03-31", "tolerance_ratio_scaled": 10}'
```

## Do NOT

- Do NOT import covenant_ml or covenant_domain directly - use the API
- Do NOT run worker jobs manually - use the API endpoints
- Do NOT guess model paths - hit `/ml/models/active`

## Typical Workflow

1. **Optimize** - Find best hyperparameters:
   ```bash
   curl -X POST .../ml/optimize -d '{"dataset": "taiwan", "backend": "xgboost", "n_trials": 50}'
   ```

2. **Poll** - Wait for completion:
   ```bash
   curl .../ml/jobs/{job_id}
   ```

3. **Train** - Use `recommended_config` from result:
   ```bash
   curl -X POST .../ml/train-external -d '{...recommended_config...}'
   ```

4. **Explain** - Get feature importance (optional):
   ```bash
   curl -X POST .../ml/explain -d '{"dataset": "taiwan", "backend": "xgboost", "model_path": "...", "explainer": "shap_tree"}'
   ```

5. **Predict** - Use trained model:
   ```bash
   curl -X POST .../ml/predict -d '{"deal_id": "..."}'
   ```

## Health Checks

```bash
# Liveness
curl "https://covenant-radar-api-production.up.railway.app/healthz"

# Readiness (checks Redis + workers)
curl "https://covenant-radar-api-production.up.railway.app/readyz"

# Full status (Redis, Postgres, model, deal count)
curl "https://covenant-radar-api-production.up.railway.app/status"
```
