# Covenant Radar API - API Reference

Complete API documentation for the covenant-radar-api service.

**Base URL:** `http://localhost:8007` (default)

---

## Health Endpoints

### GET /healthz

Liveness probe for container orchestration.

**Response (200):**
```json
{
  "status": "ok"
}
```

---

### GET /readyz

Readiness probe. Returns 503 if Redis unavailable or no workers registered.

**Response (200):**
```json
{
  "status": "ready",
  "reason": null
}
```

**Response (503 - Redis unavailable):**
```json
{
  "status": "degraded",
  "reason": "redis-unavailable"
}
```

**Response (503 - No workers):**
```json
{
  "status": "degraded",
  "reason": "no-worker"
}
```

---

### GET /status

Comprehensive service status with dependency health, model info, and data counts.

**Response (200):**
```json
{
  "service": "covenant-radar-api",
  "version": "0.1.0",
  "dependencies": [
    {
      "name": "redis",
      "status": "ok",
      "message": null
    },
    {
      "name": "postgres",
      "status": "ok",
      "message": null
    }
  ],
  "model": {
    "model_id": "default",
    "model_path": "/data/models/active.ubj",
    "is_loaded": false
  },
  "data": {
    "deals": 5
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `service` | string | Service name |
| `version` | string | Service version |
| `dependencies` | array | List of dependency health checks |
| `dependencies[].name` | string | Dependency name (`redis`, `postgres`) |
| `dependencies[].status` | string | `ok` or `error` |
| `dependencies[].message` | string\|null | Error message if status is `error` |
| `model.model_id` | string | Active model identifier |
| `model.model_path` | string | Path to active model file |
| `model.is_loaded` | boolean | Whether model is loaded in memory |
| `data.deals` | int | Total deal count in database |

---

## Deal Endpoints

### GET /deals

List all deals.

**Response (200):**
```json
[
  {
    "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "name": "TechCorp Senior Credit Facility",
    "borrower": "TechCorp Inc",
    "sector": "Technology",
    "region": "North America",
    "commitment_amount_cents": 50000000000,
    "currency": "USD",
    "maturity_date_iso": "2027-12-31"
  }
]
```

---

### POST /deals

Create a new deal.

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | object | Yes | Deal identifier object |
| `id.value` | string | Yes | UUID for the deal |
| `name` | string | Yes | Deal name |
| `borrower` | string | Yes | Borrower company name |
| `sector` | string | Yes | Industry sector |
| `region` | string | Yes | Geographic region |
| `commitment_amount_cents` | int | Yes | Commitment amount in cents |
| `currency` | string | Yes | Currency code (e.g., `USD`, `EUR`) |
| `maturity_date_iso` | string | Yes | Maturity date in ISO format (YYYY-MM-DD) |

**Request Example:**
```json
{
  "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Demo Leveraged Buyout",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 75000000000,
  "currency": "USD",
  "maturity_date_iso": "2029-06-30"
}
```

**Response (201):**
```json
{
  "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Demo Leveraged Buyout",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 75000000000,
  "currency": "USD",
  "maturity_date_iso": "2029-06-30"
}
```

**Example - curl:**
```bash
curl -X POST http://localhost:8007/deals \
  -H "Content-Type: application/json" \
  -d '{
    "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "name": "Demo Leveraged Buyout",
    "borrower": "Demo Corp",
    "sector": "Manufacturing",
    "region": "North America",
    "commitment_amount_cents": 75000000000,
    "currency": "USD",
    "maturity_date_iso": "2029-06-30"
  }'
```

---

### GET /deals/{deal_id}

Get a deal by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Response (200):**
```json
{
  "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Demo Leveraged Buyout",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 75000000000,
  "currency": "USD",
  "maturity_date_iso": "2029-06-30"
}
```

**Response (404):**
```json
{
  "code": "NOT_FOUND",
  "message": "Deal not found",
  "request_id": "uuid"
}
```

---

### PUT /deals/{deal_id}

Update an existing deal.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Deal name |
| `borrower` | string | Yes | Borrower company name |
| `sector` | string | Yes | Industry sector |
| `region` | string | Yes | Geographic region |
| `commitment_amount_cents` | int | Yes | Commitment amount in cents |
| `currency` | string | Yes | Currency code |
| `maturity_date_iso` | string | Yes | Maturity date in ISO format |

**Request Example:**
```json
{
  "name": "Updated Deal Name",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 80000000000,
  "currency": "USD",
  "maturity_date_iso": "2030-06-30"
}
```

**Response (200):**
```json
{
  "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Updated Deal Name",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 80000000000,
  "currency": "USD",
  "maturity_date_iso": "2030-06-30"
}
```

---

### DELETE /deals/{deal_id}

Delete a deal by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Response (204):** No content

---

## Covenant Endpoints

### POST /covenants

Create a new covenant.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | object | Yes | Covenant identifier object |
| `id.value` | string | Yes | UUID for the covenant |
| `deal_id` | object | Yes | Associated deal identifier |
| `deal_id.value` | string | Yes | Deal UUID |
| `name` | string | Yes | Covenant name |
| `formula` | string | Yes | Calculation formula (e.g., `total_debt / ebitda`) |
| `threshold_value_scaled` | int | Yes | Threshold value (scaled integer) |
| `threshold_direction` | string | Yes | `<=` or `>=` |
| `frequency` | string | Yes | `QUARTERLY` or `ANNUAL` |

**Request Example:**
```json
{
  "id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
  "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Max Leverage Ratio",
  "formula": "total_debt / ebitda",
  "threshold_value_scaled": 450,
  "threshold_direction": "<=",
  "frequency": "QUARTERLY"
}
```

**Response (201):**
```json
{
  "id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
  "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Max Leverage Ratio",
  "formula": "total_debt / ebitda",
  "threshold_value_scaled": 450,
  "threshold_direction": "<=",
  "frequency": "QUARTERLY"
}
```

---

### GET /covenants/by-deal/{deal_id}

List all covenants for a specific deal.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Response (200):**
```json
[
  {
    "id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
    "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "name": "Max Leverage Ratio",
    "formula": "total_debt / ebitda",
    "threshold_value_scaled": 450,
    "threshold_direction": "<=",
    "frequency": "QUARTERLY"
  }
]
```

---

### GET /covenants/{covenant_id}

Get a covenant by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `covenant_id` | string | Covenant UUID |

**Response (200):**
```json
{
  "id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
  "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Max Leverage Ratio",
  "formula": "total_debt / ebitda",
  "threshold_value_scaled": 450,
  "threshold_direction": "<=",
  "frequency": "QUARTERLY"
}
```

---

### DELETE /covenants/{covenant_id}

Delete a covenant by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `covenant_id` | string | Covenant UUID |

**Response (204):** No content

---

## Measurement Endpoints

### GET /measurements/by-deal/{deal_id}

List all measurements for a specific deal.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Response (200):**
```json
[
  {
    "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "period_start_iso": "2024-01-01",
    "period_end_iso": "2024-03-31",
    "metric_name": "total_debt",
    "metric_value_scaled": 1000000000
  }
]
```

---

### GET /measurements/by-deal/{deal_id}/period

List measurements for a deal within a specific period.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `period_start` | string | Yes | Period start date (YYYY-MM-DD) |
| `period_end` | string | Yes | Period end date (YYYY-MM-DD) |

**Response (200):**
```json
[
  {
    "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "period_start_iso": "2024-01-01",
    "period_end_iso": "2024-03-31",
    "metric_name": "total_debt",
    "metric_value_scaled": 1000000000
  }
]
```

---

### POST /measurements

Add financial measurements for deals.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `measurements` | array | Yes | List of measurement objects |
| `measurements[].deal_id` | object | Yes | Deal identifier |
| `measurements[].deal_id.value` | string | Yes | Deal UUID |
| `measurements[].period_start_iso` | string | Yes | Period start date (YYYY-MM-DD) |
| `measurements[].period_end_iso` | string | Yes | Period end date (YYYY-MM-DD) |
| `measurements[].metric_name` | string | Yes | Metric name (e.g., `total_debt`, `ebitda`) |
| `measurements[].metric_value_scaled` | int | Yes | Metric value (scaled integer) |

**Request Example:**
```json
{
  "measurements": [
    {
      "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
      "period_start_iso": "2024-01-01",
      "period_end_iso": "2024-03-31",
      "metric_name": "total_debt",
      "metric_value_scaled": 1000000000
    },
    {
      "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
      "period_start_iso": "2024-01-01",
      "period_end_iso": "2024-03-31",
      "metric_name": "ebitda",
      "metric_value_scaled": 300000000
    }
  ]
}
```

**Response (200):**
```json
{
  "count": 2
}
```

---

## Evaluation Endpoints

### POST /evaluate

Evaluate covenant compliance for a deal and period.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `deal_id` | string | Yes | Deal UUID |
| `period_start_iso` | string | Yes | Period start date (YYYY-MM-DD) |
| `period_end_iso` | string | Yes | Period end date (YYYY-MM-DD) |
| `tolerance_ratio_scaled` | int | Yes | Near-breach tolerance ratio (scaled) |

**Request Example:**
```json
{
  "deal_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
  "period_start_iso": "2024-01-01",
  "period_end_iso": "2024-03-31",
  "tolerance_ratio_scaled": 10
}
```

**Response (200):**
```json
[
  {
    "covenant_id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
    "period_start_iso": "2024-01-01",
    "period_end_iso": "2024-03-31",
    "calculated_value_scaled": 333,
    "status": "OK"
  }
]
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `covenant_id` | object | Covenant identifier |
| `period_start_iso` | string | Period start date |
| `period_end_iso` | string | Period end date |
| `calculated_value_scaled` | int | Calculated metric value |
| `status` | string | `OK`, `NEAR_BREACH`, or `BREACH` |

**Status Definitions:**

| Status | Description |
|--------|-------------|
| `OK` | Covenant is in compliance |
| `NEAR_BREACH` | Within tolerance threshold of breach |
| `BREACH` | Covenant threshold exceeded |

---

## ML Endpoints

### POST /ml/predict

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
| `risk_tier` | string | `LOW`, `MEDIUM`, or `HIGH` |

**Risk Tier Thresholds:**

| Tier | Probability Range |
|------|-------------------|
| `LOW` | 0.0 - 0.33 |
| `MEDIUM` | 0.33 - 0.66 |
| `HIGH` | 0.66 - 1.0 |

---

### POST /ml/train

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
| `scale_pos_weight` | float | No | - | Class weight for positives |

**Request Example:**
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
  "device": "auto",
  "reg_alpha": 0.0,
  "reg_lambda": 1.0,
  "scale_pos_weight": 1.5
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

### POST /ml/train-external

Train on external CSV datasets (Taiwan, US, Polish bankruptcy data) with automatic feature selection. XGBoost trains on all columns and determines feature importance.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dataset` | string | Yes | - | Dataset to use: `taiwan`, `us`, or `polish` |
| `learning_rate` | float | Yes | - | Learning rate |
| `max_depth` | int | Yes | - | XGBoost max tree depth |
| `n_estimators` | int | Yes | - | Number of trees |
| `subsample` | float | Yes | - | Row subsample ratio |
| `colsample_bytree` | float | Yes | - | Column subsample ratio |
| `random_state` | int | Yes | - | Random seed |
| `device` | string | No | `auto` | `cpu`, `cuda`, or `auto` |
| `reg_alpha` | float | No | `0.0` | L1 regularization strength |
| `reg_lambda` | float | No | `1.0` | L2 regularization strength |
| `scale_pos_weight` | float | No | - | Class weight for positives |

**Request Example:**
```json
{
  "dataset": "taiwan",
  "learning_rate": 0.1,
  "max_depth": 6,
  "n_estimators": 100,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42,
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

Poll `/ml/jobs/{job_id}` to get the result with automatic feature importance ranking:

```json
{
  "job_id": "train-job-uuid",
  "status": "finished",
  "result": {
    "status": "complete",
    "dataset": "taiwan",
    "model_id": "model-2024-01-15-143052",
    "model_path": "/data/models/model-2024-01-15-143052.ubj",
    "active_model_path": "/data/models/active.ubj",
    "samples_total": 6819,
    "samples_train": 4773,
    "samples_val": 1023,
    "samples_test": 1023,
    "n_features": 95,
    "best_val_auc": 0.94,
    "best_round": 67,
    "total_rounds": 100,
    "early_stopped": true,
    "train_metrics": {
      "loss": 0.18,
      "auc": 0.98,
      "accuracy": 0.94,
      "precision": 0.91,
      "recall": 0.88,
      "f1_score": 0.89
    },
    "val_metrics": {
      "loss": 0.24,
      "auc": 0.94,
      "accuracy": 0.91,
      "precision": 0.87,
      "recall": 0.84,
      "f1_score": 0.85
    },
    "test_metrics": {
      "loss": 0.26,
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

**Additional Result Fields (vs /ml/train):**

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | string | Dataset used (`taiwan`, `us`, or `polish`) |
| `n_features` | int | Number of features in the dataset |
| `feature_importances` | array | Ranked list of feature importances |

**Feature Importance Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Feature/column name from the dataset |
| `importance` | float | XGBoost feature importance score (0.0-1.0) |
| `rank` | int | Rank by importance (1 = most important) |

The `feature_importances` array contains ALL features ranked by importance, allowing you to identify which financial ratios are most predictive of bankruptcy/default.

---

### GET /ml/jobs/{job_id}

Get training job status.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Training job UUID |

**Response (200):**
```json
{
  "job_id": "train-job-uuid",
  "status": "completed",
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

---

### GET /ml/models/active

Get information about the currently active model.

**Response (200):**
```json
{
  "model_id": "default",
  "model_path": "/data/models/active.ubj",
  "is_loaded": true
}
```

---

## Error Handling

All errors return JSON with consistent format:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable description",
  "request_id": "uuid-for-tracing"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 422 | Invalid request body or parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `INTERNAL_ERROR` | 500 | Internal server error |

### Error Examples

**Invalid request body:**
```json
{
  "code": "INVALID_INPUT",
  "message": "Missing required field 'deal_id'",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Resource not found:**
```json
{
  "code": "NOT_FOUND",
  "message": "Deal not found",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

---

## Request ID Tracing

All requests are assigned a unique `request_id` for tracing:

- **Provided:** Pass `X-Request-ID` header
- **Generated:** UUID v4 auto-generated if header omitted

The `request_id` appears in:
- All error responses
- Structured logs

---

## Content Types

**Request content type:**
- `application/json` (for all POST/PUT endpoints)

**Response content types:**
- `application/json` (all endpoints)

---

## Domain Models

### Deal

| Field | Type | Description |
|-------|------|-------------|
| `id` | DealId | Unique identifier |
| `name` | string | Deal name |
| `borrower` | string | Borrower company name |
| `sector` | string | Industry sector |
| `region` | string | Geographic region |
| `commitment_amount_cents` | int | Commitment in cents |
| `currency` | string | Currency code |
| `maturity_date_iso` | string | Maturity date (YYYY-MM-DD) |

### Covenant

| Field | Type | Description |
|-------|------|-------------|
| `id` | CovenantId | Unique identifier |
| `deal_id` | DealId | Associated deal |
| `name` | string | Covenant name |
| `formula` | string | Calculation formula |
| `threshold_value_scaled` | int | Threshold (scaled integer) |
| `threshold_direction` | string | `<=` or `>=` |
| `frequency` | string | `QUARTERLY` or `ANNUAL` |

### Measurement

| Field | Type | Description |
|-------|------|-------------|
| `deal_id` | DealId | Associated deal |
| `period_start_iso` | string | Period start (YYYY-MM-DD) |
| `period_end_iso` | string | Period end (YYYY-MM-DD) |
| `metric_name` | string | Metric name |
| `metric_value_scaled` | int | Value (scaled integer) |

### CovenantResult

| Field | Type | Description |
|-------|------|-------------|
| `covenant_id` | CovenantId | Evaluated covenant |
| `period_start_iso` | string | Period start |
| `period_end_iso` | string | Period end |
| `calculated_value_scaled` | int | Calculated value |
| `status` | string | `OK`, `NEAR_BREACH`, `BREACH` |

---

## Scaled Integer Convention

All monetary and ratio values use scaled integers to avoid floating-point precision issues:

- **Monetary values**: Stored in cents (multiply by 100)
  - Example: $500,000,000 = `50000000000` cents
- **Ratios**: Scaled by 100 for two decimal places
  - Example: 4.5x leverage = `450` scaled

This convention ensures deterministic calculations and exact comparisons.
