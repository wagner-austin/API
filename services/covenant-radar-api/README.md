# Covenant Radar API

Loan covenant monitoring and breach prediction API service. Features deterministic rule evaluation, pluggable ML backends (XGBoost, LightGBM, MLP, LSTM), Optuna hyperparameter optimization, feature importance explainers, and PostgreSQL persistence.

## Features

- **Deal Management**: CRUD operations for loan deals with structured metadata
- **Covenant Definitions**: Configurable rules with formulas, thresholds, and frequencies
- **Financial Measurements**: Time-series metric ingestion for covenant calculations
- **Rule Evaluation**: Deterministic covenant compliance checking with OK/NEAR_BREACH/BREACH status
- **Breach Prediction**: Pluggable ML backends for risk tier prediction (LOW/MEDIUM/HIGH/CRITICAL)
  - XGBoost: Gradient boosting with feature importance ranking
  - MLP: Neural network with configurable architecture (hidden layers, dropout, precision)
  - LSTM: Recurrent network for temporal bankruptcy sequences (bidirectional support)
  - LightGBM: Fast gradient boosting for large-scale datasets
- **Hyperparameter Optimization**: Pluggable optimizers via Optuna TPE
  - DART boosting support for XGBoost and LightGBM (dropout regularization)
  - Categorical and continuous parameter spaces
  - Early stopping with validation AUC tracking
- **Model Explainability**: Feature importance extraction
  - XGBoost: Gain-based importance ranking
  - LightGBM: Split-based importance ranking
- **Background Training**: Redis + RQ worker for model training jobs
- **Kafka Streaming**: Confluent Cloud integration for real-time inference pipeline
  - Measurement event consumption from `covenant.measurements.v1`
  - Prediction event publishing to `covenant.predictions.v1`
  - Alert event publishing to `covenant.alerts.v1`
  - TypedDict schemas with encode/decode/TypeGuard functions
- **Gemini AI Integration**: Google AI (Gemini) for human-readable alert summaries
  - GeminiClient wrapper with prompt template for credit risk alerts
  - AlertContext TypedDict with deal info, risk tier, and evaluation status
  - Token usage tracking and latency metrics
  - Protocol-based DI with FakeGeminiClient for testing
- **Observability**: Datadog APM tracing and custom metrics integration
- **Type Safety**: mypy strict mode, zero `Any` types, Protocol-based DI
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- Docker Desktop (for Redis, PostgreSQL, and containerized deployment)

### Start with Docker (from repository root)

```bash
# Start infrastructure + service
make up-covenant

# Verify health
curl http://localhost:8007/healthz
curl http://localhost:8007/readyz
curl http://localhost:8007/status
```

### Local Development

```bash
cd services/covenant-radar-api
poetry install --with dev

# Start dependencies
docker compose up -d redis postgres

# Run API
poetry run hypercorn 'covenant_radar_api.api.main:create_app()' --bind 0.0.0.0:8000

# Run Worker (separate terminal)
poetry run covenant-rq-worker
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe |
| `/status` | GET | Service status with dependency health |
| `/deals` | GET | List all deals |
| `/deals` | POST | Create a new deal |
| `/deals/{deal_id}` | GET | Get deal by ID |
| `/deals/{deal_id}` | PUT | Update deal |
| `/deals/{deal_id}` | DELETE | Delete deal |
| `/covenants` | POST | Create a new covenant |
| `/covenants/by-deal/{deal_id}` | GET | List covenants for a deal |
| `/covenants/{covenant_id}` | GET | Get covenant by ID |
| `/covenants/{covenant_id}` | DELETE | Delete covenant |
| `/measurements` | POST | Add measurements |
| `/measurements/by-deal/{deal_id}` | GET | List measurements for a deal |
| `/measurements/by-deal/{deal_id}/period` | GET | List measurements for deal and period |
| `/evaluate` | POST | Evaluate covenant compliance |
| `/ml/predict` | POST | Predict breach risk |
| `/ml/train` | POST | Enqueue model training |
| `/ml/train-external` | POST | Train on external CSV datasets |
| `/ml/optimize` | POST | Optimize hyperparameters with Optuna TPE |
| `/ml/jobs/{job_id}` | GET | Get training job status |
| `/ml/models/active` | GET | Get active model info |

---

## API Examples

### Health & Status

```bash
# Basic health check
curl http://localhost:8007/healthz
# {"status":"ok"}

# Detailed status with dependency health, model info, and data counts
curl http://localhost:8007/status | python -m json.tool
# {
#     "service": "covenant-radar-api",
#     "version": "0.1.0",
#     "dependencies": [
#         {"name": "redis", "status": "ok", "message": null},
#         {"name": "postgres", "status": "ok", "message": null}
#     ],
#     "model": {
#         "model_id": "default",
#         "model_path": "/data/models/active.ubj",
#         "is_loaded": false
#     },
#     "data": {"deals": 5}
# }
```

### Deals

```bash
# List all deals
curl http://localhost:8007/deals | python -m json.tool

# Create a deal (ID must be a valid UUID)
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

# Get a specific deal
curl http://localhost:8007/deals/a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d

# Update a deal
curl -X PUT http://localhost:8007/deals/a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Deal Name",
    "borrower": "Demo Corp",
    "sector": "Manufacturing",
    "region": "North America",
    "commitment_amount_cents": 80000000000,
    "currency": "USD",
    "maturity_date_iso": "2030-06-30"
  }'

# Delete a deal
curl -X DELETE http://localhost:8007/deals/a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d
```

### Covenants

```bash
# Create a covenant (linked to a deal)
curl -X POST http://localhost:8007/covenants \
  -H "Content-Type: application/json" \
  -d '{
    "id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
    "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "name": "Max Leverage Ratio",
    "formula": "total_debt / ebitda",
    "threshold_value_scaled": 450,
    "threshold_direction": "<=",
    "frequency": "QUARTERLY"
  }'

# List covenants for a deal
curl http://localhost:8007/covenants/by-deal/a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d

# Get a specific covenant
curl http://localhost:8007/covenants/c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f
```

### Measurements

```bash
# List all measurements for a deal
curl http://localhost:8007/measurements/by-deal/a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d

# List measurements for a deal within a specific period
curl "http://localhost:8007/measurements/by-deal/a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d/period?period_start=2024-01-01&period_end=2024-03-31"

# Add financial measurements for a deal
curl -X POST http://localhost:8007/measurements \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
# {"count": 2}
```

### Covenant Evaluation

```bash
# Evaluate covenant compliance for a deal and period
curl -X POST http://localhost:8007/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "deal_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
    "period_start_iso": "2024-01-01",
    "period_end_iso": "2024-03-31",
    "tolerance_ratio_scaled": 10
  }'
# Returns covenant results with status: "OK", "NEAR_BREACH", or "BREACH"
```

### ML Prediction

```bash
# Predict breach risk for a deal
curl -X POST http://localhost:8007/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"deal_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"}'
# {
#   "deal_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
#   "probability": 0.23,
#   "risk_tier": "LOW"
# }

# Get active model info
curl http://localhost:8007/ml/models/active
```

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_ENV` | string | `dev` | Application environment (`dev` or `prod`) |
| `DATABASE_URL` | string | - | PostgreSQL connection URL (required) |
| `REDIS_URL` or `REDIS__URL` | string | `redis://redis:6379/0` | Redis connection URL |
| `REDIS__ENABLED` | bool | `true` | Enable Redis |
| `RQ__QUEUE_NAME` | string | `covenant` | RQ queue name |
| `RQ__JOB_TIMEOUT_SEC` | int | `3600` | Job timeout in seconds |
| `RQ__RESULT_TTL_SEC` | int | `86400` | Result TTL in seconds |
| `RQ__FAILURE_TTL_SEC` | int | `604800` | Failure TTL in seconds |
| `APP__DATA_ROOT` | string | `/data` | Data root directory |
| `APP__MODELS_ROOT` | string | `/data/models` | Models directory |
| `APP__LOGS_ROOT` | string | `/data/logs` | Logs directory |
| `APP__ACTIVE_MODEL_PATH` | string | `/data/models/active.ubj` | Active model path |
| `LOGGING__LEVEL` | string | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `DATADOG__ENABLED` | bool | `false` | Enable Datadog integration |
| `DATADOG__SERVICE` | string | `covenant-radar-api` | Service name for traces |
| `DATADOG__ENV` | string | `dev` | Environment (`dev`, `staging`, `production`) |
| `DATADOG__VERSION` | string | `0.0.0` | Service version |
| `DATADOG__AGENT_HOST` | string | `localhost` | Datadog agent host |
| `DATADOG__DOGSTATSD_PORT` | int | `8125` | DogStatsD UDP port |
| `DATADOG__TRACE_ENABLED` | bool | `true` | Enable APM tracing |
| `STREAMING__ENABLED` | bool | `false` | Enable Kafka streaming |
| `CONFLUENT__BOOTSTRAP_SERVERS` | string | - | Confluent Cloud bootstrap servers |
| `CONFLUENT__API_KEY` | string | - | Confluent Cloud API key |
| `CONFLUENT__API_SECRET` | string | - | Confluent Cloud API secret |
| `CONFLUENT__SCHEMA_REGISTRY_URL` | string | - | Schema Registry URL (optional) |
| `CONFLUENT__SCHEMA_REGISTRY_API_KEY` | string | - | Schema Registry API key |
| `CONFLUENT__SCHEMA_REGISTRY_API_SECRET` | string | - | Schema Registry API secret |
| `KAFKA__TOPIC_MEASUREMENTS` | string | `covenant.measurements.v1` | Input topic |
| `KAFKA__TOPIC_PREDICTIONS` | string | `covenant.predictions.v1` | Predictions output topic |
| `KAFKA__TOPIC_ALERTS` | string | `covenant.alerts.v1` | Alerts output topic |
| `KAFKA__CONSUMER_GROUP_ID` | string | `covenant-radar-api` | Consumer group ID |
| `KAFKA__AUTO_OFFSET_RESET` | string | `earliest` | Offset reset policy |
| `KAFKA__ENABLE_AUTO_COMMIT` | bool | `false` | Auto-commit offsets |
| `KAFKA__FETCH_MIN_BYTES` | int | `1` | Minimum fetch bytes |
| `KAFKA__SESSION_TIMEOUT_MS` | int | `45000` | Session timeout |
| `KAFKA__HEARTBEAT_INTERVAL_MS` | int | `15000` | Heartbeat interval |
| `KAFKA__PRODUCER_ACKS` | string | `all` | Producer acknowledgment |
| `KAFKA__PRODUCER_RETRIES` | int | `3` | Producer retries |
| `KAFKA__PRODUCER_LINGER_MS` | int | `5` | Producer linger time |
| `KAFKA__PRODUCER_BATCH_SIZE` | int | `16384` | Producer batch size |
| `KAFKA__COMPRESSION_TYPE` | string | `gzip` | Compression type |
| `GEMINI_API_KEY` | string | - | Google AI API key for Gemini integration |

### Example .env

```bash
APP_ENV=dev
DATABASE_URL=postgresql://covenant:covenant@postgres:5432/covenant
REDIS_URL=redis://redis:6379/0
RQ__QUEUE_NAME=covenant
APP__ACTIVE_MODEL_PATH=/data/models/active.ubj
LOGGING__LEVEL=INFO

# Datadog (optional, disabled by default)
DATADOG__ENABLED=false
DATADOG__SERVICE=covenant-radar-api
DATADOG__ENV=dev

# Kafka Streaming (optional, disabled by default)
STREAMING__ENABLED=false
CONFLUENT__BOOTSTRAP_SERVERS=pkc-xxxxx.us-east-1.aws.confluent.cloud:9092
CONFLUENT__API_KEY=your-api-key
CONFLUENT__API_SECRET=your-api-secret

# Google AI / Gemini (optional, for alert summaries)
GEMINI_API_KEY=your-gemini-api-key
```

---

## Architecture

### Component Overview

```
covenant_radar_api/
├── api/                    # FastAPI routes
│   ├── main.py            # App factory
│   ├── decode.py          # Request parsing
│   └── routes/            # Endpoint handlers
│       ├── health.py
│       ├── status.py
│       ├── deals.py
│       ├── covenants.py
│       ├── measurements.py
│       ├── evaluate.py
│       └── ml.py
├── core/
│   ├── config.py          # Settings
│   └── container.py       # DI container
├── worker/
│   ├── evaluate_job.py    # Batch evaluation
│   ├── train_job.py       # Model training (internal data)
│   └── train_external_job.py # Model training (external datasets)
├── integrations/
│   ├── datadog/           # Datadog APM and metrics
│   │   ├── metrics.py     # DogStatsD client
│   │   ├── tracing.py     # APM tracing setup
│   │   └── _test_hooks.py # Dependency injection
│   └── google_ai/         # Google AI (Gemini) integration
│       ├── client.py      # GeminiClient wrapper
│       ├── schemas.py     # AlertContext, Request/Response TypedDicts
│       └── _test_hooks.py # Real/Fake client implementations
├── streaming/             # Kafka streaming infrastructure
│   ├── config.py          # TypedDicts for Confluent/Kafka config
│   ├── schemas.py         # Event TypedDicts with encode/decode
│   ├── producer.py        # StreamingProducer wrapper
│   ├── consumer.py        # StreamingConsumer wrapper
│   └── _test_hooks.py     # Protocol-based DI (Real/Fake)
└── seeding/               # Database seeding
```

### Queue Architecture

```
┌─────────────────┐
│    FastAPI      │
│    API Server   │
└────────┬────────┘
         │ enqueue
         ▼
┌─────────────────┐     ┌─────────────────┐
│     Redis       │◄────│   RQ Worker     │
│   Job Queue     │     │                 │
│                 │     │  - Evaluate     │
│                 │     │  - Train        │
└─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │
│   (Persistence) │
└─────────────────┘
```

### Domain Models

```python
@dataclass
class Deal:
    id: DealId
    name: str
    borrower: str
    sector: str
    region: str
    commitment_amount_cents: int
    currency: str
    maturity_date_iso: str

@dataclass
class Covenant:
    id: CovenantId
    deal_id: DealId
    name: str
    formula: str
    threshold_value_scaled: int
    threshold_direction: Literal["<=", ">="]
    frequency: Literal["QUARTERLY", "ANNUAL"]

@dataclass
class Measurement:
    deal_id: DealId
    period_start_iso: str
    period_end_iso: str
    metric_name: str
    metric_value_scaled: int
```

---

## Development

### Commands

```bash
make install      # Install dependencies
make install-dev  # Install with dev dependencies
make lint         # Run guards + ruff + mypy
make test         # Run pytest with coverage
make check        # Run lint + test
```

### Quality Gates

All code must pass:

1. **Guard Scripts**: No `Any`, no `cast`, no `type: ignore`
2. **Ruff**: Linting and formatting
3. **Mypy**: Strict type checking
4. **Pytest**: 100% statement and branch coverage

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/test_routes_deals.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Docker

### Build and Run

```bash
# Build and run (from service directory)
docker compose up -d --build

# Or from repository root
make up-covenant

# View logs
docker compose logs -f

# Stop service
docker compose down
```

### Health Checks

- **API**: `/healthz` (liveness) and `/readyz` (readiness)
- **Worker**: Monitored via RQ heartbeats

---

## Observability

### Datadog Integration

The service integrates with Datadog for APM tracing and custom metrics.

**Enable in Production:**

```bash
export DATADOG__ENABLED=true
export DATADOG__SERVICE=covenant-radar-api
export DATADOG__ENV=production
export DATADOG__VERSION=1.0.0
export DATADOG__AGENT_HOST=datadog-agent
```

**Features:**

| Feature | Description |
|---------|-------------|
| APM Tracing | Distributed tracing via ddtrace auto-instrumentation |
| Custom Metrics | Application metrics via DogStatsD |
| Log Correlation | Automatic trace ID injection into structured logs |

**Auto-Instrumented Libraries:**

- FastAPI request lifecycle
- httpx outbound HTTP calls
- Redis operations
- Logging (trace ID injection)

**Custom Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `covenant.measurement.received` | counter | Measurement ingestion count |
| `covenant.evaluation.latency_ms` | histogram | Covenant evaluation time |
| `covenant.prediction.latency_ms` | histogram | ML prediction time |
| `covenant.prediction.risk_probability` | gauge | Current risk level per deal |
| `covenant.alert.triggered` | counter | Alert volume |

For complete documentation, see [docs/integrations/datadog.md](./docs/integrations/datadog.md).

---

## Database Seeding

Seed the database with synthetic test data:

```bash
# From service directory
poetry run python -m scripts.seed

# Verbose output
poetry run python -m scripts.seed -v
```

This creates 12 sample deals (6 safe, 6 risky) with covenants, measurements, and evaluation results across Technology, Finance, and Healthcare sectors.

---

## ML Model Training

### Pluggable Backends

The API supports four ML backends for breach risk prediction:

| Backend | Format | Best For | Feature Importance |
|---------|--------|----------|-------------------|
| `xgboost` | `.ubj` | Tabular data, interpretability | Yes (ranked list) |
| `mlp` | `.pt` | Non-linear patterns, deep learning | No |
| `lstm` | `.pt` | Temporal sequences, time-series | No |
| `lightgbm` | `.txt` | Large datasets, fast training | Yes (ranked list) |

### GPU Training

All four backends support GPU acceleration via the `device` parameter:
- `"cpu"` - Force CPU training
- `"cuda"` - Force GPU training (requires NVIDIA GPU with CUDA)
- `"auto"` - Auto-detect: uses GPU if available, falls back to CPU (default)

MLP and LSTM also support precision modes for GPU training:
- `"fp32"` - Full precision (most compatible)
- `"fp16"` - Half precision (faster on GPU)
- `"bf16"` - BFloat16 (faster on Ampere+ GPUs)
- `"auto"` - Auto-detect based on device

### Class Imbalance Handling

All backends handle imbalanced classes (few bankruptcies vs many healthy companies):
- XGBoost: `scale_pos_weight` parameter (auto-calculated if omitted)
- MLP/LSTM: Weighted BCE loss based on class distribution
- LightGBM: Auto-computed class weights

### Automatic Preprocessing

All backends apply automatic preprocessing to improve model quality. The preprocessing pipeline fits on training data only to prevent data leakage.

| Step | Description |
|------|-------------|
| Special Code Detection | Replaces sentinel values (96, 98, 999, -1, -9, -999) with NaN |
| Outlier Capping | Caps extreme values at 1st/99th percentile bounds |
| Missing Imputation | Fills NaN with per-feature median from training data |
| Z-Score Normalization | Standardizes features to mean=0, std=1 |

This happens transparently - no configuration needed. The preprocessing state is computed from training data and applied consistently to validation and test sets.

### Train on Internal Data

Train an XGBoost model on seeded deal/measurement data:

```bash
# Trigger training job (GPU auto-detect, auto class balancing)
curl -X POST http://localhost:8007/ml/train \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
# {"job_id": "uuid", "status": "queued"}

# Poll for completion - returns full metrics
curl http://localhost:8007/ml/jobs/{job_id}
# {
#   "job_id": "uuid",
#   "status": "finished",
#   "result": {
#     "model_id": "model-2024-01-15",
#     "best_val_auc": 0.89,
#     "scale_pos_weight": 14.5,
#     "test_metrics": {"auc": 0.87, "accuracy": 0.82, ...},
#     ...
#   }
# }
```

### Train on External Datasets - XGBoost

Train on real-world bankruptcy datasets with automatic feature importance:

```bash
# Train on Taiwan bankruptcy data (GPU, auto class balancing)
curl -X POST http://localhost:8007/ml/train-external \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Poll for results with feature importance ranking
curl http://localhost:8007/ml/jobs/{job_id}
# {
#   "result": {
#     "backend": "xgboost",
#     "model_format": "ubj",
#     "test_metrics": {"auc": 0.93, ...},
#     "feature_importances": [
#       {"name": "X6", "importance": 0.18, "rank": 1},
#       {"name": "X1", "importance": 0.09, "rank": 2},
#       ...
#     ]
#   }
# }
```

### Train on External Datasets - MLP Neural Network

Train an MLP neural network with configurable architecture:

```bash
# Train MLP on Taiwan bankruptcy data
curl -X POST http://localhost:8007/ml/train-external \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Poll for results (no feature importances for MLP)
curl http://localhost:8007/ml/jobs/{job_id}
# {
#   "result": {
#     "backend": "mlp",
#     "model_format": "pt",
#     "test_metrics": {"auc": 0.91, ...},
#     "feature_importances": []
#   }
# }
```

### Train on External Datasets - LightGBM

Train a LightGBM gradient boosting model with fast training and feature importance:

```bash
# Train LightGBM on Taiwan bankruptcy data
curl -X POST http://localhost:8007/ml/train-external \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Poll for results with feature importance ranking
curl http://localhost:8007/ml/jobs/{job_id}
# {
#   "result": {
#     "backend": "lightgbm",
#     "model_format": "txt",
#     "test_metrics": {"auc": 0.92, ...},
#     "feature_importances": [
#       {"name": "X6", "importance": 0.15, "rank": 1},
#       {"name": "X1", "importance": 0.08, "rank": 2},
#       ...
#     ]
#   }
# }
```

### Train on External Datasets - LSTM

Train an LSTM recurrent network for temporal sequence modeling:

```bash
# Train LSTM on Taiwan bankruptcy data (bidirectional)
curl -X POST http://localhost:8007/ml/train-external \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Poll for results (no feature importances for LSTM)
curl http://localhost:8007/ml/jobs/{job_id}
# {
#   "result": {
#     "backend": "lstm",
#     "model_format": "pt",
#     "test_metrics": {"auc": 0.90, ...},
#     "feature_importances": []
#   }
# }
```

**Standard Datasets:**
- `taiwan` - Taiwan bankruptcy data (6,819 samples, 95 features)
- `us` - US bankruptcy data (78,682 samples, 18 features)
- `polish` - Polish bankruptcy data (7,027 samples, 64 features)

**Time-Series Datasets:**
- `kaggle_amex_default` - AMEX Default Prediction (458,913 entities, 188 features, ~13 time steps)

### Hyperparameter Optimization with Optuna

Automatically find optimal hyperparameters for any backend using Bayesian optimization:

```bash
# XGBoost optimization (default backend)
curl -X POST http://localhost:8007/ml/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "taiwan",
    "backend": "xgboost",
    "n_trials": 50,
    "device": "auto",
    "space_profile": "default",
    "feature_preset": "full"
  }'

# MLP optimization
curl -X POST http://localhost:8007/ml/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "taiwan",
    "backend": "mlp",
    "n_trials": 50,
    "precision": "fp16",
    "optimizer": "adamw",
    "n_epochs": 100,
    "early_stopping_patience": 15
  }'

# LightGBM optimization
curl -X POST http://localhost:8007/ml/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "polish",
    "backend": "lightgbm",
    "n_trials": 30,
    "early_stopping_rounds": 20
  }'

# LSTM optimization
curl -X POST http://localhost:8007/ml/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "us",
    "backend": "lstm",
    "n_trials": 25,
    "precision": "bf16",
    "sequence_length": 10,
    "bidirectional": true
  }'

# Poll for best hyperparameters
curl http://localhost:8007/ml/jobs/{job_id}
# {
#   "result": {
#     "backend": "xgboost",
#     "status": "complete",
#     "best_val_auc": 0.94,
#     "recommended_config": {...}
#   }
# }
```

**Backend-Specific Options:**

| Backend | Specific Options |
|---------|------------------|
| `xgboost` | `space_profile` (`default`, `categorical`) |
| `mlp` | `precision`, `optimizer`, `n_epochs`, `early_stopping_patience` |
| `lightgbm` | `early_stopping_rounds` |
| `lstm` | `precision`, `n_epochs`, `early_stopping_patience`, `sequence_length`, `bidirectional` |

**DART Boosting Support:**

XGBoost and LightGBM optimization automatically includes DART (Dropouts meet Multiple Additive Regression Trees) in the search space. DART applies dropout regularization during boosting to reduce overfitting.

| Backend | DART Parameter | Values | Description |
|---------|----------------|--------|-------------|
| XGBoost | `booster` | `gbtree`, `dart` | Enables DART when set to "dart" |
| XGBoost | `rate_drop` | 0.0-0.5 | Tree dropout rate (DART only) |
| XGBoost | `skip_drop` | 0.0-0.5 | Probability of skipping dropout (DART only) |
| LightGBM | `boosting_type` | `gbdt`, `dart` | Enables DART when set to "dart" |
| LightGBM | `drop_rate` | 0.0-0.5 | Tree dropout rate (DART only) |
| LightGBM | `skip_drop` | 0.0-0.5 | Probability of skipping dropout (DART only) |
| LightGBM | `feature_fraction` | 0.02-0.1 | Aggressive feature subsampling (DART only) |

Optuna conditionally samples DART parameters only when DART boosting is selected, allowing exploration of both standard and DART configurations.

**Note:** Early stopping is automatically disabled for LightGBM DART mode, as DART's random tree dropout makes early stopping unreliable.

The `recommended_config` in the result can be used directly with `/ml/train-external`.

### Submit CLI (Kaggle Submissions)

Backend-agnostic CLI for training models on time-series data and generating Kaggle submission files:

```bash
# Default (LightGBM backend)
poetry run python -m scripts.submit --train-dir data/train --test-dir data/test -o submission.csv

# With specific settings
poetry run python -m scripts.submit -b lightgbm -n 1000 -l 0.05 --num-leaves 31

# Other backends
poetry run python -m scripts.submit -b xgboost -n 100 -l 0.1
poetry run python -m scripts.submit -b mlp -n 50 -l 0.001
poetry run python -m scripts.submit -b lstm -n 50 -l 0.001

# Feature engineering options
poetry run python -m scripts.submit --no-rank-features --no-diff-features
poetry run python -m scripts.submit -a statistics  # Aggregation: last, first, mean, statistics
```

**CLI Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backend` | `-b` | `lightgbm` | Backend: `lightgbm`, `xgboost`, `mlp`, `lstm` |
| `--n-estimators` | `-n` | `1000` | Boosting rounds (tree) or epochs (neural) |
| `--learning-rate` | `-l` | `0.05` | Learning rate |
| `--num-leaves` | | `31` | Max leaves per tree (LightGBM only) |
| `--max-depth` | | `-1` | Max tree depth (-1 = unlimited) |
| `--aggregation` | `-a` | `statistics` | Aggregation: `last`, `first`, `mean`, `statistics` |
| `--no-rank-features` | | False | Disable per-entity rank features |
| `--no-diff-features` | | False | Disable row-to-row diff features |
| `--train-dir` | | `data/external/amex_train` | Training data directory |
| `--test-dir` | | `data/external/amex_test` | Test data directory |
| `--output` | `-o` | `data/submissions/submission.csv` | Output CSV path |

For complete documentation, see [scripts/submit/README.md](./scripts/submit/README.md).

### AMEX Competition Pipeline

Ensemble pipeline for Kaggle AMEX Default Prediction competition with multi-backend support and weight optimization:

```bash
# Full pipeline with defaults
poetry run python -m scripts.amex

# Custom configuration
poetry run python -m scripts.amex \
    --backends lightgbm,xgboost \
    --n-folds 5 \
    --n-estimators 1000 \
    --learning-rate 0.05 \
    --aggregation statistics \
    --window-sizes 3,6 \
    --output submission.csv

# Minimal test run
poetry run python -m scripts.amex -b lightgbm -k 2 -n 10
```

**CLI Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backends` | `-b` | `lightgbm,xgboost` | Comma-separated backends |
| `--n-folds` | `-k` | `5` | Number of CV folds |
| `--n-estimators` | `-n` | `1000` | Boosting rounds |
| `--learning-rate` | `-l` | `0.05` | Learning rate |
| `--aggregation` | `-a` | `statistics` | Aggregation strategy |
| `--window-sizes` | `-w` | `3,6` | Comma-separated window sizes |
| `--no-rank-features` | | False | Disable rank features |
| `--no-diff-features` | | False | Disable diff features |
| `--no-window-features` | | False | Disable window features |
| `--output` | `-o` | `data/submissions/amex_submission.csv` | Output CSV path |

**Pipeline Steps:**
1. Load training data with competition features (rank, diff, window)
2. Train each backend with GroupKFold CV (no customer leakage)
3. Optimize ensemble weights to maximize AMEX metric
4. Generate weighted predictions on test data
5. Write Kaggle submission.csv

For complete documentation, see [scripts/amex/README.md](./scripts/amex/README.md).

### Optimization CLI

For local development and benchmarking, use the CLI directly instead of the API:

```bash
# Run optimization with defaults (taiwan, 300 trials, full features, cuda)
poetry run python -m scripts.optimize

# Quick test run
poetry run python -m scripts.optimize -n 10 -d taiwan -f full --device cpu

# Compare all feature presets on a dataset
poetry run python -m scripts.optimize -c -n 50 -d us

# Run on all standard datasets (taiwan, us, polish)
poetry run python -m scripts.optimize -a -n 100

# Run on time-series dataset (AMEX)
poetry run python -m scripts.optimize -d kaggle_amex_default -n 50 -b xgboost

# With timeout (seconds)
poetry run python -m scripts.optimize -n 300 -t 3600

# Verbose logging
poetry run python -m scripts.optimize -v -n 50
```

**CLI Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backend` | `-b` | `xgboost` | Backend: `xgboost`, `mlp`, `lightgbm`, `lstm` |
| `--dataset` | `-d` | `taiwan` | Dataset name (standard or time-series) |
| `--n-trials` | `-n` | `300` | Number of Optuna trials |
| `--feature-preset` | `-f` | `full` | Features: `none`, `log_only`, `ratios_only`, `full` |
| `--device` | | `cuda` | Device: `cpu`, `cuda`, `auto` |
| `--timeout` | `-t` | None | Timeout in seconds |
| `--compare-presets` | `-c` | False | Compare all 4 feature presets |
| `--all-datasets` | `-a` | False | Run on all standard datasets |
| `--verbose` | `-v` | False | Enable verbose logging |

**Standard Datasets:** `taiwan`, `us`, `polish`
**Time-Series Datasets:** `kaggle_amex_default`

**Feature Presets:**

| Preset | Description |
|--------|-------------|
| `none` | Original features only |
| `log_only` | Original + log transforms |
| `ratios_only` | Original + financial ratios |
| `full` | Original + log + ratios + products |

**History Tracking:**

The CLI tracks all optimization runs in `models/optimization_history.jsonl` and displays progression:
- Compares current run against previous run for the same dataset/preset
- Shows all-time best AUC with delta indicators
- Marks new records with visual indicators

### Predict Breach Risk

```bash
curl -X POST http://localhost:8007/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"deal_id": "your-deal-uuid"}'
# {"deal_id": "...", "probability": 0.82, "risk_tier": "CRITICAL"}
```

**Risk Tiers:**
- `LOW`: probability < 0.25
- `MEDIUM`: 0.25 <= probability < 0.50
- `HIGH`: 0.50 <= probability < 0.80
- `CRITICAL`: probability >= 0.80

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `redis` | Job queue backend |
| `rq` | Redis Queue |
| `psycopg[binary,pool]` | PostgreSQL driver |
| `xgboost` | Gradient boosting backend |
| `torch` | MLP neural network backend |
| `scikit-learn` | Feature processing |
| `numpy` | Numerical operations |
| `platform-core` | Logging, errors, config |
| `platform-workers` | RQ worker harness |
| `covenant-domain` | Domain models |
| `covenant-persistence` | Repository layer |
| `covenant-ml` | Pluggable ML backends |
| `google-genai` | Google AI (Gemini) SDK |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

---

## Quality Standards

### Type Safety

- **mypy strict mode**: All code passes `--strict` type checking
- **No `Any`**: Zero use of `Any` type anywhere in codebase
- **No `cast`**: No type casting workarounds
- **No `type: ignore`**: No suppression comments
- **No `.pyi` stubs**: All types inline in source files
- **TypedDict over dataclass**: Immutable typed dictionaries for structured data
- **Protocol-based DI**: Dependency injection via Protocol types

### Test Coverage

- **100% statements and branches**: Full coverage across all modules
- **No mocks**: Tests use real fake implementations via dependency injection
- **Strong assertions**: Tests verify exact behavior, not just "no errors"
- **Callback testing**: Nested callbacks exercised via fake runners

### Test Architecture

Tests use the `_test_hooks.py` pattern for dependency injection:

```python
# Production: Hooks set to real implementations at startup
_hooks.xgboost_runner = run_xgboost_optimization

# Tests: Hooks set to fakes for isolated testing
def fake_runner(...) -> XGBoostOptimizationResult:
    # Invoke callbacks to exercise nested functions
    if loading_progress_callback is not None:
        loading_progress_callback(progress_info)
    return _make_fake_result()

_hooks.xgboost_runner = fake_runner
```

This pattern enables:
- Full coverage of nested callback functions
- Isolation without mock objects
- Type-safe test doubles

### Guard Rules

Enforced via `scripts/guard.py`:
- `typing`: No `Any`, `cast`, `type: ignore`
- `imports`: Proper module structure
- `tests`: No mocks, strong assertions
- `exceptions`: Explicit error handling
- `patterns`: Consistent code patterns

### Other Standards

- **Logging**: Structured JSON via platform_core
- **Errors**: Consistent `{code, message, request_id}` format
- **No try/except recovery**: Errors propagate, no best-effort handling
- **No backwards compatibility**: Clean breaks, no shims

---

## Port Map

- **8007**: covenant-radar-api

---

## License

Apache-2.0
