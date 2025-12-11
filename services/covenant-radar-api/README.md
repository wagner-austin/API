# Covenant Radar API

Loan covenant monitoring and breach prediction API service. Features deterministic rule evaluation, XGBoost-based breach risk prediction, and PostgreSQL persistence.

## Features

- **Deal Management**: CRUD operations for loan deals with structured metadata
- **Covenant Definitions**: Configurable rules with formulas, thresholds, and frequencies
- **Financial Measurements**: Time-series metric ingestion for covenant calculations
- **Rule Evaluation**: Deterministic covenant compliance checking with OK/NEAR_BREACH/BREACH status
- **Breach Prediction**: XGBoost classifier for risk tier prediction (LOW/MEDIUM/HIGH)
- **Background Training**: Redis + RQ worker for model training jobs
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
| `/evaluate` | POST | Evaluate covenant compliance |
| `/ml/predict` | POST | Predict breach risk |
| `/ml/train` | POST | Enqueue model training |
| `/ml/train-external` | POST | Train on external CSV datasets |
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

### Example .env

```bash
APP_ENV=dev
DATABASE_URL=postgresql://covenant:covenant@postgres:5432/covenant
REDIS_URL=redis://redis:6379/0
RQ__QUEUE_NAME=covenant
APP__ACTIVE_MODEL_PATH=/data/models/active.ubj
LOGGING__LEVEL=INFO
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
│   └── train_job.py       # Model training
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

### Train on Internal Data

Train an XGBoost model on seeded deal/measurement data:

```bash
# Trigger training job
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
    "early_stopping_rounds": 10
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
#     "test_metrics": {"auc": 0.87, "accuracy": 0.82, ...},
#     ...
#   }
# }
```

### Train on External Datasets (with Automatic Feature Selection)

Train on real-world bankruptcy datasets. XGBoost automatically determines which features are most predictive:

```bash
# Train on Taiwan bankruptcy data (95 financial ratios)
curl -X POST http://localhost:8007/ml/train-external \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "taiwan",
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
  }'

# Poll for results with feature importance ranking
curl http://localhost:8007/ml/jobs/{job_id}
# {
#   "result": {
#     "test_metrics": {"auc": 0.93, ...},
#     "feature_importances": [
#       {"name": "X6", "importance": 0.18, "rank": 1},
#       {"name": "X1", "importance": 0.09, "rank": 2},
#       ...
#     ]
#   }
# }
```

**Available Datasets:**
- `taiwan` - Taiwan bankruptcy data (6,819 samples, 95 features)
- `us` - US bankruptcy data
- `polish` - Polish bankruptcy data

### Predict Breach Risk

```bash
curl -X POST http://localhost:8007/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"deal_id": "your-deal-uuid"}'
# {"deal_id": "...", "probability": 0.82, "risk_tier": "HIGH"}
```

**Risk Tiers:**
- `LOW`: probability < 0.3
- `MEDIUM`: 0.3 <= probability < 0.7
- `HIGH`: probability >= 0.7

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
| `xgboost` | ML model |
| `scikit-learn` | Feature processing |
| `numpy` | Numerical operations |
| `platform-core` | Logging, errors, config |
| `platform-workers` | RQ worker harness |
| `covenant-domain` | Domain models |
| `covenant-persistence` | Repository layer |
| `covenant-ml` | ML utilities |

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

- **Type Safety**: mypy strict mode, no `Any`, no `cast`
- **Coverage**: 100% statements and branches
- **Guard Rules**: Enforced via `scripts/guard.py`
- **Logging**: Structured JSON via platform_core
- **Errors**: Consistent `{code, message, request_id}` format

---

## Port Map

- **8007**: covenant-radar-api

---

## License

Apache-2.0
