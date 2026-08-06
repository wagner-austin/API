# Covenant Radar API

**Multi-domain risk prediction behind one protocol.** This began as loan-covenant
breach monitoring and was then generalized: adding a risk domain is a
*registration*, not a fork. Three ship today — `covenant`, `weather`, `esports` —
each its own package, reusing the same training, hyperparameter-search,
explainability and streaming machinery without modifying any of it.

Loan covenants remain the reference domain, and the CRUD plus rule-evaluation
surface below is specific to them. Everything under `/ml/*` and the streaming
worker is domain-agnostic.

**Detailed documentation lives alongside the code:**

| Topic | Document |
|-------|----------|
| Every environment variable, with defaults | [docs/configuration.md](./docs/configuration.md) |
| Full API reference (every field, every response) | [docs/api.md](./docs/api.md) |
| ML endpoints — train, optimize, explain, regression | [docs/api/ml.md](./docs/api/ml.md) |
| CRUD + evaluation endpoints | [deals](./docs/api/deals.md), [covenants](./docs/api/covenants.md), [measurements](./docs/api/measurements.md), [evaluation](./docs/api/evaluation.md) |
| Health, dashboard | [health](./docs/api/health.md), [dashboard](./docs/api/dashboard.md) |
| Kafka streaming integration | [docs/integrations/streaming.md](./docs/integrations/streaming.md) |
| Datadog APM + metrics | [docs/integrations/datadog.md](./docs/integrations/datadog.md) |
| Google AI (Gemini) alert summaries | [docs/integrations/google_ai.md](./docs/integrations/google_ai.md) |
| Optimization / submission / AMEX CLIs | [optimize](./scripts/optimize/README.md), [submit](./scripts/submit/README.md), [amex](./scripts/amex/README.md) |

## The design decisions worth knowing

- **A domain is a package, not a branch.** One domain-agnostic streaming worker
  serves every registered domain, selected at runtime by `STREAMING__DOMAIN`.
  Adding a domain means writing a package that satisfies the protocol — the
  training, explainability and streaming code is untouched. That constraint is
  what took this from a loan-covenant tool to a risk-prediction platform.
- **Deterministic rules and learned models are kept apart.** Covenant compliance
  is exact rule evaluation (`OK` / `NEAR_BREACH` / `BREACH`) with formulas,
  thresholds and frequencies. Breach *risk* is the ML path. Conflating the two
  would make a compliance answer probabilistic, which is worse than useless.
- **Eleven model backends behind one interface** — seven classifiers (XGBoost,
  LightGBM, ClearGBM, logistic regression, random forest, PyTorch MLP, and a
  bidirectional LSTM for temporal sequences) plus four regressors, swappable
  per request.
- **Explainability adapts to the backend.** One `/ml/explain` endpoint, three
  strategies chosen by model type: permutation importance for anything, SHAP
  TreeExplainer for tree models, input and integrated gradients for the neural
  ones.
- **Optuna TPE search** over categorical and continuous spaces, with DART
  boosting for XGBoost and LightGBM and early stopping on validation AUC.
- **Kafka on Confluent Cloud** for real-time inference, with a dead-letter topic
  so an undecodable payload can't stall the stream.
- **Gemini** for human-readable alert summaries, with token-usage and latency
  metrics attached.

Also here: deal / covenant / measurement CRUD, Redis + RQ background training, a
`/dashboard` monitoring UI with vendored Chart.js, Datadog APM tracing, and
PostgreSQL persistence. Strict mypy and full statement + branch coverage, as
everywhere in this monorepo.

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

# Start dependencies. Redis and PostgreSQL live in the ROOT compose file, not
# this service's -- `make infra` from the repository root brings them up.
make -C ../.. infra

# Run API
poetry run hypercorn 'covenant_radar_api.api.main:create_app()' --bind 0.0.0.0:8000

# Run Worker (separate terminal)
poetry run covenant-rq-worker

# Run the streaming worker (separate terminal; requires STREAMING__ENABLED=true)
poetry run covenant-streaming-worker
```

## API Reference

Full request/response detail is in [docs/api.md](./docs/api.md). The endpoint
surface:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe |
| `/status` | GET | Service status with dependency health |
| `/deals` | GET, POST | List all deals, create a deal |
| `/deals/{deal_id}` | GET, PUT, DELETE | Get, update, delete a deal |
| `/covenants` | POST | Create a new covenant |
| `/covenants/by-deal/{deal_id}` | GET | List covenants for a deal |
| `/covenants/{covenant_id}` | GET, DELETE | Get, delete a covenant |
| `/measurements` | POST | Add measurements |
| `/measurements/by-deal/{deal_id}` | GET | List measurements for a deal |
| `/measurements/by-deal/{deal_id}/period` | GET | List measurements for deal and period |
| `/evaluate` | POST | Evaluate covenant compliance |
| `/ml/predict` | POST | Predict breach risk |
| `/ml/train` | POST | Enqueue model training on internal data |
| `/ml/train-external` | POST | Train on external CSV datasets |
| `/ml/optimize` | POST | Optimize hyperparameters with Optuna TPE |
| `/ml/explain` | POST | Compute feature importance explanations |
| `/ml/train-external-regression` | POST | Train a regressor (`xgboost_reg`, `lightgbm_reg`) |
| `/ml/optimize-regression` | POST | Optimize regressor hyperparameters (all four `*_reg` backends) |
| `/ml/explain-regression` | POST | Feature importance for a trained regressor |
| `/ml/predict-regression` | POST | Predict continuous values from a feature matrix |
| `/ml/jobs/{job_id}` | GET | Get training job status |
| `/ml/models/active` | GET | Get active model info |
| `/dashboard` | GET | Real-time monitoring UI (HTML) |

### Worked Example

```bash
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
#         "model_path": "/data/models/active_xgb.ubj",
#         "is_loaded": false
#     },
#     "data": {"deals": 5}
# }

# Predict breach risk for a deal
curl -X POST http://localhost:8007/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"deal_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"}'
# {"deal_id": "...", "probability": 0.23, "risk_tier": "LOW"}
```

**Risk Tiers:** `LOW` < 0.25, `MEDIUM` 0.25-0.50, `HIGH` 0.50-0.80, `CRITICAL` >= 0.80.

---

## Configuration

Every environment variable, with defaults and an example `.env`, is in
[docs/configuration.md](./docs/configuration.md). The ones that matter for a
first run:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection URL. **Required** — the service refuses to start without it |
| `REDIS_URL` or `REDIS__URL` | `redis://redis:6379/0` | Redis connection URL |
| `APP__ML_BACKEND` | `xgboost` | Inference backend (`xgboost`, `mlp`, `lstm`, `lightgbm`); picks which active model path is used |
| `APP__MODELS_ROOT` | `/data/models` | Models directory; every request-supplied `model_path` must resolve inside it |
| `STREAMING__ENABLED` | `false` | Enable Kafka streaming |
| `STREAMING__DOMAIN` | `weather` | Which domain the streaming worker runs (`weather` or `esports`) |
| `DATADOG__ENABLED` | `false` | Enable Datadog APM + metrics |

There is no `APP__ACTIVE_MODEL_PATH`; the active path is derived from
`APP__ML_BACKEND` plus `APP__ACTIVE_MODEL_PATH_XGB` / `_MLP`.

---

## Architecture

```
covenant_radar_api/
├── api/                    # FastAPI routes
│   ├── main.py            # App factory
│   ├── decode.py          # Request parsing
│   └── routes/            # health, status, deals, covenants, measurements,
│                          # evaluate, ml, dashboard
├── core/
│   ├── config.py          # Settings
│   ├── container.py       # DI container
│   └── model_paths.py     # Rejects model paths outside APP__MODELS_ROOT
├── worker/                # RQ jobs: evaluate, train, train_external,
│                          # optimize, explain (+ _regression variants)
├── domains/               # Pluggable streaming domains
│   ├── protocols.py       # DomainProtocol
│   ├── registry.py        # Name to domain-factory registry
│   ├── weather/           # McKinnon-style temporal features
│   └── esports/           # Live win-probability inference
├── integrations/          # datadog/ (APM + DogStatsD), google_ai/ (Gemini)
├── streaming/             # Kafka: config, schemas, producer, consumer,
│                          # worker (covenant), generic_worker (any domain)
├── generic_worker_entry.py # covenant-streaming-worker entry point
├── worker_entry.py         # covenant-rq-worker entry point
└── seeding/               # Database seeding
```

### Entry Points

| Command | Module | Purpose |
|---------|--------|---------|
| `hypercorn 'covenant_radar_api.api.main:create_app()'` | `api.main` | FastAPI app |
| `covenant-rq-worker` | `worker_entry:main` | RQ worker for training / evaluation jobs |
| `covenant-streaming-worker` | `generic_worker_entry:main` | Kafka consumer for the selected domain |

### Queue Architecture

```
FastAPI API server --enqueue--> Redis job queue <---- RQ Worker (evaluate, train)
                                                          |
                                                          v
                                                   PostgreSQL (persistence)
```

### Streaming Dead-Letter Topic

A measurement the worker cannot decode is published to `KAFKA__TOPIC_DLQ`
(default `covenant.dlq.v1`) rather than crashing the consumer, with `reason`
`undecodable_payload`. Each record carries the original payload plus the source
topic / partition / offset, so it can be inspected and replayed once the cause
is fixed.

The offset is recorded as seen but never marked pending, so the commit position
advances past it only after the dead-letter record is published. Without the
dead-letter topic there is nowhere safe to move the offset to, and a single bad
message is redelivered on every restart forever.

Domain models (`Deal`, `Covenant`, `Measurement`, `CovenantResult`) and the
scaled-integer convention are documented in [docs/api.md](./docs/api.md).

---

## ML Model Training

Request and response detail for every ML endpoint is in
[docs/api/ml.md](./docs/api/ml.md). This section covers what the backends are
and how they behave.

### Classification Backends

Tree and linear backends come from `covenant_ml`; `mlp` and `lstm` come from
`covenant_nn`.

| Backend | Package | Format | GPU | Best For | Feature Importance |
|---------|---------|--------|-----|----------|-------------------|
| `xgboost` | covenant_ml | `.ubj` | CUDA | Tabular data, interpretability | ranked, shap_tree, permutation |
| `lightgbm` | covenant_ml | `.txt` | CUDA | Large datasets, fast training | ranked, shap_tree, permutation |
| `cleargbm` | covenant_ml | `.json` | No | Interpretable boosting | shap_tree, permutation |
| `logreg` | covenant_ml | `.joblib` | No | Linear baselines | permutation |
| `random_forest` | covenant_ml | `.joblib` | No | Ensemble baselines | shap_tree, permutation |
| `mlp` | covenant_nn | `.pt` | CUDA | Non-linear patterns, deep learning | gradient, integrated_gradients, permutation |
| `lstm` | covenant_nn | `.pt` | CUDA | Temporal sequences, time-series | gradient, integrated_gradients, permutation |

### Regression Backends

| Backend | Optimize | Train | Explain | Predict |
|---------|----------|-------|---------|---------|
| `xgboost_reg` | Yes | Yes | Yes | Yes |
| `lightgbm_reg` | Yes | Yes | Yes | Yes |
| `mlp_reg` | Yes | No | Yes | Yes |
| `lstm_reg` | Yes | No | Yes | Yes |

`/ml/train-external-regression` accepts only `xgboost_reg` and `lightgbm_reg`;
the other three regression routes accept all four. The neural-net optimizer is
selected with `optimizer` (`adamw` | `adam` | `sgd`) — the same wire key the
classification routes use.

### Device Configuration

The CUDA-capable backends support device selection via the `device` parameter:
`"cpu"`, `"cuda"` (requires an NVIDIA GPU with CUDA libraries), or `"auto"`
(GPU if available, else CPU). `cleargbm`, `logreg` and `random_forest` are
CPU-only and ignore the setting. MLP and LSTM also support `precision` —
`fp32`, `fp16`, `bf16`, or `auto`.

**Note:** `torch` is resolved from the `pytorch-cuda` index (`cu128`) via
`covenant_nn`, so the installed wheel is a CUDA build, not a CPU-only one — the
lock file pins `torch 2.10.0+cu128`. It still runs on CPU-only hosts; the cost
is image size, not function.

### Datasets

**Standard (classification)** — accepted by `/ml/train-external` and `/ml/optimize`:
`taiwan` (6,819 samples, 95 features), `us` (78,682 / 18), `polish` (7,027 / 64),
`kaggle_company_bankruptcy`, `kaggle_credit_default`, `kaggle_credit_risk`,
`kaggle_heloc`, `kaggle_give_me_credit`, `kaggle_loan_default`.

**Time-series:** `kaggle_amex_default` (458,913 entities, 188 features, ~13 time steps).

**Regression:** `financial_distress`.

The optimization CLI validates against a shorter list than the API — see
[scripts/optimize/README.md](./scripts/optimize/README.md).

### Class Imbalance Handling

All backends handle imbalanced classes (few bankruptcies vs many healthy companies):

- XGBoost: `scale_pos_weight` parameter (auto-calculated if omitted)
- MLP/LSTM: Weighted BCE loss based on class distribution
- LightGBM: Auto-computed class weights
- ClearGBM: Auto-computed `scale_pos_weight` from the training labels
- LogReg/Random Forest: sklearn `class_weight="balanced"` via `class_weight_balanced`

Every backend also runs a fixed preprocessing pipeline (sentinel-code
detection, outlier capping, median imputation, z-score normalization), fitted
on the training split only — see
[docs/api/ml.md](./docs/api/ml.md#automatic-preprocessing).

---

## Command-Line Tools

| Command | Purpose | Docs |
|---------|---------|------|
| `python -m scripts.optimize` | Multi-backend Optuna optimization with history tracking | [README](./scripts/optimize/README.md) |
| `python -m scripts.submit` | Backend-agnostic Kaggle submission generator | [README](./scripts/submit/README.md) |
| `python -m scripts.amex` | AMEX ensemble pipeline with CV + weight optimization | [README](./scripts/amex/README.md) |
| `python -m scripts.explain` | Run the explainer registry against a trained model | — |
| `python -m scripts.replay_data` | Replay a dataset onto the Kafka measurements topic | — |
| `python -m scripts.discover_datasets` | Scan `data/external` and generate `DatasetConfig` entries | [README](./scripts/discover_datasets/README.md) |
| `python -m scripts.seed` | Seed the database with synthetic data | see below |

### Database Seeding

```bash
poetry run python -m scripts.seed      # add -v for verbose output
```

Creates 12 sample deals (6 safe, 6 risky) with covenants, measurements, and
evaluation results across Technology, Finance, and Healthcare sectors.

---

## Development

```bash
make install      # Install dependencies
make install-dev  # Install with dev dependencies
make lint         # Run guards + ruff + mypy
make test         # Run pytest with coverage
make check        # Run lint + test

poetry run pytest tests/test_routes_deals.py -v   # Single test file
poetry run pytest --cov-report=html               # HTML coverage report
```

### Docker

```bash
docker compose up -d --build   # From the service directory
make up-covenant               # Or from the repository root
docker compose logs -f
docker compose down
```

The compose file defines three services — `api`, `worker`, and
`streaming-worker` — built from three targets in the same Dockerfile (`api`,
`worker`, `streaming`). Redis and PostgreSQL are not in this compose file; they
come from the root compose (`make infra`). One streaming image serves every
domain: which one it runs is `STREAMING__DOMAIN` configuration, not a build
choice.

Health checks: `/healthz` (liveness) and `/readyz` (readiness) for the API;
RQ heartbeats for the worker.

---

## Dependencies

Direct runtime dependencies, as declared in `pyproject.toml`: `fastapi`,
`hypercorn`, `redis`, `rq`, `psycopg[binary,pool]`, `httpx`, `xgboost`,
`scikit-learn`, `numpy`, `rich`, `confluent-kafka`, `ddtrace`, `datadog`,
`google-genai`, `openpyxl`, `xlrd`, plus the workspace libs `platform-core`,
`platform-workers`, `covenant-domain`, `covenant-persistence`, `covenant-ml`
(pulls in `lightgbm`), `covenant-nn` (pulls in `torch`), and `cleargbm-rs`.

Dev: `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-asyncio`,
`pytest-timeout`, `fakeredis[lua]`, `mypy`, `ruff`, `typing-extensions`, `xlwt`.

---

## Quality Standards

All code must pass guards (`scripts/guard.py`), ruff, mypy strict, and pytest
with 100% statement and branch coverage.

- **Type Safety**: no `Any`, no `cast`, no `type: ignore`, no `.pyi` stubs;
  TypedDict over dataclass; Protocol-based DI
- **Tests**: no mocks — fakes are injected through the `_test_hooks.py` pattern,
  which also gives nested callbacks real coverage
- **Guard rules**: `typing`, `imports`, `tests`, `exceptions`, `patterns`
- **Logging**: structured JSON via platform_core
- **Errors**: consistent `{code, message, request_id}`; errors propagate rather
  than being softened — a missing row is a 404, a real defect stays a 500
- **No backwards compatibility**: clean breaks, no shims

---

## Port Map

- **8007**: covenant-radar-api

## License

Apache-2.0
