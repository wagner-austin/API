# Hackathon Implementation Plan — Covenant Radar API

**Hackathon:** AI Partner Catalyst: Accelerate Innovation
**Deadline:** December 31, 2025
**Partner Tracks:** Confluent (primary) + Datadog (observability)
**Google Cloud Requirement:** Gemini (minimal, for alert text generation)

---

## 1) Strategic Decisions

### 1.1 Partner Track Selection

We are implementing **both** Confluent and Datadog:

| Partner | Role | Rationale |
|---------|------|-----------|
| **Confluent** | Primary track submission | Streaming inference aligns with our ML expertise. Our XGBoost/LightGBM ensemble is the star, not an LLM wrapper. |
| **Datadog** | Observability layer | Useful feature regardless of hackathon. APM tracing, custom metrics, log correlation. |
| **Gemini** | Minimal integration | Required for "Google Cloud AI". Used only for generating human-readable alert text when risk > 0.8. No LLM-as-brain pattern. |

### 1.2 Architecture Philosophy

**Streaming Inference, NOT Online Learning:**
- Training remains batch (unchanged): Optuna optimization, cross-validation, model artifacts
- Inference becomes streaming: Kafka consume → model.predict() → Kafka produce
- Retrain triggers: drift detection, not continuous learning

**Why not online learning:**
- Our backends (XGBoost, LightGBM, MLP, LSTM) are batch-oriented
- Online learning is mostly hype for production systems
- Industry standard: batch retrain on triggers with validation before deploy

### 1.3 Data Source Strategy

**Synthetic Measurement Generator:**
- No real streaming loan data available
- Generator produces realistic measurement events matching our domain model
- Configurable scenarios: gradual deterioration, sudden shock, recovery, seasonality, missing data
- Sufficient for hackathon demo; architecture supports real sources later

---

## 2) Infrastructure Decisions

### 2.1 Confluent Cloud (Managed Kafka)

We use **Confluent Cloud**, not self-hosted Kafka:

| Aspect | Decision |
|--------|----------|
| Bootstrap servers | Confluent Cloud endpoint (environment variable) |
| Authentication | SASL/PLAIN with API key + secret |
| Schema Registry | Confluent Schema Registry (optional, JSON schema inline for now) |
| Topics | Created via Confluent Cloud console or Terraform |

**Configuration (environment variables):**
```
CONFLUENT_BOOTSTRAP_SERVERS=pkc-xxxxx.us-east-1.aws.confluent.cloud:9092
CONFLUENT_API_KEY=...
CONFLUENT_API_SECRET=...
CONFLUENT_SCHEMA_REGISTRY_URL=https://psrc-xxxxx.us-east-1.aws.confluent.cloud
CONFLUENT_SCHEMA_REGISTRY_API_KEY=...
CONFLUENT_SCHEMA_REGISTRY_API_SECRET=...
```

### 2.2 Datadog

| Aspect | Decision |
|--------|----------|
| APM | ddtrace auto-instrumentation for FastAPI, httpx, redis |
| Metrics | DogStatsD UDP to Datadog agent |
| Logs | Existing JSON structured logs, ddtrace adds trace_id correlation |
| Agent | Datadog agent sidecar in deployment |

**Configuration (environment variables):**
```
DD_SERVICE=covenant-radar-api
DD_ENV=production
DD_VERSION=1.0.0
DD_AGENT_HOST=localhost
DD_DOGSTATSD_PORT=8125
DD_TRACE_ENABLED=true
```

### 2.3 Google Cloud (Gemini)

| Aspect | Decision |
|--------|----------|
| Model | gemini-1.5-flash (fast, cheap) |
| Use case | Generate 1-sentence alert text when risk > 0.8 |
| Integration | Direct API call via google-generativeai SDK |
| Validation | None needed - output is advisory text, not decision-making |

**Configuration (environment variables):**
```
GOOGLE_CLOUD_PROJECT=covenant-radar
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
```

---

## 3) Kafka Topics and Event Schemas

### 3.1 Topic Design

| Topic | Direction | Key | Purpose |
|-------|-----------|-----|---------|
| `covenant.measurements.v1` | Input | deal_id | Raw measurement events from generator/source |
| `covenant.predictions.v1` | Output | deal_id | Risk predictions per deal |
| `covenant.alerts.v1` | Output | deal_id | High-severity alerts with Gemini text |

Partitioning by `deal_id` ensures all events for a deal go to the same partition, enabling ordered processing.

### 3.2 Event Schemas (TypedDict with encode/decode)

All schemas follow the existing pattern from `digits_metrics_events.py`:
- TypedDict definition with Literal type discriminator
- `encode_*()` function for serialization
- `decode_*()` function with `require_*` validation
- TypeGuard functions for type narrowing

**Input: MeasurementEventV1**
```python
class MeasurementEventV1(TypedDict):
    """Single measurement event from Kafka."""
    type: Literal["covenant.measurement.v1"]
    event_id: str           # UUID for deduplication
    deal_id: str            # Partition key
    period_start: str       # ISO date (YYYY-MM-DD)
    period_end: str         # ISO date (YYYY-MM-DD)
    metric_name: str        # e.g., "debt_to_equity", "ebitda"
    metric_value: float     # The actual value
    timestamp: str          # ISO datetime when emitted
```

**Output: PredictionEventV1**
```python
class PredictionEventV1(TypedDict):
    """Risk prediction published to Kafka."""
    type: Literal["covenant.prediction.v1"]
    event_id: str
    deal_id: str
    period_start: str
    period_end: str
    evaluation_status: Literal["OK", "BREACH", "WARNING"]
    covenants_evaluated: int
    breaches_count: int
    risk_probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    model_version: str
    evaluation_latency_ms: int
    prediction_latency_ms: int
    processed_at: str
```

**Output: AlertEventV1**
```python
class AlertEventV1(TypedDict):
    """High-severity alert with Gemini-generated text."""
    type: Literal["covenant.alert.v1"]
    event_id: str
    deal_id: str
    alert_type: Literal["breach", "high_risk"]
    severity: Literal["warning", "critical"]
    risk_probability: float
    gemini_summary: str     # 1-sentence human-readable text
    triggered_at: str
```

---

## 4) Internal Observability Events

### 4.1 New Module: `covenant_metrics_events.py`

Following the pattern in `digits_metrics_events.py` and `trainer_metrics_events.py`, we add internal observability events published to Redis for monitoring.

**Location:** `libs/platform_core/src/platform_core/covenant_metrics_events.py`

**Event Types:**
```python
CovenantMetricsEventType = Literal[
    "covenant.measurement.received.v1",
    "covenant.evaluation.completed.v1",
    "covenant.prediction.completed.v1",
    "covenant.alert.triggered.v1",
    "covenant.retrain.triggered.v1",
    "covenant.stream.lag.v1",
]
```

**Event Definitions:**

| Event | Fields | Purpose |
|-------|--------|---------|
| `MeasurementReceivedV1` | deal_id, period, metric_count, latency_ms | Track ingestion rate |
| `EvaluationCompletedV1` | deal_id, status, covenants_evaluated, breaches, latency_ms | Track eval performance |
| `PredictionCompletedV1` | deal_id, risk_probability, risk_tier, model_version, latency_ms | Track inference performance |
| `AlertTriggeredV1` | deal_id, alert_type, severity, message | Track alert volume |
| `RetrainTriggeredV1` | trigger_type, current_auc, threshold_auc, samples_since_train | Track retrain triggers |
| `StreamLagV1` | topic, partition, lag_messages, lag_ms | Track consumer lag |

---

## 5) Synthetic Measurement Generator

### 5.1 Purpose

Generate realistic loan measurement events for demo and testing. The generator produces events that match our domain model and can simulate various risk scenarios.

### 5.2 Scenario Types

```python
ScenarioType = Literal[
    "stable",                 # Metrics stay healthy, no breaches
    "gradual_deterioration",  # Leverage rises 5% per period
    "sudden_shock",           # EBITDA drops 40% at period N
    "recovery",               # Metrics improve after breach
    "seasonal",               # Q4 always 20% worse
    "missing_data",           # Random gaps in reporting
]
```

### 5.3 Generator Configuration

```python
class GeneratorConfig(TypedDict):
    """Configuration for synthetic measurement generator."""
    deal_id: str
    scenario: ScenarioType
    periods: int              # How many periods to generate
    interval_seconds: float   # Time between emissions (0 for batch)
    metrics: tuple[str, ...]  # Which metrics to emit
    base_values: dict[str, float]  # Starting values per metric
    noise_pct: float          # Random noise (0.0-1.0)
```

### 5.4 Entry Point

```bash
# Generate measurements for one deal with gradual deterioration
poetry run python -m scripts.generate_measurements \
    --deal-id "deal-001" \
    --scenario gradual_deterioration \
    --periods 12 \
    --interval 1.0

# Batch generate for demo (no delay)
poetry run python -m scripts.generate_measurements \
    --deal-id "deal-demo" \
    --scenario sudden_shock \
    --periods 8 \
    --interval 0
```

---

## 6) Streaming Inference Worker

### 6.1 Architecture

```
Kafka: covenant.measurements.v1
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STREAM WORKER                                   │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │   Consume    │──▶│  Aggregate   │──▶│  Evaluate + Predict      │ │
│  │  measurement │   │  by deal_id  │   │  (existing code)         │ │
│  └──────────────┘   └──────────────┘   └──────────────────────────┘ │
│                                                    │                │
│                                                    ▼                │
│                                        ┌──────────────────────────┐ │
│                                        │  risk > 0.8?             │ │
│                                        │  ├─ Yes: Call Gemini     │ │
│                                        │  └─ No: Skip             │ │
│                                        └──────────────────────────┘ │
│                                                    │                │
│         ┌──────────────────────────────────────────┤                │
│         ▼                                          ▼                │
│  ┌─────────────────┐                    ┌──────────────────────┐   │
│  │ Produce         │                    │ Produce              │   │
│  │ predictions.v1  │                    │ alerts.v1            │   │
│  └─────────────────┘                    └──────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Emit: covenant_metrics_events (Redis)                        │   │
│  │ Emit: Datadog metrics (DogStatsD)                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Worker Lifecycle

1. **Startup:**
   - Load ML model from disk (cached in memory)
   - Initialize Kafka consumer for `covenant.measurements.v1`
   - Initialize Kafka producer for output topics
   - Initialize Datadog metrics client
   - Initialize Gemini client (lazy, only if needed)

2. **Main Loop:**
   - Poll Kafka for messages (batch of N)
   - Group messages by deal_id
   - For each deal:
     - Aggregate measurements for period
     - Run deterministic covenant evaluation
     - Run ML prediction
     - If risk > 0.8: call Gemini for alert text
     - Produce prediction event
     - Produce alert event (if applicable)
     - Emit internal metrics events
     - Emit Datadog metrics
   - Commit offsets

3. **Shutdown:**
   - Flush producer
   - Commit final offsets
   - Close connections

### 6.3 Entry Point

```bash
poetry run covenant-stream-worker
```

Defined in `pyproject.toml`:
```toml
[tool.poetry.scripts]
covenant-stream-worker = "covenant_radar_api.streaming.worker:main"
```

---

## 7) Retrain Monitoring

### 7.1 Trigger Conditions

| Trigger | Condition | Action |
|---------|-----------|--------|
| **Drift** | Rolling AUC on predictions drops below 0.75 | Emit `RetrainTriggeredV1` event |
| **Data Volume** | 10,000 new labeled samples accumulated | Emit `RetrainTriggeredV1` event |
| **Scheduled** | Weekly (configurable) | Emit `RetrainTriggeredV1` event |

### 7.2 Drift Detection

We track prediction accuracy using delayed ground truth:
- Stream worker makes prediction at time T
- Ground truth (actual default/no-default) arrives at time T+N
- Compare prediction to ground truth
- Compute rolling AUC over last 1000 samples
- If AUC < threshold, trigger retrain

### 7.3 Retrain Flow

```
RetrainTriggeredV1 event
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RETRAIN PIPELINE (existing)                     │
│                                                                     │
│  1. Load accumulated data                                           │
│  2. Run Optuna optimization (existing code)                         │
│  3. Train final model with best hyperparams                         │
│  4. Validate on holdout set                                         │
│  5. Compare to production model                                     │
│  6. If better: deploy new model                                     │
│  7. Emit RetrainCompletedV1 event                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8) Datadog Integration

### 8.1 APM Tracing

ddtrace auto-instruments:
- FastAPI request lifecycle
- httpx outbound calls (Gemini API)
- redis operations
- Kafka producer/consumer (via confluent-kafka patch)

**Setup in `api/main.py`:**
```python
from ..integrations.datadog.tracing import setup_datadog_tracing

def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or settings_from_env()

    # Setup Datadog tracing before anything else
    if cfg["app"].get("datadog_enabled", False):
        setup_datadog_tracing(
            service=cfg["app"].get("datadog_service", "covenant-radar-api"),
            env=cfg["app"].get("datadog_env", "development"),
            version=cfg["app"].get("datadog_version", "0.0.0"),
        )

    # ... rest of app setup
```

### 8.2 Custom Metrics

| Metric | Type | Tags | Purpose |
|--------|------|------|---------|
| `covenant.measurement.received` | counter | deal_id, metric_name | Ingestion volume |
| `covenant.evaluation.latency_ms` | histogram | deal_id, status | Eval performance |
| `covenant.prediction.latency_ms` | histogram | deal_id, risk_tier | Inference performance |
| `covenant.prediction.risk_probability` | gauge | deal_id | Current risk level |
| `covenant.alert.triggered` | counter | deal_id, severity | Alert volume |
| `covenant.stream.lag_messages` | gauge | topic, partition | Consumer lag |
| `covenant.gemini.latency_ms` | histogram | model | LLM call performance |
| `covenant.gemini.tokens` | counter | model, direction | Token usage |

### 8.3 Log Correlation

ddtrace automatically adds `dd.trace_id` and `dd.span_id` to log records. Our existing `JsonFormatter` will include these in structured logs.

---

## 9) File Structure

### 9.1 New Files in `libs/platform_core/`

```
libs/platform_core/src/platform_core/
├── covenant_metrics_events.py    # ~450 lines (TypedDicts + encode/decode)

libs/platform_core/tests/
├── test_covenant_metrics_events.py  # ~300 lines
```

### 9.2 New Files in `services/covenant-radar-api/`

```
services/covenant-radar-api/src/covenant_radar_api/
├── streaming/
│   ├── __init__.py                    # 20 lines (exports)
│   ├── _test_hooks.py                 # 120 lines (DI for producer/consumer/gemini)
│   ├── config.py                      # 100 lines (KafkaConfig, GeneratorConfig TypedDicts)
│   ├── schemas.py                     # 350 lines (Kafka events + encode/decode/TypeGuards)
│   ├── producer.py                    # 200 lines (KafkaProducer wrapper)
│   ├── consumer.py                    # 250 lines (KafkaConsumer wrapper)
│   ├── generator.py                   # 250 lines (synthetic measurement generator)
│   ├── worker.py                      # 350 lines (main consume-predict-produce loop)
│   └── retrain_monitor.py             # 200 lines (drift detection, trigger logic)
│
├── integrations/
│   ├── __init__.py                    # 5 lines
│   ├── datadog/
│   │   ├── __init__.py                # 10 lines
│   │   ├── _test_hooks.py             # 80 lines (fake metrics sink)
│   │   ├── tracing.py                 # 100 lines (ddtrace setup)
│   │   └── metrics.py                 # 150 lines (DogStatsD wrapper)
│   │
│   └── google_ai/
│       ├── __init__.py                # 10 lines
│       ├── _test_hooks.py             # 60 lines (fake Gemini client)
│       ├── client.py                  # 150 lines (Gemini client wrapper)
│       └── schemas.py                 # 80 lines (request/response TypedDicts)

scripts/
├── generate_measurements/
│   ├── __init__.py                    # 10 lines
│   └── __main__.py                    # 80 lines (CLI entry point)
```

### 9.3 New Test Files

```
services/covenant-radar-api/tests/
├── streaming/
│   ├── __init__.py
│   ├── conftest.py                    # 100 lines (shared fixtures)
│   ├── test_schemas.py                # 200 lines
│   ├── test_producer.py               # 180 lines
│   ├── test_consumer.py               # 220 lines
│   ├── test_generator.py              # 200 lines
│   ├── test_worker.py                 # 350 lines
│   └── test_retrain_monitor.py        # 200 lines
│
├── integrations/
│   ├── datadog/
│   │   ├── test_tracing.py            # 100 lines
│   │   └── test_metrics.py            # 120 lines
│   │
│   └── google_ai/
│       ├── test_client.py             # 150 lines
│       └── test_schemas.py            # 80 lines
```

---

## 10) Dependency Additions

### 10.1 pyproject.toml Changes

```toml
[tool.poetry.dependencies]
# Existing dependencies...

# Confluent Cloud (Kafka)
confluent-kafka = "^2.6"

# Datadog
ddtrace = "^2.14"

# Google AI (Gemini)
google-generativeai = "^0.8"

[tool.poetry.scripts]
# Existing scripts...
covenant-stream-worker = "covenant_radar_api.streaming.worker:main"
```

### 10.2 Config TypedDict Additions

Add to `libs/platform_core/src/platform_core/config/covenant_radar.py`:

```python
class ConfluentConfig(TypedDict, total=False):
    """Confluent Cloud configuration."""
    bootstrap_servers: str
    api_key: str
    api_secret: str
    schema_registry_url: str
    schema_registry_api_key: str
    schema_registry_api_secret: str

class DatadogConfig(TypedDict, total=False):
    """Datadog configuration."""
    enabled: bool
    service: str
    env: str
    version: str
    agent_host: str
    dogstatsd_port: int
    trace_enabled: bool

class GeminiConfig(TypedDict, total=False):
    """Google Gemini configuration."""
    api_key: str
    model: str
    project: str

class CovenantRadarAppConfig(TypedDict, total=False):
    # Existing fields...
    data_root: str
    models_root: str
    ml_backend: str
    active_model_path_xgboost: str
    active_model_path_mlp: str
    active_model_path_lstm: str
    active_model_path_lightgbm: str

    # New fields
    confluent: ConfluentConfig
    datadog: DatadogConfig
    gemini: GeminiConfig
```

---

## 11) Implementation Order

### Phase 1: Foundation ✅ COMPLETE
1. Add `covenant_metrics_events.py` to platform_core ✅
2. Add config TypedDicts for Confluent, Datadog, Gemini ✅ (Datadog done)
3. Add dependencies to pyproject.toml ✅
4. Run `make check` to verify no regressions ✅

### Phase 2: Datadog Integration ✅ COMPLETE
1. Implement `integrations/datadog/tracing.py` ✅
2. Implement `integrations/datadog/metrics.py` ✅
3. Wire into `api/main.py` ✅ (tracing wired, metrics for Phase 4)
4. Add tests ✅ (100% coverage)
5. Run `make check` ✅

### Phase 3: Kafka Infrastructure (2-3 days)
1. Implement `streaming/config.py`
2. Implement `streaming/schemas.py` (Kafka events)
3. Implement `streaming/producer.py`
4. Implement `streaming/consumer.py`
5. Add tests with fake producer/consumer
6. Run `make check`

### Phase 4: Stream Worker (2-3 days)
1. Implement `streaming/worker.py`
2. Wire existing evaluation + prediction code
3. Implement metric emission (internal + Datadog)
4. Add tests
5. Run `make check`

### Phase 5: Gemini Integration (1 day)
1. Implement `integrations/google_ai/client.py`
2. Wire into stream worker for high-risk alerts
3. Add tests with fake Gemini client
4. Run `make check`

### Phase 6: Generator + Demo (1-2 days)
1. Implement `streaming/generator.py`
2. Implement `scripts/generate_measurements/`
3. Implement `streaming/retrain_monitor.py`
4. Create demo scenario
5. Run `make check`

### Phase 7: Documentation + Submission (1 day)
1. Update README with streaming architecture
2. Update docs/api.md with new endpoints
3. Create demo video
4. Prepare Devpost submission

**Total: ~10-14 days**

---

## 12) Code Standards Checklist

Every new file must satisfy:

- [ ] No `Any` type annotations
- [ ] No `cast()` calls
- [ ] No `type: ignore` comments
- [ ] No `.pyi` stub files
- [ ] No `# noqa` comments
- [ ] TypedDict for all structured data (no dataclasses)
- [ ] `encode_*()` function for each TypedDict
- [ ] `decode_*()` function with `require_*` validation
- [ ] TypeGuard functions for type narrowing
- [ ] `_test_hooks.py` for DI in service modules
- [ ] `testing.py` for public test utilities in lib modules
- [ ] Production code sets hooks to real implementations at startup
- [ ] Tests set hooks to fakes (no mocks)
- [ ] Google-style docstrings with Args, Returns, Raises
- [ ] 100% statement coverage
- [ ] 100% branch coverage
- [ ] No try/except for "best effort" recovery
- [ ] No fallback logic
- [ ] No backwards compatibility shims
- [ ] Explicit error propagation

---

## 13) Acceptance Criteria

### Confluent Track
- [ ] Synthetic generator produces measurement events to Kafka
- [ ] Stream worker consumes and processes events
- [ ] Predictions published to output topic
- [ ] Alerts published when risk > 0.8
- [ ] Consumer lag < 1 second under normal load

### Datadog Track
- [ ] APM traces visible in Datadog UI
- [ ] Custom metrics visible in Datadog UI
- [ ] Logs correlated with trace IDs
- [ ] Dashboard showing prediction latency, risk distribution, alert volume

### Google Cloud AI
- [ ] Gemini client calls work in stream worker
- [ ] Alert events include Gemini-generated summary text
- [ ] Gemini latency and token metrics tracked

### Quality Gates
- [ ] `make check` passes
- [ ] 100% test coverage
- [ ] All strict typing requirements met
- [ ] No regressions in existing functionality

---

## 14) Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Confluent Cloud rate limits | Use batch producing, respect quotas |
| Gemini API latency | Async calls, timeout handling, only for high-risk |
| Datadog agent unavailable | Graceful skip if DD_AGENT_HOST unreachable |
| Model loading slow | Load once at startup, cache in memory |
| Consumer lag buildup | Auto-scaling, batch processing, alert on lag |

---

## 15) Out of Scope

The following are explicitly NOT part of this implementation:

- Online/continuous learning
- Real financial data sources
- Production deployment automation
- Multi-region failover
- Schema evolution (v2 events)
- Avro serialization (JSON only)
- Kafka Streams / ksqlDB
- Confluent connectors

---

### End of Plan
