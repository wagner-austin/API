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
    risk_tier: RiskTier  # Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] from covenant_domain
    model_version: str
    evaluation_latency_ms: int
    prediction_latency_ms: int
    processed_at: str
```

**Risk Tier Thresholds** (defined in `covenant_domain.features`):
| Tier | Probability | Description |
|------|-------------|-------------|
| LOW | < 0.25 | Normal risk |
| MEDIUM | 0.25 - 0.50 | Elevated, monitor |
| HIGH | 0.50 - 0.80 | Review required |
| CRITICAL | >= 0.80 | Immediate action, triggers alert |

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
│   ├── __init__.py                    # ~20 lines (exports)
│   ├── _test_hooks.py                 # ~180 lines (DI for producer/consumer, Protocols, Real/Fake Kafka implementations)
│   ├── _test_hooks_model.py           # ~30 lines (FakePredictor, FakeMetricsSink)
│   ├── _test_hooks_repositories.py    # ~100 lines (FakeDealRepository, FakeCovenantRepository, etc.)
│   ├── config.py                      # ~80 lines (KafkaConfig TypedDicts, env parsing)
│   ├── schemas.py                     # ~160 lines (Kafka events + encode/decode/TypeGuards)
│   ├── producer.py                    # ~35 lines (StreamingProducer wrapper)
│   ├── consumer.py                    # ~60 lines (StreamingConsumer wrapper)
│   ├── worker.py                      # ~245 lines (StreamingWorker consume-predict-produce loop)
│   ├── generator.py                   # (Phase 6) synthetic measurement generator
│   └── retrain_monitor.py             # (Phase 6) drift detection, trigger logic
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
│   ├── _test_worker_fixtures.py       # ~200 lines (shared fixtures: REQUIRED_METRICS, make_*, factories)
│   ├── test_schemas.py                # ~160 lines (encode/decode, TypeGuards, make_* functions)
│   ├── test_hooks.py                  # ~200 lines (Real/Fake Kafka implementations, hook switching)
│   ├── test_producer.py               # ~50 lines (StreamingProducer with FakeKafkaProducer)
│   ├── test_consumer.py               # ~100 lines (StreamingConsumer with FakeKafkaConsumer)
│   ├── test_worker_helpers.py         # ~200 lines (helper function tests: buffer key, status, etc.)
│   ├── test_worker_fakes.py           # ~250 lines (FakeDealRepository, FakePredictor, etc.)
│   ├── test_worker_core.py            # ~500 lines (StreamingWorker init, buffer, processing, run, edge cases)
│   ├── test_generator.py              # (Phase 6) synthetic measurement generator tests
│   └── test_retrain_monitor.py        # (Phase 6) drift detection tests
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
confluent-kafka = "^2.12"

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

### Phase 3: Kafka Infrastructure ✅ COMPLETE
1. Implement `streaming/config.py` ✅
   - ConfluentConfig, ConsumerConfig, ProducerConfig, KafkaTopicsConfig, StreamingConfig TypedDicts
   - ConfluentSchemaRegistryConfig for optional schema registry
   - load_streaming_config() with environment variable parsing
   - Parsing functions for Literal types (_parse_auto_offset_reset, _parse_acks, _parse_compression_type)
2. Implement `streaming/schemas.py` (Kafka events) ✅
   - MeasurementEventV1, PredictionEventV1, AlertEventV1 TypedDicts with Literal type discriminators
   - make_* factory functions for event creation
   - encode_* functions for JSON serialization
   - decode_* functions with require_* validation (from platform_core.json_utils)
   - TypeGuard functions (is_measurement_event, is_prediction_event, is_alert_event)
   - RiskTier and classify_risk_tier() imported from covenant_domain.features (single source of truth)
3. Implement `streaming/_test_hooks.py` ✅
   - Protocol definitions: KafkaProducerProtocol, KafkaConsumerProtocol, ConsumedMessageProtocol
   - Raw Protocol definitions: RawKafkaProducerProtocol, RawKafkaConsumerProtocol, RawKafkaMessageProtocol
   - RealKafkaProducer, RealKafkaConsumer, RealConsumedMessage (real implementations via confluent-kafka)
   - FakeKafkaProducer, FakeKafkaConsumer, FakeConsumedMessage (test fakes)
   - use_fake_kafka(), use_real_kafka() hook switching functions
   - Dynamic import pattern with _get_confluent_kafka() helper to avoid mypy untyped module errors
4. Implement `streaming/producer.py` ✅
   - StreamingProducer class wrapping KafkaProducerProtocol
   - produce_prediction(), produce_alert(), produce_event() methods
   - create_streaming_producer(), create_producer_from_parts() factory functions
5. Implement `streaming/consumer.py` ✅
   - StreamingConsumer class wrapping KafkaConsumerProtocol
   - ConsumedMeasurement TypedDict with event data and metadata (topic, partition, offset, key)
   - subscribe(), poll(), poll_batch(), commit(), close() methods
   - create_streaming_consumer(), create_consumer_from_parts() factory functions
6. Add `streaming/__init__.py` with exports ✅
7. Add tests with fake producer/consumer ✅
   - tests/streaming/test_config.py (config parsing, defaults, custom values)
   - tests/streaming/test_schemas.py (encode/decode, TypeGuards, make_* functions)
   - tests/streaming/test_hooks.py (fake implementations, real implementations, hook switching)
   - tests/streaming/test_producer.py (StreamingProducer with FakeKafkaProducer)
   - tests/streaming/test_consumer.py (StreamingConsumer with FakeKafkaConsumer)
8. Run `make check` ✅ (100% statement and branch coverage)

### Phase 4: Stream Worker ✅ COMPLETE
1. Implement `streaming/worker.py` ✅
   - StreamingWorker class with consume-evaluate-predict-produce loop
   - WorkerConfig TypedDict for configuration
   - Buffer management for aggregating measurements by period
   - Integration with existing evaluation + ML prediction code
   - Helper functions: _make_buffer_key, _determine_evaluation_status, _count_breaches, etc.
2. Wire existing evaluation + prediction code ✅
   - Uses covenant_domain.rules.evaluate_all_covenants_for_period
   - Uses covenant_domain.features.extract_features
   - PredictorProtocol for ML model (XGBoost/MLP/LSTM compatible)
3. Implement metric emission (internal + Datadog) ✅
   - MetricsClient integration for latency histograms and gauges
   - Risk probability tracking
4. Add modularized test infrastructure ✅
   - `streaming/_test_hooks_repositories.py` - Fake repository implementations
   - `streaming/_test_hooks_model.py` - FakePredictor and FakeMetricsSink
   - `tests/streaming/_test_worker_fixtures.py` - Shared test fixtures and factories
   - `tests/streaming/test_worker_helpers.py` - Helper function tests
   - `tests/streaming/test_worker_fakes.py` - Fake implementation tests
   - `tests/streaming/test_worker_core.py` - Core StreamingWorker tests
5. Run `make check` ✅ (100% statement and branch coverage, 1639 tests passing)

### Phase 5: Gemini Integration ✅ COMPLETE
1. Implement `integrations/google_ai/client.py` ✅
2. Implement `integrations/google_ai/schemas.py` ✅
3. Implement `integrations/google_ai/_test_hooks.py` ✅
4. Add tests with fake Gemini client ✅
5. Add documentation (`docs/integrations/google_ai.md`) ✅
6. Run `make check` ✅ (100% coverage)

### Phase 6: Data Replay + Entry Point (remaining)
1. Implement `scripts/replay_data/` (data replay script)
2. Implement `streaming/main.py` (worker entry point)
3. Add tests
4. Run `make check`

### Phase 7: Dashboard + Demo (remaining)
1. Implement web UI dashboard (`static/index.html`)
2. Add API endpoints for dashboard (`api/routes/streaming.py`)
3. Create demo scenario
4. Record demo video
5. Prepare Devpost submission

**Status: Core infrastructure complete, remaining work is demo/deployment**

---

## 12) Code Standards Checklist

Every new file must satisfy:

- [x] No `Any` type annotations
- [x] No `cast()` calls
- [x] No `type: ignore` comments
- [x] No `.pyi` stub files
- [x] No `# noqa` comments
- [x] TypedDict for all structured data (no dataclasses)
- [x] `encode_*()` function for each TypedDict
- [x] `decode_*()` function with `require_*` validation
- [x] TypeGuard functions for type narrowing
- [x] `_test_hooks.py` for DI in service modules
- [x] `testing.py` for public test utilities in lib modules
- [x] Production code sets hooks to real implementations at startup
- [x] Tests set hooks to fakes (no mocks)
- [x] Google-style docstrings with Args, Returns, Raises
- [x] 100% statement coverage
- [x] 100% branch coverage
- [x] No try/except for "best effort" recovery
- [x] No fallback logic
- [x] No backwards compatibility shims
- [x] Explicit error propagation

---

## 13) Acceptance Criteria

### Confluent Track
- [ ] Synthetic generator produces measurement events to Kafka (needs replay script)
- [x] Stream worker consumes and processes events (StreamingWorker complete)
- [x] Predictions published to output topic (PredictionEventV1 schema + producer)
- [x] Alerts published when risk > 0.8 (AlertEventV1 schema + producer)
- [ ] Consumer lag < 1 second under normal load (needs live testing)

### Datadog Track
- [x] APM traces visible in Datadog UI (tracing.py complete)
- [x] Custom metrics visible in Datadog UI (metrics.py complete)
- [x] Logs correlated with trace IDs (tracing integration)
- [ ] Dashboard showing prediction latency, risk distribution, alert volume (needs deployment)

### Google Cloud AI
- [x] Gemini client calls work in stream worker (GeminiClient complete)
- [x] Alert events include Gemini-generated summary text (generate_alert_summary())
- [x] Gemini latency and token metrics tracked (GenerateAlertResponse includes latency_ms, tokens)

### Quality Gates
- [x] `make check` passes
- [x] 100% test coverage
- [x] All strict typing requirements met
- [x] No regressions in existing functionality

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
