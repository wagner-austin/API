# Datadog Integration

APM tracing and custom metrics for observability.

---

## Overview

The Datadog integration provides:

- **APM Tracing**: Distributed tracing via ddtrace auto-instrumentation
- **Custom Metrics**: Application metrics via DogStatsD
- **Log Correlation**: Automatic trace ID injection into structured logs

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATADOG__ENABLED` | `false` | Enable Datadog integration |
| `DATADOG__SERVICE` | `covenant-radar-api` | Service name for traces and metrics |
| `DATADOG__ENV` | `dev` | Environment (`dev`, `staging`, `production`) |
| `DATADOG__VERSION` | `0.0.0` | Service version for trace filtering |
| `DATADOG__AGENT_HOST` | `localhost` | Datadog agent host |
| `DATADOG__DOGSTATSD_PORT` | `8125` | DogStatsD UDP port |
| `DATADOG__TRACE_ENABLED` | `true` | Enable APM tracing (when enabled=true) |

### Example Configuration

**Local development (disabled):**
```bash
export DATADOG__ENABLED=false
```

**Production deployment:**
```bash
export DATADOG__ENABLED=true
export DATADOG__SERVICE=covenant-radar-api
export DATADOG__ENV=production
export DATADOG__VERSION=1.0.0
export DATADOG__AGENT_HOST=datadog-agent
export DATADOG__DOGSTATSD_PORT=8125
export DATADOG__TRACE_ENABLED=true
```

---

## APM Tracing

When enabled, ddtrace auto-instruments:

| Library | What's Traced |
|---------|---------------|
| FastAPI | Request lifecycle, route handlers |
| httpx | Outbound HTTP calls |
| redis | Redis operations |
| logging | Log correlation with trace IDs |

### Setup

Tracing is configured automatically at application startup when `DATADOG__ENABLED=true` and `DATADOG__TRACE_ENABLED=true`.

```python
# In api/main.py (already wired)
from ..integrations.datadog.tracing import setup_datadog_tracing

if datadog_cfg["enabled"] and datadog_cfg["trace_enabled"]:
    setup_datadog_tracing(
        service=datadog_cfg["service"],
        env=datadog_cfg["env"],
        version=datadog_cfg["version"],
    )
```

### Trace Tags

All traces include:

| Tag | Description |
|-----|-------------|
| `service` | Service name |
| `env` | Environment name |
| `version` | Service version |
| `request_id` | Request correlation ID |

---

## Custom Metrics

Application metrics emitted via DogStatsD.

### Metric Types

| Type | Description | Example |
|------|-------------|---------|
| Counter | Cumulative count | Measurements received |
| Gauge | Current value | Consumer lag |
| Histogram | Distribution | Prediction latency |

### Available Metrics

| Metric | Type | Tags | Description |
|--------|------|------|-------------|
| `covenant.measurement.received` | counter | `deal_id`, `metric_name` | Measurement ingestion count |
| `covenant.evaluation.latency_ms` | histogram | `deal_id`, `status` | Covenant evaluation time |
| `covenant.prediction.latency_ms` | histogram | `deal_id`, `risk_tier` | ML prediction time |
| `covenant.prediction.risk_probability` | gauge | `deal_id` | Current risk level per deal |
| `covenant.alert.triggered` | counter | `deal_id`, `severity`, `alert_type` | Alert volume |
| `covenant.stream.lag_messages` | gauge | `topic`, `partition` | Kafka consumer lag |
| `covenant.gemini.latency_ms` | histogram | `model` | Gemini API call time |
| `covenant.gemini.tokens` | counter | `model`, `direction` | Token usage (input/output) |

### Usage

```python
from covenant_radar_api.integrations.datadog import (
    MetricsClient,
    MetricsConfig,
    create_metrics_client,
)

# Create client
config: MetricsConfig = {
    "host": "localhost",
    "port": 8125,
    "namespace": "covenant",
}
client = create_metrics_client(config)

# Emit metrics
client.increment_measurement_received("deal-123", "debt_to_equity")
client.record_prediction_latency("deal-123", "HIGH", 45.2)
client.set_prediction_risk("deal-123", 0.85)
client.increment_alert_triggered("deal-123", "critical", "high_risk")
```

---

## Log Correlation

When tracing is enabled, ddtrace automatically injects trace context into logs:

| Log Field | Description |
|-----------|-------------|
| `dd.trace_id` | Trace ID for correlation |
| `dd.span_id` | Current span ID |

### Example Log Output

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Prediction completed",
  "service": "covenant-radar-api",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "dd.trace_id": "1234567890123456789",
  "dd.span_id": "9876543210987654321"
}
```

This enables clicking from a log entry directly to the associated trace in Datadog.

---

## Testing

The integration uses dependency injection for testability.

### Fake Metrics Sink

Tests override the metrics sink factory to capture metrics without sending UDP packets:

```python
from covenant_radar_api.integrations.datadog import _test_hooks

# In test setup
def fake_factory(host: str, port: int, namespace: str) -> FakeMetricsSink:
    return FakeMetricsSink(host, port, namespace)

_test_hooks.metrics_sink_factory = fake_factory
```

### Fake Tracing Setup

Tests override the tracing setup to skip ddtrace configuration:

```python
def fake_tracing_setup(service: str, env: str, version: str) -> bool:
    return True  # Pretend tracing is configured

_test_hooks.tracing_setup = fake_tracing_setup
```

---

## Deployment

### Docker Compose

```yaml
services:
  covenant-radar-api:
    environment:
      DATADOG__ENABLED: "true"
      DATADOG__SERVICE: "covenant-radar-api"
      DATADOG__ENV: "production"
      DATADOG__VERSION: "1.0.0"
      DATADOG__AGENT_HOST: "datadog-agent"
    depends_on:
      - datadog-agent

  datadog-agent:
    image: datadog/agent:latest
    environment:
      DD_API_KEY: "${DD_API_KEY}"
      DD_SITE: "datadoghq.com"
      DD_APM_ENABLED: "true"
      DD_DOGSTATSD_NON_LOCAL_TRAFFIC: "true"
    ports:
      - "8125:8125/udp"  # DogStatsD
      - "8126:8126"      # APM traces
```

### Kubernetes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: covenant-radar-api
spec:
  containers:
    - name: api
      env:
        - name: DATADOG__ENABLED
          value: "true"
        - name: DATADOG__SERVICE
          value: "covenant-radar-api"
        - name: DATADOG__ENV
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['env']
        - name: DATADOG__VERSION
          value: "1.0.0"
        - name: DATADOG__AGENT_HOST
          valueFrom:
            fieldRef:
              fieldPath: status.hostIP
```

---

## Datadog Dashboard

Recommended widgets for the covenant-radar-api dashboard:

### Request Performance
- **Latency P50/P95/P99**: `avg:trace.fastapi.request.duration{service:covenant-radar-api}`
- **Request Rate**: `sum:trace.fastapi.request.hits{service:covenant-radar-api}.as_rate()`
- **Error Rate**: `sum:trace.fastapi.request.errors{service:covenant-radar-api}.as_rate()`

### ML Performance
- **Prediction Latency**: `avg:covenant.prediction.latency_ms{*}`
- **Evaluation Latency**: `avg:covenant.evaluation.latency_ms{*}`
- **Risk Distribution**: `avg:covenant.prediction.risk_probability{*} by {deal_id}`

### Alert Volume
- **Alerts by Severity**: `sum:covenant.alert.triggered{*} by {severity}.as_count()`
- **Alerts by Type**: `sum:covenant.alert.triggered{*} by {alert_type}.as_count()`

### Streaming Health (when Kafka integration is added)
- **Consumer Lag**: `max:covenant.stream.lag_messages{*} by {topic,partition}`
- **Measurement Ingestion**: `sum:covenant.measurement.received{*}.as_rate()`

---

## Troubleshooting

### No Traces Appearing

1. Verify agent connectivity:
   ```bash
   curl -v http://${DATADOG__AGENT_HOST}:8126/info
   ```

2. Check environment variables are set:
   ```bash
   env | grep DATADOG
   ```

3. Verify `DATADOG__ENABLED=true` and `DATADOG__TRACE_ENABLED=true`

### No Metrics Appearing

1. Verify DogStatsD port is open:
   ```bash
   nc -vzu ${DATADOG__AGENT_HOST} 8125
   ```

2. Check agent DogStatsD config:
   ```bash
   docker exec datadog-agent agent status | grep DogStatsD
   ```

### Log Correlation Not Working

1. Ensure tracing is enabled before logging is configured
2. Check logs for `dd.trace_id` field presence
3. Verify log format is JSON (not plaintext)
