# Datadog Integration Package

APM tracing and custom metrics integration for observability.

## Overview

This package provides:

- **APM Tracing**: Distributed tracing via ddtrace auto-instrumentation
- **Custom Metrics**: Application metrics via DogStatsD
- **Log Correlation**: Automatic trace ID injection into structured logs

## Package Structure

```
integrations/datadog/
├── __init__.py       # Package exports (MetricsClient, MetricsConfig, etc.)
├── _test_hooks.py    # Dependency injection hooks for testing
├── metrics.py        # DogStatsD metrics client and types
├── tracing.py        # APM tracing setup
└── README.md         # This file
```

## Module Responsibilities

### metrics.py

DogStatsD metrics client with typed configuration:

| Type | Description |
|------|-------------|
| `MetricsConfig` | TypedDict for client configuration (host, port, namespace) |
| `MetricsSinkProtocol` | Protocol for metrics emission (increment, gauge, histogram) |
| `MetricsClient` | Production client wrapping DogStatsd |
| `create_metrics_client()` | Factory function for creating metrics clients |

**Custom Metrics:**

| Metric | Type | Tags | Description |
|--------|------|------|-------------|
| `covenant.measurement.received` | counter | `deal_id`, `metric_name` | Measurement ingestion count |
| `covenant.evaluation.latency_ms` | histogram | `deal_id`, `status` | Evaluation time |
| `covenant.prediction.latency_ms` | histogram | `deal_id`, `risk_tier` | Prediction time |
| `covenant.prediction.risk_probability` | gauge | `deal_id` | Current risk level |
| `covenant.alert.triggered` | counter | `deal_id`, `severity`, `alert_type` | Alert volume |
| `covenant.stream.lag_messages` | gauge | `topic`, `partition` | Consumer lag |
| `covenant.gemini.latency_ms` | histogram | `model` | Gemini API call time |
| `covenant.gemini.tokens` | counter | `model`, `direction` | Token usage |

### tracing.py

APM tracing setup and state management:

| Function | Description |
|----------|-------------|
| `setup_datadog_tracing()` | Configure ddtrace with service, env, version |
| `reset_tracing_state()` | Reset tracing state (for testing) |

**Auto-instrumented libraries:**

| Library | What's Traced |
|---------|---------------|
| FastAPI | Request lifecycle, route handlers |
| httpx | Outbound HTTP calls |
| redis | Redis operations |
| logging | Log correlation with trace IDs |

### _test_hooks.py

Dependency injection for isolated testing:

| Hook | Production Implementation | Description |
|------|--------------------------|-------------|
| `metrics_sink_factory` | Creates `RealMetricsSink` | Factory for DogStatsD sinks |
| `tracing_setup` | Calls `ddtrace.patch()` | Sets up APM tracing |

## Usage

### Metrics Client

```python
from covenant_radar_api.integrations.datadog import (
    MetricsClient,
    MetricsConfig,
    create_metrics_client,
)

config: MetricsConfig = {
    "host": "localhost",
    "port": 8125,
    "namespace": "covenant",
}
client = create_metrics_client(config)

# Emit metrics
client.increment_measurement_received("deal-123", "debt_to_equity")
client.record_evaluation_latency("deal-123", "OK", 15.2)
client.record_prediction_latency("deal-123", "HIGH", 45.2)
client.set_prediction_risk("deal-123", 0.85)
client.increment_alert_triggered("deal-123", "critical", "high_risk")
```

### Tracing Setup

Tracing is configured automatically at application startup in `api/main.py`:

```python
from covenant_radar_api.integrations.datadog.tracing import setup_datadog_tracing

datadog_cfg = settings["datadog"]
if datadog_cfg["enabled"] and datadog_cfg["trace_enabled"]:
    setup_datadog_tracing(
        service=datadog_cfg["service"],
        env=datadog_cfg["env"],
        version=datadog_cfg["version"],
    )
```

## Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATADOG__ENABLED` | `false` | Enable Datadog integration |
| `DATADOG__SERVICE` | `covenant-radar-api` | Service name |
| `DATADOG__ENV` | `dev` | Environment (`dev`, `staging`, `production`) |
| `DATADOG__VERSION` | `0.0.0` | Service version |
| `DATADOG__AGENT_HOST` | `localhost` | Datadog agent host |
| `DATADOG__DOGSTATSD_PORT` | `8125` | DogStatsD UDP port |
| `DATADOG__TRACE_ENABLED` | `true` | Enable APM tracing |

## Testing

Tests use dependency injection via `_test_hooks.py`:

### Fake Metrics Sink

```python
from covenant_radar_api.integrations.datadog import _test_hooks


class FakeMetricsSink:
    def __init__(self, host: str, port: int, namespace: str) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    def increment(self, name: str, tags: list[str]) -> None:
        self.calls.append(("increment", name, tags))


def fake_factory(host: str, port: int, namespace: str) -> FakeMetricsSink:
    return FakeMetricsSink(host, port, namespace)


_test_hooks.metrics_sink_factory = fake_factory
```

### Fake Tracing Setup

```python
def fake_tracing_setup(service: str, env: str, version: str) -> bool:
    return True  # Pretend tracing is configured


_test_hooks.tracing_setup = fake_tracing_setup
```

## Type Safety

All modules follow strict typing:

- No `Any` types
- No `cast()` calls
- No `type: ignore` comments
- TypedDict for structured data
- Protocol types for dependency injection
