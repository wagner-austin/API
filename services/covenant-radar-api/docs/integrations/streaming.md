# Kafka Streaming Integration

Real-time inference pipeline via Confluent Cloud Kafka.

---

## Overview

The streaming integration provides:

- **Measurement Consumption**: Receive financial measurements from `covenant.measurements.v1`
- **Prediction Publishing**: Publish breach risk predictions to `covenant.predictions.v1`
- **Alert Publishing**: Publish covenant alerts to `covenant.alerts.v1`
- **TypedDict Schemas**: Type-safe event structures with encode/decode functions

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMING__ENABLED` | `false` | Enable Kafka streaming |
| `CONFLUENT__BOOTSTRAP_SERVERS` | - | Confluent Cloud bootstrap servers |
| `CONFLUENT__API_KEY` | - | Confluent Cloud API key |
| `CONFLUENT__API_SECRET` | - | Confluent Cloud API secret |
| `CONFLUENT__SCHEMA_REGISTRY_URL` | - | Schema Registry URL (optional) |
| `CONFLUENT__SCHEMA_REGISTRY_API_KEY` | - | Schema Registry API key |
| `CONFLUENT__SCHEMA_REGISTRY_API_SECRET` | - | Schema Registry API secret |
| `KAFKA__TOPIC_MEASUREMENTS` | `covenant.measurements.v1` | Input topic |
| `KAFKA__TOPIC_PREDICTIONS` | `covenant.predictions.v1` | Predictions output topic |
| `KAFKA__TOPIC_ALERTS` | `covenant.alerts.v1` | Alerts output topic |
| `KAFKA__CONSUMER_GROUP_ID` | `covenant-radar-api` | Consumer group ID |
| `KAFKA__AUTO_OFFSET_RESET` | `earliest` | Offset reset policy |
| `KAFKA__ENABLE_AUTO_COMMIT` | `false` | Auto-commit offsets |
| `KAFKA__FETCH_MIN_BYTES` | `1` | Minimum fetch bytes |
| `KAFKA__SESSION_TIMEOUT_MS` | `45000` | Session timeout |
| `KAFKA__HEARTBEAT_INTERVAL_MS` | `15000` | Heartbeat interval |
| `KAFKA__PRODUCER_ACKS` | `all` | Producer acknowledgment |
| `KAFKA__PRODUCER_RETRIES` | `3` | Producer retries |
| `KAFKA__PRODUCER_LINGER_MS` | `5` | Producer linger time |
| `KAFKA__PRODUCER_BATCH_SIZE` | `16384` | Producer batch size |
| `KAFKA__COMPRESSION_TYPE` | `gzip` | Compression type |

### Example Configuration

**Local development (disabled):**
```bash
export STREAMING__ENABLED=false
```

**Production deployment:**
```bash
export STREAMING__ENABLED=true
export CONFLUENT__BOOTSTRAP_SERVERS=pkc-xxxxx.us-east-1.aws.confluent.cloud:9092
export CONFLUENT__API_KEY=your-api-key
export CONFLUENT__API_SECRET=your-api-secret
export KAFKA__CONSUMER_GROUP_ID=covenant-radar-api-prod
export KAFKA__AUTO_OFFSET_RESET=earliest
export KAFKA__ENABLE_AUTO_COMMIT=false
```

---

## Topics

### covenant.measurements.v1 (Input)

Financial measurements for covenant evaluation.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"covenant.measurement.v1"` (discriminator) |
| `event_id` | string | UUID for deduplication |
| `deal_id` | string | Deal identifier (partition key) |
| `period_start` | string | ISO date (YYYY-MM-DD) for period start |
| `period_end` | string | ISO date (YYYY-MM-DD) for period end |
| `metric_name` | string | Metric identifier (e.g., "debt_to_equity") |
| `metric_value` | float | Numeric value of the metric |
| `timestamp` | string | ISO datetime when event was emitted |

### covenant.predictions.v1 (Output)

Breach risk predictions from ML models with covenant evaluation results.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"covenant.prediction.v1"` (discriminator) |
| `event_id` | string | UUID for this event |
| `deal_id` | string | Deal identifier (partition key) |
| `period_start` | string | ISO date (YYYY-MM-DD) for period start |
| `period_end` | string | ISO date (YYYY-MM-DD) for period end |
| `evaluation_status` | string | Deterministic result: `OK`, `BREACH`, `WARNING` |
| `covenants_evaluated` | int | Number of covenants checked |
| `breaches_count` | int | Number of covenant breaches detected |
| `risk_probability` | float | ML-predicted probability (0.0-1.0) |
| `risk_tier` | string | `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` |
| `model_version` | string | Version string of ML model used |
| `evaluation_latency_ms` | int | Time spent on deterministic evaluation |
| `prediction_latency_ms` | int | Time spent on ML inference |
| `processed_at` | string | ISO datetime when processing completed |

### covenant.alerts.v1 (Output)

High-severity alerts with Gemini-generated summaries.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"covenant.alert.v1"` (discriminator) |
| `event_id` | string | UUID for this event |
| `deal_id` | string | Deal identifier (partition key) |
| `alert_type` | string | `breach` or `high_risk` |
| `severity` | string | `warning` or `critical` |
| `risk_probability` | float | ML-predicted probability that triggered alert |
| `gemini_summary` | string | Human-readable 1-sentence summary from Gemini |
| `triggered_at` | string | ISO datetime when alert was triggered |

---

## Event Schemas

### TypedDict Definitions

All events use TypedDict for type safety with V1 suffix:

```python
from covenant_radar_api.streaming.schemas import (
    MeasurementEventV1,
    PredictionEventV1,
    AlertEventV1,
    make_measurement_event,
)

# Type-safe event creation using factory function
measurement: MeasurementEventV1 = make_measurement_event(
    event_id="evt-001",
    deal_id="deal-123",
    period_start="2024-01-01",
    period_end="2024-03-31",
    metric_name="debt_to_equity",
    metric_value=1.5,
    timestamp="2024-04-01T09:00:00Z",
)
```

### Encode/Decode Functions

```python
from covenant_radar_api.streaming.schemas import (
    encode_measurement_event,
    decode_measurement_event,
    encode_prediction_event,
    decode_prediction_event,
    encode_alert_event,
    decode_alert_event,
)

# Encode to JSON string
json_str = encode_measurement_event(measurement)

# Decode from JSON string (validates type discriminator)
parsed = decode_measurement_event(json_str)
```

### TypeGuard Validation

```python
from covenant_radar_api.streaming.schemas import (
    KafkaEventV1,
    decode_kafka_event,
    is_measurement_event,
    is_prediction_event,
    is_alert_event,
)

# Decode any event type
event: KafkaEventV1 = decode_kafka_event(raw_json)

if is_measurement_event(event):
    # event is narrowed to MeasurementEventV1
    process_measurement(event)
elif is_prediction_event(event):
    # event is narrowed to PredictionEventV1
    process_prediction(event)
```

---

## Producer Usage

### Create Producer

```python
from covenant_radar_api.streaming.producer import (
    create_streaming_producer,
    create_producer_from_parts,
)
from covenant_radar_api.streaming.config import load_streaming_config

# From full config
config = load_streaming_config()
producer = create_streaming_producer(config)

# From individual parts
producer = create_producer_from_parts(
    confluent_config=confluent_cfg,
    producer_config=producer_cfg,
    predictions_topic="covenant.predictions.v1",
    alerts_topic="covenant.alerts.v1",
)
```

### Publish Events

```python
from covenant_radar_api.streaming.schemas import (
    make_prediction_event,
    make_alert_event,
)

# Publish prediction (with full evaluation results)
producer.produce_prediction(
    make_prediction_event(
        event_id="pred-001",
        deal_id="deal-123",
        period_start="2024-01-01",
        period_end="2024-03-31",
        evaluation_status="OK",
        covenants_evaluated=3,
        breaches_count=0,
        risk_probability=0.15,
        risk_tier="LOW",
        model_version="model-2024-01",
        evaluation_latency_ms=12,
        prediction_latency_ms=45,
        processed_at="2024-04-01T10:00:00Z",
    )
)

# Publish alert (with Gemini summary)
producer.produce_alert(
    make_alert_event(
        event_id="alert-001",
        deal_id="deal-123",
        alert_type="high_risk",
        severity="critical",
        risk_probability=0.92,
        gemini_summary="Deal 123 shows elevated default risk due to declining EBITDA.",
        triggered_at="2024-04-01T10:00:00Z",
    )
)

# Flush pending messages
producer.flush(timeout_seconds=10.0)
```

---

## Consumer Usage

### Create Consumer

```python
from covenant_radar_api.streaming.consumer import (
    create_streaming_consumer,
    create_consumer_from_parts,
)
from covenant_radar_api.streaming.config import load_streaming_config

# From full config
config = load_streaming_config()
consumer = create_streaming_consumer(config)

# From individual parts
consumer = create_consumer_from_parts(
    confluent_config=confluent_cfg,
    consumer_config=consumer_cfg,
    measurements_topic="covenant.measurements.v1",
)
```

### Consume Messages

```python
# Subscribe (optional - auto-subscribes on first poll)
consumer.subscribe()

# Poll single message
result = consumer.poll(timeout_seconds=1.0)
if result is not None:
    event = result["event"]
    print(f"Received: {event['deal_id']} - {event['metric_name']}")

    # Process and commit
    process_measurement(event)
    consumer.commit()

# Poll batch
batch = consumer.poll_batch(max_messages=100, timeout_seconds=5.0)
for item in batch:
    process_measurement(item["event"])
consumer.commit()

# Cleanup
consumer.close()
```

### ConsumedMeasurement Structure

```python
from covenant_radar_api.streaming.consumer import ConsumedMeasurement

# Poll returns ConsumedMeasurement with metadata
result: ConsumedMeasurement = {
    "event": {...},  # MeasurementEvent
    "topic": "covenant.measurements.v1",
    "partition": 2,
    "offset": 12345,
    "key": "deal-123",  # Optional message key
}
```

---

## Testing

The integration uses Protocol-based dependency injection for testability.

### Fake Implementations

Tests use fake Kafka clients that don't require a real broker:

```python
from covenant_radar_api.streaming._test_hooks import (
    use_fake_kafka,
    use_real_kafka,
    get_fake_producer,
    get_fake_consumer,
    FakeKafkaProducer,
    FakeKafkaConsumer,
)

# Switch to fake implementations
use_fake_kafka()

# Create consumer (uses fake internally)
consumer = create_streaming_consumer(config)

# Add test messages to fake consumer
fake = get_fake_consumer()
fake.add_message(
    value=b'{"event_id":"evt-1",...}',
    key=b"deal-123",
    topic="test.measurements",
    partition=0,
    offset=0,
)

# Poll returns the test message
result = consumer.poll(1.0)
assert result["event"]["event_id"] == "evt-1"

# Verify producer behavior
producer = create_streaming_producer(config)
producer.produce_prediction(prediction_event)
fake_producer = get_fake_producer()
assert len(fake_producer.messages) == 1
```

### Reset to Real Implementations

```python
# In production startup
use_real_kafka()
```

### Protocol-Based Design

The `_test_hooks.py` module defines Protocols for type-safe fakes:

```python
class KafkaProducerProtocol(Protocol):
    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None: ...
    def poll(self, timeout: float = 0) -> int: ...
    def flush(self, timeout: float | None = None) -> int: ...


class KafkaConsumerProtocol(Protocol):
    def subscribe(self, topics: list[str]) -> None: ...
    def poll(self, timeout: float = 1.0) -> ConsumedMessageProtocol | None: ...
    def commit(self) -> None: ...
    def close(self) -> None: ...
```

---

## Deployment

### Docker Compose

```yaml
services:
  covenant-radar-api:
    environment:
      STREAMING__ENABLED: "true"
      CONFLUENT__BOOTSTRAP_SERVERS: "pkc-xxxxx.us-east-1.aws.confluent.cloud:9092"
      CONFLUENT__API_KEY: "${CONFLUENT_API_KEY}"
      CONFLUENT__API_SECRET: "${CONFLUENT_API_SECRET}"
      KAFKA__CONSUMER_GROUP_ID: "covenant-radar-api"
      KAFKA__AUTO_OFFSET_RESET: "earliest"
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
        - name: STREAMING__ENABLED
          value: "true"
        - name: CONFLUENT__BOOTSTRAP_SERVERS
          valueFrom:
            secretKeyRef:
              name: confluent-creds
              key: bootstrap-servers
        - name: CONFLUENT__API_KEY
          valueFrom:
            secretKeyRef:
              name: confluent-creds
              key: api-key
        - name: CONFLUENT__API_SECRET
          valueFrom:
            secretKeyRef:
              name: confluent-creds
              key: api-secret
```

---

## Pipeline Architecture

```
                        ┌───────────────────────────┐
                        │   External Systems        │
                        │   (Data Providers)        │
                        └───────────┬───────────────┘
                                    │ produce
                                    ▼
                        ┌───────────────────────────┐
                        │ covenant.measurements.v1  │
                        │      (Input Topic)        │
                        └───────────┬───────────────┘
                                    │ consume
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                     covenant-radar-api                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │  Streaming   │───▶│   Evaluate   │───▶│   ML Predict     │    │
│  │  Consumer    │    │   Covenants  │    │   (XGBoost/MLP)  │    │
│  └──────────────┘    └──────────────┘    └───────┬──────────┘    │
│                                                   │               │
│  ┌──────────────────────────────────────────────┴─────────────┐  │
│  │                    Streaming Producer                       │  │
│  └────────────────┬───────────────────────┬───────────────────┘  │
└───────────────────┼───────────────────────┼───────────────────────┘
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │ covenant.         │   │ covenant.         │
        │ predictions.v1    │   │ alerts.v1         │
        │ (Output Topic)    │   │ (Output Topic)    │
        └───────────────────┘   └───────────────────┘
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │   Downstream      │   │   Alert           │
        │   Consumers       │   │   Handlers        │
        └───────────────────┘   └───────────────────┘
```

---

## Troubleshooting

### Connection Errors

1. Verify bootstrap servers are correct:
   ```bash
   echo $CONFLUENT__BOOTSTRAP_SERVERS
   ```

2. Check API credentials:
   ```bash
   # Test connectivity (requires kafkacat/kcat)
   kcat -b $CONFLUENT__BOOTSTRAP_SERVERS \
        -X security.protocol=SASL_SSL \
        -X sasl.mechanism=PLAIN \
        -X sasl.username=$CONFLUENT__API_KEY \
        -X sasl.password=$CONFLUENT__API_SECRET \
        -L
   ```

### Consumer Not Receiving Messages

1. Verify topic exists and has messages:
   ```bash
   kcat -b $CONFLUENT__BOOTSTRAP_SERVERS ... -t covenant.measurements.v1 -C -c 1
   ```

2. Check consumer group offset:
   ```bash
   kcat -b $CONFLUENT__BOOTSTRAP_SERVERS ... -G covenant-radar-api
   ```

3. Try resetting to earliest:
   ```bash
   export KAFKA__AUTO_OFFSET_RESET=earliest
   ```

### Producer Messages Not Delivered

1. Check producer acks setting:
   ```bash
   echo $KAFKA__PRODUCER_ACKS  # Should be "all" for durability
   ```

2. Verify topic write permissions in Confluent Cloud console

3. Check for serialization errors in logs

### High Consumer Lag

1. Increase batch size:
   ```bash
   export KAFKA__FETCH_MIN_BYTES=10000
   ```

2. Scale consumer instances (same group ID)

3. Monitor lag via Datadog metric:
   - `covenant.stream.lag_messages{topic,partition}`
