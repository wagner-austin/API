"""Tests for streaming producer module."""

from __future__ import annotations

from typing import TypeVar

from covenant_radar_api.streaming._test_hooks import (
    FakeKafkaProducer,
    get_fake_producer,
    use_fake_kafka,
)
from covenant_radar_api.streaming.config import (
    ConfluentConfig,
    ConsumerConfig,
    KafkaTopicsConfig,
    ProducerConfig,
    StreamingConfig,
)
from covenant_radar_api.streaming.producer import (
    StreamingProducer,
    create_producer_from_parts,
    create_streaming_producer,
)
from covenant_radar_api.streaming.schemas import (
    make_alert_event,
    make_measurement_event,
    make_prediction_event,
)

_T = TypeVar("_T")


def _require(value: _T | None) -> _T:
    """Narrow optional type to non-None. Raises if None."""
    if value is None:
        msg = "Expected non-None value"
        raise AssertionError(msg)
    return value


def _make_confluent_config() -> ConfluentConfig:
    """Create test confluent config."""
    return {
        "bootstrap_servers": "test-broker:9092",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "PLAIN",
    }


def _make_producer_config() -> ProducerConfig:
    """Create test producer config."""
    return {
        "acks": "all",
        "retries": 3,
        "linger_ms": 5,
        "batch_size": 16384,
        "compression_type": "gzip",
    }


def _make_consumer_config() -> ConsumerConfig:
    """Create test consumer config."""
    return {
        "group_id": "test-group",
        "auto_offset_reset": "earliest",
        "enable_auto_commit": False,
        "fetch_min_bytes": 1,
        "session_timeout_ms": 45000,
        "heartbeat_interval_ms": 15000,
    }


def _make_topics_config() -> KafkaTopicsConfig:
    """Create test topics config."""
    return {
        "measurements": "test.measurements",
        "predictions": "test.predictions",
        "alerts": "test.alerts",
        "dlq": "test.dlq",
    }


def _make_streaming_config() -> StreamingConfig:
    """Create complete test streaming config."""
    return {
        "enabled": True,
        "confluent": _make_confluent_config(),
        "schema_registry": None,
        "topics": _make_topics_config(),
        "consumer": _make_consumer_config(),
        "producer": _make_producer_config(),
    }


class TestStreamingProducer:
    """Tests for StreamingProducer class."""

    def test_init(self) -> None:
        """Initialize producer with topics."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="pred-topic",
            alerts_topic="alert-topic",
            dlq_topic="dlq-topic",
        )
        assert producer._producer is fake
        assert producer._predictions_topic == "pred-topic"
        assert producer._alerts_topic == "alert-topic"

    def test_produce_prediction(self) -> None:
        """Produce prediction event to predictions topic."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="test.predictions",
            alerts_topic="test.alerts",
            dlq_topic="dlq-topic",
        )

        event = make_prediction_event(
            event_id="evt-123",
            deal_id="deal-456",
            period_start="2024-01-01",
            period_end="2024-03-31",
            evaluation_status="OK",
            covenants_evaluated=5,
            breaches_count=0,
            risk_probability=0.15,
            risk_tier="LOW",
            model_version="1.0.0",
            evaluation_latency_ms=50,
            prediction_latency_ms=100,
            processed_at="2024-04-01T10:00:00Z",
        )

        producer.produce_prediction(event)

        assert len(fake.messages) == 1
        msg = fake.messages[0]
        assert msg.topic == "test.predictions"
        assert msg.key == b"deal-456"
        assert b'"type":"covenant.prediction.v1"' in msg.value

    def test_produce_alert(self) -> None:
        """Produce alert event to alerts topic."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="test.predictions",
            alerts_topic="test.alerts",
            dlq_topic="dlq-topic",
        )

        event = make_alert_event(
            event_id="alert-789",
            deal_id="deal-xyz",
            alert_type="high_risk",
            severity="critical",
            risk_probability=0.92,
            gemini_summary="High risk detected for deal.",
            triggered_at="2024-04-01T10:05:00Z",
        )

        producer.produce_alert(event)

        assert len(fake.messages) == 1
        msg = fake.messages[0]
        assert msg.topic == "test.alerts"
        assert msg.key == b"deal-xyz"
        assert b'"type":"covenant.alert.v1"' in msg.value
        assert b'"severity":"critical"' in msg.value

    def test_produce_event_measurement(self) -> None:
        """Produce generic event to custom topic."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="test.predictions",
            alerts_topic="test.alerts",
            dlq_topic="dlq-topic",
        )

        event = make_measurement_event(
            event_id="meas-001",
            deal_id="deal-abc",
            period_start="2024-01-01",
            period_end="2024-03-31",
            metric_name="debt_to_equity",
            metric_value=1.5,
            timestamp="2024-04-01T09:00:00Z",
        )

        producer.produce_event(event, "custom.topic")

        assert len(fake.messages) == 1
        msg = fake.messages[0]
        assert msg.topic == "custom.topic"
        assert msg.key == b"deal-abc"
        assert b'"type":"covenant.measurement.v1"' in msg.value

    def test_produce_event_prediction(self) -> None:
        """Produce prediction via generic produce_event."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="test.predictions",
            alerts_topic="test.alerts",
            dlq_topic="dlq-topic",
        )

        event = make_prediction_event(
            event_id="evt-999",
            deal_id="deal-def",
            period_start="2024-01-01",
            period_end="2024-03-31",
            evaluation_status="BREACH",
            covenants_evaluated=3,
            breaches_count=1,
            risk_probability=0.65,
            risk_tier="HIGH",
            model_version="2.0.0",
            evaluation_latency_ms=30,
            prediction_latency_ms=80,
            processed_at="2024-04-01T11:00:00Z",
        )

        producer.produce_event(event, "another.topic")

        assert len(fake.messages) == 1
        assert fake.messages[0].topic == "another.topic"

    def test_produce_event_alert(self) -> None:
        """Produce alert via generic produce_event."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="test.predictions",
            alerts_topic="test.alerts",
            dlq_topic="dlq-topic",
        )

        event = make_alert_event(
            event_id="alert-888",
            deal_id="deal-ghi",
            alert_type="breach",
            severity="warning",
            risk_probability=0.45,
            gemini_summary="Covenant breach detected.",
            triggered_at="2024-04-01T12:00:00Z",
        )

        producer.produce_event(event, "alerts.backup")

        assert len(fake.messages) == 1
        assert fake.messages[0].topic == "alerts.backup"

    def test_flush(self) -> None:
        """Flush calls underlying producer flush."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="test.predictions",
            alerts_topic="test.alerts",
            dlq_topic="dlq-topic",
        )

        assert fake.flush_called is False

        result = producer.flush(5.0)

        assert result == 0
        assert fake.flush_called is True

    def test_flush_default_timeout(self) -> None:
        """Flush with default timeout."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="test.predictions",
            alerts_topic="test.alerts",
            dlq_topic="dlq-topic",
        )

        result = producer.flush()

        assert result == 0
        assert fake.flush_called is True

    def test_poll(self) -> None:
        """Poll calls underlying producer poll."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="test.predictions",
            alerts_topic="test.alerts",
            dlq_topic="dlq-topic",
        )

        assert fake.poll_count == 0

        result = producer.poll(2.0)

        assert result == 0
        assert fake.poll_count == 1

    def test_poll_default_timeout(self) -> None:
        """Poll with default timeout."""
        fake = FakeKafkaProducer()
        producer = StreamingProducer(
            producer=fake,
            predictions_topic="test.predictions",
            alerts_topic="test.alerts",
            dlq_topic="dlq-topic",
        )

        result = producer.poll()

        assert result == 0
        assert fake.poll_count == 1


class TestCreateStreamingProducer:
    """Tests for create_streaming_producer factory."""

    def test_creates_producer(self) -> None:
        """Create producer from streaming config."""
        use_fake_kafka()
        config = _make_streaming_config()

        producer = create_streaming_producer(config)

        # Verify producer has expected attributes (StreamingProducer-specific)
        assert producer._predictions_topic == "test.predictions"
        assert producer._alerts_topic == "test.alerts"

        # Verify fake was created by checking FakeKafkaProducer-specific attribute
        fake = _require(get_fake_producer())
        assert fake.messages == []


class TestCreateProducerFromParts:
    """Tests for create_producer_from_parts factory."""

    def test_creates_producer(self) -> None:
        """Create producer from individual config parts."""
        use_fake_kafka()

        producer = create_producer_from_parts(
            confluent_config=_make_confluent_config(),
            producer_config=_make_producer_config(),
            predictions_topic="custom.predictions",
            alerts_topic="custom.alerts",
            dlq_topic="dlq-topic",
        )

        # Verify producer has expected attributes (StreamingProducer-specific)
        assert producer._predictions_topic == "custom.predictions"
        assert producer._alerts_topic == "custom.alerts"

        # Verify fake was created by checking FakeKafkaProducer-specific attribute
        fake = _require(get_fake_producer())
        assert fake.messages == []
