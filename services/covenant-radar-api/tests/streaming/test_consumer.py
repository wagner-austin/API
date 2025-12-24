"""Tests for streaming consumer module."""

from __future__ import annotations

from typing import TypeVar

from covenant_radar_api.streaming._test_hooks import (
    FakeKafkaConsumer,
    get_fake_consumer,
    use_fake_kafka,
)
from covenant_radar_api.streaming.config import (
    ConfluentConfig,
    ConsumerConfig,
    KafkaTopicsConfig,
    ProducerConfig,
    StreamingConfig,
)
from covenant_radar_api.streaming.consumer import (
    StreamingConsumer,
    create_consumer_from_parts,
    create_streaming_consumer,
)
from covenant_radar_api.streaming.schemas import encode_measurement_event, make_measurement_event

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


def _make_measurement_json(
    event_id: str = "evt-001",
    deal_id: str = "deal-123",
) -> bytes:
    """Create measurement event as JSON bytes."""
    event = make_measurement_event(
        event_id=event_id,
        deal_id=deal_id,
        period_start="2024-01-01",
        period_end="2024-03-31",
        metric_name="debt_to_equity",
        metric_value=1.5,
        timestamp="2024-04-01T09:00:00Z",
    )
    return encode_measurement_event(event).encode("utf-8")


class TestStreamingConsumer:
    """Tests for StreamingConsumer class."""

    def test_init(self) -> None:
        """Initialize consumer with topic."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )
        assert consumer._consumer is fake
        assert consumer._measurements_topic == "test.measurements"
        assert consumer._subscribed is False

    def test_subscribe(self) -> None:
        """Subscribe to measurements topic."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        assert consumer.is_subscribed is False
        assert fake.subscribed_topics == ()

        consumer.subscribe()

        assert consumer.is_subscribed is True
        assert fake.subscribed_topics == ("test.measurements",)

    def test_subscribe_idempotent(self) -> None:
        """Subscribe is idempotent."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        consumer.subscribe()
        consumer.subscribe()
        consumer.subscribe()

        # Should only subscribe once
        assert consumer.is_subscribed is True
        assert fake.subscribed_topics == ("test.measurements",)

    def test_poll_no_message(self) -> None:
        """Poll returns None when no messages."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        result = consumer.poll(1.0)

        assert result is None
        assert consumer.is_subscribed is True  # Auto-subscribed

    def test_poll_returns_consumed_measurement(self) -> None:
        """Poll returns ConsumedMeasurement with event and metadata."""
        fake = FakeKafkaConsumer()
        fake.add_message(
            value=_make_measurement_json("evt-123", "deal-456"),
            key=b"deal-456",
            topic="test.measurements",
            partition=2,
            offset=100,
        )

        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        result = _require(consumer.poll(1.0))

        assert result["event"]["event_id"] == "evt-123"
        assert result["event"]["deal_id"] == "deal-456"
        assert result["topic"] == "test.measurements"
        assert result["partition"] == 2
        assert result["offset"] == 100
        assert result["key"] == "deal-456"

    def test_poll_auto_subscribes(self) -> None:
        """Poll automatically subscribes if not subscribed."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="my.measurements",
        )

        assert consumer.is_subscribed is False

        consumer.poll(0.1)

        assert consumer.is_subscribed is True
        assert fake.subscribed_topics == ("my.measurements",)

    def test_poll_skips_subscribe_when_already_subscribed(self) -> None:
        """Poll skips subscribe when already subscribed."""
        fake = FakeKafkaConsumer()
        fake.add_message(value=_make_measurement_json("evt-pre", "deal-pre"))

        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        # Explicitly subscribe first
        consumer.subscribe()
        assert consumer.is_subscribed is True

        # Poll should skip subscribe and directly poll
        result = _require(consumer.poll(1.0))

        assert result["event"]["event_id"] == "evt-pre"
        assert consumer.is_subscribed is True

    def test_poll_with_none_key(self) -> None:
        """Poll handles None message key."""
        fake = FakeKafkaConsumer()
        fake.add_message(
            value=_make_measurement_json(),
            key=None,
            topic="test.measurements",
            partition=0,
            offset=0,
        )

        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        result = _require(consumer.poll(1.0))

        assert result["key"] is None

    def test_poll_batch_empty(self) -> None:
        """Poll batch returns empty tuple when no messages."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        result = consumer.poll_batch(max_messages=10, timeout_seconds=0.1)

        assert result == ()
        assert consumer.is_subscribed is True

    def test_poll_batch_returns_messages(self) -> None:
        """Poll batch returns multiple messages."""
        fake = FakeKafkaConsumer()
        fake.add_message(value=_make_measurement_json("evt-1", "deal-1"))
        fake.add_message(value=_make_measurement_json("evt-2", "deal-2"))
        fake.add_message(value=_make_measurement_json("evt-3", "deal-3"))

        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        result = consumer.poll_batch(max_messages=10, timeout_seconds=1.0)

        assert len(result) == 3
        assert result[0]["event"]["event_id"] == "evt-1"
        assert result[1]["event"]["event_id"] == "evt-2"
        assert result[2]["event"]["event_id"] == "evt-3"

    def test_poll_batch_respects_max_messages(self) -> None:
        """Poll batch stops at max_messages."""
        fake = FakeKafkaConsumer()
        for i in range(5):
            fake.add_message(value=_make_measurement_json(f"evt-{i}", f"deal-{i}"))

        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        result = consumer.poll_batch(max_messages=2, timeout_seconds=1.0)

        assert len(result) == 2
        # 3 messages still in queue
        remaining = consumer.poll_batch(max_messages=10, timeout_seconds=1.0)
        assert len(remaining) == 3

    def test_poll_batch_auto_subscribes(self) -> None:
        """Poll batch automatically subscribes."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="batch.measurements",
        )

        assert consumer.is_subscribed is False

        consumer.poll_batch(max_messages=5, timeout_seconds=0.1)

        assert consumer.is_subscribed is True
        assert fake.subscribed_topics == ("batch.measurements",)

    def test_commit(self) -> None:
        """Commit calls underlying consumer commit."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        assert fake.commit_count == 0

        consumer.commit()

        assert fake.commit_count == 1

    def test_commit_multiple(self) -> None:
        """Multiple commits increment counter."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        consumer.commit()
        consumer.commit()
        consumer.commit()

        assert fake.commit_count == 3

    def test_close(self) -> None:
        """Close marks consumer as closed and unsubscribed."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )
        consumer.subscribe()

        assert consumer.is_subscribed is True
        assert fake.closed is False

        consumer.close()

        assert consumer.is_subscribed is False
        assert fake.closed is True

    def test_is_subscribed_property(self) -> None:
        """is_subscribed property reflects state."""
        fake = FakeKafkaConsumer()
        consumer = StreamingConsumer(
            consumer=fake,
            measurements_topic="test.measurements",
        )

        assert consumer.is_subscribed is False

        consumer.subscribe()
        assert consumer.is_subscribed is True

        consumer.close()
        assert consumer.is_subscribed is False


class TestCreateStreamingConsumer:
    """Tests for create_streaming_consumer factory."""

    def test_creates_consumer(self) -> None:
        """Create consumer from streaming config."""
        use_fake_kafka()
        config = _make_streaming_config()

        consumer = create_streaming_consumer(config)

        # Verify consumer has expected attributes (StreamingConsumer-specific)
        assert consumer._measurements_topic == "test.measurements"
        assert consumer.is_subscribed is False

        # Verify fake was created by checking FakeKafkaConsumer-specific attribute
        fake = _require(get_fake_consumer())
        assert fake.subscribed_topics == ()


class TestCreateConsumerFromParts:
    """Tests for create_consumer_from_parts factory."""

    def test_creates_consumer(self) -> None:
        """Create consumer from individual config parts."""
        use_fake_kafka()

        consumer = create_consumer_from_parts(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
            measurements_topic="custom.measurements",
        )

        # Verify consumer has expected attributes (StreamingConsumer-specific)
        assert consumer._measurements_topic == "custom.measurements"
        assert consumer.is_subscribed is False

        # Verify fake was created by checking FakeKafkaConsumer-specific attribute
        fake = _require(get_fake_consumer())
        assert fake.subscribed_topics == ()
