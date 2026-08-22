"""Tests for streaming test hooks module."""

from __future__ import annotations

from covenant_radar_api.streaming._hook_defaults import (
    RealConsumedMessage,
    RealKafkaConsumer,
    RealKafkaProducer,
    _get_confluent_kafka,
    _real_consumer_factory,
    _real_producer_factory,
)
from covenant_radar_api.streaming._hook_protocols import (
    KafkaErrorProtocol,
    RawKafkaMessageProtocol,
    RawTopicPartitionConstructor,
    RawTopicPartitionProtocol,
    TopicPartitionOffset,
)
from covenant_radar_api.streaming.config import (
    ConfluentConfig,
    ConsumerConfig,
    ProducerConfig,
)
from tests.streaming._hooks_fixtures import (
    _require,
)


def _make_confluent_config() -> ConfluentConfig:
    """Create test confluent config."""
    return {
        "bootstrap_servers": "localhost:9092",
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


class TestGetConfluentKafka:
    """Tests for _get_confluent_kafka helper."""

    def test_returns_module(self) -> None:
        """Returns the confluent_kafka module."""
        mod = _get_confluent_kafka()
        # Verify it's a module with the expected name
        assert mod.__name__ == "confluent_kafka"


class TestRealKafkaProducer:
    """Tests for RealKafkaProducer."""

    def test_init(self) -> None:
        """Initialize producer with config."""
        producer = RealKafkaProducer(
            confluent_config=_make_confluent_config(),
            producer_config=_make_producer_config(),
        )
        # Verify producer can perform operations (stronger than is not None)
        assert producer.poll(0.0) == 0

    def test_produce(self) -> None:
        """Produce buffers message (no broker connection needed)."""
        producer = RealKafkaProducer(
            confluent_config=_make_confluent_config(),
            producer_config=_make_producer_config(),
        )
        # Produce buffers locally, doesn't require broker
        producer.produce("test-topic", b"test-value", b"test-key")
        # No exception means success (buffered)

    def test_produce_no_key(self) -> None:
        """Produce without key."""
        producer = RealKafkaProducer(
            confluent_config=_make_confluent_config(),
            producer_config=_make_producer_config(),
        )
        producer.produce("test-topic", b"test-value", None)

    def test_poll(self) -> None:
        """Poll returns 0 when no callbacks pending."""
        producer = RealKafkaProducer(
            confluent_config=_make_confluent_config(),
            producer_config=_make_producer_config(),
        )
        result = producer.poll(0.0)
        assert result == 0

    def test_flush(self) -> None:
        """Flush returns 0 when no messages pending."""
        producer = RealKafkaProducer(
            confluent_config=_make_confluent_config(),
            producer_config=_make_producer_config(),
        )
        result = producer.flush(0.1)
        # Returns number of messages still in queue (should be 0 or more)
        assert result >= 0


class TestRealKafkaConsumer:
    """Tests for RealKafkaConsumer."""

    def test_init(self) -> None:
        """Initialize consumer with config."""
        consumer = RealKafkaConsumer(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
        )
        # Verify consumer can perform operations (stronger than is not None)
        consumer.subscribe(("test-topic",))
        consumer.close()

    def test_subscribe(self) -> None:
        """Subscribe to topics."""
        consumer = RealKafkaConsumer(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
        )
        consumer.subscribe(("test-topic-1", "test-topic-2"))
        consumer.close()

    def test_poll_returns_none_no_broker(self) -> None:
        """Poll returns None when no broker/messages."""
        consumer = RealKafkaConsumer(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
        )
        consumer.subscribe(("test-topic",))
        # Short timeout - returns None since no broker
        result = consumer.poll(0.01)
        assert result is None
        consumer.close()

    def test_close(self) -> None:
        """Close consumer."""
        consumer = RealKafkaConsumer(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
        )
        consumer.close()


class _TestRawMessage:
    """Test implementation of RawKafkaMessageProtocol."""

    def __init__(
        self,
        value: bytes | None,
        key: bytes | None,
        topic: str | None,
        partition: int,
        offset: int,
        error: KafkaErrorProtocol | None,
    ) -> None:
        """Initialize test message."""
        self._value = value
        self._key = key
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._error = error

    def value(self) -> bytes | None:
        """Get value."""
        return self._value

    def key(self) -> bytes | None:
        """Get key."""
        return self._key

    def topic(self) -> str | None:
        """Get topic."""
        return self._topic

    def partition(self) -> int:
        """Get partition."""
        return self._partition

    def offset(self) -> int:
        """Get offset."""
        return self._offset

    def error(self) -> KafkaErrorProtocol | None:
        """Get error."""
        return self._error


class TestRealConsumedMessage:
    """Tests for RealConsumedMessage wrapper."""

    def test_value(self) -> None:
        """Get message value."""
        raw: RawKafkaMessageProtocol = _TestRawMessage(
            value=b"test-payload",
            key=b"test-key",
            topic="test-topic",
            partition=1,
            offset=100,
            error=None,
        )
        msg = RealConsumedMessage(raw)
        assert msg.value() == b"test-payload"

    def test_value_none_returns_empty_bytes(self) -> None:
        """Get empty bytes when value is None."""
        raw: RawKafkaMessageProtocol = _TestRawMessage(
            value=None,
            key=None,
            topic="test-topic",
            partition=0,
            offset=0,
            error=None,
        )
        msg = RealConsumedMessage(raw)
        assert msg.value() == b""

    def test_key(self) -> None:
        """Get message key."""
        raw: RawKafkaMessageProtocol = _TestRawMessage(
            value=b"test",
            key=b"my-key",
            topic="test-topic",
            partition=0,
            offset=0,
            error=None,
        )
        msg = RealConsumedMessage(raw)
        assert msg.key() == b"my-key"

    def test_key_none(self) -> None:
        """Get None key."""
        raw: RawKafkaMessageProtocol = _TestRawMessage(
            value=b"test",
            key=None,
            topic="test-topic",
            partition=0,
            offset=0,
            error=None,
        )
        msg = RealConsumedMessage(raw)
        assert msg.key() is None

    def test_topic(self) -> None:
        """Get topic name."""
        raw: RawKafkaMessageProtocol = _TestRawMessage(
            value=b"test",
            key=None,
            topic="my-topic",
            partition=0,
            offset=0,
            error=None,
        )
        msg = RealConsumedMessage(raw)
        assert msg.topic() == "my-topic"

    def test_topic_none_returns_empty_string(self) -> None:
        """Get empty string when topic is None."""
        raw: RawKafkaMessageProtocol = _TestRawMessage(
            value=b"test",
            key=None,
            topic=None,
            partition=0,
            offset=0,
            error=None,
        )
        msg = RealConsumedMessage(raw)
        assert msg.topic() == ""

    def test_partition(self) -> None:
        """Get partition number."""
        raw: RawKafkaMessageProtocol = _TestRawMessage(
            value=b"test",
            key=None,
            topic="test",
            partition=5,
            offset=0,
            error=None,
        )
        msg = RealConsumedMessage(raw)
        assert msg.partition() == 5

    def test_offset(self) -> None:
        """Get offset."""
        raw: RawKafkaMessageProtocol = _TestRawMessage(
            value=b"test",
            key=None,
            topic="test",
            partition=0,
            offset=12345,
            error=None,
        )
        msg = RealConsumedMessage(raw)
        assert msg.offset() == 12345


class TestRealFactoryFunctions:
    """Tests for real factory functions."""

    def test_real_producer_factory(self) -> None:
        """Create producer via factory function."""
        producer = _real_producer_factory(
            confluent_config=_make_confluent_config(),
            producer_config=_make_producer_config(),
        )
        # Verify producer works
        assert producer.poll(0.0) == 0

    def test_real_consumer_factory(self) -> None:
        """Create consumer via factory function."""
        consumer = _real_consumer_factory(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
        )
        # Verify consumer works
        consumer.subscribe(("test-topic",))
        consumer.close()


class TestRealKafkaConsumerCommit:
    """Tests for RealKafkaConsumer commit."""

    def test_commit(self) -> None:
        """Commit offsets (no-op without actual consumption)."""
        consumer = RealKafkaConsumer(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
        )
        consumer.subscribe(("test-topic",))
        # Committing no positions is a no-op: librdkafka rejects a commit that
        # carries no partitions, so RealKafkaConsumer must not forward it.
        consumer.commit(())
        consumer.close()

    def test_commit_forwards_positions_as_topic_partitions(self) -> None:
        """Positions are converted to confluent TopicPartitions and committed.

        Exercises the real conversion against the real confluent_kafka
        TopicPartition constructor, with only the raw consumer replaced.
        """
        fake_raw = _FakeRawConsumer()
        consumer = RealKafkaConsumer(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
        )
        consumer._consumer = fake_raw

        first: TopicPartitionOffset = {"topic": "measurements", "partition": 0, "offset": 7}
        second: TopicPartitionOffset = {"topic": "measurements", "partition": 3, "offset": 41}
        consumer.commit((first, second))

        assert len(fake_raw.committed) == 1
        forwarded = fake_raw.committed[0]
        assert len(forwarded) == 2

        # Build a reference through the same Protocol-typed constructor rather
        # than reading confluent_kafka.TopicPartition as an untyped attribute.
        confluent_kafka = _get_confluent_kafka()
        topic_partition: RawTopicPartitionConstructor = confluent_kafka.TopicPartition
        reference = topic_partition("measurements", 0, 7)
        assert type(forwarded[0]) is type(reference)
        assert type(forwarded[1]) is type(reference)


class _TestKafkaError:
    """Test implementation of KafkaErrorProtocol."""

    def __init__(self, code: int, message: str) -> None:
        """Initialize test error."""
        self._code = code
        self._message = message

    def code(self) -> int:
        """Get error code."""
        return self._code

    def str(self) -> str:
        """Get error string."""
        return self._message


class _FakeRawConsumer:
    """Fake raw consumer that can inject messages including errors."""

    def __init__(self) -> None:
        """Initialize fake consumer."""
        self._messages: list[RawKafkaMessageProtocol | None] = []
        self._subscribed_topics: list[str] = []
        self.committed: list[list[RawTopicPartitionProtocol]] = []

    def subscribe(self, topics: list[str]) -> None:
        """Subscribe to topics."""
        self._subscribed_topics = topics

    def poll(self, timeout: float = -1) -> RawKafkaMessageProtocol | None:
        """Return next message from queue."""
        if not self._messages:
            return None
        return self._messages.pop(0)

    def commit(
        self,
        *,
        offsets: list[RawTopicPartitionProtocol],
        asynchronous: bool,
    ) -> None:
        """Record the committed offsets.

        Args:
            offsets: Positions handed to the consumer.
            asynchronous: Whether the commit was requested asynchronously.
        """
        del asynchronous
        self.committed.append(offsets)

    def close(self) -> None:
        """Close consumer."""

    def add_message(self, msg: RawKafkaMessageProtocol | None) -> None:
        """Add a message to the queue."""
        self._messages.append(msg)


class TestRealKafkaConsumerPollError:
    """Tests for RealKafkaConsumer poll with error messages."""

    def test_poll_with_error_returns_none(self) -> None:
        """Poll returns None when message has error."""
        # Create a fake raw consumer
        fake_raw = _FakeRawConsumer()

        # Create a RealKafkaConsumer but inject our fake
        consumer = RealKafkaConsumer(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
        )
        # Replace the internal consumer with our fake
        consumer._consumer = fake_raw

        # Create a message with an error
        error: KafkaErrorProtocol = _TestKafkaError(-191, "Local: No offset stored")
        error_msg: RawKafkaMessageProtocol = _TestRawMessage(
            value=None,
            key=None,
            topic=None,
            partition=-1,
            offset=-1,
            error=error,
        )

        # Add the error message to the fake
        fake_raw.add_message(error_msg)

        # Poll should return None due to error
        result = consumer.poll(0.1)
        assert result is None

    def test_poll_with_valid_message_returns_wrapped(self) -> None:
        """Poll returns wrapped message when no error."""
        # Create a fake raw consumer
        fake_raw = _FakeRawConsumer()

        # Create a RealKafkaConsumer but inject our fake
        consumer = RealKafkaConsumer(
            confluent_config=_make_confluent_config(),
            consumer_config=_make_consumer_config(),
        )
        consumer._consumer = fake_raw

        # Create a valid message (no error)
        valid_msg: RawKafkaMessageProtocol = _TestRawMessage(
            value=b"test-value",
            key=b"test-key",
            topic="test-topic",
            partition=1,
            offset=100,
            error=None,
        )

        fake_raw.add_message(valid_msg)

        # Poll should return the wrapped message
        result = _require(consumer.poll(0.1))
        assert result.value() == b"test-value"
        assert result.topic() == "test-topic"
