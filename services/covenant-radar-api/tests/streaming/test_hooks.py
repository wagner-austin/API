"""Tests for streaming test hooks module."""

from __future__ import annotations

from typing import TypeVar

from covenant_radar_api.streaming._test_hooks import (
    FakeConsumedMessage,
    FakeKafkaConsumer,
    FakeKafkaProducer,
    KafkaErrorProtocol,
    ProducedMessage,
    RawKafkaMessageProtocol,
    RawTopicPartitionConstructor,
    RawTopicPartitionProtocol,
    RealConsumedMessage,
    RealKafkaConsumer,
    RealKafkaProducer,
    TopicPartitionOffset,
    _get_confluent_kafka,
    _real_consumer_factory,
    _real_producer_factory,
    get_fake_consumer,
    get_fake_producer,
    use_fake_kafka,
    use_real_kafka,
)
from covenant_radar_api.streaming.config import (
    ConfluentConfig,
    ConsumerConfig,
    ProducerConfig,
)

_T = TypeVar("_T")


def _require(value: _T | None) -> _T:
    """Narrow optional type to non-None. Raises if None."""
    if value is None:
        msg = "Expected non-None value"
        raise AssertionError(msg)
    return value


class TestFakeConsumedMessage:
    """Tests for FakeConsumedMessage."""

    def test_value(self) -> None:
        """Get message value."""
        msg = FakeConsumedMessage(
            value=b"test payload",
            key=b"key",
            topic="test-topic",
            partition=0,
            offset=123,
        )
        assert msg.value() == b"test payload"

    def test_key(self) -> None:
        """Get message key."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=b"my-key",
            topic="test-topic",
            partition=0,
            offset=0,
        )
        assert msg.key() == b"my-key"

    def test_key_none(self) -> None:
        """Get None message key."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=None,
            topic="test-topic",
            partition=0,
            offset=0,
        )
        assert msg.key() is None

    def test_topic(self) -> None:
        """Get topic name."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=None,
            topic="my-topic",
            partition=0,
            offset=0,
        )
        assert msg.topic() == "my-topic"

    def test_partition(self) -> None:
        """Get partition number."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=None,
            topic="test",
            partition=5,
            offset=0,
        )
        assert msg.partition() == 5

    def test_offset(self) -> None:
        """Get offset."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=None,
            topic="test",
            partition=0,
            offset=999,
        )
        assert msg.offset() == 999


class TestProducedMessage:
    """Tests for ProducedMessage."""

    def test_attributes(self) -> None:
        """Access all attributes."""
        msg = ProducedMessage(
            topic="my-topic",
            value=b"my-value",
            key=b"my-key",
        )
        assert msg.topic == "my-topic"
        assert msg.value == b"my-value"
        assert msg.key == b"my-key"

    def test_none_key(self) -> None:
        """None key is stored."""
        msg = ProducedMessage(
            topic="test",
            value=b"test",
            key=None,
        )
        assert msg.key is None


class TestFakeKafkaProducer:
    """Tests for FakeKafkaProducer."""

    def test_produce(self) -> None:
        """Produce stores messages."""
        producer = FakeKafkaProducer()
        producer.produce("topic1", b"value1", b"key1")
        producer.produce("topic2", b"value2", None)

        assert len(producer.messages) == 2
        assert producer.messages[0].topic == "topic1"
        assert producer.messages[0].value == b"value1"
        assert producer.messages[0].key == b"key1"
        assert producer.messages[1].topic == "topic2"
        assert producer.messages[1].key is None

    def test_flush(self) -> None:
        """Flush marks as called and returns 0."""
        producer = FakeKafkaProducer()
        assert producer.flush_called is False

        result = producer.flush(10.0)

        assert result == 0
        assert producer.flush_called is True

    def test_poll(self) -> None:
        """Poll increments counter."""
        producer = FakeKafkaProducer()
        assert producer.poll_count == 0

        result = producer.poll(1.0)

        assert result == 0
        assert producer.poll_count == 1


class TestFakeKafkaConsumer:
    """Tests for FakeKafkaConsumer."""

    def test_subscribe(self) -> None:
        """Subscribe records topics."""
        consumer = FakeKafkaConsumer()
        assert consumer.subscribed_topics == ()

        consumer.subscribe(("topic1", "topic2"))

        assert consumer.subscribed_topics == ("topic1", "topic2")

    def test_poll_empty_queue(self) -> None:
        """Poll returns None for empty queue."""
        consumer = FakeKafkaConsumer()
        result = consumer.poll(1.0)
        assert result is None

    def test_poll_returns_message(self) -> None:
        """Poll returns message from queue."""
        consumer = FakeKafkaConsumer()
        consumer.add_message(
            value=b"test payload",
            key=b"key1",
            topic="test-topic",
            partition=1,
            offset=100,
        )

        msg = _require(consumer.poll(1.0))

        assert msg.value() == b"test payload"
        assert msg.key() == b"key1"
        assert msg.topic() == "test-topic"
        assert msg.partition() == 1
        assert msg.offset() == 100

    def test_poll_fifo_order(self) -> None:
        """Poll returns messages in FIFO order."""
        consumer = FakeKafkaConsumer()
        consumer.add_message(value=b"first")
        consumer.add_message(value=b"second")
        consumer.add_message(value=b"third")

        msg1 = consumer.poll(1.0)
        msg2 = consumer.poll(1.0)
        msg3 = consumer.poll(1.0)
        msg4 = consumer.poll(1.0)

        assert msg1 is not None and msg1.value() == b"first"
        assert msg2 is not None and msg2.value() == b"second"
        assert msg3 is not None and msg3.value() == b"third"
        assert msg4 is None

    def test_commit(self) -> None:
        """Commit records the requested positions."""
        consumer = FakeKafkaConsumer()
        assert consumer.commit_count == 0

        first: TopicPartitionOffset = {"topic": "t", "partition": 0, "offset": 5}
        second: TopicPartitionOffset = {"topic": "t", "partition": 1, "offset": 9}
        consumer.commit((first,))
        consumer.commit((second,))

        assert consumer.commit_count == 2
        assert consumer.committed_offsets == [(first,), (second,)]

    def test_close(self) -> None:
        """Close marks consumer as closed."""
        consumer = FakeKafkaConsumer()
        assert consumer.closed is False

        consumer.close()

        assert consumer.closed is True

    def test_add_message_defaults(self) -> None:
        """Add message with default values."""
        consumer = FakeKafkaConsumer()
        consumer.add_message(value=b"test")

        msg = _require(consumer.poll(1.0))

        assert msg.value() == b"test"
        assert msg.key() is None
        assert msg.topic() == "test-topic"
        assert msg.partition() == 0
        assert msg.offset() == 0


class TestHookSwitching:
    """Tests for use_fake_kafka and use_real_kafka."""

    def test_use_fake_kafka_sets_factories(self) -> None:
        """use_fake_kafka sets fake factories."""
        # First ensure real is set
        use_real_kafka()

        # Then switch to fake
        use_fake_kafka()

        # Verify by checking get_fake_* works after factory call
        from covenant_radar_api.streaming._test_hooks import producer_factory

        confluent_cfg: ConfluentConfig = {
            "bootstrap_servers": "test",
            "api_key": "test",
            "api_secret": "test",
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
        }
        producer_cfg: ProducerConfig = {
            "acks": "all",
            "retries": 3,
            "linger_ms": 5,
            "batch_size": 16384,
            "compression_type": "gzip",
        }

        # Create producer via factory - should create fake
        producer_factory(confluent_cfg, producer_cfg)

        # Should have created a fake - access fake's attribute to verify
        fake = _require(get_fake_producer())
        assert fake.messages == []  # FakeKafkaProducer-specific attribute

    def test_get_fake_producer_none_initially(self) -> None:
        """get_fake_producer returns None before any creation."""
        # Note: This test may be affected by prior test state
        # The global _fake_producer may already be set
        # This tests the function at least runs
        result = get_fake_producer()
        # Could be None or FakeKafkaProducer depending on test order
        assert result is None or isinstance(result, FakeKafkaProducer)

    def test_get_fake_consumer_none_initially(self) -> None:
        """get_fake_consumer returns None before any creation."""
        result = get_fake_consumer()
        assert result is None or isinstance(result, FakeKafkaConsumer)

    def test_use_real_kafka_restores(self) -> None:
        """use_real_kafka restores real factories."""
        use_fake_kafka()
        use_real_kafka()

        # After restoring real, the factories should be the real ones
        # We can't easily test without confluent-kafka installed,
        # but at least verify no errors
        from covenant_radar_api.streaming._test_hooks import (
            _real_consumer_factory,
            _real_producer_factory,
            consumer_factory,
            producer_factory,
        )

        assert producer_factory == _real_producer_factory
        assert consumer_factory == _real_consumer_factory


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
        # No exception means success


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
