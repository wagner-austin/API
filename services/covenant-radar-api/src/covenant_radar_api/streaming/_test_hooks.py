"""Test hooks for Kafka streaming infrastructure.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Protocol, TypedDict

from ._test_hooks_model import (
    FakeMetricsSink,
    FakePredictor,
)
from ._test_hooks_repositories import (
    FakeCovenantRepository,
    FakeCovenantResultRepository,
    FakeDealRepository,
    FakeMeasurementRepository,
)
from .config import ConfluentConfig, ConsumerConfig, ProducerConfig


def _get_confluent_kafka() -> ModuleType:
    """Get the confluent_kafka module.

    Returns:
        The confluent_kafka module.
    """
    return __import__("confluent_kafka")


# Type alias for raw Kafka config dict
KafkaConfigDict = dict[str, str | int | bool]

# =============================================================================
# Raw Kafka Message Protocol (includes error method)
# =============================================================================


class RawKafkaMessageProtocol(Protocol):
    """Protocol for raw confluent-kafka Message including error method.

    This extends the public ConsumedMessageProtocol with the error() method
    that confluent-kafka messages have.
    """

    def value(self) -> bytes | None:
        """Get message value as bytes or None."""
        ...

    def key(self) -> bytes | None:
        """Get message key as bytes or None."""
        ...

    def topic(self) -> str | None:
        """Get topic name or None."""
        ...

    def partition(self) -> int:
        """Get partition number."""
        ...

    def offset(self) -> int:
        """Get message offset."""
        ...

    def error(self) -> KafkaErrorProtocol | None:
        """Get error if this is an error message.

        Returns:
            KafkaError if error, None if successful message.
        """
        ...


class KafkaErrorProtocol(Protocol):
    """Protocol for Kafka error."""

    def code(self) -> int:
        """Get error code."""
        ...

    def str(self) -> str:
        """Get error string."""
        ...


# =============================================================================
# Raw Kafka Client Protocols (for constructor calls)
# =============================================================================


class RawKafkaProducerProtocol(Protocol):
    """Protocol for raw confluent-kafka Producer.

    Includes produce() with different signature than our wrapper.
    """

    def produce(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Produce a message."""
        ...

    def flush(self, timeout: float = -1) -> int:
        """Flush pending messages."""
        ...

    def poll(self, timeout: float = 0) -> int:
        """Poll for delivery reports."""
        ...


class TopicPartitionOffset(TypedDict, total=True):
    """A commit position for one topic partition.

    Fields:
        topic: Topic name.
        partition: Partition number.
        offset: Offset of the NEXT message to consume, i.e. one past the last
            message that has been fully processed.
    """

    topic: str
    partition: int
    offset: int


class RawTopicPartitionProtocol(Protocol):
    """Protocol for a confluent_kafka.TopicPartition instance.

    Opaque to this codebase: it is constructed and handed straight back to
    confluent-kafka, so no attributes are read from it.
    """


class RawTopicPartitionConstructor(Protocol):
    """Protocol for the confluent_kafka.TopicPartition constructor."""

    def __call__(self, topic: str, partition: int, offset: int) -> RawTopicPartitionProtocol:
        """Construct a TopicPartition.

        Args:
            topic: Topic name.
            partition: Partition number.
            offset: Offset to associate with the partition.

        Returns:
            A confluent-kafka TopicPartition.
        """
        ...


class RawKafkaConsumerProtocol(Protocol):
    """Protocol for raw confluent-kafka Consumer."""

    def subscribe(self, topics: list[str]) -> None:
        """Subscribe to topics."""
        ...

    def poll(self, timeout: float = -1) -> RawKafkaMessageProtocol | None:
        """Poll for a message."""
        ...

    def commit(
        self,
        *,
        offsets: list[RawTopicPartitionProtocol],
        asynchronous: bool,
    ) -> None:
        """Commit the given offsets.

        Args:
            offsets: Explicit positions to commit.
            asynchronous: Whether to return before the broker acknowledges.
        """
        ...

    def close(self) -> None:
        """Close consumer."""
        ...


# Callable types for constructors
RawProducerConstructor = Callable[[KafkaConfigDict], RawKafkaProducerProtocol]
RawConsumerConstructor = Callable[[KafkaConfigDict], RawKafkaConsumerProtocol]


# =============================================================================
# Consumed Message Protocol
# =============================================================================


class ConsumedMessageProtocol(Protocol):
    """Protocol for a consumed Kafka message.

    Provides access to message payload, key, and metadata without
    exposing confluent-kafka internals.
    """

    def value(self) -> bytes:
        """Get message value as bytes.

        Returns:
            Raw message payload bytes.
        """
        ...

    def key(self) -> bytes | None:
        """Get message key as bytes.

        Returns:
            Message key bytes or None if no key.
        """
        ...

    def topic(self) -> str:
        """Get topic name.

        Returns:
            Name of the topic this message was consumed from.
        """
        ...

    def partition(self) -> int:
        """Get partition number.

        Returns:
            Partition number within the topic.
        """
        ...

    def offset(self) -> int:
        """Get message offset.

        Returns:
            Offset of this message within the partition.
        """
        ...


# =============================================================================
# Producer Protocol
# =============================================================================


class KafkaProducerProtocol(Protocol):
    """Protocol for Kafka producer.

    Defines the interface for producing messages to Kafka topics.
    """

    def produce(
        self,
        topic: str,
        value: bytes,
        key: bytes | None = None,
    ) -> None:
        """Produce a message to a topic.

        Args:
            topic: Target topic name.
            value: Message payload as bytes.
            key: Optional message key for partitioning.
        """
        ...

    def flush(self, timeout_seconds: float) -> int:
        """Flush pending messages.

        Blocks until all pending messages are delivered or timeout.

        Args:
            timeout_seconds: Maximum time to wait.

        Returns:
            Number of messages still in queue (0 if all flushed).
        """
        ...

    def poll(self, timeout_seconds: float) -> int:
        """Poll for delivery reports.

        Args:
            timeout_seconds: Maximum time to wait.

        Returns:
            Number of events processed.
        """
        ...


# =============================================================================
# Consumer Protocol
# =============================================================================


class KafkaConsumerProtocol(Protocol):
    """Protocol for Kafka consumer.

    Defines the interface for consuming messages from Kafka topics.
    """

    def subscribe(self, topics: tuple[str, ...]) -> None:
        """Subscribe to topics.

        Args:
            topics: Tuple of topic names to subscribe to.
        """
        ...

    def poll(self, timeout_seconds: float) -> ConsumedMessageProtocol | None:
        """Poll for a single message.

        Args:
            timeout_seconds: Maximum time to wait for a message.

        Returns:
            Consumed message or None if timeout.
        """
        ...

    def commit(self, offsets: tuple[TopicPartitionOffset, ...]) -> None:
        """Commit the given positions synchronously.

        Positions are explicit rather than implicit: an argument-less commit
        would commit the consumed position on every assigned partition,
        including messages that have been polled but not yet processed.

        Args:
            offsets: Positions to commit.
        """
        ...

    def close(self) -> None:
        """Close the consumer and leave consumer group."""
        ...


# =============================================================================
# Producer Factory Protocol
# =============================================================================


class ProducerFactoryProtocol(Protocol):
    """Protocol for creating Kafka producers."""

    def __call__(
        self,
        confluent_config: ConfluentConfig,
        producer_config: ProducerConfig,
    ) -> KafkaProducerProtocol:
        """Create a Kafka producer.

        Args:
            confluent_config: Confluent Cloud connection settings.
            producer_config: Producer-specific settings.

        Returns:
            KafkaProducerProtocol implementation.
        """
        ...


# =============================================================================
# Consumer Factory Protocol
# =============================================================================


class ConsumerFactoryProtocol(Protocol):
    """Protocol for creating Kafka consumers."""

    def __call__(
        self,
        confluent_config: ConfluentConfig,
        consumer_config: ConsumerConfig,
    ) -> KafkaConsumerProtocol:
        """Create a Kafka consumer.

        Args:
            confluent_config: Confluent Cloud connection settings.
            consumer_config: Consumer-specific settings.

        Returns:
            KafkaConsumerProtocol implementation.
        """
        ...


# =============================================================================
# Real Producer Implementation
# =============================================================================


class RealKafkaProducer:
    """Real Kafka producer using confluent-kafka.

    Wraps confluent_kafka.Producer with Protocol-compatible interface.
    """

    def __init__(
        self,
        confluent_config: ConfluentConfig,
        producer_config: ProducerConfig,
    ) -> None:
        """Initialize producer with Confluent Cloud settings.

        Args:
            confluent_config: Confluent Cloud connection settings.
            producer_config: Producer-specific settings.
        """
        config: KafkaConfigDict = {
            "bootstrap.servers": confluent_config["bootstrap_servers"],
            "security.protocol": confluent_config["security_protocol"],
            "sasl.mechanisms": confluent_config["sasl_mechanism"],
            "sasl.username": confluent_config["api_key"],
            "sasl.password": confluent_config["api_secret"],
            "acks": producer_config["acks"],
            "retries": producer_config["retries"],
            "linger.ms": producer_config["linger_ms"],
            "batch.size": producer_config["batch_size"],
            "compression.type": producer_config["compression_type"],
        }

        confluent_kafka = _get_confluent_kafka()
        producer_constructor: RawProducerConstructor = confluent_kafka.Producer
        self._producer: RawKafkaProducerProtocol = producer_constructor(config)

    def produce(
        self,
        topic: str,
        value: bytes,
        key: bytes | None = None,
    ) -> None:
        """Produce a message to a topic.

        Args:
            topic: Target topic name.
            value: Message payload as bytes.
            key: Optional message key for partitioning.
        """
        self._producer.produce(topic=topic, value=value, key=key)

    def flush(self, timeout_seconds: float) -> int:
        """Flush pending messages.

        Args:
            timeout_seconds: Maximum time to wait.

        Returns:
            Number of messages still in queue.
        """
        return self._producer.flush(timeout_seconds)

    def poll(self, timeout_seconds: float) -> int:
        """Poll for delivery reports.

        Args:
            timeout_seconds: Maximum time to wait.

        Returns:
            Number of events processed.
        """
        return self._producer.poll(timeout_seconds)


# =============================================================================
# Real Consumer Implementation
# =============================================================================


class RealConsumedMessage:
    """Wrapper for confluent_kafka Message with Protocol interface."""

    def __init__(self, message: RawKafkaMessageProtocol) -> None:
        """Initialize wrapper.

        Args:
            message: Raw confluent_kafka Message.
        """
        self._message = message

    def value(self) -> bytes:
        """Get message value."""
        raw = self._message.value()
        if raw is None:
            return b""
        return raw

    def key(self) -> bytes | None:
        """Get message key."""
        return self._message.key()

    def topic(self) -> str:
        """Get topic name."""
        result = self._message.topic()
        if result is None:
            return ""
        return result

    def partition(self) -> int:
        """Get partition number."""
        return self._message.partition()

    def offset(self) -> int:
        """Get message offset."""
        return self._message.offset()


class RealKafkaConsumer:
    """Real Kafka consumer using confluent-kafka.

    Wraps confluent_kafka.Consumer with Protocol-compatible interface.
    """

    def __init__(
        self,
        confluent_config: ConfluentConfig,
        consumer_config: ConsumerConfig,
    ) -> None:
        """Initialize consumer with Confluent Cloud settings.

        Args:
            confluent_config: Confluent Cloud connection settings.
            consumer_config: Consumer-specific settings.
        """
        config: KafkaConfigDict = {
            "bootstrap.servers": confluent_config["bootstrap_servers"],
            "security.protocol": confluent_config["security_protocol"],
            "sasl.mechanisms": confluent_config["sasl_mechanism"],
            "sasl.username": confluent_config["api_key"],
            "sasl.password": confluent_config["api_secret"],
            "group.id": consumer_config["group_id"],
            "auto.offset.reset": consumer_config["auto_offset_reset"],
            "enable.auto.commit": consumer_config["enable_auto_commit"],
            "fetch.min.bytes": consumer_config["fetch_min_bytes"],
            "session.timeout.ms": consumer_config["session_timeout_ms"],
            "heartbeat.interval.ms": consumer_config["heartbeat_interval_ms"],
        }

        confluent_kafka = _get_confluent_kafka()
        consumer_constructor: RawConsumerConstructor = confluent_kafka.Consumer
        self._consumer: RawKafkaConsumerProtocol = consumer_constructor(config)

    def subscribe(self, topics: tuple[str, ...]) -> None:
        """Subscribe to topics.

        Args:
            topics: Tuple of topic names.
        """
        # Real consumer expects list, our Protocol uses tuple for immutability
        self._consumer.subscribe(list(topics))

    def poll(self, timeout_seconds: float) -> ConsumedMessageProtocol | None:
        """Poll for a single message.

        Args:
            timeout_seconds: Maximum time to wait.

        Returns:
            Consumed message or None.
        """
        msg = self._consumer.poll(timeout_seconds)
        if msg is None:
            return None
        # Check for errors - confluent-kafka returns error messages too
        # The RawKafkaMessageProtocol defines error() -> KafkaErrorProtocol | None
        if msg.error() is not None:
            return None
        return RealConsumedMessage(msg)

    def commit(self, offsets: tuple[TopicPartitionOffset, ...]) -> None:
        """Commit the given positions synchronously.

        Args:
            offsets: Positions to commit. An empty tuple is a no-op; librdkafka
                rejects a commit carrying no partitions.
        """
        if len(offsets) == 0:
            return
        confluent_kafka = _get_confluent_kafka()
        topic_partition: RawTopicPartitionConstructor = confluent_kafka.TopicPartition
        raw_offsets: list[RawTopicPartitionProtocol] = []
        for position in offsets:
            raw_offsets.append(
                topic_partition(
                    position["topic"],
                    position["partition"],
                    position["offset"],
                )
            )
        self._consumer.commit(offsets=raw_offsets, asynchronous=False)

    def close(self) -> None:
        """Close consumer."""
        self._consumer.close()


# =============================================================================
# Real Factory Functions
# =============================================================================


def _real_producer_factory(
    confluent_config: ConfluentConfig,
    producer_config: ProducerConfig,
) -> KafkaProducerProtocol:
    """Create a real Kafka producer.

    Args:
        confluent_config: Confluent Cloud connection settings.
        producer_config: Producer-specific settings.

    Returns:
        RealKafkaProducer instance.
    """
    return RealKafkaProducer(confluent_config, producer_config)


def _real_consumer_factory(
    confluent_config: ConfluentConfig,
    consumer_config: ConsumerConfig,
) -> KafkaConsumerProtocol:
    """Create a real Kafka consumer.

    Args:
        confluent_config: Confluent Cloud connection settings.
        consumer_config: Consumer-specific settings.

    Returns:
        RealKafkaConsumer instance.
    """
    return RealKafkaConsumer(confluent_config, consumer_config)


# =============================================================================
# Module-Level Injectable Hooks
# =============================================================================

# Production code calls these; tests override before calling.
producer_factory: ProducerFactoryProtocol = _real_producer_factory
consumer_factory: ConsumerFactoryProtocol = _real_consumer_factory


# =============================================================================
# Fake Implementations for Testing
# =============================================================================


class FakeConsumedMessage:
    """Fake consumed message for testing.

    Stores message data in memory without Kafka dependency.
    """

    def __init__(
        self,
        value: bytes,
        key: bytes | None,
        topic: str,
        partition: int,
        offset: int,
    ) -> None:
        """Initialize fake message.

        Args:
            value: Message payload.
            key: Message key.
            topic: Topic name.
            partition: Partition number.
            offset: Message offset.
        """
        self._value = value
        self._key = key
        self._topic = topic
        self._partition = partition
        self._offset = offset

    def value(self) -> bytes:
        """Get message value."""
        return self._value

    def key(self) -> bytes | None:
        """Get message key."""
        return self._key

    def topic(self) -> str:
        """Get topic name."""
        return self._topic

    def partition(self) -> int:
        """Get partition number."""
        return self._partition

    def offset(self) -> int:
        """Get message offset."""
        return self._offset


class ProducedMessage:
    """Record of a produced message for test verification."""

    def __init__(
        self,
        topic: str,
        value: bytes,
        key: bytes | None,
    ) -> None:
        """Initialize produced message record.

        Args:
            topic: Target topic.
            value: Message payload.
            key: Message key.
        """
        self.topic = topic
        self.value = value
        self.key = key


class FakeKafkaProducer:
    """Fake Kafka producer for testing.

    Stores produced messages in memory for verification.
    """

    def __init__(self) -> None:
        """Initialize fake producer."""
        self.messages: list[ProducedMessage] = []
        self.flush_called = False
        self.poll_count = 0

    def produce(
        self,
        topic: str,
        value: bytes,
        key: bytes | None = None,
    ) -> None:
        """Store produced message.

        Args:
            topic: Target topic.
            value: Message payload.
            key: Message key.
        """
        self.messages.append(ProducedMessage(topic, value, key))

    def flush(self, timeout_seconds: float) -> int:
        """Mark flush as called.

        Args:
            timeout_seconds: Ignored in fake.

        Returns:
            Always 0 (all flushed).
        """
        self.flush_called = True
        return 0

    def poll(self, timeout_seconds: float) -> int:
        """Increment poll counter.

        Args:
            timeout_seconds: Ignored in fake.

        Returns:
            Always 0.
        """
        self.poll_count += 1
        return 0


class FakeKafkaConsumer:
    """Fake Kafka consumer for testing.

    Returns pre-configured messages from queue.
    """

    def __init__(self) -> None:
        """Initialize fake consumer."""
        self.subscribed_topics: tuple[str, ...] = ()
        self.message_queue: list[FakeConsumedMessage] = []
        self.commit_count = 0
        self.committed_offsets: list[tuple[TopicPartitionOffset, ...]] = []
        self.closed = False
        self.poll_count = 0
        self._on_poll: Callable[[], None] | None = None

    def subscribe(self, topics: tuple[str, ...]) -> None:
        """Record subscribed topics.

        Args:
            topics: Topics to subscribe to.
        """
        self.subscribed_topics = topics

    def poll(self, timeout_seconds: float) -> ConsumedMessageProtocol | None:
        """Return next message from queue.

        Args:
            timeout_seconds: Ignored in fake.

        Returns:
            Next message or None if queue empty.
        """
        self.poll_count += 1
        if self._on_poll is not None:
            self._on_poll()
        if not self.message_queue:
            return None
        return self.message_queue.pop(0)

    def set_on_poll(self, callback: Callable[[], None] | None) -> None:
        """Set callback to invoke on each poll.

        Args:
            callback: Function to call on poll, or None to clear.
        """
        self._on_poll = callback

    def commit(self, offsets: tuple[TopicPartitionOffset, ...]) -> None:
        """Record the committed positions.

        Args:
            offsets: Positions the worker asked to commit.
        """
        self.commit_count += 1
        self.committed_offsets.append(offsets)

    def close(self) -> None:
        """Mark consumer as closed."""
        self.closed = True

    def add_message(
        self,
        value: bytes,
        key: bytes | None = None,
        topic: str = "test-topic",
        partition: int = 0,
        offset: int = 0,
    ) -> None:
        """Add a message to the queue for testing.

        Args:
            value: Message payload.
            key: Message key.
            topic: Topic name.
            partition: Partition number.
            offset: Message offset.
        """
        msg = FakeConsumedMessage(value, key, topic, partition, offset)
        self.message_queue.append(msg)


# Global fake instances for test access
_fake_producer: FakeKafkaProducer | None = None
_fake_consumer: FakeKafkaConsumer | None = None


def _fake_producer_factory(
    confluent_config: ConfluentConfig,
    producer_config: ProducerConfig,
) -> KafkaProducerProtocol:
    """Create fake producer for testing.

    Args:
        confluent_config: Ignored in fake.
        producer_config: Ignored in fake.

    Returns:
        FakeKafkaProducer instance.
    """
    global _fake_producer
    _fake_producer = FakeKafkaProducer()
    return _fake_producer


def _fake_consumer_factory(
    confluent_config: ConfluentConfig,
    consumer_config: ConsumerConfig,
) -> KafkaConsumerProtocol:
    """Create fake consumer for testing.

    Args:
        confluent_config: Ignored in fake.
        consumer_config: Ignored in fake.

    Returns:
        FakeKafkaConsumer instance.
    """
    global _fake_consumer
    _fake_consumer = FakeKafkaConsumer()
    return _fake_consumer


def get_fake_producer() -> FakeKafkaProducer | None:
    """Get the current fake producer instance.

    Returns:
        FakeKafkaProducer if one was created, None otherwise.
    """
    return _fake_producer


def get_fake_consumer() -> FakeKafkaConsumer | None:
    """Get the current fake consumer instance.

    Returns:
        FakeKafkaConsumer if one was created, None otherwise.
    """
    return _fake_consumer


def use_fake_kafka() -> None:
    """Configure module to use fake Kafka implementations.

    Call this in test setup to inject fake producer/consumer factories.
    """
    global producer_factory, consumer_factory
    producer_factory = _fake_producer_factory
    consumer_factory = _fake_consumer_factory


def use_real_kafka() -> None:
    """Configure module to use real Kafka implementations.

    Call this at application startup to use real confluent-kafka.
    """
    global producer_factory, consumer_factory
    producer_factory = _real_producer_factory
    consumer_factory = _real_consumer_factory


__all__ = [
    "ConsumedMessageProtocol",
    "ConsumerFactoryProtocol",
    "FakeConsumedMessage",
    "FakeCovenantRepository",
    "FakeCovenantResultRepository",
    "FakeDealRepository",
    "FakeKafkaConsumer",
    "FakeKafkaProducer",
    "FakeMeasurementRepository",
    "FakeMetricsSink",
    "FakePredictor",
    "KafkaConsumerProtocol",
    "KafkaProducerProtocol",
    "ProducedMessage",
    "ProducerFactoryProtocol",
    "RealConsumedMessage",
    "RealKafkaConsumer",
    "RealKafkaProducer",
    "consumer_factory",
    "get_fake_consumer",
    "get_fake_producer",
    "producer_factory",
    "use_fake_kafka",
    "use_real_kafka",
]
