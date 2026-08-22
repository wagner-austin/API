"""Test hooks for Kafka streaming infrastructure.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Callable

from covenant_radar_api.streaming._hook_defaults import (
    RealKafkaConsumer,
    RealKafkaProducer,
    _real_consumer_factory,
    _real_producer_factory,
)
from covenant_radar_api.streaming._hook_protocols import (
    ConsumedMessageProtocol,
    ConsumerFactoryProtocol,
    KafkaConsumerProtocol,
    KafkaProducerProtocol,
    ProducerFactoryProtocol,
    TopicPartitionOffset,
)

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

producer_factory: ProducerFactoryProtocol = _real_producer_factory

consumer_factory: ConsumerFactoryProtocol = _real_consumer_factory


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
    "RealKafkaConsumer",
    "RealKafkaProducer",
    "consumer_factory",
    "get_fake_consumer",
    "get_fake_producer",
    "producer_factory",
    "use_fake_kafka",
    "use_real_kafka",
]
