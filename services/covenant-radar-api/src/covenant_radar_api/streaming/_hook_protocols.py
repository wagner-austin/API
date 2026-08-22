"""Hook protocols for covenant_radar_api.streaming."""

from __future__ import annotations

from typing import Protocol, TypedDict

from .config import ConfluentConfig, ConsumerConfig, ProducerConfig


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
