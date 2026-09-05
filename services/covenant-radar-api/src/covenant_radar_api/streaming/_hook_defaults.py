"""Default (production) implementations for covenant_radar_api.streaming hooks."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

from covenant_radar_api.streaming._hook_protocols import (
    ConsumedMessageProtocol,
    KafkaConsumerProtocol,
    KafkaProducerProtocol,
    RawKafkaConsumerProtocol,
    RawKafkaMessageProtocol,
    RawKafkaProducerProtocol,
    RawTopicPartitionConstructor,
    RawTopicPartitionProtocol,
    TopicPartitionOffset,
)

from .config import ConfluentConfig, ConsumerConfig, ProducerConfig

KafkaConfigDict = dict[str, str | int | bool]


def _get_confluent_kafka() -> ModuleType:
    """Get the confluent_kafka module.

    Returns:
        The confluent_kafka module.
    """
    return __import__("confluent_kafka")


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


def _connection_config(confluent_config: ConfluentConfig) -> KafkaConfigDict:
    """Build the connection and authentication keys shared by both clients.

    Under PLAINTEXT the sasl.* keys are omitted entirely rather than sent
    empty. librdkafka validates them against the selected protocol, so an
    empty sasl.username with a SASL protocol is rejected at construction, and
    sending SASL credentials to an unauthenticated broker is meaningless.

    Args:
        confluent_config: Kafka connection settings.

    Returns:
        Config keys common to the producer and consumer.
    """
    config: KafkaConfigDict = {
        "bootstrap.servers": confluent_config["bootstrap_servers"],
        "security.protocol": confluent_config["security_protocol"],
    }
    if confluent_config["security_protocol"] == "SASL_SSL":
        config["sasl.mechanisms"] = confluent_config["sasl_mechanism"]
        config["sasl.username"] = confluent_config["api_key"]
        config["sasl.password"] = confluent_config["api_secret"]
    return config


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
        config: KafkaConfigDict = _connection_config(confluent_config)
        config["acks"] = producer_config["acks"]
        config["retries"] = producer_config["retries"]
        config["linger.ms"] = producer_config["linger_ms"]
        config["batch.size"] = producer_config["batch_size"]
        config["compression.type"] = producer_config["compression_type"]

        confluent_kafka = _get_confluent_kafka()
        producer_constructor: Callable[[KafkaConfigDict], RawKafkaProducerProtocol] = (
            confluent_kafka.Producer
        )
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
        config: KafkaConfigDict = _connection_config(confluent_config)
        config["group.id"] = consumer_config["group_id"]
        config["auto.offset.reset"] = consumer_config["auto_offset_reset"]
        config["enable.auto.commit"] = consumer_config["enable_auto_commit"]
        config["fetch.min.bytes"] = consumer_config["fetch_min_bytes"]
        config["session.timeout.ms"] = consumer_config["session_timeout_ms"]
        config["heartbeat.interval.ms"] = consumer_config["heartbeat_interval_ms"]

        confluent_kafka = _get_confluent_kafka()
        consumer_constructor: Callable[[KafkaConfigDict], RawKafkaConsumerProtocol] = (
            confluent_kafka.Consumer
        )
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
