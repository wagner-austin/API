"""Kafka consumer wrapper for streaming pipeline.

This module provides a high-level wrapper around the Kafka consumer
for consuming measurement events from Confluent Cloud.

Uses dependency injection via _test_hooks for testability.
Strict typing only: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import TypedDict

from . import _test_hooks
from ._test_hooks import (
    ConsumedMessageProtocol,
    KafkaConsumerProtocol,
)
from .config import ConfluentConfig, ConsumerConfig, StreamingConfig
from .schemas import MeasurementEventV1, decode_measurement_event


class ConsumedMeasurement(TypedDict):
    """Result of consuming a measurement event.

    Combines the deserialized event with Kafka metadata for
    tracking and debugging.

    Fields:
        event: Deserialized measurement event.
        topic: Topic the message was consumed from.
        partition: Partition number.
        offset: Message offset within partition.
        key: Message key (deal_id) or None.
    """

    event: MeasurementEventV1
    topic: str
    partition: int
    offset: int
    key: str | None


class StreamingConsumer:
    """High-level Kafka consumer for covenant measurement events.

    Wraps the low-level Kafka consumer protocol with typed methods
    for consuming and deserializing measurement events.

    Handles JSON deserialization and provides Kafka metadata alongside
    the event payload.
    """

    def __init__(
        self,
        consumer: KafkaConsumerProtocol,
        measurements_topic: str,
    ) -> None:
        """Initialize streaming consumer.

        Args:
            consumer: Underlying Kafka consumer.
            measurements_topic: Topic name for measurement events.
        """
        self._consumer = consumer
        self._measurements_topic = measurements_topic
        self._subscribed = False

    def subscribe(self) -> None:
        """Subscribe to the measurements topic.

        Must be called before polling for messages. Safe to call
        multiple times (idempotent).
        """
        if not self._subscribed:
            self._consumer.subscribe((self._measurements_topic,))
            self._subscribed = True

    def poll(self, timeout_seconds: float = 1.0) -> ConsumedMeasurement | None:
        """Poll for a single measurement event.

        Automatically subscribes if not already subscribed.

        Args:
            timeout_seconds: Maximum time to wait for a message.

        Returns:
            ConsumedMeasurement with event and metadata, or None if timeout.

        Raises:
            JSONTypeError: If message payload is not a valid measurement event.
        """
        if not self._subscribed:
            self.subscribe()

        message = self._consumer.poll(timeout_seconds)
        if message is None:
            return None

        return self._deserialize_message(message)

    def poll_batch(
        self,
        max_messages: int,
        timeout_seconds: float = 1.0,
    ) -> tuple[ConsumedMeasurement, ...]:
        """Poll for multiple measurement events.

        Polls up to max_messages, returning early if timeout reached
        or no more messages available.

        Args:
            max_messages: Maximum number of messages to return.
            timeout_seconds: Timeout for each individual poll.

        Returns:
            Tuple of ConsumedMeasurement instances (may be empty).

        Raises:
            JSONTypeError: If any message payload is not valid.
        """
        if not self._subscribed:
            self.subscribe()

        messages: list[ConsumedMeasurement] = []
        for _ in range(max_messages):
            message = self._consumer.poll(timeout_seconds)
            if message is None:
                break
            messages.append(self._deserialize_message(message))

        return tuple(messages)

    def _deserialize_message(
        self,
        message: ConsumedMessageProtocol,
    ) -> ConsumedMeasurement:
        """Deserialize a Kafka message to ConsumedMeasurement.

        Args:
            message: Raw Kafka message.

        Returns:
            ConsumedMeasurement with event and metadata.

        Raises:
            JSONTypeError: If payload is not a valid measurement event.
        """
        payload = message.value().decode("utf-8")
        event = decode_measurement_event(payload)

        key_bytes = message.key()
        key = key_bytes.decode("utf-8") if key_bytes else None

        return {
            "event": event,
            "topic": message.topic(),
            "partition": message.partition(),
            "offset": message.offset(),
            "key": key,
        }

    def commit(self) -> None:
        """Commit current offsets synchronously.

        Should be called after successfully processing a batch of
        messages to mark them as consumed.
        """
        self._consumer.commit()

    def close(self) -> None:
        """Close the consumer and leave consumer group.

        Should be called during graceful shutdown to ensure
        clean consumer group rebalancing.
        """
        self._consumer.close()
        self._subscribed = False

    @property
    def is_subscribed(self) -> bool:
        """Check if consumer is currently subscribed.

        Returns:
            True if subscribed, False otherwise.
        """
        return self._subscribed


def create_streaming_consumer(
    config: StreamingConfig,
) -> StreamingConsumer:
    """Create a streaming consumer from configuration.

    Uses the injected consumer factory from _test_hooks to create
    the underlying Kafka consumer.

    Args:
        config: Complete streaming configuration.

    Returns:
        Configured StreamingConsumer instance.
    """
    consumer = _test_hooks.consumer_factory(
        config["confluent"],
        config["consumer"],
    )
    return StreamingConsumer(
        consumer=consumer,
        measurements_topic=config["topics"]["measurements"],
    )


def create_consumer_from_parts(
    confluent_config: ConfluentConfig,
    consumer_config: ConsumerConfig,
    measurements_topic: str,
) -> StreamingConsumer:
    """Create a streaming consumer from individual config parts.

    Useful when configuration components are provided separately.

    Args:
        confluent_config: Confluent Cloud connection settings.
        consumer_config: Consumer-specific settings.
        measurements_topic: Topic name for measurements.

    Returns:
        Configured StreamingConsumer instance.
    """
    consumer = _test_hooks.consumer_factory(confluent_config, consumer_config)
    return StreamingConsumer(
        consumer=consumer,
        measurements_topic=measurements_topic,
    )


__all__ = [
    "ConsumedMeasurement",
    "StreamingConsumer",
    "create_consumer_from_parts",
    "create_streaming_consumer",
]
