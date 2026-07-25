"""Kafka consumer wrapper for streaming pipeline.

This module provides a high-level wrapper around the Kafka consumer
for consuming measurement events from Confluent Cloud.

Uses dependency injection via _test_hooks for testability.
Strict typing only: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import InvalidJsonError, JSONTypeError
from platform_core.logging import get_logger

from . import _test_hooks
from ._test_hooks import (
    ConsumedMessageProtocol,
    KafkaConsumerProtocol,
    TopicPartitionOffset,
)
from .config import ConfluentConfig, ConsumerConfig, StreamingConfig
from .schemas import MeasurementEventV1, decode_measurement_event

_log = get_logger(__name__)


class ConsumedMeasurement(TypedDict):
    """Result of consuming a measurement event.

    Combines the deserialized event with Kafka metadata for
    tracking and debugging.

    Fields:
        kind: Discriminator distinguishing this from an UndecodableMessage.
        event: Deserialized measurement event.
        topic: Topic the message was consumed from.
        partition: Partition number.
        offset: Message offset within partition.
        key: Message key (deal_id) or None.
    """

    kind: Literal["measurement"]
    event: MeasurementEventV1
    topic: str
    partition: int
    offset: int
    key: str | None


class UndecodableMessage(TypedDict):
    """A polled message whose payload is not a valid measurement event.

    Returned instead of raising so the worker can route the message to the
    dead-letter topic and advance past it. Raising here would abort the poll
    loop, and because the offset would never advance the same message would be
    redelivered on every restart.

    Fields:
        kind: Discriminator distinguishing this from a ConsumedMeasurement.
        payload: Raw payload decoded as UTF-8, invalid bytes replaced, so the
            value is always renderable and always serialisable.
        reason: Human-readable description of why decoding failed.
        topic: Topic the message was consumed from.
        partition: Partition number.
        offset: Message offset within partition.
    """

    kind: Literal["undecodable"]
    payload: str
    reason: str
    topic: str
    partition: int
    offset: int


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

    def poll(self, timeout_seconds: float = 1.0) -> ConsumedMeasurement | UndecodableMessage | None:
        """Poll for a single measurement event.

        Automatically subscribes if not already subscribed.

        A payload that does not decode is returned as an UndecodableMessage
        rather than raised. That is deliberate: the caller needs the raw bytes
        and the Kafka position in order to dead-letter the message and move
        past it, neither of which survives an exception.

        Args:
            timeout_seconds: Maximum time to wait for a message.

        Returns:
            ConsumedMeasurement on success, UndecodableMessage if the payload
            is not a valid measurement event, or None if the poll timed out.
        """
        if not self._subscribed:
            self.subscribe()

        message = self._consumer.poll(timeout_seconds)
        if message is None:
            return None

        return self._deserialize_or_report(message)

    def _deserialize_or_report(
        self,
        message: ConsumedMessageProtocol,
    ) -> ConsumedMeasurement | UndecodableMessage:
        """Decode a message, reporting failure as a value rather than raising.

        This is the pipeline's single decode boundary. JSONTypeError,
        InvalidJsonError and UnicodeDecodeError all describe bad input data
        rather than a defect in this service, so they are converted to an
        UndecodableMessage here and handled explicitly by the caller. Any other
        exception is a real defect and propagates untouched.

        UnicodeDecodeError is included because the payload is decoded before it
        is parsed: a tombstone or a corrupt byte sequence fails there, ahead of
        any JSON error, and would otherwise still take the worker down.

        Args:
            message: Raw Kafka message.

        Returns:
            ConsumedMeasurement, or UndecodableMessage describing the failure.
        """
        try:
            return self._deserialize_message(message)
        except (JSONTypeError, InvalidJsonError, UnicodeDecodeError) as exc:
            _log.warning(
                "Undecodable message routed for dead-lettering",
                extra={
                    "topic": message.topic(),
                    "partition": str(message.partition()),
                    "offset": str(message.offset()),
                    "reason": str(exc),
                },
            )
            return {
                "kind": "undecodable",
                "payload": message.value().decode("utf-8", errors="replace"),
                "reason": str(exc),
                "topic": message.topic(),
                "partition": message.partition(),
                "offset": message.offset(),
            }

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
            "kind": "measurement",
            "event": event,
            "topic": message.topic(),
            "partition": message.partition(),
            "offset": message.offset(),
            "key": key,
        }

    def commit(self, offsets: tuple[TopicPartitionOffset, ...]) -> None:
        """Commit explicit positions synchronously.

        Positions must be supplied by the caller, which is the only component
        that knows which messages have actually been processed. An
        argument-less commit would advance every assigned partition to its
        consumed position, acknowledging messages still held in memory.

        Args:
            offsets: Positions to commit, one per topic partition.
        """
        self._consumer.commit(offsets)

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
