"""Kafka producer wrapper for streaming pipeline.

This module provides a high-level wrapper around the Kafka producer
for publishing prediction and alert events to Confluent Cloud.

Uses dependency injection via _test_hooks for testability.
Strict typing only: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from . import _test_hooks
from ._test_hooks import KafkaProducerProtocol
from .config import ConfluentConfig, ProducerConfig, StreamingConfig
from .schemas import (
    AlertEventV1,
    DlqEventV1,
    KafkaEventV1,
    PredictionEventV1,
    encode_alert_event,
    encode_dlq_event,
    encode_kafka_event,
    encode_prediction_event,
)


class StreamingProducer:
    """High-level Kafka producer for covenant streaming events.

    Wraps the low-level Kafka producer protocol with typed methods
    for producing prediction and alert events.

    Uses string keys (deal_id) for consistent partitioning.
    """

    def __init__(
        self,
        producer: KafkaProducerProtocol,
        predictions_topic: str,
        alerts_topic: str,
        dlq_topic: str,
    ) -> None:
        """Initialize streaming producer.

        Args:
            producer: Underlying Kafka producer.
            predictions_topic: Topic name for prediction events.
            alerts_topic: Topic name for alert events.
            dlq_topic: Topic name for dead-lettered messages.
        """
        self._producer = producer
        self._predictions_topic = predictions_topic
        self._alerts_topic = alerts_topic
        self._dlq_topic = dlq_topic

    def produce_prediction(self, event: PredictionEventV1) -> None:
        """Produce a prediction event.

        Serializes the event to JSON and publishes to the predictions topic.
        Uses deal_id as the message key for partition affinity.

        Args:
            event: Prediction event to publish.
        """
        value = encode_prediction_event(event).encode("utf-8")
        key = event["deal_id"].encode("utf-8")
        self._producer.produce(
            topic=self._predictions_topic,
            value=value,
            key=key,
        )

    def produce_alert(self, event: AlertEventV1) -> None:
        """Produce an alert event.

        Serializes the event to JSON and publishes to the alerts topic.
        Uses deal_id as the message key for partition affinity.

        Args:
            event: Alert event to publish.
        """
        value = encode_alert_event(event).encode("utf-8")
        key = event["deal_id"].encode("utf-8")
        self._producer.produce(
            topic=self._alerts_topic,
            value=value,
            key=key,
        )

    def produce_dlq(self, event: DlqEventV1) -> None:
        """Produce a dead-letter event.

        Keyed by source topic-partition rather than deal_id, because a
        dead-lettered message may be one whose deal could not be identified.

        Args:
            event: Dead-letter event to publish.
        """
        value = encode_dlq_event(event).encode("utf-8")
        key = f"{event['source_topic']}:{event['source_partition']}".encode()
        self._producer.produce(
            topic=self._dlq_topic,
            value=value,
            key=key,
        )

    def produce_event(self, event: KafkaEventV1, topic: str) -> None:
        """Produce any Kafka event to a specified topic.

        Serializes the event to JSON and publishes to the given topic.
        Uses deal_id as the message key for partition affinity.

        Args:
            event: Any Kafka event to publish.
            topic: Target topic name.
        """
        value = encode_kafka_event(event).encode("utf-8")
        key = event["deal_id"].encode("utf-8")
        self._producer.produce(
            topic=topic,
            value=value,
            key=key,
        )

    def flush(self, timeout_seconds: float = 10.0) -> int:
        """Flush pending messages.

        Blocks until all buffered messages are delivered or timeout.

        Args:
            timeout_seconds: Maximum time to wait for flush.

        Returns:
            Number of messages still in queue (0 if all flushed).
        """
        return self._producer.flush(timeout_seconds)

    def poll(self, timeout_seconds: float = 0.0) -> int:
        """Poll for delivery reports.

        Should be called periodically to handle delivery callbacks.

        Args:
            timeout_seconds: Maximum time to wait for events.

        Returns:
            Number of events processed.
        """
        return self._producer.poll(timeout_seconds)


def create_streaming_producer(
    config: StreamingConfig,
) -> StreamingProducer:
    """Create a streaming producer from configuration.

    Uses the injected producer factory from _test_hooks to create
    the underlying Kafka producer.

    Args:
        config: Complete streaming configuration.

    Returns:
        Configured StreamingProducer instance.
    """
    producer = _test_hooks.producer_factory(
        config["confluent"],
        config["producer"],
    )
    return StreamingProducer(
        producer=producer,
        predictions_topic=config["topics"]["predictions"],
        alerts_topic=config["topics"]["alerts"],
        dlq_topic=config["topics"]["dlq"],
    )


def create_producer_from_parts(
    confluent_config: ConfluentConfig,
    producer_config: ProducerConfig,
    predictions_topic: str,
    alerts_topic: str,
    dlq_topic: str,
) -> StreamingProducer:
    """Create a streaming producer from individual config parts.

    Useful when configuration components are provided separately.

    Args:
        confluent_config: Confluent Cloud connection settings.
        producer_config: Producer-specific settings.
        predictions_topic: Topic name for predictions.
        alerts_topic: Topic name for alerts.
        dlq_topic: Topic name for dead-lettered messages.

    Returns:
        Configured StreamingProducer instance.
    """
    producer = _test_hooks.producer_factory(confluent_config, producer_config)
    return StreamingProducer(
        producer=producer,
        predictions_topic=predictions_topic,
        alerts_topic=alerts_topic,
        dlq_topic=dlq_topic,
    )


__all__ = [
    "StreamingProducer",
    "create_producer_from_parts",
    "create_streaming_producer",
]
