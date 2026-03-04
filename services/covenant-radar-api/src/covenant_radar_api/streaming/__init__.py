"""Kafka streaming infrastructure for covenant monitoring.

This package provides TypedDict-based Kafka integration for:
- Consuming measurement events from Confluent Cloud
- Producing prediction and alert events
- Strict schema validation with encode/decode functions
- Streaming inference worker for real-time prediction
- Generic streaming worker for multi-domain ML prediction

Modules:
- config: Configuration TypedDicts and environment parsing
- schemas: Kafka event TypedDicts with encode/decode/TypeGuards
- producer: High-level producer wrapper
- consumer: High-level consumer wrapper
- worker: Streaming inference worker (covenant-specific)
- generic_worker: Domain-agnostic streaming worker
- _test_hooks: Dependency injection for testing (private)

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from .config import (
    DEFAULT_ALERTS_TOPIC,
    DEFAULT_MEASUREMENTS_TOPIC,
    DEFAULT_PREDICTIONS_TOPIC,
    ConfluentConfig,
    ConfluentSchemaRegistryConfig,
    ConsumerConfig,
    KafkaTopicsConfig,
    ProducerConfig,
    StreamingConfig,
    load_streaming_config,
)
from .consumer import (
    ConsumedMeasurement,
    StreamingConsumer,
    create_consumer_from_parts,
    create_streaming_consumer,
)
from .generic_worker import (
    GenericProcessingResult,
    GenericStreamingWorker,
    GenericWorkerConfig,
    make_generic_worker_config,
)
from .producer import (
    StreamingProducer,
    create_producer_from_parts,
    create_streaming_producer,
)
from .schemas import (
    AlertEventType,
    AlertEventV1,
    AlertSeverity,
    AlertType,
    EvaluationStatus,
    KafkaEventType,
    KafkaEventV1,
    MeasurementEventType,
    MeasurementEventV1,
    PredictionEventType,
    PredictionEventV1,
    RiskTier,
    classify_risk_tier,
    decode_alert_event,
    decode_kafka_event,
    decode_measurement_event,
    decode_prediction_event,
    encode_alert_event,
    encode_kafka_event,
    encode_measurement_event,
    encode_prediction_event,
    is_alert_event,
    is_measurement_event,
    is_prediction_event,
    make_alert_event,
    make_measurement_event,
    make_prediction_event,
)
from .worker import (
    ProcessingResult,
    StreamingWorker,
    WorkerConfig,
    make_default_worker_config,
)

__all__ = [
    "DEFAULT_ALERTS_TOPIC",
    "DEFAULT_MEASUREMENTS_TOPIC",
    "DEFAULT_PREDICTIONS_TOPIC",
    "AlertEventType",
    "AlertEventV1",
    "AlertSeverity",
    "AlertType",
    "ConfluentConfig",
    "ConfluentSchemaRegistryConfig",
    "ConsumedMeasurement",
    "ConsumerConfig",
    "EvaluationStatus",
    "GenericProcessingResult",
    "GenericStreamingWorker",
    "GenericWorkerConfig",
    "KafkaEventType",
    "KafkaEventV1",
    "KafkaTopicsConfig",
    "MeasurementEventType",
    "MeasurementEventV1",
    "PredictionEventType",
    "PredictionEventV1",
    "ProcessingResult",
    "ProducerConfig",
    "RiskTier",
    "StreamingConfig",
    "StreamingConsumer",
    "StreamingProducer",
    "StreamingWorker",
    "WorkerConfig",
    "classify_risk_tier",
    "create_consumer_from_parts",
    "create_producer_from_parts",
    "create_streaming_consumer",
    "create_streaming_producer",
    "decode_alert_event",
    "decode_kafka_event",
    "decode_measurement_event",
    "decode_prediction_event",
    "encode_alert_event",
    "encode_kafka_event",
    "encode_measurement_event",
    "encode_prediction_event",
    "is_alert_event",
    "is_measurement_event",
    "is_prediction_event",
    "load_streaming_config",
    "make_alert_event",
    "make_default_worker_config",
    "make_generic_worker_config",
    "make_measurement_event",
    "make_prediction_event",
]
