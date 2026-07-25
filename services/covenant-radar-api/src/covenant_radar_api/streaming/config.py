"""Configuration TypedDicts for Kafka streaming infrastructure.

This module provides TypedDict definitions for Confluent Cloud Kafka
configuration and environment variable parsing.

All configuration is immutable and strictly typed with no Any types.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.config._utils import _parse_bool, _parse_int, _parse_str

# =============================================================================
# Confluent Cloud Configuration
# =============================================================================


class ConfluentConfig(TypedDict):
    """Confluent Cloud Kafka connection configuration.

    All fields are required for production Kafka connectivity via Confluent Cloud.
    Authentication uses SASL/PLAIN with API key and secret.

    Fields:
        bootstrap_servers: Confluent Cloud bootstrap server endpoint.
        api_key: SASL username for authentication.
        api_secret: SASL password for authentication.
        security_protocol: Security protocol (always SASL_SSL for Confluent Cloud).
        sasl_mechanism: SASL mechanism (always PLAIN for Confluent Cloud).
    """

    bootstrap_servers: str
    api_key: str
    api_secret: str
    security_protocol: Literal["SASL_SSL"]
    sasl_mechanism: Literal["PLAIN"]


class ConfluentSchemaRegistryConfig(TypedDict):
    """Confluent Schema Registry configuration.

    Used for schema validation and evolution if Schema Registry is enabled.

    Fields:
        url: Schema Registry endpoint URL.
        api_key: Schema Registry API key.
        api_secret: Schema Registry API secret.
    """

    url: str
    api_key: str
    api_secret: str


# =============================================================================
# Kafka Topic Configuration
# =============================================================================


class KafkaTopicsConfig(TypedDict):
    """Kafka topic names for streaming pipeline.

    Fields:
        measurements: Input topic for measurement events.
        predictions: Output topic for prediction events.
        alerts: Output topic for high-severity alert events.
        dlq: Dead-letter topic for messages that cannot be processed. Without
            it a single undecodable message is replayed forever, because its
            offset can never be advanced past.
    """

    measurements: str
    predictions: str
    alerts: str
    dlq: str


# =============================================================================
# Consumer Configuration
# =============================================================================


class ConsumerConfig(TypedDict):
    """Kafka consumer configuration.

    Fields:
        group_id: Consumer group ID for coordinated consumption.
        auto_offset_reset: Offset reset policy for new consumer groups.
        enable_auto_commit: Whether to auto-commit offsets.
        fetch_min_bytes: Minimum amount of data broker should return.
        session_timeout_ms: Session timeout in milliseconds.
        heartbeat_interval_ms: Heartbeat interval in milliseconds.
    """

    group_id: str
    auto_offset_reset: Literal["earliest", "latest"]
    enable_auto_commit: bool
    fetch_min_bytes: int
    session_timeout_ms: int
    heartbeat_interval_ms: int


# =============================================================================
# Producer Configuration
# =============================================================================


class ProducerConfig(TypedDict):
    """Kafka producer configuration.

    Fields:
        acks: Acknowledgment level for produced messages.
        retries: Number of retries for failed sends.
        linger_ms: Batching linger time in milliseconds.
        batch_size: Maximum batch size in bytes.
        compression_type: Compression algorithm for messages.
    """

    acks: Literal["all", "0", "1"]
    retries: int
    linger_ms: int
    batch_size: int
    compression_type: Literal["none", "gzip", "snappy", "lz4", "zstd"]


# =============================================================================
# Combined Streaming Configuration
# =============================================================================


class StreamingConfig(TypedDict):
    """Complete streaming configuration for Kafka pipeline.

    Combines all configuration components for the streaming infrastructure.

    Fields:
        confluent: Confluent Cloud connection settings.
        schema_registry: Optional Schema Registry settings.
        topics: Kafka topic names.
        consumer: Consumer configuration.
        producer: Producer configuration.
        enabled: Whether streaming is enabled.
    """

    confluent: ConfluentConfig
    schema_registry: ConfluentSchemaRegistryConfig | None
    topics: KafkaTopicsConfig
    consumer: ConsumerConfig
    producer: ProducerConfig
    enabled: bool


# =============================================================================
# Environment Variable Parsing
# =============================================================================


def _parse_auto_offset_reset(value: str) -> Literal["earliest", "latest"]:
    """Parse auto offset reset value.

    Args:
        value: Raw string value.

    Returns:
        Validated Literal type.

    Raises:
        ValueError: If value is not valid.
    """
    if value == "earliest":
        return "earliest"
    if value == "latest":
        return "latest"
    raise ValueError(f"Invalid auto_offset_reset: '{value}', must be 'earliest' or 'latest'")


def _parse_acks(value: str) -> Literal["all", "0", "1"]:
    """Parse producer acks value.

    Args:
        value: Raw string value.

    Returns:
        Validated Literal type.

    Raises:
        ValueError: If value is not valid.
    """
    if value == "all":
        return "all"
    if value == "0":
        return "0"
    if value == "1":
        return "1"
    raise ValueError(f"Invalid acks: '{value}', must be 'all', '0', or '1'")


def _parse_compression_type(value: str) -> Literal["none", "gzip", "snappy", "lz4", "zstd"]:
    """Parse compression type value.

    Args:
        value: Raw string value.

    Returns:
        Validated Literal type.

    Raises:
        ValueError: If value is not valid.
    """
    if value == "none":
        return "none"
    if value == "gzip":
        return "gzip"
    if value == "snappy":
        return "snappy"
    if value == "lz4":
        return "lz4"
    if value == "zstd":
        return "zstd"
    raise ValueError(
        f"Invalid compression_type: '{value}', must be 'none', 'gzip', 'snappy', 'lz4', or 'zstd'"
    )


def load_streaming_config() -> StreamingConfig:
    """Load streaming configuration from environment variables.

    Environment variables:
        STREAMING__ENABLED: Enable streaming (default: false)
        CONFLUENT__BOOTSTRAP_SERVERS: Kafka bootstrap servers
        CONFLUENT__API_KEY: SASL username
        CONFLUENT__API_SECRET: SASL password
        CONFLUENT__SCHEMA_REGISTRY_URL: Schema Registry URL (optional)
        CONFLUENT__SCHEMA_REGISTRY_API_KEY: Schema Registry API key
        CONFLUENT__SCHEMA_REGISTRY_API_SECRET: Schema Registry API secret
        KAFKA__TOPIC_MEASUREMENTS: Measurements topic (default: covenant.measurements.v1)
        KAFKA__TOPIC_PREDICTIONS: Predictions topic (default: covenant.predictions.v1)
        KAFKA__TOPIC_ALERTS: Alerts topic (default: covenant.alerts.v1)
        KAFKA__CONSUMER_GROUP_ID: Consumer group ID (default: covenant-radar-api)
        KAFKA__AUTO_OFFSET_RESET: Offset reset policy (default: earliest)
        KAFKA__ENABLE_AUTO_COMMIT: Auto-commit offsets (default: false)
        KAFKA__FETCH_MIN_BYTES: Minimum fetch bytes (default: 1)
        KAFKA__SESSION_TIMEOUT_MS: Session timeout (default: 45000)
        KAFKA__HEARTBEAT_INTERVAL_MS: Heartbeat interval (default: 15000)
        KAFKA__PRODUCER_ACKS: Producer acks (default: all)
        KAFKA__PRODUCER_RETRIES: Producer retries (default: 3)
        KAFKA__PRODUCER_LINGER_MS: Producer linger (default: 5)
        KAFKA__PRODUCER_BATCH_SIZE: Producer batch size (default: 16384)
        KAFKA__COMPRESSION_TYPE: Compression type (default: gzip)

    Returns:
        Complete StreamingConfig with all settings.
    """
    # Parse Confluent Cloud config
    confluent: ConfluentConfig = {
        "bootstrap_servers": _parse_str("CONFLUENT__BOOTSTRAP_SERVERS", ""),
        "api_key": _parse_str("CONFLUENT__API_KEY", ""),
        "api_secret": _parse_str("CONFLUENT__API_SECRET", ""),
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "PLAIN",
    }

    # Parse Schema Registry config (optional)
    schema_registry_url = _parse_str("CONFLUENT__SCHEMA_REGISTRY_URL", "")
    schema_registry: ConfluentSchemaRegistryConfig | None = None
    if schema_registry_url:
        schema_registry = {
            "url": schema_registry_url,
            "api_key": _parse_str("CONFLUENT__SCHEMA_REGISTRY_API_KEY", ""),
            "api_secret": _parse_str("CONFLUENT__SCHEMA_REGISTRY_API_SECRET", ""),
        }

    # Parse topic config
    topics: KafkaTopicsConfig = {
        "measurements": _parse_str("KAFKA__TOPIC_MEASUREMENTS", "covenant.measurements.v1"),
        "predictions": _parse_str("KAFKA__TOPIC_PREDICTIONS", "covenant.predictions.v1"),
        "alerts": _parse_str("KAFKA__TOPIC_ALERTS", "covenant.alerts.v1"),
        "dlq": _parse_str("KAFKA__TOPIC_DLQ", "covenant.dlq.v1"),
    }

    # Parse consumer config
    auto_offset_reset = _parse_auto_offset_reset(_parse_str("KAFKA__AUTO_OFFSET_RESET", "earliest"))
    consumer: ConsumerConfig = {
        "group_id": _parse_str("KAFKA__CONSUMER_GROUP_ID", "covenant-radar-api"),
        "auto_offset_reset": auto_offset_reset,
        "enable_auto_commit": _parse_bool("KAFKA__ENABLE_AUTO_COMMIT", False),
        "fetch_min_bytes": _parse_int("KAFKA__FETCH_MIN_BYTES", 1),
        "session_timeout_ms": _parse_int("KAFKA__SESSION_TIMEOUT_MS", 45000),
        "heartbeat_interval_ms": _parse_int("KAFKA__HEARTBEAT_INTERVAL_MS", 15000),
    }

    # Parse producer config
    acks = _parse_acks(_parse_str("KAFKA__PRODUCER_ACKS", "all"))
    compression_type = _parse_compression_type(_parse_str("KAFKA__COMPRESSION_TYPE", "gzip"))
    producer: ProducerConfig = {
        "acks": acks,
        "retries": _parse_int("KAFKA__PRODUCER_RETRIES", 3),
        "linger_ms": _parse_int("KAFKA__PRODUCER_LINGER_MS", 5),
        "batch_size": _parse_int("KAFKA__PRODUCER_BATCH_SIZE", 16384),
        "compression_type": compression_type,
    }

    return {
        "confluent": confluent,
        "schema_registry": schema_registry,
        "topics": topics,
        "consumer": consumer,
        "producer": producer,
        "enabled": _parse_bool("STREAMING__ENABLED", False),
    }


# =============================================================================
# Default Topic Names
# =============================================================================

DEFAULT_MEASUREMENTS_TOPIC: str = "covenant.measurements.v1"
DEFAULT_PREDICTIONS_TOPIC: str = "covenant.predictions.v1"
DEFAULT_ALERTS_TOPIC: str = "covenant.alerts.v1"


__all__ = [
    "DEFAULT_ALERTS_TOPIC",
    "DEFAULT_MEASUREMENTS_TOPIC",
    "DEFAULT_PREDICTIONS_TOPIC",
    "ConfluentConfig",
    "ConfluentSchemaRegistryConfig",
    "ConsumerConfig",
    "KafkaTopicsConfig",
    "ProducerConfig",
    "StreamingConfig",
    "load_streaming_config",
]
