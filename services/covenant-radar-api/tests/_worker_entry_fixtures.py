"""Shared fixtures and helpers for test_streaming_worker_entry splits."""

from __future__ import annotations

from covenant_radar_api.streaming.config import StreamingConfig


class _RecordingLogger:
    """Logger that records calls for testing."""

    def __init__(self) -> None:
        self.info_messages: list[tuple[str, dict[str, str] | None]] = []
        self.error_messages: list[tuple[str, dict[str, str] | None]] = []

    def info(self, message: str, *, extra: dict[str, str] | None = None) -> None:
        """Record info message."""
        self.info_messages.append((message, extra))

    def error(self, message: str, *, extra: dict[str, str] | None = None) -> None:
        """Record error message."""
        self.error_messages.append((message, extra))


def _make_test_streaming_config(enabled: bool = True) -> StreamingConfig:
    """Create a test streaming config.

    Args:
        enabled: Whether streaming is enabled.

    Returns:
        StreamingConfig for testing.
    """
    return {
        "confluent": {
            "bootstrap_servers": "test:9092",
            "api_key": "test-key",
            "api_secret": "test-secret",
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
        },
        "schema_registry": None,
        "topics": {
            "measurements": "test.measurements.v1",
            "predictions": "test.predictions.v1",
            "alerts": "test.alerts.v1",
            "dlq": "test.dlq.v1",
        },
        "consumer": {
            "group_id": "test-group",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
            "fetch_min_bytes": 1,
            "session_timeout_ms": 45000,
            "heartbeat_interval_ms": 15000,
        },
        "producer": {
            "acks": "all",
            "retries": 3,
            "linger_ms": 5,
            "batch_size": 16384,
            "compression_type": "gzip",
        },
        "enabled": enabled,
    }
