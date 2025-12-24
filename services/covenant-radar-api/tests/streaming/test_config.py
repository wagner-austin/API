"""Tests for streaming configuration module."""

from __future__ import annotations

from typing import TypeVar

import pytest
from platform_core.config import _test_hooks as env_hooks

from covenant_radar_api.streaming.config import (
    DEFAULT_ALERTS_TOPIC,
    DEFAULT_MEASUREMENTS_TOPIC,
    DEFAULT_PREDICTIONS_TOPIC,
    _parse_acks,
    _parse_auto_offset_reset,
    _parse_compression_type,
    load_streaming_config,
)

_T = TypeVar("_T")


def _require(value: _T | None) -> _T:
    """Narrow optional type to non-None. Raises if None."""
    if value is None:
        msg = "Expected non-None value"
        raise AssertionError(msg)
    return value


class TestDefaultTopics:
    """Tests for default topic constants."""

    def test_default_measurements_topic(self) -> None:
        """Default measurements topic should be covenant.measurements.v1."""
        assert DEFAULT_MEASUREMENTS_TOPIC == "covenant.measurements.v1"

    def test_default_predictions_topic(self) -> None:
        """Default predictions topic should be covenant.predictions.v1."""
        assert DEFAULT_PREDICTIONS_TOPIC == "covenant.predictions.v1"

    def test_default_alerts_topic(self) -> None:
        """Default alerts topic should be covenant.alerts.v1."""
        assert DEFAULT_ALERTS_TOPIC == "covenant.alerts.v1"


class TestParseAutoOffsetReset:
    """Tests for _parse_auto_offset_reset."""

    def test_earliest(self) -> None:
        """Parse 'earliest' value."""
        result = _parse_auto_offset_reset("earliest")
        assert result == "earliest"

    def test_latest(self) -> None:
        """Parse 'latest' value."""
        result = _parse_auto_offset_reset("latest")
        assert result == "latest"

    def test_invalid_raises(self) -> None:
        """Invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid auto_offset_reset"):
            _parse_auto_offset_reset("invalid")


class TestParseAcks:
    """Tests for _parse_acks."""

    def test_all(self) -> None:
        """Parse 'all' value."""
        result = _parse_acks("all")
        assert result == "all"

    def test_zero(self) -> None:
        """Parse '0' value."""
        result = _parse_acks("0")
        assert result == "0"

    def test_one(self) -> None:
        """Parse '1' value."""
        result = _parse_acks("1")
        assert result == "1"

    def test_invalid_raises(self) -> None:
        """Invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid acks"):
            _parse_acks("invalid")


class TestParseCompressionType:
    """Tests for _parse_compression_type."""

    def test_none(self) -> None:
        """Parse 'none' value."""
        result = _parse_compression_type("none")
        assert result == "none"

    def test_gzip(self) -> None:
        """Parse 'gzip' value."""
        result = _parse_compression_type("gzip")
        assert result == "gzip"

    def test_snappy(self) -> None:
        """Parse 'snappy' value."""
        result = _parse_compression_type("snappy")
        assert result == "snappy"

    def test_lz4(self) -> None:
        """Parse 'lz4' value."""
        result = _parse_compression_type("lz4")
        assert result == "lz4"

    def test_zstd(self) -> None:
        """Parse 'zstd' value."""
        result = _parse_compression_type("zstd")
        assert result == "zstd"

    def test_invalid_raises(self) -> None:
        """Invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid compression_type"):
            _parse_compression_type("invalid")


class TestLoadStreamingConfig:
    """Tests for load_streaming_config."""

    def test_defaults(self) -> None:
        """Load config with default values."""
        # Set up empty env
        fake_env: dict[str, str] = {}
        env_hooks.get_env = lambda key: fake_env.get(key)

        config = load_streaming_config()

        # Check defaults
        assert config["enabled"] is False
        assert config["confluent"]["bootstrap_servers"] == ""
        assert config["confluent"]["api_key"] == ""
        assert config["confluent"]["api_secret"] == ""
        assert config["confluent"]["security_protocol"] == "SASL_SSL"
        assert config["confluent"]["sasl_mechanism"] == "PLAIN"
        assert config["schema_registry"] is None
        assert config["topics"]["measurements"] == "covenant.measurements.v1"
        assert config["topics"]["predictions"] == "covenant.predictions.v1"
        assert config["topics"]["alerts"] == "covenant.alerts.v1"
        assert config["consumer"]["group_id"] == "covenant-radar-api"
        assert config["consumer"]["auto_offset_reset"] == "earliest"
        assert config["consumer"]["enable_auto_commit"] is False
        assert config["consumer"]["fetch_min_bytes"] == 1
        assert config["consumer"]["session_timeout_ms"] == 45000
        assert config["consumer"]["heartbeat_interval_ms"] == 15000
        assert config["producer"]["acks"] == "all"
        assert config["producer"]["retries"] == 3
        assert config["producer"]["linger_ms"] == 5
        assert config["producer"]["batch_size"] == 16384
        assert config["producer"]["compression_type"] == "gzip"

    def test_custom_values(self) -> None:
        """Load config with custom values from env."""
        fake_env: dict[str, str] = {
            "STREAMING__ENABLED": "true",
            "CONFLUENT__BOOTSTRAP_SERVERS": "broker.cloud:9092",
            "CONFLUENT__API_KEY": "mykey",
            "CONFLUENT__API_SECRET": "mysecret",
            "KAFKA__TOPIC_MEASUREMENTS": "custom.measurements",
            "KAFKA__TOPIC_PREDICTIONS": "custom.predictions",
            "KAFKA__TOPIC_ALERTS": "custom.alerts",
            "KAFKA__CONSUMER_GROUP_ID": "my-group",
            "KAFKA__AUTO_OFFSET_RESET": "latest",
            "KAFKA__ENABLE_AUTO_COMMIT": "true",
            "KAFKA__FETCH_MIN_BYTES": "500",
            "KAFKA__SESSION_TIMEOUT_MS": "60000",
            "KAFKA__HEARTBEAT_INTERVAL_MS": "20000",
            "KAFKA__PRODUCER_ACKS": "1",
            "KAFKA__PRODUCER_RETRIES": "5",
            "KAFKA__PRODUCER_LINGER_MS": "10",
            "KAFKA__PRODUCER_BATCH_SIZE": "32768",
            "KAFKA__COMPRESSION_TYPE": "snappy",
        }
        env_hooks.get_env = lambda key: fake_env.get(key)

        config = load_streaming_config()

        assert config["enabled"] is True
        assert config["confluent"]["bootstrap_servers"] == "broker.cloud:9092"
        assert config["confluent"]["api_key"] == "mykey"
        assert config["confluent"]["api_secret"] == "mysecret"
        assert config["topics"]["measurements"] == "custom.measurements"
        assert config["topics"]["predictions"] == "custom.predictions"
        assert config["topics"]["alerts"] == "custom.alerts"
        assert config["consumer"]["group_id"] == "my-group"
        assert config["consumer"]["auto_offset_reset"] == "latest"
        assert config["consumer"]["enable_auto_commit"] is True
        assert config["consumer"]["fetch_min_bytes"] == 500
        assert config["consumer"]["session_timeout_ms"] == 60000
        assert config["consumer"]["heartbeat_interval_ms"] == 20000
        assert config["producer"]["acks"] == "1"
        assert config["producer"]["retries"] == 5
        assert config["producer"]["linger_ms"] == 10
        assert config["producer"]["batch_size"] == 32768
        assert config["producer"]["compression_type"] == "snappy"

    def test_schema_registry_config(self) -> None:
        """Load config with schema registry settings."""
        fake_env: dict[str, str] = {
            "CONFLUENT__SCHEMA_REGISTRY_URL": "https://sr.cloud:443",
            "CONFLUENT__SCHEMA_REGISTRY_API_KEY": "srkey",
            "CONFLUENT__SCHEMA_REGISTRY_API_SECRET": "srsecret",
        }
        env_hooks.get_env = lambda key: fake_env.get(key)

        config = load_streaming_config()

        # Narrow schema_registry to non-None
        schema_registry = _require(config["schema_registry"])
        assert schema_registry["url"] == "https://sr.cloud:443"
        assert schema_registry["api_key"] == "srkey"
        assert schema_registry["api_secret"] == "srsecret"
