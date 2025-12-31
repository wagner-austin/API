"""Tests for replay script types."""

from __future__ import annotations

from scripts.replay_data.types import (
    ReplaySpeed,
    make_replay_config,
    make_replay_stats,
)


class TestReplaySpeed:
    """Tests for ReplaySpeed literal type."""

    def test_realtime_is_valid(self) -> None:
        """Test realtime speed is valid."""
        speed: ReplaySpeed = "realtime"
        assert speed == "realtime"

    def test_fast_is_valid(self) -> None:
        """Test fast speed is valid."""
        speed: ReplaySpeed = "fast"
        assert speed == "fast"

    def test_instant_is_valid(self) -> None:
        """Test instant speed is valid."""
        speed: ReplaySpeed = "instant"
        assert speed == "instant"


class TestMakeReplayConfig:
    """Tests for make_replay_config factory."""

    def test_creates_config_with_defaults(self) -> None:
        """Test config creation with default values."""
        config = make_replay_config(dataset="taiwan")

        assert config["dataset"] == "taiwan"
        assert config["topic"] == "covenant.measurements.v1"
        assert config["speed"] == "fast"
        assert config["batch_size"] == 100
        assert config["deal_id_prefix"] == "replay"
        assert config["max_rows"] == 0

    def test_creates_config_with_custom_values(self) -> None:
        """Test config creation with custom values."""
        config = make_replay_config(
            dataset="kaggle_amex_default",
            topic="custom.topic",
            speed="instant",
            batch_size=500,
            deal_id_prefix="amex",
            max_rows=1000,
        )

        assert config["dataset"] == "kaggle_amex_default"
        assert config["topic"] == "custom.topic"
        assert config["speed"] == "instant"
        assert config["batch_size"] == 500
        assert config["deal_id_prefix"] == "amex"
        assert config["max_rows"] == 1000

    def test_config_is_typed_dict(self) -> None:
        """Test that config is a proper TypedDict."""
        config = make_replay_config(dataset="test")

        # Verify all required keys exist
        assert "dataset" in config
        assert "topic" in config
        assert "speed" in config
        assert "batch_size" in config
        assert "deal_id_prefix" in config
        assert "max_rows" in config


class TestMakeReplayStats:
    """Tests for make_replay_stats factory."""

    def test_creates_stats_with_values(self) -> None:
        """Test stats creation with provided values."""
        stats = make_replay_stats(
            rows_processed=100,
            events_sent=500,
            batches_sent=5,
            elapsed_seconds=10.0,
        )

        assert stats["rows_processed"] == 100
        assert stats["events_sent"] == 500
        assert stats["batches_sent"] == 5
        assert stats["elapsed_seconds"] == 10.0
        assert stats["events_per_second"] == 50.0

    def test_computes_throughput(self) -> None:
        """Test throughput calculation."""
        stats = make_replay_stats(
            rows_processed=50,
            events_sent=200,
            batches_sent=2,
            elapsed_seconds=4.0,
        )

        assert stats["events_per_second"] == 50.0

    def test_zero_elapsed_time(self) -> None:
        """Test throughput with zero elapsed time."""
        stats = make_replay_stats(
            rows_processed=10,
            events_sent=30,
            batches_sent=1,
            elapsed_seconds=0.0,
        )

        assert stats["events_per_second"] == 0.0

    def test_fractional_throughput(self) -> None:
        """Test throughput with fractional result."""
        stats = make_replay_stats(
            rows_processed=10,
            events_sent=33,
            batches_sent=1,
            elapsed_seconds=3.0,
        )

        assert stats["events_per_second"] == 11.0

    def test_stats_is_typed_dict(self) -> None:
        """Test that stats is a proper TypedDict."""
        stats = make_replay_stats(
            rows_processed=0,
            events_sent=0,
            batches_sent=0,
            elapsed_seconds=0.0,
        )

        # Verify all required keys exist
        assert "rows_processed" in stats
        assert "events_sent" in stats
        assert "batches_sent" in stats
        assert "elapsed_seconds" in stats
        assert "events_per_second" in stats
