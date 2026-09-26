"""Tests for replay script types."""

from __future__ import annotations

from scripts.replay_data.types import (
    ReplaySpeed,
    make_replay_config,
    make_replay_stats,
)


class TestReplaySpeed:
    """Tests for the ReplaySpeed vocabulary."""

    def test_members_are_their_cli_words(self) -> None:
        """Each speed reads as the word --speed takes."""
        assert [str(m) for m in ReplaySpeed] == ["realtime", "fast", "instant"]

    def test_realtime_waits_a_second_between_batches(self) -> None:
        """Realtime pauses one second per batch."""
        assert ReplaySpeed.REALTIME.delay_seconds == 1.0

    def test_fast_waits_a_tenth_of_a_second(self) -> None:
        """Fast pauses a tenth of a second per batch."""
        assert ReplaySpeed.FAST.delay_seconds == 0.1

    def test_instant_does_not_wait(self) -> None:
        """Instant publishes batches back to back."""
        assert ReplaySpeed.INSTANT.delay_seconds == 0.0


class TestMakeReplayConfig:
    """Tests for make_replay_config factory."""

    def test_creates_config_with_defaults(self) -> None:
        """Test config creation with default values."""
        config = make_replay_config(dataset="taiwan")

        assert config["dataset"] == "taiwan"
        assert config["topic"] == "covenant.measurements.v1"
        assert config["speed"] is ReplaySpeed.FAST
        assert config["batch_size"] == 100
        assert config["deal_id_prefix"] == "replay"
        assert config["max_rows"] == 0

    def test_creates_config_with_custom_values(self) -> None:
        """Test config creation with custom values."""
        config = make_replay_config(
            dataset="kaggle_amex_default",
            topic="custom.topic",
            speed=ReplaySpeed.INSTANT,
            batch_size=500,
            deal_id_prefix="amex",
            max_rows=1000,
        )

        assert config["dataset"] == "kaggle_amex_default"
        assert config["topic"] == "custom.topic"
        assert config["speed"] is ReplaySpeed.INSTANT
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
