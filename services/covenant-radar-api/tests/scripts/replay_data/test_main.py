"""Tests for replay script CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from scripts.replay_data.__main__ import (
    _parse_args,
    _parse_speed,
    _StreamingProducerAdapter,
)


class TestParseSpeed:
    """Tests for _parse_speed function."""

    def test_realtime_valid(self) -> None:
        """Test realtime speed is valid."""
        result = _parse_speed("realtime")
        assert result == "realtime"

    def test_fast_valid(self) -> None:
        """Test fast speed is valid."""
        result = _parse_speed("fast")
        assert result == "fast"

    def test_instant_valid(self) -> None:
        """Test instant speed is valid."""
        result = _parse_speed("instant")
        assert result == "instant"

    def test_invalid_raises_error(self) -> None:
        """Test invalid speed raises ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid speed"):
            _parse_speed("slow")

    def test_case_sensitive(self) -> None:
        """Test speed parsing is case sensitive."""
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_speed("FAST")


class TestParseArgs:
    """Tests for _parse_args function."""

    def test_requires_dataset(self) -> None:
        """Test dataset argument is required."""
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_parses_dataset(self) -> None:
        """Test parses dataset argument."""
        args = _parse_args(["--dataset", "taiwan"])
        assert args["dataset"] == "taiwan"

    def test_parses_dataset_short(self) -> None:
        """Test parses short dataset flag."""
        args = _parse_args(["-d", "kaggle_amex_default"])
        assert args["dataset"] == "kaggle_amex_default"

    def test_default_speed(self) -> None:
        """Test default speed is fast."""
        args = _parse_args(["--dataset", "test"])
        assert args["speed"] == "fast"

    def test_parses_speed(self) -> None:
        """Test parses speed argument."""
        args = _parse_args(["--dataset", "test", "--speed", "instant"])
        assert args["speed"] == "instant"

    def test_parses_speed_short(self) -> None:
        """Test parses short speed flag."""
        args = _parse_args(["-d", "test", "-s", "realtime"])
        assert args["speed"] == "realtime"

    def test_default_batch_size(self) -> None:
        """Test default batch size is 100."""
        args = _parse_args(["--dataset", "test"])
        assert args["batch_size"] == 100

    def test_parses_batch_size(self) -> None:
        """Test parses batch size argument."""
        args = _parse_args(["--dataset", "test", "--batch-size", "500"])
        assert args["batch_size"] == 500

    def test_parses_batch_size_short(self) -> None:
        """Test parses short batch size flag."""
        args = _parse_args(["-d", "test", "-b", "250"])
        assert args["batch_size"] == 250

    def test_default_max_rows(self) -> None:
        """Test default max rows is 0 (unlimited)."""
        args = _parse_args(["--dataset", "test"])
        assert args["max_rows"] == 0

    def test_parses_max_rows(self) -> None:
        """Test parses max rows argument."""
        args = _parse_args(["--dataset", "test", "--max-rows", "1000"])
        assert args["max_rows"] == 1000

    def test_parses_max_rows_short(self) -> None:
        """Test parses short max rows flag."""
        args = _parse_args(["-d", "test", "-m", "500"])
        assert args["max_rows"] == 500

    def test_default_deal_prefix(self) -> None:
        """Test default deal prefix is 'replay'."""
        args = _parse_args(["--dataset", "test"])
        assert args["deal_prefix"] == "replay"

    def test_parses_deal_prefix(self) -> None:
        """Test parses deal prefix argument."""
        args = _parse_args(["--dataset", "test", "--deal-prefix", "demo"])
        assert args["deal_prefix"] == "demo"

    def test_parses_deal_prefix_short(self) -> None:
        """Test parses short deal prefix flag."""
        args = _parse_args(["-d", "test", "-p", "amex"])
        assert args["deal_prefix"] == "amex"

    def test_default_external_dir(self) -> None:
        """Test default external dir is data/external."""
        args = _parse_args(["--dataset", "test"])
        assert args["external_dir"] == Path("data/external")

    def test_parses_external_dir(self) -> None:
        """Test parses external dir argument."""
        args = _parse_args(["--dataset", "test", "--external-dir", "/tmp/data"])
        assert args["external_dir"] == Path("/tmp/data")

    def test_parses_external_dir_short(self) -> None:
        """Test parses short external dir flag."""
        args = _parse_args(["-d", "test", "-e", "/data/ext"])
        assert args["external_dir"] == Path("/data/ext")


class TestStreamingProducerAdapter:
    """Tests for _StreamingProducerAdapter class."""

    # Note: Full integration tests for the adapter require the real
    # StreamingProducer which needs Kafka configuration. These tests
    # verify the adapter's interface exists.

    def test_adapter_produce_event_is_method(self) -> None:
        """Test adapter produce_event is callable method."""
        produce_event = _StreamingProducerAdapter.produce_event
        assert callable(produce_event)

    def test_adapter_poll_is_method(self) -> None:
        """Test adapter poll is callable method."""
        poll = _StreamingProducerAdapter.poll
        assert callable(poll)

    def test_adapter_flush_is_method(self) -> None:
        """Test adapter flush is callable method."""
        flush = _StreamingProducerAdapter.flush
        assert callable(flush)
