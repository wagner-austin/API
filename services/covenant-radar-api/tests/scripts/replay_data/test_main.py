"""Tests for replay script CLI entry point."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import pytest
from platform_core.logging import setup_rich_logging
from scripts.replay_data.__main__ import (
    _parse_args,
    _parse_speed,
    _StreamingProducerAdapter,
    main,
)

from covenant_radar_api.streaming._test_hooks import (
    FakeKafkaProducer,
    use_fake_kafka,
    use_real_kafka,
)
from covenant_radar_api.streaming.producer import StreamingProducer
from covenant_radar_api.streaming.schemas import make_measurement_event
from scripts.replay_data import _test_hooks as replay_hooks

from .conftest import (
    FakeDatasetLoader,
    FakeDatasetRegistry,
    FakeTimeSeriesRegistry,
    make_test_dataset_config,
    make_test_loaded_dataset,
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

    def test_produce_event_delegates(self) -> None:
        """Test produce_event delegates to StreamingProducer."""
        fake_kafka = FakeKafkaProducer()
        streaming = StreamingProducer(fake_kafka, "pred", "alert")
        adapter = _StreamingProducerAdapter(streaming)

        event = make_measurement_event(
            event_id="evt-1",
            deal_id="deal-001",
            period_start="2024-01-01",
            period_end="2024-01-31",
            metric_name="ratio",
            metric_value=1.5,
            timestamp="2024-01-15T12:00:00Z",
        )

        adapter.produce_event(event, "test.topic")

        assert len(fake_kafka.messages) == 1
        assert fake_kafka.messages[0].topic == "test.topic"

    def test_poll_delegates(self) -> None:
        """Test poll delegates to StreamingProducer."""
        fake_kafka = FakeKafkaProducer()
        streaming = StreamingProducer(fake_kafka, "pred", "alert")
        adapter = _StreamingProducerAdapter(streaming)

        result = adapter.poll(1.0)

        assert result == 0
        assert fake_kafka.poll_count == 1

    def test_flush_delegates(self) -> None:
        """Test flush delegates to StreamingProducer."""
        fake_kafka = FakeKafkaProducer()
        streaming = StreamingProducer(fake_kafka, "pred", "alert")
        adapter = _StreamingProducerAdapter(streaming)

        result = adapter.flush(5.0)

        assert result == 0
        assert fake_kafka.flush_called is True


class TestMain:
    """Tests for main() function."""

    def test_main_returns_zero(
        self,
        restore_hooks: None,
        tmp_path: Path,
    ) -> None:
        """Test main() returns 0 on success."""
        # Ensure rich console is available
        setup_rich_logging()

        # Set up replay hooks
        dataset = make_test_loaded_dataset(n_samples=1, n_features=1)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        replay_hooks.dataset_loader_factory = lambda: loader
        replay_hooks.registry_factory = lambda: registry
        replay_hooks.timeseries_registry_factory = lambda: ts_registry
        replay_hooks.perf_counter = lambda: 1.0
        replay_hooks.generate_uuid = lambda: "test-uuid"

        # Set up streaming hooks (fake Kafka producer)
        use_fake_kafka()
        try:
            result = main(
                [
                    "--dataset",
                    "test",
                    "--speed",
                    "instant",
                    "--external-dir",
                    str(tmp_path),
                ]
            )

            assert result == 0
        finally:
            use_real_kafka()

    def test_main_resolves_relative_external_dir(
        self,
        restore_hooks: None,
    ) -> None:
        """Test main() resolves relative external_dir to absolute."""
        # Ensure rich console is available
        setup_rich_logging()

        # Set up replay hooks
        dataset = make_test_loaded_dataset(n_samples=1, n_features=1)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        replay_hooks.dataset_loader_factory = lambda: loader
        replay_hooks.registry_factory = lambda: registry
        replay_hooks.timeseries_registry_factory = lambda: ts_registry
        replay_hooks.perf_counter = lambda: 1.0
        replay_hooks.generate_uuid = lambda: "test-uuid"

        # Set up streaming hooks (fake Kafka producer)
        use_fake_kafka()
        try:
            # Use default --external-dir (relative "data/external")
            result = main(
                [
                    "--dataset",
                    "test",
                    "--speed",
                    "instant",
                ]
            )

            assert result == 0
        finally:
            use_real_kafka()


class TestMainGuard:
    """Tests for __main__ guard."""

    def test_main_guard_executes(
        self,
        restore_hooks: None,
        tmp_path: Path,
    ) -> None:
        """Test __main__ guard calls main() and exits with 0."""
        # Ensure rich console is available
        setup_rich_logging()

        # Set up replay hooks
        dataset = make_test_loaded_dataset(n_samples=1, n_features=1)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        replay_hooks.dataset_loader_factory = lambda: loader
        replay_hooks.registry_factory = lambda: registry
        replay_hooks.timeseries_registry_factory = lambda: ts_registry
        replay_hooks.perf_counter = lambda: 1.0
        replay_hooks.generate_uuid = lambda: "test-uuid"

        # Set up streaming hooks
        use_fake_kafka()
        orig_argv = sys.argv
        try:
            sys.argv = [
                "scripts.replay_data",
                "--dataset",
                "test",
                "--speed",
                "instant",
                "--external-dir",
                str(tmp_path),
            ]
            # Remove cached __main__ so runpy doesn't find it
            # already loaded from our top-level import.
            sys.modules.pop("scripts.replay_data.__main__", None)
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module(
                    "scripts.replay_data",
                    run_name="__main__",
                )
            assert exc_info.value.code == 0
        finally:
            sys.argv = orig_argv
            use_real_kafka()
