"""Tests for replay script runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.replay_data._test_hooks import FakeProducer
from scripts.replay_data.runner import (
    DataReplayRunner,
    _current_iso_timestamp,
    _get_delay_seconds,
    _make_deal_id,
    _make_period_dates,
    _row_to_events,
    run_replay,
)
from scripts.replay_data.types import make_replay_config

from scripts.replay_data import _test_hooks

from .conftest import (
    FakeDatasetLoader,
    FakeDatasetRegistry,
    FakeTimeSeriesRegistry,
    make_test_dataset_config,
    make_test_loaded_dataset,
    make_test_timeseries_config,
)


class TestGetDelaySeconds:
    """Tests for _get_delay_seconds helper."""

    def test_realtime_delay(self) -> None:
        """Test realtime speed returns 1.0 second delay."""
        delay = _get_delay_seconds("realtime")
        assert delay == 1.0

    def test_fast_delay(self) -> None:
        """Test fast speed returns 0.1 second delay."""
        delay = _get_delay_seconds("fast")
        assert delay == 0.1

    def test_instant_delay(self) -> None:
        """Test instant speed returns 0 delay."""
        delay = _get_delay_seconds("instant")
        assert delay == 0.0


class TestMakeDealId:
    """Tests for _make_deal_id helper."""

    def test_formats_with_prefix(self) -> None:
        """Test deal ID formatted with prefix."""
        deal_id = _make_deal_id("replay", 42)
        assert deal_id == "replay-000042"

    def test_zero_padded(self) -> None:
        """Test deal ID is zero-padded to 6 digits."""
        deal_id = _make_deal_id("test", 1)
        assert deal_id == "test-000001"

    def test_large_index(self) -> None:
        """Test deal ID with large index."""
        deal_id = _make_deal_id("demo", 123456)
        assert deal_id == "demo-123456"


class TestMakePeriodDates:
    """Tests for _make_period_dates helper."""

    def test_first_row_january(self) -> None:
        """Test first row creates January period."""
        start, end = _make_period_dates(0)
        assert start == "2024-01-01"
        assert end == "2024-01-28"

    def test_cycles_through_months(self) -> None:
        """Test period dates cycle through 12 months."""
        # Row 11 -> December (index 11 % 12 + 1 = 12)
        start, end = _make_period_dates(11)
        assert start == "2024-12-01"
        assert end == "2024-12-28"

    def test_wraps_after_december(self) -> None:
        """Test period wraps to January after December."""
        # Row 12 -> January (index 12 % 12 + 1 = 1)
        start, end = _make_period_dates(12)
        assert start == "2024-01-01"
        assert end == "2024-01-28"


class TestCurrentIsoTimestamp:
    """Tests for _current_iso_timestamp helper."""

    def test_returns_iso_format(self) -> None:
        """Test returns ISO formatted timestamp."""
        ts = _current_iso_timestamp()
        # Should match pattern: 2024-01-15T12:30:45Z
        assert len(ts) == 20
        assert ts[4] == "-"
        assert ts[7] == "-"
        assert ts[10] == "T"
        assert ts[13] == ":"
        assert ts[16] == ":"
        assert ts[19] == "Z"


class TestRowToEvents:
    """Tests for _row_to_events helper."""

    def test_creates_event_per_feature(
        self,
        restore_hooks: None,
    ) -> None:
        """Test creates one event per feature."""
        # Set up deterministic UUID
        uuid_counter = 0

        def fake_uuid() -> str:
            nonlocal uuid_counter
            uuid_counter += 1
            return f"uuid-{uuid_counter:04d}"

        _test_hooks.generate_uuid = fake_uuid

        feature_names = ("feat_a", "feat_b", "feat_c")
        feature_values = (1.0, 2.0, 3.0)

        events = _row_to_events(
            row_index=0,
            feature_names=feature_names,
            feature_values=feature_values,
            deal_id_prefix="test",
        )

        assert len(events) == 3
        assert events[0]["metric_name"] == "feat_a"
        assert events[0]["metric_value"] == 1.0
        assert events[1]["metric_name"] == "feat_b"
        assert events[1]["metric_value"] == 2.0
        assert events[2]["metric_name"] == "feat_c"
        assert events[2]["metric_value"] == 3.0

    def test_same_deal_id_for_row(
        self,
        restore_hooks: None,
    ) -> None:
        """Test all events share same deal_id."""
        _test_hooks.generate_uuid = lambda: "test-uuid"

        events = _row_to_events(
            row_index=5,
            feature_names=("a", "b"),
            feature_values=(1.0, 2.0),
            deal_id_prefix="replay",
        )

        assert events[0]["deal_id"] == "replay-000005"
        assert events[1]["deal_id"] == "replay-000005"

    def test_same_period_for_row(
        self,
        restore_hooks: None,
    ) -> None:
        """Test all events share same period."""
        _test_hooks.generate_uuid = lambda: "test-uuid"

        events = _row_to_events(
            row_index=3,
            feature_names=("x", "y"),
            feature_values=(1.0, 2.0),
            deal_id_prefix="test",
        )

        # Row 3 -> month 4 (April)
        assert events[0]["period_start"] == "2024-04-01"
        assert events[0]["period_end"] == "2024-04-28"
        assert events[1]["period_start"] == "2024-04-01"
        assert events[1]["period_end"] == "2024-04-28"

    def test_event_type_discriminator(
        self,
        restore_hooks: None,
    ) -> None:
        """Test events have correct type discriminator."""
        _test_hooks.generate_uuid = lambda: "test-uuid"

        events = _row_to_events(
            row_index=0,
            feature_names=("a",),
            feature_values=(1.0,),
            deal_id_prefix="test",
        )

        assert events[0]["type"] == "covenant.measurement.v1"


class TestDataReplayRunner:
    """Tests for DataReplayRunner class."""

    def test_run_processes_all_rows(
        self,
        restore_hooks: None,
        fake_producer: FakeProducer,
        external_dir: Path,
    ) -> None:
        """Test runner processes all dataset rows."""
        # Set up fakes
        dataset = make_test_loaded_dataset(n_samples=3, n_features=2)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        _test_hooks.dataset_loader_factory = lambda: loader
        _test_hooks.registry_factory = lambda: registry
        _test_hooks.timeseries_registry_factory = lambda: ts_registry
        _test_hooks.perf_counter = lambda: 1.0
        _test_hooks.generate_uuid = lambda: "test-uuid"

        replay_config = make_replay_config(
            dataset="test",
            speed="instant",
            batch_size=100,
        )

        runner = DataReplayRunner(fake_producer, replay_config, external_dir)
        stats = runner.run()

        # 3 rows * 2 features = 6 events
        assert stats["rows_processed"] == 3
        assert stats["events_sent"] == 6
        assert len(fake_producer.events) == 6

    def test_run_respects_max_rows(
        self,
        restore_hooks: None,
        fake_producer: FakeProducer,
        external_dir: Path,
    ) -> None:
        """Test runner respects max_rows limit."""
        dataset = make_test_loaded_dataset(n_samples=10, n_features=2)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        _test_hooks.dataset_loader_factory = lambda: loader
        _test_hooks.registry_factory = lambda: registry
        _test_hooks.timeseries_registry_factory = lambda: ts_registry
        _test_hooks.perf_counter = lambda: 1.0
        _test_hooks.generate_uuid = lambda: "test-uuid"

        replay_config = make_replay_config(
            dataset="test",
            speed="instant",
            max_rows=3,
        )

        runner = DataReplayRunner(fake_producer, replay_config, external_dir)
        stats = runner.run()

        assert stats["rows_processed"] == 3
        # 3 rows * 2 features = 6 events
        assert stats["events_sent"] == 6

    def test_run_uses_correct_topic(
        self,
        restore_hooks: None,
        fake_producer: FakeProducer,
        external_dir: Path,
    ) -> None:
        """Test runner uses configured topic."""
        dataset = make_test_loaded_dataset(n_samples=1, n_features=1)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        _test_hooks.dataset_loader_factory = lambda: loader
        _test_hooks.registry_factory = lambda: registry
        _test_hooks.timeseries_registry_factory = lambda: ts_registry
        _test_hooks.perf_counter = lambda: 1.0
        _test_hooks.generate_uuid = lambda: "test-uuid"

        replay_config = make_replay_config(
            dataset="test",
            topic="custom.topic.v1",
            speed="instant",
        )

        runner = DataReplayRunner(fake_producer, replay_config, external_dir)
        runner.run()

        assert len(fake_producer.events) == 1
        _event, topic = fake_producer.events[0]
        assert topic == "custom.topic.v1"

    def test_run_batches_events(
        self,
        restore_hooks: None,
        fake_producer: FakeProducer,
        external_dir: Path,
    ) -> None:
        """Test runner batches events correctly."""
        # 5 samples * 2 features = 10 events, batch size 3 = 4 batches
        dataset = make_test_loaded_dataset(n_samples=5, n_features=2)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        _test_hooks.dataset_loader_factory = lambda: loader
        _test_hooks.registry_factory = lambda: registry
        _test_hooks.timeseries_registry_factory = lambda: ts_registry
        _test_hooks.perf_counter = lambda: 1.0
        _test_hooks.generate_uuid = lambda: "test-uuid"
        _test_hooks.sleep = lambda s: None

        replay_config = make_replay_config(
            dataset="test",
            speed="fast",
            batch_size=3,
        )

        runner = DataReplayRunner(fake_producer, replay_config, external_dir)
        stats = runner.run()

        # 10 events / 3 per batch = 4 batches (3, 3, 3, 1)
        assert stats["batches_sent"] == 4
        assert stats["events_sent"] == 10

    def test_run_loads_timeseries_dataset(
        self,
        restore_hooks: None,
        fake_producer: FakeProducer,
        external_dir: Path,
    ) -> None:
        """Test runner loads time-series dataset correctly."""
        dataset = make_test_loaded_dataset(n_samples=2, n_features=2)
        loader = FakeDatasetLoader(dataset)
        ts_config = make_test_timeseries_config(name="test_ts")
        std_registry = FakeDatasetRegistry(make_test_dataset_config(name="other"))
        ts_registry = FakeTimeSeriesRegistry(ts_config)

        _test_hooks.dataset_loader_factory = lambda: loader
        _test_hooks.registry_factory = lambda: std_registry
        _test_hooks.timeseries_registry_factory = lambda: ts_registry
        _test_hooks.perf_counter = lambda: 1.0
        _test_hooks.generate_uuid = lambda: "test-uuid"

        replay_config = make_replay_config(
            dataset="test_ts",
            speed="instant",
        )

        runner = DataReplayRunner(fake_producer, replay_config, external_dir)
        stats = runner.run()

        # Should call load_timeseries, not load
        assert len(loader.load_timeseries_calls) == 1
        assert len(loader.load_calls) == 0
        assert stats["rows_processed"] == 2

    def test_run_raises_for_unknown_dataset(
        self,
        restore_hooks: None,
        fake_producer: FakeProducer,
        external_dir: Path,
    ) -> None:
        """Test runner raises KeyError for unknown dataset."""
        dataset = make_test_loaded_dataset()
        loader = FakeDatasetLoader(dataset)
        std_registry = FakeDatasetRegistry(make_test_dataset_config(name="known"))
        ts_registry = FakeTimeSeriesRegistry()

        _test_hooks.dataset_loader_factory = lambda: loader
        _test_hooks.registry_factory = lambda: std_registry
        _test_hooks.timeseries_registry_factory = lambda: ts_registry

        replay_config = make_replay_config(
            dataset="unknown",
            speed="instant",
        )

        runner = DataReplayRunner(fake_producer, replay_config, external_dir)

        with pytest.raises(KeyError, match="unknown"):
            runner.run()

    def test_run_flushes_producer(
        self,
        restore_hooks: None,
        fake_producer: FakeProducer,
        external_dir: Path,
    ) -> None:
        """Test runner flushes producer at end."""
        dataset = make_test_loaded_dataset(n_samples=1, n_features=1)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        _test_hooks.dataset_loader_factory = lambda: loader
        _test_hooks.registry_factory = lambda: registry
        _test_hooks.timeseries_registry_factory = lambda: ts_registry
        _test_hooks.perf_counter = lambda: 1.0
        _test_hooks.generate_uuid = lambda: "test-uuid"

        replay_config = make_replay_config(
            dataset="test",
            speed="instant",
        )

        runner = DataReplayRunner(fake_producer, replay_config, external_dir)
        runner.run()

        assert fake_producer.flush_count == 1

    def test_run_no_delay_with_full_batches(
        self,
        restore_hooks: None,
        fake_producer: FakeProducer,
        external_dir: Path,
    ) -> None:
        """Test runner with instant speed and exact batch multiple (delay=0)."""
        # 3 samples * 2 features = 6 events, batch size 3 = 2 full batches, no remainder
        dataset = make_test_loaded_dataset(n_samples=3, n_features=2)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        _test_hooks.dataset_loader_factory = lambda: loader
        _test_hooks.registry_factory = lambda: registry
        _test_hooks.timeseries_registry_factory = lambda: ts_registry
        _test_hooks.perf_counter = lambda: 1.0
        _test_hooks.generate_uuid = lambda: "test-uuid"

        replay_config = make_replay_config(
            dataset="test",
            speed="instant",
            batch_size=3,
        )

        runner = DataReplayRunner(fake_producer, replay_config, external_dir)
        stats = runner.run()

        # 6 events / 3 per batch = 2 batches, no remainder
        assert stats["batches_sent"] == 2
        assert stats["events_sent"] == 6


class TestRunReplay:
    """Tests for run_replay convenience function."""

    def test_creates_runner_and_runs(
        self,
        restore_hooks: None,
        fake_producer: FakeProducer,
        external_dir: Path,
    ) -> None:
        """Test run_replay creates runner and executes."""
        dataset = make_test_loaded_dataset(n_samples=2, n_features=2)
        loader = FakeDatasetLoader(dataset)
        config = make_test_dataset_config(name="test")
        registry = FakeDatasetRegistry(config)
        ts_registry = FakeTimeSeriesRegistry()

        _test_hooks.dataset_loader_factory = lambda: loader
        _test_hooks.registry_factory = lambda: registry
        _test_hooks.timeseries_registry_factory = lambda: ts_registry
        _test_hooks.perf_counter = lambda: 1.0
        _test_hooks.generate_uuid = lambda: "test-uuid"

        replay_config = make_replay_config(
            dataset="test",
            speed="instant",
        )

        stats = run_replay(fake_producer, replay_config, external_dir)

        assert stats["rows_processed"] == 2
        assert stats["events_sent"] == 4
