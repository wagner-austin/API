"""Tests for replay script _test_hooks module."""

from __future__ import annotations

from scripts.replay_data._test_hooks import (
    FakeProducer,
    _real_dataset_loader_factory,
    _real_registry_factory,
    _real_sleep,
    _real_time,
    _real_timeseries_registry_factory,
    _real_uuid,
    dataset_loader_factory,
    generate_uuid,
    perf_counter,
    registry_factory,
    sleep,
    timeseries_registry_factory,
)

from covenant_radar_api.streaming.schemas import make_measurement_event


class TestRealTime:
    """Tests for _real_time function."""

    def test_returns_positive_float(self) -> None:
        """Test returns positive float value."""
        result = _real_time()
        # Type is float and value is positive
        assert result > 0.0
        assert result == float(result)


class TestRealSleep:
    """Tests for _real_sleep function."""

    def test_accepts_float_seconds(self) -> None:
        """Test accepts float seconds argument."""
        # Very short sleep to avoid slowing tests
        _real_sleep(0.001)


class TestRealUUID:
    """Tests for _real_uuid function."""

    def test_returns_uuid_format(self) -> None:
        """Test returns UUID format string."""
        result = _real_uuid()
        # UUID format: 8-4-4-4-12 hex chars with dashes
        assert len(result) == 36
        assert result[8] == "-"
        assert result[13] == "-"
        assert result[18] == "-"
        assert result[23] == "-"

    def test_returns_unique_values(self) -> None:
        """Test returns unique values each call."""
        results = [_real_uuid() for _ in range(10)]
        assert len(set(results)) == 10


class TestModuleLevelHooks:
    """Tests for module-level hook defaults."""

    def test_perf_counter_returns_positive_float(self) -> None:
        """Test perf_counter hook returns positive float."""
        result = perf_counter()
        assert result > 0.0
        assert result == float(result)

    def test_sleep_accepts_zero(self) -> None:
        """Test sleep hook accepts zero delay."""
        # Should not raise
        sleep(0.0)

    def test_generate_uuid_returns_uuid_format(self) -> None:
        """Test generate_uuid hook returns UUID format."""
        result = generate_uuid()
        assert len(result) == 36
        assert result[8] == "-"


class TestDatasetLoaderFactory:
    """Tests for dataset loader factory hooks."""

    def test_real_factory_returns_loader_with_load_method(self) -> None:
        """Test real factory returns loader with load method."""
        loader = _real_dataset_loader_factory()
        # Verify methods exist by checking they are callable
        load_method = loader.load
        assert callable(load_method)

    def test_real_factory_returns_loader_with_load_timeseries_method(self) -> None:
        """Test real factory returns loader with load_timeseries method."""
        loader = _real_dataset_loader_factory()
        load_timeseries_method = loader.load_timeseries
        assert callable(load_timeseries_method)

    def test_module_hook_returns_loader_with_load_method(self) -> None:
        """Test module-level factory hook returns loader with load method."""
        loader = dataset_loader_factory()
        load_method = loader.load
        assert callable(load_method)


class TestRegistryFactory:
    """Tests for registry factory hooks."""

    def test_real_factory_returns_registry_with_get_method(self) -> None:
        """Test real factory returns registry with get method."""
        registry = _real_registry_factory()
        get_method = registry.get
        assert callable(get_method)

    def test_real_factory_returns_registry_with_list_names_method(self) -> None:
        """Test real factory returns registry with list_names method."""
        registry = _real_registry_factory()
        list_names_method = registry.list_names
        assert callable(list_names_method)

    def test_module_hook_returns_registry_with_get_method(self) -> None:
        """Test module-level factory hook returns registry with get method."""
        registry = registry_factory()
        get_method = registry.get
        assert callable(get_method)


class TestTimeSeriesRegistryFactory:
    """Tests for time-series registry factory hooks."""

    def test_real_factory_returns_registry_with_get_method(self) -> None:
        """Test real factory returns registry with get method."""
        registry = _real_timeseries_registry_factory()
        get_method = registry.get
        assert callable(get_method)

    def test_real_factory_returns_registry_with_list_names_method(self) -> None:
        """Test real factory returns registry with list_names method."""
        registry = _real_timeseries_registry_factory()
        list_names_method = registry.list_names
        assert callable(list_names_method)

    def test_module_hook_returns_registry_with_get_method(self) -> None:
        """Test module-level factory hook returns registry with get method."""
        registry = timeseries_registry_factory()
        get_method = registry.get
        assert callable(get_method)


class TestFakeProducer:
    """Tests for FakeProducer class."""

    def test_init_creates_empty_events_list(self) -> None:
        """Test init creates empty events list."""
        fake = FakeProducer()
        assert fake.events == []

    def test_init_sets_poll_count_zero(self) -> None:
        """Test init sets poll count to zero."""
        fake = FakeProducer()
        assert fake.poll_count == 0

    def test_init_sets_flush_count_zero(self) -> None:
        """Test init sets flush count to zero."""
        fake = FakeProducer()
        assert fake.flush_count == 0

    def test_produce_event_records_event(self) -> None:
        """Test produce_event records event and topic."""
        fake = FakeProducer()
        event = make_measurement_event(
            event_id="test-id",
            deal_id="deal-001",
            period_start="2024-01-01",
            period_end="2024-01-31",
            metric_name="ratio",
            metric_value=1.5,
            timestamp="2024-01-15T12:00:00Z",
        )

        fake.produce_event(event, "test.topic")

        assert len(fake.events) == 1
        recorded_event, recorded_topic = fake.events[0]
        assert recorded_event == event
        assert recorded_topic == "test.topic"

    def test_produce_event_records_multiple(self) -> None:
        """Test produce_event records multiple events."""
        fake = FakeProducer()

        for i in range(5):
            event = make_measurement_event(
                event_id=f"id-{i}",
                deal_id=f"deal-{i:03d}",
                period_start="2024-01-01",
                period_end="2024-01-31",
                metric_name="metric",
                metric_value=float(i),
                timestamp="2024-01-15T12:00:00Z",
            )
            fake.produce_event(event, f"topic-{i}")

        assert len(fake.events) == 5

    def test_poll_increments_count(self) -> None:
        """Test poll increments poll_count."""
        fake = FakeProducer()

        fake.poll(1.0)
        assert fake.poll_count == 1

        fake.poll(0.5)
        assert fake.poll_count == 2

    def test_poll_returns_zero(self) -> None:
        """Test poll returns zero."""
        fake = FakeProducer()
        result = fake.poll(1.0)
        assert result == 0

    def test_flush_increments_count(self) -> None:
        """Test flush increments flush_count."""
        fake = FakeProducer()

        fake.flush(10.0)
        assert fake.flush_count == 1

        fake.flush(5.0)
        assert fake.flush_count == 2

    def test_flush_returns_zero(self) -> None:
        """Test flush returns zero."""
        fake = FakeProducer()
        result = fake.flush(10.0)
        assert result == 0
