"""Tests for StreamingWorker core functionality.

Covers initialization, buffer management, processing, run loop, and data loading.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from covenant_radar_api.streaming.schemas import encode_measurement_event

from ._test_worker_fixtures import (
    REQUIRED_METRICS,
    make_covenant,
    make_covenant_result,
    make_deal,
    make_measurement,
    make_measurement_event,
    make_streaming_worker,
)


class TestStreamingWorkerInit:
    """Tests for StreamingWorker initialization."""

    def test_creates_worker(self) -> None:
        """Creates worker with dependencies."""
        (
            worker,
            _fake_consumer,
            _fake_producer,
            _fake_sink,
            _fake_predictor,
            _deal_repo,
            _covenant_repo,
            _measurement_repo,
            _result_repo,
        ) = make_streaming_worker()
        assert worker.is_running is False
        assert worker.buffer_size == 0


class TestStreamingWorkerBuffer:
    """Tests for StreamingWorker buffer management."""

    def test_add_to_buffer(self) -> None:
        """Adding events increases buffer size."""
        (
            worker,
            fake_consumer,
            _,
            _,
            _,
            deal_repo,
            covenant_repo,
            _,
            _,
        ) = make_streaming_worker()

        # Setup deal and covenant
        deal_repo.create(make_deal("deal-001"))
        covenant_repo.create(make_covenant("cov-001", "deal-001"))

        # Add measurement event to consumer
        event = make_measurement_event("deal-001", "debt_to_equity")
        encoded = encode_measurement_event(event)
        fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")

        # Run one iteration
        messages, _periods = worker.run_once()

        assert messages == 1
        assert worker.buffer_size == 1  # Event buffered, not processed yet

    def test_buffer_processes_when_enough_metrics(self) -> None:
        """Buffer processes when min_metrics_per_period reached."""
        (
            worker,
            fake_consumer,
            fake_producer,
            _,
            _,
            deal_repo,
            covenant_repo,
            _,
            _,
        ) = make_streaming_worker()

        # Setup deal and covenant
        deal_repo.create(make_deal("deal-001"))
        covenant_repo.create(make_covenant("cov-001", "deal-001"))

        # Add all 5 required metrics
        for metric_name, value in REQUIRED_METRICS.items():
            event = make_measurement_event("deal-001", metric_name, value)
            encoded = encode_measurement_event(event)
            fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")

        # Run iterations to consume all messages and process
        for _ in range(5):
            worker.run_once()

        # Should have processed and produced prediction
        assert len(fake_producer.messages) == 1


class TestStreamingWorkerProcessing:
    """Tests for StreamingWorker processing."""

    def test_process_buffered_period(self) -> None:
        """Processes buffered period and returns result."""
        (
            worker,
            _,
            _,
            fake_sink,
            _,
            deal_repo,
            covenant_repo,
            _,
            _,
        ) = make_streaming_worker()

        # Setup deal and covenant
        deal_repo.create(make_deal("deal-001", "Test Deal"))
        covenant_repo.create(make_covenant("cov-001", "deal-001"))

        # Process directly
        result = worker.process_buffered_period(
            deal_id="deal-001",
            period_start="2024-01-01",
            period_end="2024-03-31",
            metrics=REQUIRED_METRICS,
        )

        # Verify prediction event fields
        prediction = result["prediction"]
        assert prediction["deal_id"] == "deal-001"
        assert prediction["period_start"] == "2024-01-01"
        assert prediction["period_end"] == "2024-03-31"
        assert prediction["model_version"] == "test-v1"

        # Verify latencies are non-negative
        assert result["evaluation_latency_ms"] >= 0
        assert result["prediction_latency_ms"] >= 0

        # Verify metrics were recorded (at least eval + prediction latency)
        assert len(fake_sink.histograms) == 2
        assert len(fake_sink.gauges) == 1  # Risk probability

    def test_process_generates_alert_on_breach(self) -> None:
        """Generates alert when covenant is breached."""
        (
            worker,
            _,
            _,
            _,
            _,
            deal_repo,
            covenant_repo,
            _,
            _,
        ) = make_streaming_worker()

        # Setup deal and covenant with threshold that will be breached
        # REQUIRED_METRICS: total_debt=5M, ebitda=1M, ratio=5.0
        # Threshold 4.0, so 5.0 > 4.0 is a breach
        deal_repo.create(make_deal("deal-001", "Test Deal"))
        covenant_repo.create(
            make_covenant(
                "cov-001",
                "deal-001",
                "total_debt / ebitda",  # Uses available metrics
                threshold_value_scaled=4_000_000,  # 4.0 threshold (5.0 > 4.0 = breach)
                threshold_direction="<=",
            )
        )

        # Process with metrics that breach covenant (5.0 > 4.0)
        result = worker.process_buffered_period(
            deal_id="deal-001",
            period_start="2024-01-01",
            period_end="2024-03-31",
            metrics=REQUIRED_METRICS,
        )

        # Should have alert with specific fields
        alert = result["alert"]
        if alert is None:
            raise AssertionError("Expected alert to be generated")
        assert alert["deal_id"] == "deal-001"
        assert alert["alert_type"] == "breach"

    def test_process_generates_alert_on_high_risk(self) -> None:
        """Generates alert when risk exceeds threshold."""
        (
            worker,
            _,
            _,
            _,
            fake_predictor,
            deal_repo,
            covenant_repo,
            _,
            _,
        ) = make_streaming_worker()

        # Setup deal and covenant
        deal_repo.create(make_deal("deal-001", "Test Deal"))
        covenant_repo.create(make_covenant("cov-001", "deal-001"))

        # Create worker with high-risk predictor
        fake_predictor._default_probability = 0.95  # Above 0.80 threshold

        # Process
        result = worker.process_buffered_period(
            deal_id="deal-001",
            period_start="2024-01-01",
            period_end="2024-03-31",
            metrics=REQUIRED_METRICS,
        )

        # Should have alert due to high risk with specific fields
        alert = result["alert"]
        if alert is None:
            raise AssertionError("Expected alert due to high risk")
        assert alert["alert_type"] == "high_risk"
        assert alert["severity"] == "critical"  # 0.95 >= 0.90


class TestStreamingWorkerDataLoading:
    """Tests for StreamingWorker data loading methods."""

    def test_load_historical_metrics(self) -> None:
        """Loads historical metrics grouped by period."""
        (
            worker,
            _,
            _,
            _,
            _,
            deal_repo,
            _,
            measurement_repo,
            _,
        ) = make_streaming_worker()

        # Setup
        deal_repo.create(make_deal("deal-001"))

        # Add historical measurements
        measurement_repo.add_many(
            [
                make_measurement(
                    "deal-001", "debt_to_equity", 1_500_000, "2024-01-01", "2024-03-31"
                ),
                make_measurement(
                    "deal-001", "current_ratio", 2_000_000, "2024-01-01", "2024-03-31"
                ),
                make_measurement(
                    "deal-001", "debt_to_equity", 1_600_000, "2024-04-01", "2024-06-30"
                ),
            ]
        )

        # Load historical
        historical = worker._load_historical_metrics("deal-001", periods_back=5)

        assert len(historical) == 2
        assert historical["2024-03-31"]["debt_to_equity"] == 1_500_000
        assert historical["2024-06-30"]["debt_to_equity"] == 1_600_000

    def test_load_recent_results(self) -> None:
        """Loads recent covenant results sorted by period."""
        (
            worker,
            _,
            _,
            _,
            _,
            deal_repo,
            covenant_repo,
            _,
            result_repo,
        ) = make_streaming_worker()

        # Setup
        deal_repo.create(make_deal("deal-001"))
        covenant_repo.create(make_covenant("cov-001", "deal-001"))

        # Add results
        result_repo.save(make_covenant_result("cov-001", period_end="2024-03-31"))
        result_repo.save(make_covenant_result("cov-001", period_end="2024-06-30"))
        result_repo.save(make_covenant_result("cov-001", period_end="2024-09-30"))

        # Load recent
        recent = worker._load_recent_results("deal-001", limit=2)

        assert len(recent) == 2
        # Should be sorted descending by period_end
        assert recent[0]["period_end_iso"] == "2024-09-30"
        assert recent[1]["period_end_iso"] == "2024-06-30"
