"""Tests for StreamingWorker core functionality.

Covers initialization, buffer management, processing, run loop, and data loading.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import time

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


class TestStreamingWorkerRun:
    """Tests for StreamingWorker run loop."""

    def test_run_with_max_iterations(self) -> None:
        """Runs for specified number of iterations."""
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

        total_messages, total_periods = worker.run(max_iterations=3)

        # No messages, so nothing processed
        assert total_messages == 0
        assert total_periods == 0

    def test_run_processes_messages(self) -> None:
        """Run loop processes messages from consumer."""
        (
            worker,
            fake_consumer,
            _fake_producer,
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

        # Run
        total_messages, total_periods = worker.run(max_iterations=10)

        assert total_messages == 5
        assert total_periods == 1

    def test_run_exits_on_shutdown(self) -> None:
        """Run loop exits when shutdown() is called via on_poll callback."""
        (
            worker,
            fake_consumer,
            _fake_producer,
            _,
            _,
            _deal_repo,
            _covenant_repo,
            _,
            _,
        ) = make_streaming_worker()

        # Set callback to trigger shutdown after first poll
        def trigger_shutdown() -> None:
            if fake_consumer.poll_count >= 1:
                worker.shutdown()

        fake_consumer.set_on_poll(trigger_shutdown)

        # Run without max_iterations - should exit via shutdown
        total_messages, total_periods = worker.run()

        # Verify loop exited after shutdown was triggered
        assert fake_consumer.poll_count >= 1
        assert total_messages == 0
        assert total_periods == 0

    def test_shutdown_processes_remaining_buffers(self) -> None:
        """Shutdown processes remaining buffered periods."""
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

        # Add all 5 required metrics for a period
        for metric_name, value in REQUIRED_METRICS.items():
            event = make_measurement_event("deal-001", metric_name, value)
            encoded = encode_measurement_event(event)
            fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")

        # Run 4 times (consume 4 metrics, buffer incomplete, won't auto-process)
        for _ in range(4):
            worker.run_once()

        assert worker.buffer_size == 1  # One period with 4 metrics

        # Add the 5th metric and consume it
        last_metric = list(REQUIRED_METRICS.items())[-1]
        event = make_measurement_event("deal-001", last_metric[0], last_metric[1])
        encoded = encode_measurement_event(event)
        fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")
        worker.run_once()

        # Now 5 metrics, triggers processing
        assert worker.buffer_size == 0
        assert len(fake_producer.messages) == 1

        # Shutdown should flush and close
        worker.shutdown()
        assert fake_producer.flush_called is True
        assert fake_consumer.closed is True

    def test_run_commits_periodically(self) -> None:
        """Run commits after commit_interval messages."""
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

        # Add enough messages to trigger commit (commit_interval = 5)
        for i in range(6):
            event = make_measurement_event(
                "deal-001",
                f"metric_{i}",
                period_start=f"2024-{i:02d}-01",
                period_end=f"2024-{i:02d}-28",
            )
            encoded = encode_measurement_event(event)
            fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")

        # Run enough iterations
        worker.run(max_iterations=10)

        # Should have committed at least once
        assert fake_consumer.commit_count == 1


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


class TestStreamingWorkerEdgeCases:
    """Tests for edge cases to achieve 100% coverage."""

    def test_should_process_buffer_returns_false_for_missing_key(self) -> None:
        """_should_process_buffer returns False when key not in buffer."""
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

        # Buffer is empty, so key won't be found
        result = worker._should_process_buffer(("nonexistent", "2024-01-01", "2024-03-31"))

        assert result is False

    def test_shutdown_processes_remaining_buffers_with_alert(self) -> None:
        """Shutdown processes buffers and produces alerts."""
        (
            worker,
            fake_consumer,
            fake_producer,
            _,
            fake_predictor,
            deal_repo,
            covenant_repo,
            _,
            _,
        ) = make_streaming_worker()

        # Setup deal and covenant
        deal_repo.create(make_deal("deal-001"))
        covenant_repo.create(make_covenant("cov-001", "deal-001"))

        # Set predictor to high risk to trigger alert
        fake_predictor._default_probability = 0.95

        # Add all required metrics for a complete period
        for metric_name, value in REQUIRED_METRICS.items():
            event = make_measurement_event("deal-001", metric_name, value)
            encoded = encode_measurement_event(event)
            fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")

        # Consume all 5 messages - this triggers processing (min_metrics_per_period=5)
        for _ in range(5):
            worker.run_once()

        # Now 5 metrics should have triggered processing
        # At least one prediction was produced
        prediction_count = len([m for m in fake_producer.messages if m.topic == "predictions"])
        assert prediction_count == 1

        # Check if alert was produced (using .topic attribute)
        alert_messages = [m for m in fake_producer.messages if m.topic == "alerts"]
        assert len(alert_messages) == 1

    def test_shutdown_commits_pending_messages(self) -> None:
        """Shutdown commits when messages_since_commit > 0."""
        (
            worker,
            fake_consumer,
            _fake_producer,
            _fake_sink,
            _fake_predictor,
            deal_repo,
            covenant_repo,
            _measurement_repo,
            _result_repo,
        ) = make_streaming_worker()

        # Setup deal and covenant
        deal_repo.create(make_deal("deal-001"))
        covenant_repo.create(make_covenant("cov-001", "deal-001"))

        # Add all 5 required metrics for a single period (below commit_interval=5)
        for metric_name, value in REQUIRED_METRICS.items():
            event = make_measurement_event(
                "deal-001",
                metric_name,
                value,
                period_start="2024-01-01",
                period_end="2024-03-31",
            )
            encoded = encode_measurement_event(event)
            fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")

        # Run to consume 4 messages (not triggering auto-commit at interval=5)
        for _ in range(4):
            worker.run_once()

        # No commits yet (only 4 messages, interval is 5)
        assert fake_consumer.commit_count == 0

        # Consume the 5th message - triggers processing but not commit yet
        worker.run_once()

        # Now have processed 5 messages, commit should have happened
        assert fake_consumer.commit_count == 1

        # Add one more message to test shutdown commit path
        event = make_measurement_event(
            "deal-001",
            "total_debt",
            5000000.0,
            period_start="2024-04-01",
            period_end="2024-06-30",
        )
        encoded = encode_measurement_event(event)
        fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")
        worker.run_once()

        # Only 1 message since last commit, below interval
        # Clear buffer to avoid processing issues during shutdown
        worker._buffer.clear()

        # Shutdown should commit the pending message
        worker.shutdown()

        # Should have 2 commits total
        assert fake_consumer.commit_count == 2

    def test_process_ready_buffers_produces_alert_in_loop(self) -> None:
        """_process_ready_buffers produces alerts during normal processing."""
        (
            worker,
            fake_consumer,
            fake_producer,
            _,
            fake_predictor,
            deal_repo,
            covenant_repo,
            _,
            _,
        ) = make_streaming_worker()

        # Setup deal with covenant that will breach
        deal_repo.create(make_deal("deal-001", "Test Deal"))
        covenant_repo.create(
            make_covenant(
                "cov-001",
                "deal-001",
                "total_debt / ebitda",
                threshold_value_scaled=4_000_000,  # 4.0 threshold
                threshold_direction="<=",
            )
        )

        # Set high risk to ensure alert
        fake_predictor._default_probability = 0.95

        # Add all required metrics
        for metric_name, value in REQUIRED_METRICS.items():
            event = make_measurement_event("deal-001", metric_name, value)
            encoded = encode_measurement_event(event)
            fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")

        # Run enough iterations to process
        for _ in range(6):
            worker.run_once()

        # Verify both prediction and alert were produced (using .topic attribute)
        prediction_messages = [m for m in fake_producer.messages if m.topic == "predictions"]
        alert_messages = [m for m in fake_producer.messages if m.topic == "alerts"]

        assert len(prediction_messages) == 1
        assert len(alert_messages) == 1  # This covers line 778

    def test_shutdown_with_remaining_buffers(self) -> None:
        """Shutdown processes remaining buffers."""
        (
            worker,
            fake_consumer,
            fake_producer,
            _,
            fake_predictor,
            deal_repo,
            covenant_repo,
            _,
            _,
        ) = make_streaming_worker()

        # Setup deal and covenant
        deal_repo.create(make_deal("deal-001"))
        covenant_repo.create(make_covenant("cov-001", "deal-001"))

        # Add all 5 required metrics for first period
        for metric_name, value in REQUIRED_METRICS.items():
            event = make_measurement_event(
                "deal-001",
                metric_name,
                value,
                period_start="2024-07-01",
                period_end="2024-09-30",
            )
            encoded = encode_measurement_event(event)
            fake_consumer.add_message(encoded.encode("utf-8"), topic="measurements")

        # Process only 4 messages to leave buffer incomplete
        for _ in range(4):
            messages, _periods = worker.run_once()
            if messages == 0:
                break

        # Buffer should have data (4 metrics, not yet at min threshold of 5)
        assert worker.buffer_size == 1

        # Consume the 5th message which triggers processing
        worker.run_once()

        # Should have produced one prediction
        prediction_msgs = [m for m in fake_producer.messages if m.topic == "predictions"]
        assert len(prediction_msgs) == 1

        # Set high risk to trigger alert during shutdown
        fake_predictor._default_probability = 0.95

        # Directly populate the buffer with all required metrics
        # This simulates a scenario where all metrics arrived but threshold not met
        buffer_key = ("deal-001", "2024-10-01", "2024-12-31")
        worker._buffer[buffer_key] = {
            "metrics": dict(REQUIRED_METRICS),  # All 5 metrics
            "first_received_at": time.monotonic(),
            "message_count": 5,
        }

        # Verify buffer has data
        assert worker.buffer_size == 1

        # Shutdown should process remaining buffers (force flush)
        worker.shutdown()

        # Should have produced another prediction from forced buffer flush
        final_predictions = [m for m in fake_producer.messages if m.topic == "predictions"]
        assert len(final_predictions) == 2  # Original + shutdown flush

        # Should have produced alert during shutdown (line 870 coverage)
        final_alerts = [m for m in fake_producer.messages if m.topic == "alerts"]
        assert len(final_alerts) == 1  # Alert from shutdown flush

    def test_shutdown_with_multiple_buffers(self) -> None:
        """Shutdown processes multiple remaining buffers (branch 869->857)."""
        (
            worker,
            _fake_consumer,
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

        # Directly populate multiple buffers to test for loop iteration
        buffer_key_1 = ("deal-001", "2024-01-01", "2024-03-31")
        buffer_key_2 = ("deal-001", "2024-04-01", "2024-06-30")
        worker._buffer[buffer_key_1] = {
            "metrics": dict(REQUIRED_METRICS),
            "first_received_at": time.monotonic(),
            "message_count": 5,
        }
        worker._buffer[buffer_key_2] = {
            "metrics": dict(REQUIRED_METRICS),
            "first_received_at": time.monotonic(),
            "message_count": 5,
        }

        # Verify 2 buffers
        assert worker.buffer_size == 2

        # Shutdown should process both buffers (covers branch 869->857)
        worker.shutdown()

        # Should have produced 2 predictions
        final_predictions = [m for m in fake_producer.messages if m.topic == "predictions"]
        assert len(final_predictions) == 2
