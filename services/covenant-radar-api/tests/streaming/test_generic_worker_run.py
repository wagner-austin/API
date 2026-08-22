"""Tests for GenericStreamingWorker."""

from __future__ import annotations

import covenant_radar_api.streaming._test_hooks_generic_worker as _hooks
from covenant_radar_api.domains.base_schemas import (
    BaseAlertEventV1,
    BasePredictionEventV1,
    decode_base_alert_event,
    decode_base_prediction_event,
)

from ._test_generic_worker_fixtures import (
    make_base_input_payload,
    make_generic_streaming_worker,
)


class TestRunOnce:
    """Tests for GenericStreamingWorker.run_once."""

    def setup_method(self) -> None:
        """Set up deterministic hooks."""
        _hooks.perf_counter = lambda: 100.0
        _hooks.generate_uuid = lambda: "test-uuid-001"
        _hooks.current_iso_timestamp = lambda: "2026-01-15T12:00:00Z"

    def teardown_method(self) -> None:
        """Restore real hooks."""
        _hooks.use_real_hooks()

    def test_no_message_returns_zero(self) -> None:
        """Empty consumer returns (0, 0)."""
        worker, _, _, _, _, _ = make_generic_streaming_worker()

        consumed, produced = worker.run_once()

        assert consumed == 0
        assert produced == 0

    def test_processes_message(self) -> None:
        """Consumer with message returns (1, produced_count)."""
        worker, _, consumer, _, _, _ = make_generic_streaming_worker()
        payload: str = make_base_input_payload()
        consumer.add_message(value=payload.encode("utf-8"))

        consumed, produced = worker.run_once()

        assert consumed == 1
        assert produced == 1  # prediction only (below threshold)

    def test_produces_prediction(self) -> None:
        """Producer receives prediction on prediction topic."""
        worker, _, consumer, producer, _, _ = make_generic_streaming_worker()
        payload: str = make_base_input_payload(entity_id="entity-abc")
        consumer.add_message(value=payload.encode("utf-8"))

        worker.run_once()

        assert len(producer.messages) == 1
        msg = producer.messages[0]
        assert msg.topic == "test-predictions"
        decoded: BasePredictionEventV1 = decode_base_prediction_event(msg.value.decode("utf-8"))
        assert decoded["entity_id"] == "entity-abc"

    def test_produces_alert_when_threshold_exceeded(self) -> None:
        """Producer receives alert when prediction exceeds threshold."""
        worker, _, consumer, producer, _, _ = make_generic_streaming_worker(
            probability=0.90,
            alert_threshold=0.80,
        )
        payload: str = make_base_input_payload()
        consumer.add_message(value=payload.encode("utf-8"))

        consumed, produced = worker.run_once()

        assert consumed == 1
        assert produced == 2  # prediction + alert
        assert len(producer.messages) == 2
        assert producer.messages[0].topic == "test-predictions"
        assert producer.messages[1].topic == "test-alerts"

    def test_alert_decodable(self) -> None:
        """Produced alert is a valid BaseAlertEventV1 JSON."""
        worker, _, consumer, producer, _, _ = make_generic_streaming_worker(
            probability=0.90,
            alert_threshold=0.80,
        )
        payload: str = make_base_input_payload(entity_id="entity-xyz")
        consumer.add_message(value=payload.encode("utf-8"))

        worker.run_once()

        alert_msg = producer.messages[1]
        decoded: BaseAlertEventV1 = decode_base_alert_event(alert_msg.value.decode("utf-8"))
        assert decoded["entity_id"] == "entity-xyz"
        assert decoded["severity"] == "critical"

    def test_message_key_is_entity_id(self) -> None:
        """Producer message key is entity_id bytes."""
        worker, _, consumer, producer, _, _ = make_generic_streaming_worker()
        payload: str = make_base_input_payload(entity_id="entity-key-test")
        consumer.add_message(value=payload.encode("utf-8"))

        worker.run_once()

        assert producer.messages[0].key == b"entity-key-test"


class TestRun:
    """Tests for GenericStreamingWorker.run."""

    def setup_method(self) -> None:
        """Set up deterministic hooks."""
        _hooks.perf_counter = lambda: 100.0
        _hooks.generate_uuid = lambda: "test-uuid-001"
        _hooks.current_iso_timestamp = lambda: "2026-01-15T12:00:00Z"

    def teardown_method(self) -> None:
        """Restore real hooks."""
        _hooks.use_real_hooks()

    def test_runs_max_iterations(self) -> None:
        """run(max_iterations=3) processes exactly 3 iterations."""
        worker, _, consumer, _, _, _ = make_generic_streaming_worker()
        # Add 5 messages, but only 3 iterations
        for i in range(5):
            payload: str = make_base_input_payload(entity_id=f"entity-{i:03d}")
            consumer.add_message(value=payload.encode("utf-8"))

        worker.run(max_iterations=3)

        # 3 iterations consumed 3 messages
        assert consumer.poll_count == 3

    def test_returns_totals(self) -> None:
        """Run returns (total_consumed, total_produced)."""
        worker, _, consumer, _, _, _ = make_generic_streaming_worker()
        for i in range(2):
            payload: str = make_base_input_payload(entity_id=f"entity-{i:03d}")
            consumer.add_message(value=payload.encode("utf-8"))

        total_consumed, total_produced = worker.run(max_iterations=2)

        assert total_consumed == 2
        assert total_produced == 2  # 2 predictions, no alerts

    def test_returns_totals_with_alerts(self) -> None:
        """Totals include alert events."""
        worker, _, consumer, _, _, _ = make_generic_streaming_worker(
            probability=0.90,
            alert_threshold=0.80,
        )
        payload: str = make_base_input_payload()
        consumer.add_message(value=payload.encode("utf-8"))

        total_consumed, total_produced = worker.run(max_iterations=1)

        assert total_consumed == 1
        assert total_produced == 2  # prediction + alert

    def test_handles_empty_iterations(self) -> None:
        """Empty iterations add zero to totals."""
        worker, _, _, _, _, _ = make_generic_streaming_worker()

        total_consumed, total_produced = worker.run(max_iterations=3)

        assert total_consumed == 0
        assert total_produced == 0

    def test_stops_when_shutdown_called(self) -> None:
        """Shutdown from poll callback exits the run loop."""
        worker, _, consumer, _, _, _ = make_generic_streaming_worker()

        # Add many messages so the loop wouldn't stop naturally
        for i in range(100):
            payload: str = make_base_input_payload(entity_id=f"entity-{i:03d}")
            consumer.add_message(value=payload.encode("utf-8"))

        # Shutdown after 2 polls
        poll_calls: int = 0

        def on_poll() -> None:
            nonlocal poll_calls
            poll_calls += 1
            if poll_calls >= 2:
                worker.shutdown()

        consumer.set_on_poll(on_poll)

        worker.run()

        # Should have stopped early (not processed all 100)
        assert consumer.poll_count <= 3

    def test_is_running_true_during_run(self) -> None:
        """is_running is True while run loop is active."""
        worker, _, consumer, _, _, _ = make_generic_streaming_worker()
        observed_running: list[bool] = []

        def on_poll() -> None:
            observed_running.append(worker.is_running)
            worker.shutdown()

        consumer.set_on_poll(on_poll)
        worker.run()

        assert len(observed_running) == 1
        assert observed_running[0] is True


class TestShutdown:
    """Tests for GenericStreamingWorker.shutdown."""

    def test_flush_producer(self) -> None:
        """Shutdown flushes the producer."""
        worker, _, _, producer, _, _ = make_generic_streaming_worker()

        worker.shutdown()

        assert producer.flush_called is True

    def test_close_consumer(self) -> None:
        """Shutdown closes the consumer."""
        worker, _, consumer, _, _, _ = make_generic_streaming_worker()

        worker.shutdown()

        assert consumer.closed is True

    def test_sets_running_false(self) -> None:
        """Shutdown sets is_running to False."""
        worker, _, _, _, _, _ = make_generic_streaming_worker()

        worker.shutdown()

        assert worker.is_running is False


class TestSubscribesToTheDomainTopic:
    """run() subscribes the consumer to the domain's own input topic."""

    def test_subscribes_before_polling(self) -> None:
        """The worker subscribes itself rather than trusting the caller.

        Both output topics already come from domain.config. Leaving the
        input topic to the caller meant a worker could poll an unsubscribed
        consumer forever, reporting zero messages and no error.
        """
        worker, domain, consumer, _, _, _ = make_generic_streaming_worker()

        worker.run(max_iterations=0)

        assert consumer.subscribed_topics == (domain.config["input_topic"],)
