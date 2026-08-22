"""Streaming inference worker for covenant breach prediction.

Consumes measurement events from Kafka, aggregates by deal/period,
runs covenant evaluation and ML prediction, publishes results.

Architecture:
1. Consume measurement events from Kafka
2. Buffer measurements by (deal_id, period_start, period_end)
3. When buffer has enough metrics or timeout, process the batch
4. Query database for deal/covenant data
5. Run deterministic covenant evaluation
6. Run ML prediction
7. Produce prediction/alert events to Kafka
8. Emit metrics to Datadog

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import time

from covenant_domain import (
    DealId,
)
from platform_core.logging import get_logger

from covenant_radar_api.streaming.worker_evaluation import _StreamingWorkerEvaluation
from covenant_radar_api.streaming.worker_events import (
    ProcessingResult,
    _count_breaches,
    _current_iso_timestamp,
    _determine_alert_severity,
    _determine_alert_type,
    _determine_evaluation_status,
    _generate_alert_message,
    _generate_event_id,
    _scale_metrics,
)

from .consumer import ConsumedMeasurement, UndecodableMessage
from .schemas import (
    AlertEventV1,
    make_alert_event,
    make_prediction_event,
)

_log = get_logger(__name__)


class StreamingWorker(_StreamingWorkerEvaluation):
    """Kafka streaming worker for real-time inference.

    Consumes measurement events, buffers by deal/period, runs evaluation
    and ML prediction, and produces prediction/alert events.

    The worker maintains an in-memory buffer of measurements grouped by
    (deal_id, period_start, period_end). When a buffer has enough metrics
    or times out, it processes the batch:

    1. Load deal and covenant data from database
    2. Run deterministic covenant evaluation
    3. Extract ML features
    4. Run ML prediction
    5. Produce prediction event to Kafka
    6. If high risk, produce alert event
    7. Emit metrics to Datadog

    Example:
        worker = StreamingWorker(
            consumer=consumer,
            producer=producer,
            metrics=metrics_client,
            model=model,
            deal_repo=deal_repo,
            covenant_repo=covenant_repo,
            measurement_repo=measurement_repo,
            result_repo=result_repo,
            sector_encoder={"Technology": 0},
            region_encoder={"North America": 0},
            config=make_default_worker_config(),
        )
        worker.run()
    """

    def process_buffered_period(
        self,
        deal_id: str,
        period_start: str,
        period_end: str,
        metrics: dict[str, float],
    ) -> ProcessingResult:
        """Process a buffered period and generate prediction.

        Args:
            deal_id: Deal identifier.
            period_start: Period start date.
            period_end: Period end date.
            metrics: Metrics for this period (float values).

        Returns:
            ProcessingResult with prediction and optional alert.

        Raises:
            KeyError: If deal does not exist.
        """
        deal_id_typed = DealId(value=deal_id)

        # Load deal data
        deal = self._load_deal(deal_id)

        # Load covenants
        covenants = self._load_covenants(deal_id)

        # Load historical data for features
        recent_results = self._load_recent_results(deal_id)
        historical = self._load_historical_metrics(deal_id, periods_back=5)

        # Scale current metrics
        metrics_scaled = _scale_metrics(metrics)

        # Run covenant evaluation
        eval_start = time.perf_counter()
        results = self._run_evaluation(
            deal_id_typed, covenants, period_start, period_end, metrics_scaled
        )
        evaluation_status = _determine_evaluation_status(results)
        breaches_count = _count_breaches(results)
        covenants_evaluated = len(results)
        evaluation_latency_ms = int((time.perf_counter() - eval_start) * 1000)

        # Emit evaluation latency metric
        self._metrics.record_evaluation_latency(
            deal_id, evaluation_status, float(evaluation_latency_ms)
        )

        # Build features and run ML prediction
        features = self._build_features(deal, metrics_scaled, historical, recent_results)
        risk_probability, risk_tier, prediction_latency_ms = self._run_prediction(features)

        # Emit prediction metrics
        self._metrics.record_prediction_latency(deal_id, risk_tier, float(prediction_latency_ms))
        self._metrics.set_prediction_risk(deal_id, risk_probability)

        # Create prediction event
        prediction = make_prediction_event(
            event_id=_generate_event_id(),
            deal_id=deal_id,
            period_start=period_start,
            period_end=period_end,
            evaluation_status=evaluation_status,
            covenants_evaluated=covenants_evaluated,
            breaches_count=breaches_count,
            risk_probability=risk_probability,
            risk_tier=risk_tier,
            model_version=self._config["model_version"],
            evaluation_latency_ms=evaluation_latency_ms,
            prediction_latency_ms=prediction_latency_ms,
            processed_at=_current_iso_timestamp(),
        )

        # Check if alert needed
        alert: AlertEventV1 | None = None
        if self._should_alert(evaluation_status, risk_probability):
            alert_type = _determine_alert_type(evaluation_status)
            severity = _determine_alert_severity(risk_probability)
            message = _generate_alert_message(
                deal_id=deal_id,
                deal_name=deal["name"],
                risk_probability=risk_probability,
                evaluation_status=evaluation_status,
                breaches_count=breaches_count,
            )

            alert = make_alert_event(
                event_id=_generate_event_id(),
                deal_id=deal_id,
                alert_type=alert_type,
                severity=severity,
                risk_probability=risk_probability,
                gemini_summary=message,
                triggered_at=_current_iso_timestamp(),
            )

            # Emit alert metric
            self._metrics.increment_alert_triggered(deal_id, severity, alert_type)

        return {
            "prediction": prediction,
            "alert": alert,
            "evaluation_latency_ms": evaluation_latency_ms,
            "prediction_latency_ms": prediction_latency_ms,
        }

    def _process_ready_buffers(self) -> int:
        """Process all ready buffers and produce events.

        Returns:
            Number of periods processed.

        Raises:
            KeyError: If a deal does not exist.
        """
        ready_keys = self._get_ready_buffers()
        processed = 0

        for key in ready_keys:
            deal_id, period_start, period_end = key
            buffered = self._buffer.pop(key)

            missing = self._missing_required_metrics(buffered)
            if len(missing) > 0:
                self._discard_incomplete_period(key, buffered, missing)
                continue

            result = self.process_buffered_period(
                deal_id=deal_id,
                period_start=period_start,
                period_end=period_end,
                metrics=buffered["metrics"],
            )

            # Produce prediction event
            self._producer.produce_prediction(result["prediction"])

            # Produce alert event if present
            if result["alert"] is not None:
                self._producer.produce_alert(result["alert"])

            # Only now may these offsets be committed.
            self._release_offsets(buffered)
            processed += 1

        # Poll producer for delivery callbacks
        if processed > 0:
            self._producer.poll(0.0)

        return processed

    def run_once(self) -> tuple[int, int]:
        """Run a single iteration of the worker loop.

        Returns:
            Tuple of (messages_consumed, periods_processed).
        """
        # Poll for new messages
        polled = self._consumer.poll(self._config["poll_timeout_seconds"])

        messages_consumed = 0
        if polled is not None:
            messages_consumed = 1
            self._messages_since_commit += 1
            if polled["kind"] == "measurement":
                consumed: ConsumedMeasurement = polled
                event = consumed["event"]

                # Emit measurement received metric
                self._metrics.increment_measurement_received(event["deal_id"], event["metric_name"])

                # Add to buffer
                self._add_to_buffer(consumed)
            else:
                undecodable: UndecodableMessage = polled
                self._dead_letter_undecodable(undecodable)

        # Process ready buffers
        periods_processed = self._process_ready_buffers()

        # Commit periodically. The positions exclude anything still buffered,
        # so a crash after this point replays unprocessed messages rather than
        # dropping them.
        if self._messages_since_commit >= self._config["commit_interval"]:
            self._consumer.commit(self._commit_positions())
            self._messages_since_commit = 0

        return messages_consumed, periods_processed

    def run(self, max_iterations: int | None = None) -> tuple[int, int]:
        """Run the worker main loop.

        Args:
            max_iterations: Optional limit on iterations (for testing).
                None means run until shutdown() is called.

        Returns:
            Tuple of (total_messages_consumed, total_periods_processed).
        """
        self._running = True
        total_messages = 0
        total_periods = 0
        iterations = 0

        while self._running:
            if max_iterations is not None and iterations >= max_iterations:
                break

            messages, periods = self.run_once()
            total_messages += messages
            total_periods += periods
            iterations += 1

        return total_messages, total_periods

    def request_stop(self) -> None:
        """Ask the run loop to stop after the current iteration.

        Safe to call from a signal handler: it only flips a flag, so no Kafka,
        database or producer call is made at an arbitrary bytecode boundary in
        the middle of run_once. Call shutdown() afterwards, once run() has
        returned, to drain and close.
        """
        self._running = False

    def shutdown(self) -> None:
        """Graceful shutdown of the worker.

        Processes any remaining buffered periods, flushes the producer, commits
        the safe positions and closes the consumer. Must be called from the
        main flow after run() returns, never from a signal handler.

        Raises:
            KeyError: If a deal does not exist during processing.
        """
        self._running = False

        # Process any remaining buffers (force timeout)
        for key in list(self._buffer.keys()):
            deal_id, period_start, period_end = key
            buffered = self._buffer.pop(key)

            missing = self._missing_required_metrics(buffered)
            if len(missing) > 0:
                self._discard_incomplete_period(key, buffered, missing)
                continue

            result = self.process_buffered_period(
                deal_id=deal_id,
                period_start=period_start,
                period_end=period_end,
                metrics=buffered["metrics"],
            )

            self._producer.produce_prediction(result["prediction"])
            if result["alert"] is not None:
                self._producer.produce_alert(result["alert"])

            self._release_offsets(buffered)

        # Flush before committing: a position must not be acknowledged until
        # the events derived from it have actually reached the broker.
        self._producer.flush(timeout_seconds=10.0)

        self._consumer.commit(self._commit_positions())

        # Cleared so a caller that resumes the loop after shutdown does not
        # commit again on a closed consumer.
        self._messages_since_commit = 0

        self._consumer.close()


__all__ = [
    "StreamingWorker",
]
