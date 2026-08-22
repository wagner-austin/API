"""Streaming worker base: measurement buffering, offsets, and buffer readiness."""

from __future__ import annotations

import time

from covenant_domain.features import (
    REQUIRED_CURRENT_METRICS,
)
from covenant_ml.types import PredictorProtocol
from covenant_persistence import (
    CovenantRepository,
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)
from platform_core.logging import get_logger

from covenant_radar_api.streaming.worker_events import (
    BufferedPeriod,
    WorkerConfig,
    _current_iso_timestamp,
    _generate_event_id,
    _make_buffer_key,
)

from ..integrations.datadog.metrics import MetricsClient
from ._hook_protocols import TopicPartitionOffset
from .consumer import ConsumedMeasurement, StreamingConsumer, UndecodableMessage
from .producer import StreamingProducer
from .schemas import (
    make_dlq_event,
)

_log = get_logger(__name__)


class _StreamingWorkerBuffers:
    """Buffering and offset-tracking base of the streaming worker.

    Owns the in-memory measurement buffers keyed by (deal_id, period_start,
    period_end), dead-lettering of undecodable payloads, offset release and
    commit bookkeeping, and the readiness rules that decide when a buffered
    period may be processed.
    """

    def __init__(
        self,
        consumer: StreamingConsumer,
        producer: StreamingProducer,
        metrics: MetricsClient,
        model: PredictorProtocol,
        deal_repo: DealRepository,
        covenant_repo: CovenantRepository,
        measurement_repo: MeasurementRepository,
        result_repo: CovenantResultRepository,
        sector_encoder: dict[str, int],
        region_encoder: dict[str, int],
        config: WorkerConfig,
    ) -> None:
        """Initialize the streaming worker.

        Args:
            consumer: Kafka consumer for measurements.
            producer: Kafka producer for predictions/alerts.
            metrics: Datadog metrics client.
            model: ML model for predictions.
            deal_repo: Repository for deal data.
            covenant_repo: Repository for covenant data.
            measurement_repo: Repository for historical measurements.
            result_repo: Repository for covenant results.
            sector_encoder: Sector to integer mapping.
            region_encoder: Region to integer mapping.
            config: Worker configuration.
        """
        self._consumer = consumer
        self._producer = producer
        self._metrics = metrics
        self._model = model
        self._deal_repo = deal_repo
        self._covenant_repo = covenant_repo
        self._measurement_repo = measurement_repo
        self._result_repo = result_repo
        self._sector_encoder = sector_encoder
        self._region_encoder = region_encoder
        self._config = config
        self._running = False
        self._messages_since_commit = 0

        # Buffer: (deal_id, period_start, period_end) -> BufferedPeriod
        self._buffer: dict[tuple[str, str, str], BufferedPeriod] = {}

        # Offsets polled but still sitting in _buffer, per (topic, partition).
        # A position may only be committed once it has left this set.
        self._pending_offsets: dict[tuple[str, int], set[int]] = {}

        # Highest offset seen per (topic, partition), used to derive the commit
        # position once nothing is pending for that partition.
        self._highest_offset: dict[tuple[str, int], int] = {}

    @property
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self._running

    @property
    def buffer_size(self) -> int:
        """Get number of periods currently buffered."""
        return len(self._buffer)

    def _add_to_buffer(self, consumed: ConsumedMeasurement) -> None:
        """Add a consumed measurement to the buffer and mark its offset pending.

        Args:
            consumed: Measurement event plus the Kafka position it came from.
        """
        event = consumed["event"]
        key = _make_buffer_key(event)

        if key not in self._buffer:
            self._buffer[key] = {
                "metrics": {},
                "first_received_at": time.monotonic(),
                "message_count": 0,
                "offsets": [],
            }

        buffered = self._buffer[key]
        buffered["metrics"][event["metric_name"]] = event["metric_value"]
        buffered["message_count"] += 1

        topic = consumed["topic"]
        partition = consumed["partition"]
        offset = consumed["offset"]
        buffered["offsets"].append((topic, partition, offset))

        tp = (topic, partition)
        pending = self._pending_offsets.get(tp)
        if pending is None:
            pending = set()
            self._pending_offsets[tp] = pending
        pending.add(offset)

        highest = self._highest_offset.get(tp)
        if highest is None or offset > highest:
            self._highest_offset[tp] = offset

    def _dead_letter_undecodable(self, message: UndecodableMessage) -> None:
        """Publish an undecodable message to the dead-letter topic.

        The offset is recorded as seen but never marked pending, so the commit
        position advances past it once the surrounding messages are processed.
        That is the whole point of the dead-letter topic: without a durable
        copy there is nowhere safe to move the offset to, and the same message
        is redelivered on every restart forever.

        Args:
            message: The message that could not be decoded.
        """
        topic = message["topic"]
        partition = message["partition"]
        offset = message["offset"]

        self._producer.produce_dlq(
            make_dlq_event(
                event_id=_generate_event_id(),
                reason="undecodable_payload",
                detail=message["reason"],
                source_topic=topic,
                source_partition=partition,
                source_offset=offset,
                payload=message["payload"],
                failed_at=_current_iso_timestamp(),
            )
        )

        tp = (topic, partition)
        highest = self._highest_offset.get(tp)
        if highest is None or offset > highest:
            self._highest_offset[tp] = offset

        _log.warning(
            "Dead-lettered undecodable message",
            extra={
                "topic": topic,
                "partition": str(partition),
                "offset": str(offset),
                "reason": message["reason"],
            },
        )

    def _release_offsets(self, buffered: BufferedPeriod) -> None:
        """Mark a flushed period's offsets as no longer pending.

        Args:
            buffered: The period whose messages have been fully processed.
        """
        # Indexed directly, not .get(): every offset recorded on a period was
        # put there by _add_to_buffer, which creates the partition's set first.
        for topic, partition, offset in buffered["offsets"]:
            self._pending_offsets[(topic, partition)].discard(offset)

    def _commit_positions(self) -> tuple[TopicPartitionOffset, ...]:
        """Compute the highest position safe to commit on each partition.

        For a partition still holding buffered messages, the safe position is
        the lowest pending offset: everything below it has been processed, that
        message has not. With nothing pending, everything through the highest
        offset seen is done, so the next position is one past it.

        Returns:
            One position per assigned partition, possibly empty.
        """
        positions: list[TopicPartitionOffset] = []
        for tp, highest in self._highest_offset.items():
            pending = self._pending_offsets.get(tp)
            safe = min(pending) if pending else highest + 1
            position: TopicPartitionOffset = {
                "topic": tp[0],
                "partition": tp[1],
                "offset": safe,
            }
            positions.append(position)
        return tuple(positions)

    def _should_process_buffer(self, key: tuple[str, str, str]) -> bool:
        """Check if a buffered period should be processed.

        A period is ready for processing if:
        1. It has minimum required metrics, OR
        2. It has timed out

        Args:
            key: Buffer key (deal_id, period_start, period_end).

        Returns:
            True if buffer should be processed.
        """
        if key not in self._buffer:
            return False

        buffered = self._buffer[key]
        metric_count = len(buffered["metrics"])
        age_seconds = time.monotonic() - buffered["first_received_at"]

        # Process if we have enough metrics or buffer has timed out
        has_enough_metrics = metric_count >= self._config["min_metrics_per_period"]
        timed_out = age_seconds >= self._config["buffer_timeout_seconds"]
        return has_enough_metrics or timed_out

    def _get_ready_buffers(self) -> list[tuple[str, str, str]]:
        """Get list of buffer keys ready for processing.

        Returns:
            List of buffer keys that should be processed.
        """
        ready: list[tuple[str, str, str]] = []
        for key in self._buffer:
            if self._should_process_buffer(key):
                ready.append(key)
        return ready

    def _missing_required_metrics(self, buffered: BufferedPeriod) -> tuple[str, ...]:
        """Report which feature-extraction metrics a period still lacks.

        min_metrics_per_period governs when a period is *considered*, which is
        not the same as having everything extract_features reads. Checking the
        published contract up front keeps a partial period from reaching
        feature extraction and raising KeyError out of the run loop.

        Args:
            buffered: The buffered period to inspect.

        Returns:
            The missing metric names, empty if the period is complete.
        """
        present = buffered["metrics"].keys()
        return tuple(name for name in REQUIRED_CURRENT_METRICS if name not in present)

    def _discard_incomplete_period(
        self,
        key: tuple[str, str, str],
        buffered: BufferedPeriod,
        missing: tuple[str, ...],
    ) -> None:
        """Drop a period that can never produce a prediction, and say so.

        A period only reaches here once it has timed out, so the measurements it
        is missing are not going to arrive. Its offsets are released rather than
        left pending, otherwise the same incomplete period would be replayed on
        every restart and block the partition's commit position forever.

        Args:
            key: Buffer key (deal_id, period_start, period_end).
            buffered: The period being discarded.
            missing: Metric names that never arrived.
        """
        deal_id, period_start, period_end = key
        _log.warning(
            "Discarding incomplete period; required metrics never arrived",
            extra={
                "deal_id": deal_id,
                "period_start": period_start,
                "period_end": period_end,
                "missing_metrics": ",".join(missing),
                "metrics_received": ",".join(sorted(buffered["metrics"].keys())),
            },
        )
        self._release_offsets(buffered)
