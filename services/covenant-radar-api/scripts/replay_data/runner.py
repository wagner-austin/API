"""Data replay runner for streaming datasets to Kafka.

Loads datasets via covenant_ml.datasets, converts rows to measurement events,
and publishes to Kafka topic for streaming inference demonstration.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import time
from pathlib import Path

from covenant_radar_api.streaming.schemas import (
    MeasurementEventV1,
    make_measurement_event,
)
from scripts.replay_data import _test_hooks
from scripts.replay_data._test_hooks import ProducerProtocol
from scripts.replay_data.types import (
    ReplayConfig,
    ReplaySpeed,
    ReplayStats,
    make_replay_stats,
)

# =============================================================================
# Constants
# =============================================================================

# Delay between batches for each speed mode
_SPEED_DELAYS: dict[ReplaySpeed, float] = {
    "realtime": 1.0,
    "fast": 0.1,
    "instant": 0.0,
}


# =============================================================================
# Helper Functions
# =============================================================================


def _get_delay_seconds(speed: ReplaySpeed) -> float:
    """Get delay between batches for replay speed.

    Args:
        speed: Replay speed mode.

    Returns:
        Delay in seconds between batches.
    """
    return _SPEED_DELAYS[speed]


def _current_iso_timestamp() -> str:
    """Get current UTC timestamp in ISO format.

    Returns:
        ISO-8601 formatted timestamp string.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _make_deal_id(prefix: str, row_index: int) -> str:
    """Generate deal ID from prefix and row index.

    Args:
        prefix: Deal ID prefix from config.
        row_index: Row number in dataset (0-indexed).

    Returns:
        Formatted deal ID like "replay-000042".
    """
    return f"{prefix}-{row_index:06d}"


def _make_period_dates(row_index: int) -> tuple[str, str]:
    """Generate period start and end dates from row index.

    Creates synthetic quarterly periods cycling through months 1-12.

    Args:
        row_index: Row number used to vary the period.

    Returns:
        Tuple of (period_start, period_end) in YYYY-MM-DD format.
    """
    month = (row_index % 12) + 1
    period_start = f"2024-{month:02d}-01"
    period_end = f"2024-{month:02d}-28"
    return period_start, period_end


def _row_to_events(
    row_index: int,
    feature_names: tuple[str, ...],
    feature_values: tuple[float, ...],
    deal_id_prefix: str,
) -> list[MeasurementEventV1]:
    """Convert a dataset row to measurement events.

    Each feature becomes a separate measurement event, all sharing
    the same deal_id and period.

    Args:
        row_index: Row number in dataset.
        feature_names: Ordered feature column names.
        feature_values: Feature values for this row.
        deal_id_prefix: Prefix for generated deal IDs.

    Returns:
        List of MeasurementEventV1 events, one per feature.
    """
    deal_id = _make_deal_id(deal_id_prefix, row_index)
    period_start, period_end = _make_period_dates(row_index)
    timestamp = _current_iso_timestamp()

    events: list[MeasurementEventV1] = []
    for name, value in zip(feature_names, feature_values, strict=True):
        event = make_measurement_event(
            event_id=_test_hooks.generate_uuid(),
            deal_id=deal_id,
            period_start=period_start,
            period_end=period_end,
            metric_name=name,
            metric_value=float(value),
            timestamp=timestamp,
        )
        events.append(event)

    return events


# =============================================================================
# Replay Runner
# =============================================================================


class DataReplayRunner:
    """Replays dataset rows as Kafka measurement events.

    Loads dataset via covenant_ml.datasets, iterates rows,
    converts to measurement events, and publishes to Kafka.

    Uses dependency injection via _test_hooks for testability.
    """

    def __init__(
        self,
        producer: ProducerProtocol,
        config: ReplayConfig,
        external_dir: Path,
    ) -> None:
        """Initialize replay runner.

        Args:
            producer: Kafka producer for publishing events.
            config: Replay configuration.
            external_dir: Path to data/external directory with datasets.
        """
        self._producer = producer
        self._config = config
        self._external_dir = external_dir

    def run(self) -> ReplayStats:
        """Execute data replay.

        Loads the dataset, converts rows to measurement events,
        and publishes to Kafka in batches with configured delay.

        Returns:
            ReplayStats with run statistics.

        Raises:
            KeyError: If dataset not found in registry.
            FileNotFoundError: If dataset file doesn't exist.
        """
        start_time = _test_hooks.perf_counter()

        # Load dataset
        loader = _test_hooks.dataset_loader_factory()
        registry = _test_hooks.registry_factory()
        ts_registry = _test_hooks.timeseries_registry_factory()

        dataset_name = self._config["dataset"]

        # Check which registry has the dataset
        if dataset_name in ts_registry:
            ts_config = ts_registry.get(dataset_name)
            loaded = loader.load_timeseries(ts_config, self._external_dir)
        elif dataset_name in registry:
            config = registry.get(dataset_name)
            loaded = loader.load(config, self._external_dir)
        else:
            # Build combined list of available datasets
            std_names = registry.list_names()
            ts_names = ts_registry.list_names()
            all_names = sorted(set(std_names) | set(ts_names))
            available = ", ".join(all_names)
            raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")

        meta = loaded["meta"]
        features = _test_hooks.wrap_features(loaded["x"])
        feature_names = meta["feature_names"]
        n_rows = meta["n_samples"]

        # Apply max_rows limit
        max_rows = self._config["max_rows"]
        if max_rows > 0:
            n_rows = min(n_rows, max_rows)

        # Replay loop
        delay = _get_delay_seconds(self._config["speed"])
        topic = self._config["topic"]
        batch_size = self._config["batch_size"]
        deal_id_prefix = self._config["deal_id_prefix"]

        events_sent = 0
        batches_sent = 0
        batch: list[MeasurementEventV1] = []

        n_cols = len(feature_names)
        for row_idx in range(n_rows):
            # Extract row values as tuple of floats using typed protocol indexing
            row_values = tuple(features[row_idx, col] for col in range(n_cols))

            # Convert row to measurement events
            row_events = _row_to_events(
                row_index=row_idx,
                feature_names=feature_names,
                feature_values=row_values,
                deal_id_prefix=deal_id_prefix,
            )

            # Add to batch
            for event in row_events:
                batch.append(event)
                events_sent += 1

                # Send batch when full
                if len(batch) >= batch_size:
                    self._send_batch(batch, topic)
                    batches_sent += 1
                    batch = []

                    # Delay between batches
                    if delay > 0:
                        _test_hooks.sleep(delay)

        # Send remaining batch
        if batch:
            self._send_batch(batch, topic)
            batches_sent += 1

        # Flush producer
        self._producer.flush(timeout_seconds=10.0)

        elapsed = _test_hooks.perf_counter() - start_time
        return make_replay_stats(
            rows_processed=n_rows,
            events_sent=events_sent,
            batches_sent=batches_sent,
            elapsed_seconds=elapsed,
        )

    def _send_batch(self, batch: list[MeasurementEventV1], topic: str) -> None:
        """Send batch of events to Kafka.

        Args:
            batch: List of measurement events to send.
            topic: Kafka topic to publish to.
        """
        for event in batch:
            self._producer.produce_event(event, topic)
        self._producer.poll(0.0)


# =============================================================================
# Factory Function
# =============================================================================


def run_replay(
    producer: ProducerProtocol,
    config: ReplayConfig,
    external_dir: Path,
) -> ReplayStats:
    """Run data replay with provided dependencies.

    Convenience function that creates runner and executes replay.

    Args:
        producer: Kafka producer for publishing events.
        config: Replay configuration.
        external_dir: Path to data/external directory.

    Returns:
        ReplayStats with run statistics.

    Raises:
        KeyError: If dataset not found.
        FileNotFoundError: If dataset file doesn't exist.
    """
    runner = DataReplayRunner(producer, config, external_dir)
    return runner.run()


__all__ = [
    "DataReplayRunner",
    "run_replay",
]
