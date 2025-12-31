"""TypedDict definitions for data replay script.

Provides configuration and statistics types for replaying
datasets as Kafka measurement events.

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# =============================================================================
# Type Aliases
# =============================================================================

ReplaySpeed = Literal["realtime", "fast", "instant"]


# =============================================================================
# Configuration TypedDicts
# =============================================================================


class ReplayConfig(TypedDict, total=True):
    """Configuration for data replay.

    Fields:
        dataset: Dataset name from registry (e.g., "taiwan", "kaggle_amex_default").
        topic: Kafka topic to publish measurement events to.
        speed: Replay speed mode (realtime=1s delay, fast=0.1s, instant=0).
        batch_size: Number of measurement events per batch.
        deal_id_prefix: Prefix for generated deal IDs.
        max_rows: Maximum dataset rows to process (0 = unlimited).
    """

    dataset: str
    topic: str
    speed: ReplaySpeed
    batch_size: int
    deal_id_prefix: str
    max_rows: int


class ReplayStats(TypedDict, total=True):
    """Statistics from replay run.

    Fields:
        rows_processed: Number of dataset rows processed.
        events_sent: Total measurement events published.
        batches_sent: Number of batches sent.
        elapsed_seconds: Total runtime in seconds.
        events_per_second: Throughput (events / elapsed).
    """

    rows_processed: int
    events_sent: int
    batches_sent: int
    elapsed_seconds: float
    events_per_second: float


# =============================================================================
# Factory Functions
# =============================================================================


def make_replay_config(
    *,
    dataset: str,
    topic: str = "covenant.measurements.v1",
    speed: ReplaySpeed = "fast",
    batch_size: int = 100,
    deal_id_prefix: str = "replay",
    max_rows: int = 0,
) -> ReplayConfig:
    """Create replay configuration.

    Args:
        dataset: Dataset name from registry.
        topic: Kafka topic for measurements.
        speed: Replay speed mode.
        batch_size: Events per batch.
        deal_id_prefix: Prefix for deal IDs.
        max_rows: Maximum rows (0 = unlimited).

    Returns:
        ReplayConfig instance.
    """
    return {
        "dataset": dataset,
        "topic": topic,
        "speed": speed,
        "batch_size": batch_size,
        "deal_id_prefix": deal_id_prefix,
        "max_rows": max_rows,
    }


def make_replay_stats(
    *,
    rows_processed: int,
    events_sent: int,
    batches_sent: int,
    elapsed_seconds: float,
) -> ReplayStats:
    """Create replay statistics.

    Args:
        rows_processed: Rows processed from dataset.
        events_sent: Total events sent.
        batches_sent: Total batches sent.
        elapsed_seconds: Runtime in seconds.

    Returns:
        ReplayStats instance with computed throughput.
    """
    eps = events_sent / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return {
        "rows_processed": rows_processed,
        "events_sent": events_sent,
        "batches_sent": batches_sent,
        "elapsed_seconds": elapsed_seconds,
        "events_per_second": eps,
    }


__all__ = [
    "ReplayConfig",
    "ReplaySpeed",
    "ReplayStats",
    "make_replay_config",
    "make_replay_stats",
]
