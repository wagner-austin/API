"""CLI entry point for data replay script.

Loads external datasets and streams them as measurement events
to Kafka for demonstrating the streaming inference pipeline.

Usage:
    poetry run python -m scripts.replay_data --dataset taiwan --speed fast
    poetry run python -m scripts.replay_data -d kaggle_amex_default -s instant -m 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TypedDict

from platform_core.logging import get_rich_console

from covenant_radar_api.streaming.config import load_streaming_config
from covenant_radar_api.streaming.producer import (
    StreamingProducer,
    create_streaming_producer,
)
from covenant_radar_api.streaming.schemas import MeasurementEventV1
from scripts.replay_data._test_hooks import ProducerProtocol
from scripts.replay_data.runner import DataReplayRunner
from scripts.replay_data.types import ReplaySpeed, make_replay_config

# =============================================================================
# Parsed Arguments TypedDict
# =============================================================================


class ParsedArgs(TypedDict):
    """Parsed command line arguments.

    Fields:
        dataset: Dataset name.
        speed: Replay speed mode.
        batch_size: Events per batch.
        max_rows: Maximum rows to process.
        deal_prefix: Deal ID prefix.
        external_dir: Path to external data directory.
    """

    dataset: str
    speed: ReplaySpeed
    batch_size: int
    max_rows: int
    deal_prefix: str
    external_dir: Path


# =============================================================================
# Argument Parsing
# =============================================================================


def _parse_speed(value: str) -> ReplaySpeed:
    """Parse replay speed argument.

    Args:
        value: Speed string from command line.

    Returns:
        Validated ReplaySpeed literal.

    Raises:
        argparse.ArgumentTypeError: If value is not valid.
    """
    if value == "realtime":
        return "realtime"
    if value == "fast":
        return "fast"
    if value == "instant":
        return "instant"
    raise argparse.ArgumentTypeError(
        f"Invalid speed: {value}. Must be one of: realtime, fast, instant"
    )


def _parse_args(args: list[str] | None = None) -> ParsedArgs:
    """Parse command line arguments.

    Args:
        args: Command line arguments (None for sys.argv).

    Returns:
        ParsedArgs TypedDict with validated arguments.
    """
    parser = argparse.ArgumentParser(description="Replay dataset as Kafka measurement events")
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset name (e.g., taiwan, us, kaggle_amex_default)",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=_parse_speed,
        default="fast",
        help="Replay speed: realtime (1s), fast (0.1s), instant (0) [default: fast]",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Events per batch [default: 100]",
    )
    parser.add_argument(
        "--max-rows",
        "-m",
        type=int,
        default=0,
        help="Maximum dataset rows to process (0 = unlimited) [default: 0]",
    )
    parser.add_argument(
        "--deal-prefix",
        "-p",
        default="replay",
        help="Deal ID prefix [default: replay]",
    )
    parser.add_argument(
        "--external-dir",
        "-e",
        type=Path,
        default=Path("data/external"),
        help="Path to external datasets directory [default: data/external]",
    )
    ns = parser.parse_args(args)

    # Extract typed values from namespace
    dataset: str = ns.dataset
    speed: ReplaySpeed = ns.speed
    batch_size: int = ns.batch_size
    max_rows: int = ns.max_rows
    deal_prefix: str = ns.deal_prefix
    external_dir: Path = ns.external_dir

    return ParsedArgs(
        dataset=dataset,
        speed=speed,
        batch_size=batch_size,
        max_rows=max_rows,
        deal_prefix=deal_prefix,
        external_dir=external_dir,
    )


# =============================================================================
# Producer Adapter
# =============================================================================


class _StreamingProducerAdapter:
    """Adapts StreamingProducer to ProducerProtocol.

    StreamingProducer has produce_event(event, topic) which matches ProducerProtocol.
    This adapter holds a typed reference to the StreamingProducer and delegates.
    """

    def __init__(self, producer: StreamingProducer) -> None:
        """Initialize adapter.

        Args:
            producer: StreamingProducer instance.
        """
        self._producer = producer

    def produce_event(self, event: MeasurementEventV1, topic: str) -> None:
        """Produce measurement event to Kafka.

        Args:
            event: Measurement event to publish.
            topic: Target Kafka topic.
        """
        self._producer.produce_event(event, topic)

    def poll(self, timeout_seconds: float) -> int:
        """Poll for delivery reports.

        Args:
            timeout_seconds: Maximum wait time.

        Returns:
            Number of events processed.
        """
        return self._producer.poll(timeout_seconds)

    def flush(self, timeout_seconds: float) -> int:
        """Flush pending messages.

        Args:
            timeout_seconds: Maximum wait time.

        Returns:
            Number of messages still in queue.
        """
        return self._producer.flush(timeout_seconds)


# =============================================================================
# Main Entry Point
# =============================================================================


def main(args: list[str] | None = None) -> int:
    """Main entry point for replay script.

    Args:
        args: Command line arguments (None for sys.argv).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parsed = _parse_args(args)

    # Load streaming config from environment
    streaming_config = load_streaming_config()

    # Create producer
    raw_producer = create_streaming_producer(streaming_config)
    producer: ProducerProtocol = _StreamingProducerAdapter(raw_producer)

    # Create replay config
    config = make_replay_config(
        dataset=parsed["dataset"],
        topic=streaming_config["topics"]["measurements"],
        speed=parsed["speed"],
        batch_size=parsed["batch_size"],
        deal_id_prefix=parsed["deal_prefix"],
        max_rows=parsed["max_rows"],
    )

    # Resolve external directory path
    external_dir = parsed["external_dir"]
    if not external_dir.is_absolute():
        external_dir = Path.cwd() / external_dir

    console = get_rich_console()
    console.print(f"Data Replay: {parsed['dataset']}")
    console.print(f"  Topic: {config['topic']}")
    console.print(f"  Speed: {config['speed']}")
    console.print(f"  Batch size: {config['batch_size']}")
    console.print(f"  Max rows: {config['max_rows'] or 'unlimited'}")
    console.print(f"  External dir: {external_dir}")
    console.print()

    # Run replay
    runner = DataReplayRunner(producer, config, external_dir)
    stats = runner.run()

    console.print("Replay complete:")
    console.print(f"  Rows processed: {stats['rows_processed']}")
    console.print(f"  Events sent: {stats['events_sent']}")
    console.print(f"  Batches sent: {stats['batches_sent']}")
    console.print(f"  Elapsed: {stats['elapsed_seconds']:.2f}s")
    console.print(f"  Throughput: {stats['events_per_second']:.1f} events/sec")

    return 0


if __name__ == "__main__":
    sys.exit(main())
