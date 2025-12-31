"""Data replay script for streaming datasets to Kafka.

Loads external datasets and publishes them as measurement events
for demonstrating the streaming inference pipeline.

Usage:
    poetry run python -m scripts.replay_data --dataset taiwan --speed fast
"""

from scripts.replay_data.runner import DataReplayRunner, run_replay
from scripts.replay_data.types import (
    ReplayConfig,
    ReplaySpeed,
    ReplayStats,
    make_replay_config,
    make_replay_stats,
)

__all__ = [
    "DataReplayRunner",
    "ReplayConfig",
    "ReplaySpeed",
    "ReplayStats",
    "make_replay_config",
    "make_replay_stats",
    "run_replay",
]
