"""Progress store for training job progress tracking.

This module provides Redis-backed storage for detailed training progress,
enabling real-time monitoring of training jobs through the API.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.trainer_keys import progress_key
from platform_workers.redis import RedisStrProto

from model_trainer.core.contracts.progress import (
    TrainingProgress,
    decode_training_progress,
    encode_training_progress,
)
from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry

_PROGRESS_TTL_SECONDS: Final[int] = 86400  # 24 hours


class ProgressStore:
    """Redis-backed store for training progress."""

    def __init__(self, redis: RedisStrProto) -> None:
        """Initialize progress store with Redis client.

        Args:
            redis: Redis client implementing string protocol.
        """
        self._redis = redis

    def save(self, progress: TrainingProgress) -> None:
        """Save training progress to Redis.

        Args:
            progress: Training progress to save.
        """
        key = progress_key(progress["run_id"])
        encoded = encode_training_progress(progress)
        json_str = dump_json_str(encoded)
        set_with_retry(self._redis, key, json_str)
        self._redis.expire(key, _PROGRESS_TTL_SECONDS)

    def load(self, run_id: str) -> TrainingProgress | None:
        """Load training progress from Redis.

        Args:
            run_id: Run identifier.

        Returns:
            TrainingProgress if found, None otherwise.
        """
        key = progress_key(run_id)
        raw = get_with_retry(self._redis, key)
        if raw is None:
            return None
        obj = load_json_str(str(raw))
        if not isinstance(obj, dict):
            return None
        return decode_training_progress(obj)


__all__ = ["ProgressStore"]
