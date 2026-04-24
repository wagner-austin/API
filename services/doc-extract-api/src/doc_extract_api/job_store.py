"""Redis-backed job store for extraction jobs."""

from __future__ import annotations

from platform_workers.redis import RedisStrProto

from .types import ExtractionJob, decode_extraction_job, encode_extraction_job

_DOMAIN: str = "doc-extract"


def _job_key(job_id: str) -> str:
    """Generate a Redis key for a job.

    Args:
        job_id: The job identifier.

    Returns:
        Redis key string.
    """
    return f"{_DOMAIN}:job:{job_id}"


def save_job(redis: RedisStrProto, job: ExtractionJob) -> None:
    """Save a job to Redis.

    Args:
        redis: Redis client.
        job: The extraction job to save.
    """
    key = _job_key(job["job_id"])
    redis.hset(key, mapping=encode_extraction_job(job))


def load_job(redis: RedisStrProto, job_id: str) -> ExtractionJob | None:
    """Load a job from Redis.

    Args:
        redis: Redis client.
        job_id: The job identifier.

    Returns:
        The extraction job, or None if not found.
    """
    key = _job_key(job_id)
    raw = redis.hgetall(key)
    if len(raw) == 0:
        return None
    return decode_extraction_job(raw, job_id)


def update_progress(redis: RedisStrProto, job_id: str, pages_done: int) -> None:
    """Update the progress of a running job.

    Args:
        redis: Redis client.
        job_id: The job identifier.
        pages_done: Number of pages extracted so far.
    """
    key = _job_key(job_id)
    redis.hset(key, mapping={"pages_done": str(pages_done)})


def mark_completed(redis: RedisStrProto, job_id: str, document_id: str) -> None:
    """Mark a job as completed.

    Args:
        redis: Redis client.
        job_id: The job identifier.
        document_id: UUID of the created document.
    """
    key = _job_key(job_id)
    redis.hset(
        key,
        mapping={
            "status": "completed",
            "document_id": document_id,
        },
    )


def mark_failed(redis: RedisStrProto, job_id: str, error: str) -> None:
    """Mark a job as failed.

    Args:
        redis: Redis client.
        job_id: The job identifier.
        error: Error message describing the failure.
    """
    key = _job_key(job_id)
    redis.hset(
        key,
        mapping={
            "status": "failed",
            "error": error,
        },
    )


__all__ = [
    "_job_key",
    "load_job",
    "mark_completed",
    "mark_failed",
    "save_job",
    "update_progress",
]
