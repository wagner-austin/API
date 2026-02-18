"""Redis key helpers for Art-Trainer.

This module provides functions to generate consistent Redis keys for
various service data structures.
"""

from __future__ import annotations

# Key prefixes for Art-Trainer service
JOB_STATUS_PREFIX = "art:job:status:"
JOB_PROGRESS_PREFIX = "art:job:progress:"
JOB_CANCEL_PREFIX = "art:job:cancel:"
JOB_RESULT_PREFIX = "art:job:result:"


def status_key(job_id: str) -> str:
    """Get the Redis key for job status.

    Args:
        job_id: Job identifier.

    Returns:
        Redis key string.
    """
    return f"{JOB_STATUS_PREFIX}{job_id}"


def progress_key(job_id: str) -> str:
    """Get the Redis key for job progress.

    Args:
        job_id: Job identifier.

    Returns:
        Redis key string.
    """
    return f"{JOB_PROGRESS_PREFIX}{job_id}"


def cancel_key(job_id: str) -> str:
    """Get the Redis key for job cancellation flag.

    Args:
        job_id: Job identifier.

    Returns:
        Redis key string.
    """
    return f"{JOB_CANCEL_PREFIX}{job_id}"


def result_key(job_id: str) -> str:
    """Get the Redis key for job result.

    Args:
        job_id: Job identifier.

    Returns:
        Redis key string.
    """
    return f"{JOB_RESULT_PREFIX}{job_id}"


__all__ = [
    "JOB_CANCEL_PREFIX",
    "JOB_PROGRESS_PREFIX",
    "JOB_RESULT_PREFIX",
    "JOB_STATUS_PREFIX",
    "cancel_key",
    "progress_key",
    "result_key",
    "status_key",
]
