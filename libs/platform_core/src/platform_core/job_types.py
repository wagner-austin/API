"""Persisted job status shared by every service that runs RQ jobs.

:class:`JobStatus` is the one closed vocabulary of a job's lifecycle state, a
:class:`enum.StrEnum` whose members are the same strings stored in Redis. It
replaced two identical module-level ``Literal`` aliases, here and in
``turkic_jobs``, which the operator's "no type alias" covers (MCPs board task
1374feba); an untrusted word narrows through :mod:`platform_core.members`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TypedDict


class JobStatus(StrEnum):
    """Where a job is in its lifecycle."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseJobStatus(TypedDict):
    """Base fields for persisted job status in Redis."""

    job_id: str
    user_id: int
    status: JobStatus
    progress: int
    message: str | None
    created_at: datetime
    updated_at: datetime
    error: str | None


def job_key(domain: str, job_id: str) -> str:
    """Generate a stable Redis key for a job."""
    return f"{domain}:job:{job_id}"


__all__ = ["BaseJobStatus", "JobStatus", "job_key"]
