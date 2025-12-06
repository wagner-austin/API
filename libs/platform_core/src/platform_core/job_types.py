from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

JobStatusLiteral = Literal["queued", "processing", "completed", "failed"]


class BaseJobStatus(TypedDict):
    """Base fields for persisted job status in Redis."""

    job_id: str
    user_id: int
    status: JobStatusLiteral
    progress: int
    message: str | None
    created_at: datetime
    updated_at: datetime
    error: str | None


def job_key(domain: str, job_id: str) -> str:
    """Generate a stable Redis key for a job."""
    return f"{domain}:job:{job_id}"


__all__ = ["BaseJobStatus", "JobStatusLiteral", "job_key"]
