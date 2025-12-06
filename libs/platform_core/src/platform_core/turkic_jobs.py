from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict


def turkic_job_key(job_id: str) -> str:
    return f"turkic:job:{job_id}"


JobStatusLiteral = Literal["queued", "processing", "completed", "failed"]


class TurkicJobStatus(TypedDict):
    job_id: str
    user_id: int
    status: JobStatusLiteral
    progress: int
    message: str | None
    result_url: str | None
    file_id: str | None
    upload_status: Literal["uploaded"] | None
    created_at: datetime
    updated_at: datetime
    error: str | None


__all__ = [
    "JobStatusLiteral",
    "TurkicJobStatus",
    "turkic_job_key",
]
