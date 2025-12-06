from __future__ import annotations

from datetime import UTC, datetime

from platform_core.turkic_jobs import JobStatusLiteral, TurkicJobStatus, turkic_job_key


def test_turkic_job_key() -> None:
    assert turkic_job_key("abc") == "turkic:job:abc"


def test_turkic_job_status_typeddict() -> None:
    now = datetime.now(tz=UTC)
    status: JobStatusLiteral = "processing"
    model: TurkicJobStatus = {
        "job_id": "j",
        "user_id": 42,
        "status": status,
        "progress": 42,
        "message": None,
        "result_url": None,
        "file_id": None,
        "upload_status": None,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }
    assert model["job_id"] == "j"
    assert model["user_id"] == 42
    assert model["status"] == "processing"
    assert model["progress"] == 42
