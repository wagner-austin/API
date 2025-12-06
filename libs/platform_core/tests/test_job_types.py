from __future__ import annotations

from datetime import datetime

from platform_core.job_types import BaseJobStatus, job_key


def test_job_key_builds_expected_pattern() -> None:
    assert job_key("turkic", "abc") == "turkic:job:abc"


def test_base_job_status_structure() -> None:
    status: BaseJobStatus = {
        "job_id": "j1",
        "user_id": 5,
        "status": "queued",
        "progress": 0,
        "message": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "error": None,
    }
    assert status["job_id"] == "j1"
