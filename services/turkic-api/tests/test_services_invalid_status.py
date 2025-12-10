from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError
from platform_core.logging import get_logger
from platform_core.turkic_jobs import turkic_job_key
from platform_workers.testing import FakeRedis

from turkic_api.api.services import JobService
from turkic_api.api.types import RQJobLike, RQRetryLike, _EnqCallable
from turkic_api.core.models import UnknownJson


class _Queue:
    def enqueue(
        self,
        func: str | _EnqCallable,
        *args: UnknownJson,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike:
        class _Job(RQJobLike):
            def get_id(self) -> str:
                return "test-job-id"

        return _Job()


def test_get_job_status_invalid_status_raises() -> None:
    r = FakeRedis()
    now = "2024-01-01T00:00:00"
    r._hashes[turkic_job_key("x")] = {
        "user_id": "42",
        "status": "weird",
        "progress": "0",
        "created_at": now,
        "updated_at": now,
    }
    svc = JobService(redis=r, queue=_Queue(), logger=get_logger(__name__), data_dir="/tmp")
    with pytest.raises(JSONTypeError, match="invalid status"):
        svc.get_job_status("x")
    r.assert_only_called({"hgetall"})
