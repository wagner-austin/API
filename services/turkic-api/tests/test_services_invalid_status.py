from __future__ import annotations

import pytest
from platform_core.logging import get_logger
from platform_core.turkic_jobs import turkic_job_key

from turkic_api.api.services import JobService
from turkic_api.api.types import RQJobLike, RQRetryLike, _EnqCallable
from turkic_api.core.models import UnknownJson


class _Redis:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, str]] = {}

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def hgetall(self, key: str) -> dict[str, str]:
        return self.data.get(key, {})

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        cur = self.data.get(key, {})
        cur.update(mapping)
        self.data[key] = cur
        return 1

    def publish(self, channel: str, message: str) -> int:
        return 1

    def set(self, key: str, value: str) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def close(self) -> None:
        return None

    def sadd(self, key: str, *values: str) -> int:
        return len(values)

    def scard(self, key: str) -> int:
        return 0

    def hget(self, key: str, field: str) -> str | None:
        return self.data.get(key, {}).get(field)

    def sismember(self, key: str, member: str) -> bool:
        return False


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
    r = _Redis()
    now = "2024-01-01T00:00:00"
    r.data[turkic_job_key("x")] = {
        "user_id": "42",
        "status": "weird",
        "progress": "0",
        "created_at": now,
        "updated_at": now,
    }
    svc = JobService(redis=r, queue=_Queue(), logger=get_logger(__name__), data_dir="/tmp")
    with pytest.raises(ValueError, match="invalid status"):
        svc.get_job_status("x")
