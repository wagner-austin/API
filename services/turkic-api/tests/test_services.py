from __future__ import annotations

import asyncio

from platform_core.logging import get_logger
from platform_core.turkic_jobs import turkic_job_key

from turkic_api.api.models import JobCreate
from turkic_api.api.services import JobService
from turkic_api.api.types import RQJobLike, RQRetryLike, _EnqCallable
from turkic_api.core.models import UnknownJson


class _RedisStub:
    def __init__(self) -> None:
        self.hset_calls: list[tuple[str, dict[str, str]]] = []

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self.hset_calls.append((key, mapping))
        return 1

    def hgetall(self, key: str) -> dict[str, str]:  # satisfy RedisProtocol
        return {}

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
        return None

    def sismember(self, key: str, member: str) -> bool:
        return False


class _QueueStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[UnknownJson, ...]]] = []

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
        self.calls.append((str(func), args))

        class _Job(RQJobLike):
            def get_id(self) -> str:
                return "test-job-id"

        return _Job()


def test_job_service_create_job_enqueues_and_sets_metadata() -> None:
    r = _RedisStub()
    q = _QueueStub()
    service = JobService(redis=r, logger=get_logger(__name__), queue=q)
    job: JobCreate = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "script": None,
        "max_sentences": 5,
        "transliterate": True,
        "confidence_threshold": 0.9,
    }

    resp = asyncio.run(service.create_job(job))

    assert resp["status"] == "queued"
    assert len(r.hset_calls) == 1
    key, mapping = r.hset_calls[0]
    assert key == turkic_job_key(resp["job_id"])
    assert mapping["status"] == "queued"
    assert q.calls
    assert q.calls[0][0] == "turkic_api.api.jobs.process_corpus"
