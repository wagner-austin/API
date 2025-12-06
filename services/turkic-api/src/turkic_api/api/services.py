from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from platform_core.turkic_jobs import TurkicJobStatus
from platform_workers.redis import RedisStrProto

from turkic_api.api.job_store import TurkicJobStore
from turkic_api.api.models import JobCreate, JobResponse, JobStatus
from turkic_api.api.types import LoggerProtocol, QueueProtocol, UnknownJson


class JobService:
    """Service for job lifecycle; all dependencies are injected explicitly."""

    def __init__(
        self,
        *,
        redis: RedisStrProto,
        logger: LoggerProtocol,
        queue: QueueProtocol,
        data_dir: str = "/data",
    ) -> None:
        self._redis = redis
        self._logger = logger
        self._queue = queue
        self._data_dir = data_dir
        self._store = TurkicJobStore(redis)

    async def create_job(self, job: JobCreate) -> JobResponse:
        """Create a new job and enqueue background processing."""
        job_id = str(uuid4())
        user_id = job["user_id"]
        now = datetime.utcnow()

        self._logger.debug("Enqueuing job", extra={"job_id": job_id, "language": job["language"]})

        queued_status: TurkicJobStatus = {
            "job_id": job_id,
            "user_id": user_id,
            "status": "queued",
            "progress": 0,
            "message": None,
            "result_url": None,
            "file_id": None,
            "upload_status": None,
            "created_at": now,
            "updated_at": now,
            "error": None,
        }
        self._store.save(queued_status)

        # Enqueue background job via injected queue with a typed payload
        payload: dict[str, UnknownJson] = {
            "user_id": user_id,
            "source": job["source"],
            "language": job["language"],
            "script": job["script"] if job["script"] is not None else None,
            "max_sentences": job["max_sentences"],
            "transliterate": job["transliterate"],
            "confidence_threshold": float(job["confidence_threshold"]),
        }
        self._queue.enqueue("turkic_api.api.jobs.process_corpus", job_id, payload)

        return {"job_id": job_id, "user_id": user_id, "status": "queued", "created_at": now}

    def get_job_status(self, job_id: str) -> JobStatus | None:
        """Fetch job status from Redis and build a typed response; returns None if not found."""
        stored = self._store.load(job_id)
        if stored is None:
            return None

        user_id = stored["user_id"]
        status = stored["status"]
        progress = stored["progress"]
        message = stored["message"]
        error = stored["error"]
        created_at = stored["created_at"]
        updated_at = stored["updated_at"]

        file_id = stored["file_id"]
        upload_status = stored["upload_status"]
        result_url: str | None = None
        if status == "completed" and upload_status == "uploaded":
            result_url = f"/api/v1/jobs/{job_id}/result"

        return {
            "job_id": job_id,
            "user_id": user_id,
            "status": status,
            "progress": progress,
            "message": message,
            "result_url": result_url,
            "file_id": file_id,
            "upload_status": upload_status,
            "created_at": created_at,
            "updated_at": updated_at,
            "error": error,
        }
