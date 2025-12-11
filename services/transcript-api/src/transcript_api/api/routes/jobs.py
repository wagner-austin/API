"""STT job routes for async transcript processing."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue, load_json_bytes

from ...dependencies import LoggerDep, QueueDep, RedisDep
from ...job_store import TranscriptJobStore
from ...youtube import canonicalize_youtube_url, extract_video_id


class STTJobRequest:
    """Parsed STT job submission request."""

    def __init__(self, url: str, user_id: int) -> None:
        self.url = url
        self.user_id = user_id


def _parse_stt_job_request(payload: dict[str, JSONValue]) -> STTJobRequest:
    """Parse and validate STT job submission payload."""
    url = payload.get("url")
    if not isinstance(url, str) or url.strip() == "":
        raise AppError(ErrorCode.INVALID_INPUT, "url is required", 400)
    user_id = payload.get("user_id")
    if not isinstance(user_id, int):
        raise AppError(ErrorCode.INVALID_INPUT, "user_id must be an integer", 400)
    # Validate URL format
    canonical = canonicalize_youtube_url(url)
    extract_video_id(canonical)  # Raises if invalid
    return STTJobRequest(url=canonical, user_id=user_id)


def build_router() -> APIRouter:
    """Build STT jobs router."""
    router = APIRouter()

    async def create_stt_job(
        request: Request,
        queue: QueueDep,
        redis: RedisDep,
        logger: LoggerDep,
    ) -> JSONResponse:
        body = await request.body()
        raw_payload = load_json_bytes(body)
        if not isinstance(raw_payload, dict):
            raise AppError(ErrorCode.INVALID_INPUT, "Request body must be a JSON object", 400)
        parsed = _parse_stt_job_request(raw_payload)

        job_id = str(uuid.uuid4())

        # Enqueue the STT processing job
        job = queue.enqueue(
            "transcript_api.jobs.process_stt",
            job_id,
            {"url": parsed.url, "user_id": parsed.user_id},
            job_timeout=1800,  # 30 minutes for long videos
            result_ttl=86400,  # 24 hours
            failure_ttl=86400,
            description=f"stt:{parsed.url}",
        )

        logger.info(
            "Enqueued STT job %s (rq_id=%s)",
            job_id,
            job.get_id(),
            extra={"job_id": job_id, "url": parsed.url, "user_id": parsed.user_id},
        )

        response_content: dict[str, str | int] = {
            "job_id": job_id,
            "user_id": parsed.user_id,
            "status": "queued",
            "url": parsed.url,
        }
        return JSONResponse(content=response_content, status_code=202)

    async def get_stt_job_status(
        job_id: str,
        redis: RedisDep,
    ) -> JSONResponse:
        store = TranscriptJobStore(redis)
        status = store.load(job_id)
        if status is None:
            raise AppError(ErrorCode.NOT_FOUND, "Job not found", 404)

        response: dict[str, str | int | None] = {
            "job_id": status["job_id"],
            "user_id": status["user_id"],
            "status": status["status"],
            "progress": status["progress"],
            "message": status["message"],
            "url": status["url"],
            "video_id": status["video_id"],
            "error": status["error"],
        }

        # Only include text if job is completed
        if status["status"] == "completed" and status["text"] is not None:
            response["text"] = status["text"]

        return JSONResponse(content=response)

    router.add_api_route("/v1/stt/jobs", create_stt_job, methods=["POST"])
    router.add_api_route("/v1/stt/jobs/{job_id}", get_stt_job_status, methods=["GET"])

    return router


__all__ = ["build_router"]
