"""FastAPI routes for document extraction jobs."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response
from platform_workers.redis import RedisStrProto

from doc_extract_api import _test_hooks
from doc_extract_api.job_store import load_job, save_job
from doc_extract_api.settings import get_redis_url
from doc_extract_api.types import (
    ExtractionJob,
    decode_extraction_request,
    encode_extraction_job_response,
    validate_category,
)


def _get_redis() -> RedisStrProto:
    """Get a Redis client from the hook.

    Returns:
        A Redis client for key-value operations.
    """
    redis_url = get_redis_url()
    return _test_hooks.redis_factory(redis_url)


def _create_job(
    title: str,
    file_path: str,
    category: str = "general",
    source: str = "",
) -> dict[str, str | int]:
    """Create a new extraction job.

    Args:
        title: Document title.
        file_path: Absolute path to the PDF file.
        category: Document category.
        source: Source URL or reference.

    Returns:
        JSON response with the created job details.
    """
    validate_category(category)
    request = decode_extraction_request(
        title=title,
        file_path=file_path,
        category=category,
        source=source,
    )

    job_id = str(uuid.uuid4())
    job = ExtractionJob(
        job_id=job_id,
        status="queued",
        title=request["title"],
        source=request["source"],
        category=request["category"],
        file_path=request["file_path"],
        pages_total=0,
        pages_done=0,
        document_id="",
        error="",
    )

    redis = _get_redis()
    save_job(redis, job)

    return encode_extraction_job_response(job)


def _get_job(job_id: str, response: Response) -> dict[str, str | int]:
    """Get the status of an extraction job.

    Args:
        job_id: The job identifier.
        response: FastAPI response object for setting status code.

    Returns:
        JSON response with job details, or error if not found.
    """
    redis = _get_redis()
    job = load_job(redis, job_id)
    if job is None:
        response.status_code = 404
        return {"error": f"Job not found: {job_id}"}
    return encode_extraction_job_response(job)


def build_router() -> APIRouter:
    """Build the ingest router with job creation and status endpoints.

    Returns:
        APIRouter with ingest endpoints configured.
    """
    router = APIRouter()
    router.add_api_route("/jobs", _create_job, methods=["POST"])
    router.add_api_route("/jobs/{job_id}", _get_job, methods=["GET"])
    return router


__all__ = [
    "build_router",
]
