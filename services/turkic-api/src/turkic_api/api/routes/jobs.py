from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from platform_core.errors import AppError, ErrorCode
from platform_workers.redis import RedisStrProto

from ..config import Settings
from ..models import parse_job_create
from ..provider_context import (
    get_logger_from_context,
    get_queue_from_context,
    get_redis_from_context,
    get_settings_from_context,
)
from ..services import JobService
from ..streaming import stream_data_bank_file
from ..types import JsonDict, LoggerProtocol, QueueProtocol


def _to_hash_redis(r: RedisStrProto) -> RedisStrProto:
    return r


def _create_service(
    *,
    redis: RedisStrProto,
    logger: LoggerProtocol,
    queue: QueueProtocol,
    settings: Settings,
) -> JobService:
    return JobService(
        redis=_to_hash_redis(redis),
        logger=logger,
        queue=queue,
        data_dir=settings["data_dir"],
    )


def build_router() -> APIRouter:
    router = APIRouter()

    async def create_job_endpoint(
        payload: JsonDict,
        redis: Annotated[RedisStrProto, Depends(get_redis_from_context)],
        logger: Annotated[LoggerProtocol, Depends(get_logger_from_context)],
        queue: Annotated[QueueProtocol, Depends(get_queue_from_context)],
        settings: Annotated[Settings, Depends(get_settings_from_context)],
    ) -> JSONResponse:
        job = parse_job_create(payload)
        service = _create_service(redis=redis, logger=logger, queue=queue, settings=settings)
        result = await service.create_job(job)
        serialized: dict[str, str | int | float | bool | None] = {
            "job_id": result["job_id"],
            "user_id": result["user_id"],
            "status": result["status"],
            "created_at": result["created_at"].isoformat(),
        }
        return JSONResponse(content=serialized)

    async def get_job_endpoint(
        job_id: str,
        redis: Annotated[RedisStrProto, Depends(get_redis_from_context)],
        logger: Annotated[LoggerProtocol, Depends(get_logger_from_context)],
        queue: Annotated[QueueProtocol, Depends(get_queue_from_context)],
        settings: Annotated[Settings, Depends(get_settings_from_context)],
    ) -> JSONResponse:
        service = _create_service(redis=redis, logger=logger, queue=queue, settings=settings)
        status_obj = service.get_job_status(job_id)
        if status_obj is None:
            raise AppError(code=ErrorCode.JOB_NOT_FOUND, message="Job not found", http_status=404)
        serialized: dict[str, str | int | float | bool | None] = {
            "job_id": status_obj["job_id"],
            "user_id": status_obj["user_id"],
            "status": status_obj["status"],
            "progress": status_obj["progress"],
            "message": status_obj["message"],
            "result_url": status_obj["result_url"],
            "file_id": status_obj["file_id"],
            "upload_status": status_obj["upload_status"],
            "created_at": status_obj["created_at"].isoformat(),
            "updated_at": status_obj["updated_at"].isoformat(),
            "error": status_obj["error"],
        }
        return JSONResponse(content=serialized)

    async def get_job_result_endpoint(
        job_id: str,
        redis: Annotated[RedisStrProto, Depends(get_redis_from_context)],
        logger: Annotated[LoggerProtocol, Depends(get_logger_from_context)],
        queue: Annotated[QueueProtocol, Depends(get_queue_from_context)],
        settings: Annotated[Settings, Depends(get_settings_from_context)],
    ) -> StreamingResponse:
        service = _create_service(redis=redis, logger=logger, queue=queue, settings=settings)
        status_obj = service.get_job_status(job_id)
        if status_obj is None:
            raise AppError(code=ErrorCode.JOB_NOT_FOUND, message="Job not found", http_status=404)
        if status_obj["status"] != "completed":
            raise AppError(
                code=ErrorCode.JOB_NOT_READY,
                message="Job not completed",
                http_status=425,
            )
        file_id = status_obj["file_id"]
        upload_status = status_obj["upload_status"]
        if file_id is None or upload_status != "uploaded":
            raise AppError(
                code=ErrorCode.JOB_FAILED,
                message="Job result not available",
                http_status=410,
            )
        return stream_data_bank_file(job_id, file_id, settings)

    router.add_api_route("/api/v1/jobs", create_job_endpoint, methods=["POST"])
    router.add_api_route("/api/v1/jobs/{job_id}", get_job_endpoint, methods=["GET"])
    router.add_api_route("/api/v1/jobs/{job_id}/result", get_job_result_endpoint, methods=["GET"])
    return router


__all__ = ["build_router"]
