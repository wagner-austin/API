"""Training job routes for handwriting-ai."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict
from uuid import uuid4

from fastapi import APIRouter
from fastapi.params import Depends as DependsParamType
from fastapi.responses import JSONResponse
from platform_core.errors import AppError, ErrorCode
from platform_core.security import ApiKeyCheckFn

from ..dependencies import LoggerDep, QueueDep
from ..types import JsonDict, UnknownJson


class _DependsCtor(Protocol):
    """Protocol for FastAPI Depends constructor with typed return."""

    def __call__(self, dependency: ApiKeyCheckFn) -> DependsParamType: ...


def _typed_depends(dep: ApiKeyCheckFn) -> DependsParamType:
    """Create Depends instance with proper typing via dynamic import."""
    fastapi_mod = __import__("fastapi")
    depends_ctor: _DependsCtor = fastapi_mod.Depends
    result: DependsParamType = depends_ctor(dep)
    return result


class TrainJobRequest(TypedDict):
    """Request payload for creating a training job."""

    user_id: int
    model_id: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    augment: bool
    notes: str | None


class TrainJobResponse(TypedDict):
    """Response payload after creating a training job."""

    job_id: str
    user_id: int
    model_id: str
    status: Literal["queued"]


def _validate_train_request(payload: JsonDict) -> TrainJobRequest:
    """Validate and parse training job request payload."""
    user_id = payload.get("user_id")
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        raise AppError(ErrorCode.INVALID_INPUT, "user_id must be an integer")

    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise AppError(ErrorCode.INVALID_INPUT, "model_id must be a non-empty string")

    epochs = payload.get("epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise AppError(ErrorCode.INVALID_INPUT, "epochs must be a positive integer")

    batch_size = payload.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise AppError(ErrorCode.INVALID_INPUT, "batch_size must be a positive integer")

    lr = payload.get("lr")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr <= 0:
        raise AppError(ErrorCode.INVALID_INPUT, "lr must be a positive number")

    seed = payload.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise AppError(ErrorCode.INVALID_INPUT, "seed must be an integer")

    augment = payload.get("augment", False)
    if not isinstance(augment, bool):
        raise AppError(ErrorCode.INVALID_INPUT, "augment must be a boolean")

    notes_raw = payload.get("notes")
    notes: str | None = None
    if notes_raw is not None:
        if not isinstance(notes_raw, str):
            raise AppError(ErrorCode.INVALID_INPUT, "notes must be a string or null")
        notes = notes_raw

    return {
        "user_id": user_id,
        "model_id": model_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": float(lr),
        "seed": seed,
        "augment": augment,
        "notes": notes,
    }


def build_router(api_key_dep: ApiKeyCheckFn) -> APIRouter:
    """Build training router with job submission endpoint."""
    router = APIRouter(dependencies=[_typed_depends(api_key_dep)])

    async def create_training_job(
        payload: JsonDict,
        queue: QueueDep,
        logger: LoggerDep,
    ) -> JSONResponse:
        """Create a new digit training job and enqueue for background processing."""
        req = _validate_train_request(payload)

        request_id = str(uuid4())
        user_id = req["user_id"]
        model_id = req["model_id"]

        logger.info(
            "Enqueuing training job",
            extra={"request_id": request_id, "model_id": model_id, "user_id": user_id},
        )

        # Build typed payload for the worker
        job_payload: dict[str, UnknownJson] = {
            "type": "digits.train.v1",
            "request_id": request_id,
            "user_id": user_id,
            "model_id": model_id,
            "epochs": req["epochs"],
            "batch_size": req["batch_size"],
            "lr": req["lr"],
            "seed": req["seed"],
            "augment": req["augment"],
            "notes": req["notes"],
        }

        # Enqueue the job
        job = queue.enqueue(
            "handwriting_ai.jobs.digits.process_train_job",
            job_payload,
            job_timeout=3600,  # 1 hour timeout for training
            result_ttl=86400,  # Keep results for 24 hours
            failure_ttl=86400,
            description=f"digits:train:{model_id}:{request_id}",
        )

        logger.info(
            "Training job enqueued",
            extra={"request_id": request_id, "job_id": job.get_id()},
        )

        response: dict[str, str | int | bool | None] = {
            "job_id": job.get_id(),
            "request_id": request_id,
            "user_id": user_id,
            "model_id": model_id,
            "status": "queued",
        }
        return JSONResponse(content=response, status_code=202)

    router.add_api_route("/api/v1/training/jobs", create_training_job, methods=["POST"])
    return router


__all__ = ["build_router"]
