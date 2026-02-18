"""LoRA training routes for Art-Trainer API.

This module provides endpoints for LoRA training operations.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.params import Depends as DependsParamType
from platform_core.json_utils import JSONTypeError, load_json_str
from platform_core.logging import get_logger

from art_trainer.api.middleware import api_key_dependency
from art_trainer.api.schemas.lora import (
    LoraCancelResponse,
    LoraProgressResponse,
    LoraStatusResponse,
    LoraTrainRequest,
    LoraTrainResponse,
)
from art_trainer.api.validators.lora import decode_lora_train_request
from art_trainer.core.services.container import ServiceContainer

_logger = get_logger(__name__)


class _LoraRoutes:
    """Route handlers for LoRA training endpoints."""

    _container: ServiceContainer

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize route handlers.

        Args:
            container: Service container.
        """
        self._container = container

    async def start_training(self, request: Request) -> LoraTrainResponse:
        """Start a new LoRA training job.

        Args:
            request: FastAPI request.

        Returns:
            Training response with job ID.
        """
        raw_body = await request.body()
        body = load_json_str(raw_body.decode("utf-8"))
        if not isinstance(body, dict):
            raise JSONTypeError("Request body must be a JSON object")
        req: LoraTrainRequest = decode_lora_train_request(body)

        orchestrator = self._container.lora_orchestrator
        _logger.info(
            "lora enqueue",
            extra={
                "category": "api",
                "service": "lora",
                "event": "lora_enqueue",
                "base_model": req["base_model"],
                "training_type": req["training_type"],
            },
        )
        result = orchestrator.enqueue_training(req)
        return {"job_id": result["job_id"]}

    def get_status(self, job_id: str) -> LoraStatusResponse:
        """Get the status of a training job.

        Args:
            job_id: Job identifier.

        Returns:
            Job status response with lora_file_id and lora_name if completed.
        """
        orchestrator = self._container.lora_orchestrator
        result = orchestrator.get_status(job_id)
        return {
            "job_id": result["job_id"],
            "status": result["status"],
            "message": result["message"],
            "lora_file_id": result["lora_file_id"],
            "lora_name": result["lora_name"],
        }

    def get_progress(self, job_id: str) -> LoraProgressResponse:
        """Get the progress of a training job.

        Args:
            job_id: Job identifier.

        Returns:
            Job progress response with lora_file_id and lora_name if completed.
        """
        orchestrator = self._container.lora_orchestrator
        progress = orchestrator.get_progress(job_id)
        return {
            "job_id": progress["job_id"],
            "phase": progress["phase"],
            "step": progress["step"],
            "total_steps": progress["total_steps"],
            "loss": progress["loss"],
            "learning_rate": progress["learning_rate"],
            "updated_at": progress["updated_at"],
            "lora_file_id": progress["lora_file_id"],
            "lora_name": progress["lora_name"],
        }

    def cancel_job(self, job_id: str) -> LoraCancelResponse:
        """Cancel a training job.

        Args:
            job_id: Job identifier.

        Returns:
            Cancellation response.
        """
        orchestrator = self._container.lora_orchestrator
        orchestrator.cancel_job(job_id)
        _logger.info(
            "lora cancel",
            extra={
                "category": "api",
                "service": "lora",
                "event": "lora_cancel",
                "job_id": job_id,
            },
        )
        return {"status": "cancellation-requested"}


def build_router(container: ServiceContainer) -> APIRouter:
    """Build LoRA routes router.

    Args:
        container: Service container.

    Returns:
        Configured API router.
    """
    api_dep: DependsParamType = Depends(api_key_dependency(container.settings))
    router = APIRouter(dependencies=[api_dep])
    handlers = _LoraRoutes(container)

    router.add_api_route("/train", handlers.start_training, methods=["POST"])
    router.add_api_route("/{job_id}", handlers.get_status, methods=["GET"])
    router.add_api_route("/{job_id}/progress", handlers.get_progress, methods=["GET"])
    router.add_api_route("/{job_id}/cancel", handlers.cancel_job, methods=["POST"])

    return router


__all__ = [
    "build_router",
]
