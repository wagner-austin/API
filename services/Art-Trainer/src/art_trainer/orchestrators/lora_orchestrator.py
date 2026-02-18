"""LoRA training orchestrator.

This module coordinates LoRA training jobs, handling enqueueing,
status management, and progress tracking.
"""

from __future__ import annotations

import uuid
from typing import Literal

from platform_core.json_utils import load_json_str
from platform_workers.redis import RedisStrProto
from typing_extensions import TypedDict

from art_trainer.api.schemas.lora import LoraTrainRequest
from art_trainer.core.config.settings import Settings
from art_trainer.core.contracts.job_result import JobResult, decode_job_result
from art_trainer.core.contracts.progress import (
    ArtTrainingProgress,
    decode_art_training_progress,
    encode_art_training_progress,
)
from art_trainer.core.contracts.queue import LoraTrainPayload
from art_trainer.core.contracts.queue_encoding import encode_lora_train_payload
from art_trainer.core.infra.redis_keys import (
    cancel_key,
    progress_key,
    result_key,
    status_key,
)
from art_trainer.core.services.queue.rq_adapter import RQEnqueuer
from art_trainer.core.services.registries import BackendRegistry

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class LoraTrainResponse(TypedDict, total=True):
    """Response from enqueue_training.

    Attributes:
        job_id: Unique job identifier.
    """

    job_id: str


class LoraStatusResponse(TypedDict, total=True):
    """Response from get_status.

    Attributes:
        job_id: Unique job identifier.
        status: Current job status.
        message: Status message or None.
        lora_file_id: File ID of the trained LoRA in data-bank.
        lora_name: Name of the deployed LoRA in ComfyUI.
    """

    job_id: str
    status: JobStatus
    message: str | None
    lora_file_id: str | None
    lora_name: str | None


class LoraProgressResponse(TypedDict, total=True):
    """Response from get_progress with result data.

    Attributes:
        job_id: Unique job identifier.
        phase: Current training phase.
        step: Current step number.
        total_steps: Total number of steps.
        loss: Current loss value.
        learning_rate: Current learning rate.
        updated_at: ISO 8601 timestamp of last update.
        lora_file_id: File ID of the trained LoRA in data-bank.
        lora_name: Name of the deployed LoRA in ComfyUI.
    """

    job_id: str
    phase: Literal[
        "queued",
        "preparing",
        "training",
        "saving",
        "uploading",
        "completed",
        "failed",
        "cancelled",
    ]
    step: int
    total_steps: int
    loss: float | None
    learning_rate: float
    updated_at: str
    lora_file_id: str | None
    lora_name: str | None


class LoraOrchestrator:
    """Orchestrator for LoRA training jobs.

    Coordinates job enqueueing, status tracking, and progress reporting.
    """

    _settings: Settings
    _redis: RedisStrProto
    _enqueuer: RQEnqueuer
    _backend_registry: BackendRegistry

    def __init__(
        self,
        settings: Settings,
        redis_client: RedisStrProto,
        enqueuer: RQEnqueuer,
        backend_registry: BackendRegistry,
    ) -> None:
        """Initialize orchestrator.

        Args:
            settings: Application settings.
            redis_client: Redis client.
            enqueuer: RQ enqueuer.
            backend_registry: Backend registry.
        """
        self._settings = settings
        self._redis = redis_client
        self._enqueuer = enqueuer
        self._backend_registry = backend_registry

    def enqueue_training(self, request: LoraTrainRequest) -> LoraTrainResponse:
        """Enqueue a LoRA training job.

        Args:
            request: Training request.

        Returns:
            Response with job ID.
        """
        job_id = str(uuid.uuid4())

        payload: LoraTrainPayload = {
            "job_id": job_id,
            "user_id": request["user_id"],
            "base_model": request["base_model"],
            "training_type": request["training_type"],
            "dataset_file_id": request["dataset_file_id"],
            "steps": request["steps"],
            "learning_rate": request["learning_rate"],
            "network_rank": request["network_rank"],
            "network_alpha": request["network_alpha"],
            "resolution": request["resolution"],
            "batch_size": request["batch_size"],
            "seed": request["seed"],
            "caption_extension": request["caption_extension"],
            "shuffle_caption": request["shuffle_caption"],
            "keep_tokens": request["keep_tokens"],
        }

        # Set initial status
        self._redis.set(status_key(job_id), "queued")

        # Set initial progress
        initial_progress: ArtTrainingProgress = {
            "job_id": job_id,
            "phase": "queued",
            "step": 0,
            "total_steps": request["steps"],
            "loss": None,
            "learning_rate": request["learning_rate"],
            "updated_at": "",
        }
        progress_json = encode_art_training_progress(initial_progress)
        from platform_core.json_utils import dump_json_str

        self._redis.set(progress_key(job_id), dump_json_str(progress_json))

        # Enqueue job
        encoded = encode_lora_train_payload(payload)
        self._enqueuer.enqueue(
            "art_trainer.worker.lora_train_job.run_lora_train",
            encoded,
            description=f"lora:{job_id}",
        )

        return {"job_id": job_id}

    def _get_result(self, job_id: str) -> JobResult | None:
        """Get the result of a completed training job.

        Args:
            job_id: Job identifier.

        Returns:
            Job result if available, None otherwise.
        """
        result_raw = self._redis.get(result_key(job_id))
        if result_raw is None:
            return None

        result_obj = load_json_str(result_raw)
        if not isinstance(result_obj, dict):
            return None

        return decode_job_result(result_obj)

    def get_status(self, job_id: str) -> LoraStatusResponse:
        """Get the status of a training job.

        Args:
            job_id: Job identifier.

        Returns:
            Status response with lora_file_id and lora_name if completed.
        """
        status_raw = self._redis.get(status_key(job_id))
        if status_raw is None:
            return {
                "job_id": job_id,
                "status": "failed",
                "message": "Job not found",
                "lora_file_id": None,
                "lora_name": None,
            }

        status: JobStatus
        if status_raw == "queued":
            status = "queued"
        elif status_raw == "running":
            status = "running"
        elif status_raw == "completed":
            status = "completed"
        elif status_raw == "cancelled":
            status = "cancelled"
        else:
            status = "failed"

        # Get result data if job is completed
        lora_file_id: str | None = None
        lora_name: str | None = None
        if status == "completed":
            result = self._get_result(job_id)
            if result is not None:
                lora_file_id = result["lora_file_id"]
                lora_name = result["lora_name"]

        return {
            "job_id": job_id,
            "status": status,
            "message": None,
            "lora_file_id": lora_file_id,
            "lora_name": lora_name,
        }

    def get_progress(self, job_id: str) -> LoraProgressResponse:
        """Get the progress of a training job.

        Args:
            job_id: Job identifier.

        Returns:
            Progress response with lora_file_id and lora_name if completed.
        """
        progress_raw = self._redis.get(progress_key(job_id))
        if progress_raw is None:
            return {
                "job_id": job_id,
                "phase": "failed",
                "step": 0,
                "total_steps": 0,
                "loss": None,
                "learning_rate": 0.0,
                "updated_at": "",
                "lora_file_id": None,
                "lora_name": None,
            }

        progress_obj = load_json_str(progress_raw)
        if not isinstance(progress_obj, dict):
            return {
                "job_id": job_id,
                "phase": "failed",
                "step": 0,
                "total_steps": 0,
                "loss": None,
                "learning_rate": 0.0,
                "updated_at": "",
                "lora_file_id": None,
                "lora_name": None,
            }

        progress = decode_art_training_progress(progress_obj)

        # Get result data if job is completed
        lora_file_id: str | None = None
        lora_name: str | None = None
        if progress["phase"] == "completed":
            result = self._get_result(job_id)
            if result is not None:
                lora_file_id = result["lora_file_id"]
                lora_name = result["lora_name"]

        return {
            "job_id": progress["job_id"],
            "phase": progress["phase"],
            "step": progress["step"],
            "total_steps": progress["total_steps"],
            "loss": progress["loss"],
            "learning_rate": progress["learning_rate"],
            "updated_at": progress["updated_at"],
            "lora_file_id": lora_file_id,
            "lora_name": lora_name,
        }

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of a training job.

        Args:
            job_id: Job identifier.

        Returns:
            True if cancellation was requested.
        """
        self._redis.set(cancel_key(job_id), "1")
        return True


__all__ = [
    "LoraOrchestrator",
    "LoraProgressResponse",
    "LoraStatusResponse",
    "LoraTrainResponse",
]
