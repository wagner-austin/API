"""LoRA training worker job.

This module defines the RQ job function for executing LoRA training.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str
from platform_workers.redis import RedisStrProto

from art_trainer.core import _test_hooks
from art_trainer.core.contracts.job_result import JobResult, encode_job_result
from art_trainer.core.contracts.lora import LoraTrainConfig
from art_trainer.core.contracts.progress import (
    ArtTrainingPhase,
    ArtTrainingProgress,
    encode_art_training_progress,
)
from art_trainer.core.contracts.queue_encoding import decode_lora_train_payload
from art_trainer.core.infra.paths import lora_output_dir
from art_trainer.core.infra.redis_keys import (
    cancel_key,
    progress_key,
    result_key,
    status_key,
)
from art_trainer.core.services.dataset import download_dataset
from art_trainer.core.services.dataset.uploader import upload_lora
from art_trainer.core.services.deployment import deploy_lora
from art_trainer.core.services.training.backend_factory import create_kohya_backend


def run_lora_train(payload: JSONObject) -> None:
    """Execute LoRA training job.

    Downloads dataset from data-bank, runs training, uploads LoRA to data-bank,
    and deploys to ComfyUI.

    Args:
        payload: Job payload from queue.

    Raises:
        DataBankDownloadError: If dataset download fails.
        DataBankUploadError: If LoRA upload fails.
    """
    # Decode payload
    decoded = decode_lora_train_payload(payload)
    job_id = decoded["job_id"]
    dataset_file_id = decoded["dataset_file_id"]

    # Load settings and create redis client
    settings = _test_hooks.load_settings()
    redis: RedisStrProto = _test_hooks.kv_store_factory(settings["redis"]["url"])

    # Update status to running
    redis.set(status_key(job_id), "running")

    # Phase 1: Preparing - Download dataset from data-bank
    _set_progress(redis, job_id, "preparing", 0, decoded["steps"])
    dataset_path = download_dataset(settings, dataset_file_id, job_id)

    # Build config
    config: LoraTrainConfig = {
        "job_id": job_id,
        "base_model": decoded["base_model"],
        "training_type": decoded["training_type"],
        "dataset_dir": str(dataset_path),
        "output_dir": str(lora_output_dir(settings, job_id)),
        "steps": decoded["steps"],
        "learning_rate": decoded["learning_rate"],
        "network_rank": decoded["network_rank"],
        "network_alpha": decoded["network_alpha"],
        "resolution": decoded["resolution"],
        "batch_size": decoded["batch_size"],
        "seed": decoded["seed"],
        "caption_extension": decoded["caption_extension"],
        "shuffle_caption": decoded["shuffle_caption"],
        "keep_tokens": decoded["keep_tokens"],
    }

    # Create progress callback
    def progress_callback(progress: ArtTrainingProgress) -> None:
        progress_json = encode_art_training_progress(progress)
        redis.set(progress_key(job_id), dump_json_str(progress_json))

    # Create cancellation check
    def cancelled() -> bool:
        cancel_flag = redis.get(cancel_key(job_id))
        return cancel_flag == "1"

    # Phase 2: Training - Run backend training
    backend = create_kohya_backend(settings)
    outcome = backend.train(
        config,
        progress_callback=progress_callback,
        cancelled=cancelled,
    )

    # Handle training outcome
    if outcome["success"]:
        lora_path_str = outcome["lora_path"]
        lora_file_id: str | None = None
        lora_name: str | None = None

        if lora_path_str is not None:
            lora_path = Path(lora_path_str)

            # Phase 3: Uploading - Upload LoRA to data-bank
            _set_progress(redis, job_id, "uploading", config["steps"], config["steps"])
            upload_result = upload_lora(settings, lora_path)
            lora_file_id = upload_result["file_id"]

            # Deploy LoRA to ComfyUI
            lora_name = f"lora_{job_id}"
            deploy_lora(settings, lora_path, lora_name)

        # Store result with lora_file_id
        result: JobResult = {
            "job_id": job_id,
            "lora_file_id": lora_file_id,
            "lora_name": lora_name,
        }
        redis.set(result_key(job_id), dump_json_str(encode_job_result(result)))

        redis.set(status_key(job_id), "completed")
        _set_progress(redis, job_id, "completed", config["steps"], config["steps"])
    elif outcome["error_message"] == "Training cancelled by user":
        redis.set(status_key(job_id), "cancelled")
        _set_progress(redis, job_id, "cancelled", 0, 0)
    else:
        redis.set(status_key(job_id), "failed")
        _set_progress(redis, job_id, "failed", 0, 0)


def _set_progress(
    redis: RedisStrProto,
    job_id: str,
    phase: ArtTrainingPhase,
    step: int,
    total_steps: int,
) -> None:
    """Set progress state in Redis.

    Args:
        redis: Redis client.
        job_id: Job identifier.
        phase: Current training phase.
        step: Current step number.
        total_steps: Total number of steps.
    """
    progress: ArtTrainingProgress = {
        "job_id": job_id,
        "phase": phase,
        "step": step,
        "total_steps": total_steps,
        "loss": None,
        "learning_rate": 0.0,
        "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    progress_json = encode_art_training_progress(progress)
    redis.set(progress_key(job_id), dump_json_str(progress_json))


__all__ = [
    "run_lora_train",
]
