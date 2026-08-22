"""Train-job lifecycle helpers: wandb, errors, artifacts, GGUF, post-save."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger
from platform_core.trainer_keys import artifact_file_id_key
from platform_ml.wandb_publisher import WandbPublisher, WandbUnavailableError
from platform_workers.job_context import JobContext
from platform_workers.redis import RedisStrProto

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import GgufExportConfig, TrainOutcome
from model_trainer.core.services.export.gguf_export import GgufExportResult, export_lora_to_gguf
from model_trainer.worker.job_utils import (
    emit_completed_metrics,
)
from model_trainer.worker.progress_store import ProgressStore
from model_trainer.worker.trainer_job_store import TrainerJobStore

_log = get_logger(__name__)


def _create_wandb_publisher(
    settings: Settings, run_id: str, model_family: str
) -> WandbPublisher | None:
    """Create a wandb publisher from settings.

    Args:
        settings: Application settings with wandb configuration.
        run_id: Training run identifier.
        model_family: Model family name (char_lstm, gpt2).

    Returns:
        WandbPublisher if enabled and wandb installed, None otherwise.
    """
    wandb_cfg = settings["wandb"]
    if not wandb_cfg["enabled"]:
        return None

    project = wandb_cfg["project"]
    run_name = f"{model_family}-{run_id}"

    try:
        return WandbPublisher(project=project, run_name=run_name, enabled=True)
    except WandbUnavailableError:
        _log.warning("wandb enabled but not installed, skipping wandb logging")
        return None


def _handle_train_error(
    r: RedisStrProto,
    store: TrainerJobStore,
    ctx: JobContext,
    run_id: str,
    user_id: int,
    created_at: datetime,
    error: Exception,
) -> None:
    """Handle training job error."""
    store.save(
        {
            "job_id": run_id,
            "user_id": user_id,
            "status": "failed",
            "progress": 100,
            "message": str(error),
            "created_at": created_at,
            "updated_at": datetime.utcnow(),
            "error": str(error),
            "artifact_file_id": None,
        },
    )
    get_logger(__name__).exception("Training job failed run_id=%s error=%s", run_id, error)
    ctx.publish_failed("system", str(error))


def _upload_and_persist_pointer(
    settings: Settings, r: RedisStrProto, run_id: str, out_dir: str
) -> tuple[str, int]:
    from pathlib import Path as _Path

    log = get_logger(__name__)
    log.info(
        "Artifact upload started run_id=%s out_dir=%s",
        run_id,
        out_dir,
        extra={
            "category": "artifact",
            "event": "upload_started",
            "run_id": run_id,
            "out_dir": out_dir,
        },
    )

    api_url = settings["app"]["data_bank_api_url"]
    api_key = settings["app"]["data_bank_api_key"]
    if api_url.strip() == "" or api_key.strip() == "":
        raise AppError(
            ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED,
            "data-bank-api configuration missing for artifact upload",
            model_trainer_status_for(ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED),
        )
    store = _test_hooks.artifact_store_factory(api_url, api_key)
    base = _Path(out_dir)
    fid_resp = store.upload_artifact(base, artifact_name=f"model-{run_id}", request_id=run_id)
    r.set(artifact_file_id_key(run_id), fid_resp["file_id"])

    file_id = fid_resp["file_id"]
    file_size = int(fid_resp["size"])
    log.info(
        "Artifact upload completed run_id=%s file_id=%s size=%d",
        run_id,
        file_id,
        file_size,
        extra={
            "category": "artifact",
            "event": "upload_completed",
            "run_id": run_id,
            "file_id": file_id,
            "size": file_size,
        },
    )

    return file_id, file_size


def _maybe_export_to_gguf(
    gguf_export: GgufExportConfig | None,
    out_dir: str,
    hub_model_id: str | None,
) -> GgufExportResult | None:
    """Export to GGUF format if configured.

    Args:
        gguf_export: GGUF export configuration, or None if not enabled.
        out_dir: Output directory containing the adapter weights.
        hub_model_id: HuggingFace model ID of the base model.

    Returns:
        GgufExportResult if export was performed, None otherwise.

    Raises:
        RuntimeError: If export is enabled but hub_model_id is None.
    """
    if gguf_export is None or not gguf_export["enabled"]:
        return None

    if hub_model_id is None:
        raise RuntimeError("GGUF export requires hub_model_id to be set")

    return export_lora_to_gguf(
        adapter_dir=out_dir,
        base_model_id=hub_model_id,
        output_dir=out_dir,
        output_type=gguf_export["output_type"],
    )


def _handle_post_save_or_cancel(
    *,
    r: RedisStrProto,
    settings: Settings,
    run_id: str,
    user_id: int,
    result: TrainOutcome,
    out_dir: str,
    cancelled: bool,
    store: TrainerJobStore,
    ctx: JobContext,
    created_at: datetime,
    progress_store: ProgressStore,
    total_epochs: int,
    gguf_export_result: GgufExportResult | None,
) -> None:
    from model_trainer.core.contracts.progress import TrainingProgress

    def _save_phase_progress(
        phase: Literal["uploading", "completed", "cancelled"],
    ) -> None:
        """Save progress with given phase."""
        phase_lit = phase

        now = datetime.utcnow()
        progress: TrainingProgress = {
            "run_id": run_id,
            "phase": phase_lit,
            "epoch": total_epochs,
            "total_epochs": total_epochs,
            "step": result["steps"],
            "total_steps": result["steps"],
            "train_loss": result["loss"],
            "train_ppl": result["perplexity"],
            "grad_norm": 0.0,
            "samples_per_sec": 0.0,
            "val_loss": result["best_val_loss"],
            "val_ppl": None,  # val_perplexity not tracked in TrainOutcome
            "updated_at": now.isoformat(),
        }
        progress_store.save(progress)

    if cancelled:
        _save_phase_progress("cancelled")
        now = datetime.utcnow()
        store.save(
            {
                "job_id": run_id,
                "user_id": user_id,
                "status": "failed",
                "progress": 100,
                "message": "Training cancelled",
                "created_at": created_at,
                "updated_at": now,
                "error": "Training cancelled",
                "artifact_file_id": None,
            },
        )
        get_logger(__name__).info(
            "Training cancelled run_id=%s loss=%.4f perplexity=%.2f steps=%d",
            run_id,
            result["loss"],
            result["perplexity"],
            result["steps"],
        )
        ctx.publish_failed("system", "Training cancelled")
        return

    # Transition to uploading phase
    _save_phase_progress("uploading")
    file_id, file_bytes = _upload_and_persist_pointer(settings, r, run_id, out_dir)

    # Transition to completed phase
    _save_phase_progress("completed")

    now = datetime.utcnow()
    store.save(
        {
            "job_id": run_id,
            "user_id": user_id,
            "status": "completed",
            "progress": 100,
            "message": "Training completed",
            "created_at": created_at,
            "updated_at": now,
            "error": None,
            "artifact_file_id": file_id,
        },
    )
    get_logger(__name__).info(
        "Training completed run_id=%s loss=%.4f perplexity=%.2f steps=%d",
        run_id,
        result["loss"],
        result["perplexity"],
        result["steps"],
    )

    from model_trainer.core.services.storage.artifact_cleanup import ArtifactCleanupService

    cleanup_service = ArtifactCleanupService(settings=settings, redis_client=r)
    _ = cleanup_service.cleanup_run_artifacts(run_id, out_dir)

    ctx.publish_completed(file_id, file_bytes)
    # Use test metrics if available, fallback to train metrics
    final_loss = result["test_loss"] if result["test_loss"] is not None else result["loss"]
    final_ppl = (
        result["test_perplexity"] if result["test_perplexity"] is not None else result["perplexity"]
    )
    emit_completed_metrics(
        r,
        run_id,
        user_id,
        final_loss,
        final_ppl,
        out_dir,
    )
