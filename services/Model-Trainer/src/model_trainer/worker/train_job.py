"""Training job processing."""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.job_events import JobDomain, default_events_channel
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.queues import TRAINER_QUEUE
from platform_core.trainer_keys import artifact_file_id_key, cancel_key, heartbeat_key
from platform_ml.wandb_publisher import WandbPublisher, WandbUnavailableError
from platform_workers.job_context import JobContext, make_job_context
from platform_workers.redis import RedisStrProto, is_redis_error

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import GgufExportConfig, ModelTrainConfig, TrainOutcome
from model_trainer.core.contracts.queue_encoding import decode_train_job_payload
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.infra.paths import model_dir
from model_trainer.core.services.export.gguf_export import GgufExportResult, export_lora_to_gguf
from model_trainer.worker.job_utils import (
    build_cfg,
    emit_completed_metrics,
    emit_config_event,
    emit_progress_metrics,
    load_tokenizer_for_training,
    materialize_run_artifacts,
    redis_client,
    setup_env,
    setup_job_logging,
)
from model_trainer.worker.progress_store import ProgressStore
from model_trainer.worker.trainer_job_store import TrainerJobStore

_log = get_logger(__name__)
_TRAINER_DOMAIN: JobDomain = "trainer"


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

    from model_trainer.core import _test_hooks

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


def _execute_training(
    settings: Settings,
    r: RedisStrProto,
    run_id: str,
    user_id: int,
    cfg: ModelTrainConfig,
    threads: int,
    heartbeat_fn: Callable[[float], None],
    cancelled_fn: Callable[[], bool],
    store: TrainerJobStore,
    ctx: JobContext,
    created_at: datetime,
    resume: bool,
) -> None:
    """Execute training workflow."""
    from model_trainer.core.contracts.progress import TrainingPhase, TrainingProgress

    log = get_logger(__name__)
    progress_store = ProgressStore(r)

    # Track total steps (updated during training)
    total_steps_ref: list[int] = [0]
    last_val_loss_ref: list[float | None] = [None]
    last_val_ppl_ref: list[float | None] = [None]

    def _save_progress(
        phase: TrainingPhase,
        epoch: int,
        step: int,
        train_loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
    ) -> None:
        """Save training progress to Redis."""
        now = datetime.utcnow()
        progress: TrainingProgress = {
            "run_id": run_id,
            "phase": phase,
            "epoch": epoch,
            "total_epochs": cfg["num_epochs"],
            "step": step,
            "total_steps": total_steps_ref[0],
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "grad_norm": grad_norm,
            "samples_per_sec": samples_per_sec,
            "val_loss": last_val_loss_ref[0],
            "val_ppl": last_val_ppl_ref[0],
            "updated_at": now.isoformat(),
        }
        progress_store.save(progress)

    heartbeat_fn(_time.time())
    log.info(
        "Training started run_id=%s model_family=%s model_size=%s max_seq_len=%d "
        "num_epochs=%d batch_size=%d learning_rate=%.6f tokenizer_id=%s steps=%d",
        run_id,
        cfg["model_family"],
        cfg["model_size"],
        cfg["max_seq_len"],
        cfg["num_epochs"],
        cfg["batch_size"],
        cfg["learning_rate"],
        cfg["tokenizer_id"],
        0,
    )
    ctx.publish_started()
    emit_config_event(r, run_id, user_id, cfg, threads)

    # Save initial progress
    _save_progress("queued", 0, 0, 0.0, 0.0, 0.0, 0.0)

    def _progress(
        step: int,
        epoch: int,
        train_loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
        val_loss: float | None,
        val_ppl: float | None,
    ) -> None:
        log.info(
            "Training progress run_id=%s epoch=%d steps=%d loss=%.4f ppl=%.2f grad=%.4f",
            run_id,
            epoch,
            step,
            train_loss,
            train_ppl,
            grad_norm,
        )
        # Update validation metrics (store latest values, None if validation hasn't run)
        last_val_loss_ref[0] = val_loss if val_loss is not None else last_val_loss_ref[0]
        last_val_ppl_ref[0] = val_ppl if val_ppl is not None else last_val_ppl_ref[0]

        # Save detailed progress
        _save_progress(
            "training",
            epoch,
            step,
            train_loss,
            train_ppl,
            grad_norm,
            samples_per_sec,
        )

        progress_pct = max(0, min(99, int((epoch * 100) / max(cfg["num_epochs"], 1))))
        now = datetime.utcnow()
        store.save(
            {
                "job_id": run_id,
                "user_id": user_id,
                "status": "processing",
                "progress": progress_pct,
                "message": "training",
                "created_at": created_at,
                "updated_at": now,
                "error": None,
                "artifact_file_id": None,
            },
        )
        ctx.publish_progress(progress_pct, "training")
        emit_progress_metrics(
            r,
            run_id,
            user_id,
            int(epoch),
            cfg["num_epochs"],
            int(step),
            float(train_loss),
            float(train_ppl),
            float(grad_norm),
            float(samples_per_sec),
            val_loss,
            val_ppl,
        )

    container = _test_hooks.service_container_from_settings(settings)
    backend = container.model_registry.get(cfg["model_family"])

    # Load tokenizer handle if tokenizer_id is provided.
    # For hf_lm models, tokenizer_id may be None - the backend uses the HF
    # tokenizer from hub_model_id instead.
    tokenizer_id = cfg["tokenizer_id"]
    tok_handle: TokenizerHandle | None
    if tokenizer_id is not None:
        tok_handle = load_tokenizer_for_training(settings, tokenizer_id)
    else:
        tok_handle = None

    pretrained_run_id = cfg["pretrained_run_id"]
    if pretrained_run_id is not None:
        # The source run's artifacts were uploaded and then deleted from local
        # disk by the cleanup service, so reading model_dir() straight off disk
        # failed for every completed run. Fetch them back the same way the five
        # inference job types already do.
        pretrained_dir = str(
            materialize_run_artifacts(settings, r, pretrained_run_id, purpose="continued training")
        )
        log.info(
            "Loading pretrained model run_id=%s pretrained_run_id=%s pretrained_dir=%s",
            run_id,
            pretrained_run_id,
            pretrained_dir,
        )
        prepared = backend.load(pretrained_dir, settings, tokenizer=tok_handle)
    else:
        prepared = backend.prepare(cfg, settings, tokenizer=tok_handle)

    # Create wandb publisher if enabled via settings
    wandb_pub = _create_wandb_publisher(settings, run_id, cfg["model_family"])

    result = backend.train(
        cfg,
        settings,
        run_id=run_id,
        heartbeat=heartbeat_fn,
        cancelled=cancelled_fn,
        prepared=prepared,
        resume=resume,
        progress=_progress,
        wandb_publisher=wandb_pub,
    )
    if result["cancelled"]:
        _save_progress(
            "cancelled",
            cfg["num_epochs"],
            result["steps"],
            result["loss"],
            result["perplexity"],
            0.0,
            0.0,
        )
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
        log.info(
            "Training cancelled run_id=%s loss=%.4f perplexity=%.2f steps=%d",
            run_id,
            result["loss"],
            result["perplexity"],
            result["steps"],
        )
        ctx.publish_failed("system", "Training cancelled")
        return
    # Transition to saving phase
    _save_progress(
        "saving",
        cfg["num_epochs"],
        result["steps"],
        result["loss"],
        result["perplexity"],
        0.0,
        0.0,
    )
    out_dir = str(model_dir(settings, run_id))
    _ = backend.save(prepared, out_dir)

    # GGUF export phase (for LoRA strategies only)
    gguf_export_result: GgufExportResult | None = None
    gguf_export_cfg = cfg.get("gguf_export")
    if gguf_export_cfg is not None and gguf_export_cfg["enabled"]:
        _save_progress(
            "exporting",
            cfg["num_epochs"],
            result["steps"],
            result["loss"],
            result["perplexity"],
            0.0,
            0.0,
        )
        gguf_export_result = _maybe_export_to_gguf(
            gguf_export=gguf_export_cfg,
            out_dir=out_dir,
            hub_model_id=cfg.get("hub_model_id"),
        )
        log.info(
            "GGUF export completed run_id=%s output_size=%d",
            run_id,
            gguf_export_result["output_size_bytes"] if gguf_export_result else 0,
        )

    _handle_post_save_or_cancel(
        r=r,
        settings=settings,
        run_id=run_id,
        user_id=user_id,
        result=result,
        out_dir=out_dir,
        cancelled=cancelled_fn(),
        store=store,
        ctx=ctx,
        created_at=created_at,
        progress_store=progress_store,
        total_epochs=cfg["num_epochs"],
        gguf_export_result=gguf_export_result,
    )


def process_train_job(payload_raw: JSONObject) -> None:
    """Process a training job.

    Args:
        payload_raw: Raw JSON object from RQ queue to be decoded and validated.
    """
    settings = _test_hooks.load_settings()
    setup_job_logging(settings)

    # Decode and validate the entire payload from raw JSON
    payload = decode_train_job_payload(payload_raw)

    r = redis_client(settings)
    run_id = payload["run_id"]
    user_id = int(payload["user_id"])
    created_at = datetime.utcnow()
    job_store = TrainerJobStore(r)
    job_store.save(
        {
            "job_id": run_id,
            "user_id": user_id,
            "status": "processing",
            "progress": 0,
            "message": "started",
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
            "artifact_file_id": None,
        },
    )
    ctx: JobContext = make_job_context(
        redis=r,
        domain=_TRAINER_DOMAIN,
        events_channel=default_events_channel(_TRAINER_DOMAIN),
        job_id=run_id,
        user_id=user_id,
        queue_name=TRAINER_QUEUE,
    )

    def _hb(ts: float) -> None:
        r.set(heartbeat_key(run_id), str(ts))

    def _cancelled() -> bool:
        val = r.get(cancel_key(run_id))
        return bool(val == "1")

    req = payload["request"]
    # The guard opens before corpus resolution, not just before training: a
    # fetch or config failure must record the run as failed exactly like a
    # training failure, or the job store advertises "running" forever.
    try:
        threads = setup_env(settings)
        fid = str(req["corpus_file_id"]).strip()
        fetcher = _test_hooks.corpus_fetcher_factory(
            settings["app"]["data_bank_api_url"],
            settings["app"]["data_bank_api_key"],
            Path(settings["app"]["data_root"]) / "corpus_cache",
        )
        resolved_corpus = str(fetcher.fetch(fid))
        cfg = build_cfg(req, resolved_corpus)
        _execute_training(
            settings,
            r,
            run_id,
            user_id,
            cfg,
            threads,
            _hb,
            _cancelled,
            job_store,
            ctx,
            created_at,
            payload["resume"],
        )
    except Exception as e:
        _log.exception("Training job failed", extra={"run_id": run_id, "user_id": user_id})
        try:
            _handle_train_error(r, job_store, ctx, run_id, user_id, created_at, e)
        except BaseException as record_err:
            if not is_redis_error(record_err):
                raise
            _log.warning(
                "Failed to record training error: %s",
                record_err,
                extra={"run_id": run_id, "user_id": user_id},
            )
        raise
