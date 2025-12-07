"""Training job processing."""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.job_events import JobDomain, default_events_channel
from platform_core.logging import get_logger
from platform_core.queues import TRAINER_QUEUE
from platform_core.trainer_keys import artifact_file_id_key, cancel_key, heartbeat_key
from platform_ml.wandb_publisher import WandbPublisher, WandbUnavailableError
from platform_workers.job_context import JobContext, make_job_context
from platform_workers.redis import RedisStrProto, is_redis_error

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.contracts.model import ModelTrainConfig, TrainOutcome
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.infra.paths import model_dir
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.data import corpus_fetcher as corpus_fetcher_mod
from model_trainer.worker.job_utils import (
    build_cfg,
    emit_completed_metrics,
    emit_config_event,
    emit_progress_metrics,
    load_tokenizer_for_training,
    redis_client,
    setup_env,
    setup_job_logging,
)
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

    from platform_ml import ArtifactStore

    api_url = settings["app"]["data_bank_api_url"]
    api_key = settings["app"]["data_bank_api_key"]
    if api_url.strip() == "" or api_key.strip() == "":
        raise AppError(
            ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED,
            "data-bank-api configuration missing for artifact upload",
            model_trainer_status_for(ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED),
        )
    store = ArtifactStore(api_url, api_key)
    base = _Path(out_dir)
    fid_resp = store.upload_artifact(base, artifact_name=f"model-{run_id}", request_id=run_id)
    r.set(artifact_file_id_key(run_id), fid_resp["file_id"])
    return fid_resp["file_id"], int(fid_resp["size"])


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
) -> None:
    if cancelled:
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

    file_id, file_bytes = _upload_and_persist_pointer(settings, r, run_id, out_dir)

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
) -> None:
    """Execute training workflow."""
    log = get_logger(__name__)
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

    container = ServiceContainer.from_settings(settings)
    backend = container.model_registry.get(cfg["model_family"])
    tok_handle = load_tokenizer_for_training(settings, cfg["tokenizer_id"])
    pretrained_run_id = cfg["pretrained_run_id"]
    if pretrained_run_id is not None:
        pretrained_dir = str(model_dir(settings, pretrained_run_id))
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
        progress=_progress,
        wandb_publisher=wandb_pub,
    )
    if result["cancelled"]:
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
    out_dir = str(model_dir(settings, run_id))
    _ = backend.save(prepared, out_dir)
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
    )


def process_train_job(payload: TrainJobPayload) -> None:
    """Process a training job."""
    settings = load_settings()
    setup_job_logging(settings)

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
    threads = setup_env(settings)

    req = payload["request"]
    fid = str(req["corpus_file_id"]).strip()
    fetcher = corpus_fetcher_mod.CorpusFetcher(
        api_url=settings["app"]["data_bank_api_url"],
        api_key=settings["app"]["data_bank_api_key"],
        cache_dir=Path(settings["app"]["data_root"]) / "corpus_cache",
    )
    resolved_corpus = str(fetcher.fetch(fid))
    cfg = build_cfg(req, resolved_corpus)

    def _hb(ts: float) -> None:
        r.set(heartbeat_key(run_id), str(ts))

    def _cancelled() -> bool:
        val = r.get(cancel_key(run_id))
        return bool(val == "1")

    try:
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
