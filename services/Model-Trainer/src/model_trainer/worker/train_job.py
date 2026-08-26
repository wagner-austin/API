"""Training job processing."""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from platform_core.determinism_record import DeterminismRecord
from platform_core.job_events import JobDomain, default_events_channel
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.queues import TRAINER_QUEUE
from platform_core.trainer_keys import cancel_key, heartbeat_key
from platform_workers.job_context import JobContext, make_job_context
from platform_workers.redis import RedisStrProto, is_redis_error

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.queue_encoding import decode_train_job_payload
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.infra.paths import model_dir
from model_trainer.core.services.export.gguf_export import GgufExportResult
from model_trainer.worker.job_utils import (
    build_cfg,
    emit_config_event,
    emit_progress_metrics,
    load_tokenizer_for_training,
    materialize_run_artifacts,
    redis_client,
    setup_env,
    setup_job_logging,
)
from model_trainer.worker.progress_store import ProgressStore
from model_trainer.worker.train_job_lifecycle import (
    _create_wandb_publisher,
    _handle_post_save_or_cancel,
    _handle_train_error,
    _maybe_export_to_gguf,
)
from model_trainer.worker.trainer_job_store import TrainerJobStore

_log = get_logger(__name__)
_TRAINER_DOMAIN: JobDomain = "trainer"


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
    determinism: DeterminismRecord,
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
        determinism=determinism,
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
        threads, determinism = setup_env(settings)
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
            determinism,
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
