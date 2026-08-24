"""Job utilities for training worker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from platform_core.config import _optional_env_str
from platform_core.determinism_env import DETERMINISM_ENV_VAR, determinism_requested
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.job_events import default_events_channel
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.queues import TRAINER_QUEUE
from platform_core.trainer_keys import artifact_file_id_key
from platform_core.trainer_metrics_events import (
    encode_trainer_metrics_event,
    make_completed_metrics_event,
    make_config_event,
    make_progress_metrics_event,
)
from platform_ml import RequestedDevice, ResolvedDevice, encode_determinism_report
from platform_workers.redis import RedisStrProto

from model_trainer.core import _test_hooks
from model_trainer.core.compute.device_selector import (
    RequestedPrecision,
    ResolvedPrecision,
    recommended_batch_size_for,
    resolve_device,
    resolve_precision,
)
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.compute import LocalCPUProvider
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.queue import TrainRequestPayload
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.infra.paths import models_dir
from model_trainer.core.logging.types import LOGGING_EXTRA_FIELDS
from model_trainer.core.services.tokenizer.loader import load_tokenizer_from_dir
from model_trainer.worker.trainer_job_store import TrainerJobStore

_log = get_logger(__name__)

EVENTS_CHANNEL = default_events_channel("trainer")


def redis_client(settings: Settings) -> RedisStrProto:
    """Create Redis client from settings."""
    return _test_hooks.kv_store_factory(settings["redis"]["url"])


#: Materialized run directories the models root keeps. A run dir is ~1.4 GB,
#: so four is ~6 GB. Five job types re-materialize completed runs through
#: ``materialize_run_artifacts`` and nothing used to delete them, which is how
#: the models root reached 49 GB.
MATERIALIZED_RUN_KEEP: Final = 4


def _last_used(run_dir: Path) -> float:
    """Recency key for a materialized run directory.

    Args:
        run_dir: Directory to read.

    Returns:
        Its modification time, which ``materialize_run_artifacts`` stamps on
        every use so that recency tracks use rather than download age.
    """
    return run_dir.stat().st_mtime


def _materialized_run_dirs(models_root: Path) -> list[Path]:
    """List materialized run directories, most recently used first.

    Args:
        models_root: Directory holding one subdirectory per materialized run.

    Returns:
        The run directories, newest mtime first.
    """
    if not models_root.is_dir():
        return []
    dirs = [child for child in models_root.iterdir() if child.is_dir()]
    return sorted(dirs, key=_last_used, reverse=True)


def evict_materialized_runs(
    settings: Settings,
    redis: RedisStrProto,
    *,
    keep: int = MATERIALIZED_RUN_KEEP,
) -> tuple[str, ...]:
    """Evict the least recently used materialized run directories.

    ``materialize_run_artifacts`` is deliberately cache-semantic: five job
    types, including interactive chat and generate, re-materialize a completed
    run and expect it to still be there next call. Deleting after each use
    would force a 1.4 GB re-download per chat message. So the cache is bounded
    instead of emptied.

    A directory whose run is NOT terminal is never evicted, because the
    actively-training run writes into this same models root and evicting it
    would delete a run out from under itself. That makes eviction conservative
    when a run is stuck in a non-terminal state, which is the safe direction.

    Args:
        settings: Service settings carrying the models root.
        redis: Client holding the run statuses.
        keep: How many of the most recently used directories to retain.

    Returns:
        The run ids evicted, most recently used first.
    """
    store = TrainerJobStore(redis)
    evicted: list[str] = []
    for candidate in _materialized_run_dirs(models_dir(settings))[keep:]:
        run_id = candidate.name
        status_record = store.load(run_id)
        status = None if status_record is None else status_record["status"]
        if status not in ("completed", "failed"):
            _log.info(
                "Materialized-run eviction skipped: run not terminal",
                extra={
                    "event": "materialized_evict_skipped",
                    "run_id": run_id,
                    "reason": "run_not_terminal",
                    "status": status,
                },
            )
            continue
        _test_hooks.shutil_rmtree(candidate)
        evicted.append(run_id)
        _log.info(
            "Materialized-run evicted",
            extra={
                "event": "materialized_evicted",
                "run_id": run_id,
                "path": str(candidate),
            },
        )
    return tuple(evicted)


def materialize_run_artifacts(
    settings: Settings,
    redis: RedisStrProto,
    run_id: str,
    *,
    purpose: str,
) -> Path:
    """Return the local directory holding a completed run's artifacts.

    A finished run's artifacts are uploaded to the artifact store and then
    deleted from local disk by ``ArtifactCleanupService``, so for any run that
    is not the one currently training, "already on disk" is the exception
    rather than the rule. Every consumer therefore has to be able to fetch them
    back, and five job types each grew their own copy of that fetch. This is
    the single copy; ``train_job`` had no copy at all, which is why continuing
    training from a ``pretrained_run_id`` failed with a missing-metadata error
    for every completed source run.

    Args:
        settings: Service settings carrying the artifact-store credentials.
        redis: Client holding the run's artifact pointer.
        run_id: Run whose artifacts are needed.
        purpose: What needs them, used only in the not-found message.

    Returns:
        Directory containing the run's artifacts.

    Raises:
        AppError: With ``DATA_NOT_FOUND`` when the artifacts are absent locally
            and the run has no artifact pointer, which together mean there is
            nowhere left to get them from.
    """
    models_root = models_dir(settings)
    normalized = models_root / run_id
    if normalized.exists():
        # Already on disk, so the pointer is irrelevant -- requiring it here
        # would fail a run whose artifacts are present but whose upload
        # pointer was never written.
        #
        # Touched so that recency reflects USE and not the download that first
        # created the directory. Without this the cache evicts by age and the
        # run being chatted with every minute is the one thrown away.
        _test_hooks.os_utime(normalized)
        evict_materialized_runs(settings, redis)
        return normalized

    file_id = redis.get(artifact_file_id_key(run_id))
    if not isinstance(file_id, str) or file_id.strip() == "":
        raise AppError(
            ModelTrainerErrorCode.DATA_NOT_FOUND,
            f"artifact pointer not found for {purpose} (run_id={run_id})",
            model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
        )

    api_url = settings["app"]["data_bank_api_url"]
    api_key = settings["app"]["data_bank_api_key"]
    store = _test_hooks.artifact_store_factory(api_url, api_key)
    out_root = store.download_artifact(
        file_id.strip(),
        dest_dir=models_root,
        request_id=run_id,
        expected_root=f"model-{run_id}",
    )
    out_root.rename(normalized)
    # Evicted AFTER the rename so the run just fetched is the newest entry and
    # is never itself a candidate.
    evict_materialized_runs(settings, redis)
    return normalized


def publish_metrics(r: RedisStrProto, message: str) -> None:
    """Publish a trainer metrics event to the standard events channel."""
    r.publish(EVENTS_CHANNEL, message)


def setup_env(settings: Settings) -> int:
    """Setup environment for training job.

    Also pins kernel-level numerical determinism, and does it HERE because
    this is the last point in the job that is guaranteed to precede any CUDA
    work: ``CUBLAS_WORKSPACE_CONFIG`` is read once when the cuBLAS handle is
    created, so a later call is accepted in silence and has no effect.

    Seeding is separate and already handled per-run. Seeds fix what the
    sampler draws; they do not fix the order a GPU accumulates a reduction
    in, and floating-point addition is not associative, so without this an
    identical config on identical hardware still yields a different model on
    every run. That variance was previously indistinguishable from seed
    spread in any experiment reading small between-arm differences.

    The applied settings are logged rather than assumed, because a run whose
    determinism settings are unknown cannot be compared with one whose are.
    That logging is also the ONLY truthful record of the posture: a launcher
    can declare determinism, but only the process that makes the torch calls
    knows whether they happened. A run that skips them logs that it skipped
    them, in the same field, so the two are never distinguished by absence.

    The posture comes from the environment via
    :func:`~platform_core.determinism_env.determinism_requested`, which
    defaults to ON when nothing asked. A launcher that wants speed instead
    sets the variable explicitly; an unreadable value raises rather than
    resolving to either posture.

    Args:
        settings: Application settings supplying the thread count.

    Returns:
        The resolved thread count.

    Raises:
        ValueError: If the determinism variable is present and unreadable.
            Raised before any CUDA work, so the job fails rather than running
            under a posture nobody can name.
    """
    threads_cfg = settings["app"]["threads"]
    threads = threads_cfg if threads_cfg and threads_cfg > 0 else max(1, int(os.cpu_count() or 1))
    env = LocalCPUProvider(threads_count=threads).env()
    for k, v in env.items():
        __import__("os").putenv(k, v)
    __import__("os").putenv("TOKENIZERS_PARALLELISM", "1")

    if not determinism_requested(_optional_env_str(DETERMINISM_ENV_VAR)):
        _log.info("determinism declined", extra={"determinism": "off"})
        return threads

    report = _test_hooks.apply_determinism_hook()
    _log.info("determinism pinned", extra={"determinism": encode_determinism_report(report)})
    return threads


def build_cfg(req: TrainRequestPayload, corpus_path: str) -> ModelTrainConfig:
    """Build ModelTrainConfig from request payload.

    Args:
        req: Training request payload from queue.
        corpus_path: Resolved path to corpus directory.

    Returns:
        ModelTrainConfig ready for training.

    Raises:
        RuntimeError: If fp16/bf16 precision is requested on CPU.
    """
    # Resolve device once at job start ("auto" -> concrete device)
    requested_device: RequestedDevice = req["device"]
    resolved_device: ResolvedDevice = resolve_device(requested_device)

    # Resolve precision based on device ("auto" -> fp16 on CUDA, fp32 on CPU)
    requested_precision: RequestedPrecision = req["precision"]
    resolved_precision: ResolvedPrecision = resolve_precision(requested_precision, resolved_device)

    # Resolve data loader knobs; prefer explicit values, otherwise device-based defaults
    req_workers = req.get("data_num_workers")
    req_pinmem = req.get("data_pin_memory")
    cpu_count = int(os.cpu_count() or 1)
    default_workers = min(4, cpu_count) if resolved_device == "cuda" else 0
    data_num_workers = default_workers if req_workers is None else int(req_workers)
    data_pin_memory = (resolved_device == "cuda") if req_pinmem is None else bool(req_pinmem)

    # Adjust batch size conservatively for CUDA when client used typical default
    bs_in = int(req["batch_size"])  # explicit int conversion for safety
    bs_eff = recommended_batch_size_for(req["model_family"], bs_in, resolved_device)

    cfg: ModelTrainConfig = {
        "model_family": req["model_family"],
        "model_size": req["model_size"],
        "max_seq_len": req["max_seq_len"],
        "num_epochs": req["num_epochs"],
        "batch_size": bs_eff,
        "learning_rate": req["learning_rate"],
        "tokenizer_id": req["tokenizer_id"],
        "corpus_path": corpus_path,
        "holdout_fraction": req["holdout_fraction"],
        "seed": req["seed"],
        "pretrained_run_id": req["pretrained_run_id"],
        "freeze_embed": req["freeze_embed"],
        "gradient_clipping": req["gradient_clipping"],
        "optimizer": req["optimizer"],
        "device": resolved_device,
        "precision": resolved_precision,
        "data_num_workers": data_num_workers,
        "data_pin_memory": data_pin_memory,
        "early_stopping_patience": req["early_stopping_patience"],
        "test_split_ratio": req["test_split_ratio"],
        "finetune_lr_cap": req["finetune_lr_cap"],
        "loss_mask_prefix_separator": req["loss_mask_prefix_separator"],
        "finetuning_strategy": req["finetuning_strategy"],
        "hub_model_id": req["hub_model_id"],
        "lora": req["lora"],
        "quantization": req["quantization"],
        "gguf_export": req["gguf_export"],
    }
    return cfg


def setup_job_logging(settings: Settings) -> None:
    """Re-initialize logging in RQ subprocess."""
    level: LogLevel = settings["logging"]["level"]
    format_mode: LogFormat = "json"
    setup_logging(
        level=level,
        format_mode=format_mode,
        service_name="model-trainer-job",
        instance_id=None,
        extra_fields=list(LOGGING_EXTRA_FIELDS),
    )


def load_tokenizer_for_training(settings: Settings, tokenizer_id: str) -> TokenizerHandle:
    """Load tokenizer from artifacts directory.

    Args:
        settings: Application settings containing artifacts_root path.
        tokenizer_id: Identifier for the tokenizer artifact.

    Returns:
        Loaded tokenizer handle ready for encoding/decoding.

    Raises:
        AppError: If tokenizer artifacts are not found.
    """
    tok_dir = os.path.join(settings["app"]["artifacts_root"], "tokenizers", tokenizer_id)
    return load_tokenizer_from_dir(tok_dir)


def emit_config_event(
    r: RedisStrProto,
    run_id: str,
    user_id: int,
    cfg: ModelTrainConfig,
    threads: int,
) -> None:
    """Emit trainer config metrics event at job start."""
    ev = make_config_event(
        job_id=run_id,
        user_id=user_id,
        model_family=cfg["model_family"],
        model_size=cfg["model_size"],
        total_epochs=cfg["num_epochs"],
        queue=TRAINER_QUEUE,
        cpu_cores=int(os.cpu_count() or 1),
        optimal_threads=threads,
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
    )
    publish_metrics(r, encode_trainer_metrics_event(ev))


def emit_progress_metrics(
    r: RedisStrProto,
    run_id: str,
    user_id: int,
    epoch: int,
    total_epochs: int,
    step: int,
    train_loss: float,
    train_ppl: float,
    grad_norm: float,
    samples_per_sec: float,
    val_loss: float | None = None,
    val_ppl: float | None = None,
) -> None:
    """Emit trainer progress metrics event during training."""
    ev = make_progress_metrics_event(
        job_id=run_id,
        user_id=user_id,
        epoch=int(epoch),
        total_epochs=int(total_epochs),
        step=int(step),
        train_loss=float(train_loss),
        train_ppl=float(train_ppl),
        grad_norm=float(grad_norm),
        samples_per_sec=float(samples_per_sec),
        val_loss=val_loss,
        val_ppl=val_ppl,
    )
    publish_metrics(r, encode_trainer_metrics_event(ev))


def emit_completed_metrics(
    r: RedisStrProto,
    run_id: str,
    user_id: int,
    test_loss: float,
    test_ppl: float,
    artifact_path: str,
) -> None:
    """Emit trainer completed metrics event at job completion."""
    ev = make_completed_metrics_event(
        job_id=run_id,
        user_id=user_id,
        test_loss=float(test_loss),
        test_ppl=float(test_ppl),
        artifact_path=artifact_path,
    )
    publish_metrics(r, encode_trainer_metrics_event(ev))
