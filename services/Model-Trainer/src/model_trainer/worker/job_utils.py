"""Job utilities for training worker."""

from __future__ import annotations

import os

from platform_core.job_events import default_events_channel
from platform_core.logging import LogFormat, LogLevel, setup_logging
from platform_core.queues import TRAINER_QUEUE
from platform_core.trainer_metrics_events import (
    encode_trainer_metrics_event,
    make_completed_metrics_event,
    make_config_event,
    make_progress_metrics_event,
)
from platform_ml import RequestedDevice, ResolvedDevice
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
from model_trainer.core.logging.types import LOGGING_EXTRA_FIELDS
from model_trainer.core.services.tokenizer.loader import load_tokenizer_from_dir

EVENTS_CHANNEL = default_events_channel("trainer")


def redis_client(settings: Settings) -> RedisStrProto:
    """Create Redis client from settings."""
    return _test_hooks.kv_store_factory(settings["redis"]["url"])


def publish_metrics(r: RedisStrProto, message: str) -> None:
    """Publish a trainer metrics event to the standard events channel."""
    r.publish(EVENTS_CHANNEL, message)


def setup_env(settings: Settings) -> int:
    """Setup environment for training job."""
    threads_cfg = settings["app"]["threads"]
    threads = threads_cfg if threads_cfg and threads_cfg > 0 else max(1, int(os.cpu_count() or 1))
    env = LocalCPUProvider(threads_count=threads).env()
    for k, v in env.items():
        __import__("os").putenv(k, v)
    __import__("os").putenv("TOKENIZERS_PARALLELISM", "1")
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
        "unsloth": req["unsloth"],
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
