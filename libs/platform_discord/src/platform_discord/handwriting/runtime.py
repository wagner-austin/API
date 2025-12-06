from __future__ import annotations

from typing import TypedDict

from platform_discord.discord_types import Embed as _Embed
from platform_discord.embed_helpers import (
    add_field as _add_field,
)
from platform_discord.embed_helpers import (
    create_embed as _create_embed,
)
from platform_discord.embed_helpers import (
    set_footer as _set_footer,
)

from .embeds import build_training_embed
from .types import BatchProgress, TrainingConfig, TrainingMetrics


class RequestAction(TypedDict):
    request_id: str
    user_id: int
    embed: _Embed | None


class DigitsRuntime(TypedDict):
    """In-memory state and embed composition for digits training events."""

    _configs: dict[str, TrainingConfig]
    _metrics: dict[str, TrainingMetrics]


def new_runtime() -> DigitsRuntime:
    return {"_configs": {}, "_metrics": {}}


def on_started(
    runtime: DigitsRuntime,
    *,
    user_id: int,
    request_id: str,
    model_id: str,
    total_epochs: int,
    queue: str,
    cpu_cores: int | None = None,
    optimal_threads: int | None = None,
    memory_mb: int | None = None,
    optimal_workers: int | None = None,
    max_batch_size: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    augment: bool | None = None,
    aug_rotate: float | None = None,
    aug_translate: float | None = None,
    noise_prob: float | None = None,
    dots_prob: float | None = None,
) -> RequestAction:
    _ = max_batch_size
    config: TrainingConfig = {
        "model_id": model_id,
        "total_epochs": total_epochs,
        "queue": queue,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
        "cpu_cores": cpu_cores,
        "memory_mb": memory_mb,
        "optimal_threads": optimal_threads,
        "optimal_workers": optimal_workers,
        "augment": augment,
        "aug_rotate": aug_rotate,
        "aug_translate": aug_translate,
        "noise_prob": noise_prob,
        "dots_prob": dots_prob,
    }
    runtime["_configs"][request_id] = config
    runtime["_metrics"][request_id] = {}
    embed = build_training_embed(request_id=request_id, config=config, status="starting")
    return {"request_id": request_id, "user_id": user_id, "embed": embed}


def on_progress(
    runtime: DigitsRuntime,
    *,
    user_id: int,
    request_id: str,
    epoch: int,
    total_epochs: int,
    val_acc: float | None,
    train_loss: float | None = None,
    time_s: float | None = None,
) -> RequestAction:
    current = runtime["_metrics"].get(request_id, {})
    current_loss = current.get("final_train_loss", 0.0)
    final_train_loss = train_loss if isinstance(train_loss, float) else current_loss
    added_time = time_s if isinstance(time_s, float) else 0.0
    total_time = current.get("total_time_s", 0.0) + added_time
    best_epoch = current.get("best_epoch", 0)
    if isinstance(val_acc, float) and epoch > best_epoch:
        best_epoch = epoch
    runtime["_metrics"][request_id] = {
        "final_avg_loss": current.get("final_avg_loss", 0.0),
        "final_train_loss": final_train_loss,
        "total_time_s": total_time,
        "avg_samples_per_sec": current.get("avg_samples_per_sec", 0.0),
        "best_epoch": best_epoch,
        "peak_memory_mb": current.get("peak_memory_mb", 0),
    }

    pct = (epoch / max(1, total_epochs)) * 100.0
    filled = max(0, min(20, int((epoch / max(1, total_epochs)) * 20)))
    bar = "#" * filled + "-" * (20 - filled)
    embed = _create_embed(
        title="Training Progress",
        description=f"**Epoch {epoch} of {total_epochs}** ({pct:.1f}%)\n`{bar}`",
        color=0xFEE75C,
    )
    lines: list[str] = []
    if isinstance(val_acc, float):
        lines.append(f"**Validation Accuracy:** `{val_acc:.2%}`")
    if isinstance(train_loss, float):
        lines.append(f"**Training Loss:** `{train_loss:.4f}`")
    if lines:
        _add_field(embed, name="Metrics", value="\n".join(lines), inline=True)
    if isinstance(time_s, float):
        mins = int(time_s // 60)
        secs = int(time_s % 60)
        text = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        _add_field(embed, name="Epoch Time", value=f"`{text}`", inline=True)
    _set_footer(embed, text=f"Request ID: {request_id}")
    return {"request_id": request_id, "user_id": user_id, "embed": embed}


def on_batch(
    runtime: DigitsRuntime,
    *,
    user_id: int,
    request_id: str,
    model_id: str,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    batch_loss: float,
    batch_acc: float,
    avg_loss: float,
    samples_per_sec: float,
    main_rss_mb: int,
    workers_rss_mb: int,
    worker_count: int,
    cgroup_usage_mb: int,
    cgroup_limit_mb: int,
    cgroup_pct: float,
    anon_mb: int,
    file_mb: int,
) -> RequestAction:
    config = runtime["_configs"].get(request_id)
    if config is None:
        new_config: TrainingConfig = {
            "model_id": model_id,
            "total_epochs": total_epochs,
            "queue": "digits",
            "batch_size": None,
            "learning_rate": None,
            "device": None,
            "cpu_cores": None,
            "memory_mb": None,
            "optimal_threads": None,
            "optimal_workers": None,
            "augment": None,
            "aug_rotate": None,
            "aug_translate": None,
            "noise_prob": None,
            "dots_prob": None,
        }
        runtime["_configs"][request_id] = new_config
        config = new_config

    current = runtime["_metrics"].get(request_id, {})
    total_process_mb = main_rss_mb + workers_rss_mb
    runtime["_metrics"][request_id] = {
        "final_avg_loss": avg_loss,
        "final_train_loss": current.get("final_train_loss", 0.0),
        "total_time_s": current.get("total_time_s", 0.0),
        "avg_samples_per_sec": samples_per_sec,
        "best_epoch": current.get("best_epoch", 0),
        "peak_memory_mb": max(current.get("peak_memory_mb", 0), total_process_mb),
    }

    progress: BatchProgress = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch,
        "total_batches": total_batches,
        "batch_loss": batch_loss,
        "batch_acc": batch_acc,
        "avg_loss": avg_loss,
        "samples_per_sec": samples_per_sec,
        "main_rss_mb": main_rss_mb,
        "workers_rss_mb": workers_rss_mb,
        "worker_count": worker_count,
        "cgroup_usage_mb": cgroup_usage_mb,
        "cgroup_limit_mb": cgroup_limit_mb,
        "cgroup_pct": cgroup_pct,
        "anon_mb": anon_mb,
        "file_mb": file_mb,
    }
    embed = build_training_embed(
        request_id=request_id,
        config=config,
        status="training",
        progress=progress,
    )
    return {"request_id": request_id, "user_id": user_id, "embed": embed}


def on_best(
    runtime: DigitsRuntime, *, user_id: int, request_id: str, epoch: int, val_acc: float
) -> None:
    current = runtime["_metrics"].get(request_id, {})
    runtime["_metrics"][request_id] = {
        "final_avg_loss": current.get("final_avg_loss", 0.0),
        "final_train_loss": current.get("final_train_loss", 0.0),
        "total_time_s": current.get("total_time_s", 0.0),
        "avg_samples_per_sec": current.get("avg_samples_per_sec", 0.0),
        "best_epoch": epoch,
        "peak_memory_mb": current.get("peak_memory_mb", 0),
    }
    _ = (user_id, val_acc)


def on_artifact(runtime: DigitsRuntime, *, user_id: int, request_id: str, path: str) -> None:
    _ = (user_id, request_id, path)


def on_upload(
    runtime: DigitsRuntime,
    *,
    user_id: int,
    request_id: str,
    status: int,
    model_bytes: int,
    manifest_bytes: int,
) -> None:
    _ = (user_id, request_id, status, model_bytes, manifest_bytes)


def on_prune(runtime: DigitsRuntime, *, user_id: int, request_id: str, deleted_count: int) -> None:
    _ = (user_id, request_id, deleted_count)


def on_completed(
    runtime: DigitsRuntime,
    *,
    user_id: int,
    request_id: str,
    model_id: str,
    run_id: str | None,
    val_acc: float,
) -> RequestAction | None:
    config = runtime["_configs"].get(request_id)
    if config is None:
        config = {
            "model_id": model_id,
            "total_epochs": 0,
            "queue": "digits",
            "batch_size": None,
            "learning_rate": None,
            "device": None,
            "cpu_cores": None,
            "memory_mb": None,
            "optimal_threads": None,
            "optimal_workers": None,
            "augment": None,
            "aug_rotate": None,
            "aug_translate": None,
            "noise_prob": None,
            "dots_prob": None,
        }
        runtime["_configs"][request_id] = config
    metrics = runtime["_metrics"].get(request_id)
    embed = build_training_embed(
        request_id=request_id,
        config=config,
        status="completed",
        final_val_acc=val_acc,
        final_metrics=metrics,
        run_id=run_id,
    )
    runtime["_configs"].pop(request_id, None)
    runtime["_metrics"].pop(request_id, None)
    return {"request_id": request_id, "user_id": user_id, "embed": embed}


def on_failed(
    runtime: DigitsRuntime,
    *,
    user_id: int,
    request_id: str,
    model_id: str,
    error_kind: str,
    message: str,
    queue: str,
    status: str,
) -> RequestAction:
    config = runtime["_configs"].get(request_id)
    if config is None:
        new_config: TrainingConfig = {
            "model_id": model_id,
            "total_epochs": 0,
            "queue": queue,
            "batch_size": None,
            "learning_rate": None,
            "device": None,
            "cpu_cores": None,
            "memory_mb": None,
            "optimal_threads": None,
            "optimal_workers": None,
            "augment": None,
            "aug_rotate": None,
            "aug_translate": None,
            "noise_prob": None,
            "dots_prob": None,
        }
        config = new_config
    embed = build_training_embed(
        request_id=request_id,
        config=config,
        status=status,
        error_kind=error_kind,
        error_message=message,
    )
    runtime["_configs"].pop(request_id, None)
    runtime["_metrics"].pop(request_id, None)
    return {"request_id": request_id, "user_id": user_id, "embed": embed}


__all__ = [
    "DigitsRuntime",
    "RequestAction",
    "new_runtime",
    "on_artifact",
    "on_batch",
    "on_best",
    "on_completed",
    "on_failed",
    "on_progress",
    "on_prune",
    "on_started",
    "on_upload",
]
