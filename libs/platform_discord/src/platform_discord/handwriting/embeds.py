from __future__ import annotations

from platform_discord.discord_types import Embed as _Embed
from platform_discord.embed_helpers import add_field as _add_field
from platform_discord.embed_helpers import create_embed as _create_embed
from platform_discord.embed_helpers import set_footer as _set_footer

from .types import BatchProgress, TrainingConfig, TrainingMetrics


def _status_title_and_color(status: str) -> tuple[str, int]:
    colors: dict[str, int] = {
        "starting": 0x5865F2,
        "training": 0x5865F2,
        "completed": 0x57F287,
        "failed": 0xED4245,
        "canceled": 0xFAA61A,
    }
    return f"Training {status.title()}", colors.get(status, 0x5865F2)


def _progress_bar(current: int, total: int, width: int = 20) -> str:
    total = max(total, 1)
    filled = max(0, min(width, int((current / total) * width)))
    return "#" * filled + "-" * (width - filled)


def _add_config_section(embed: _Embed, config: TrainingConfig) -> None:
    cfg_lines: list[str] = [f"**Epochs:** `{config['total_epochs']}`"]
    if isinstance(config.get("batch_size"), int):
        cfg_lines.append(f"**Batch Size:** `{config['batch_size']}`")
    if isinstance(config.get("device"), str) and config.get("device"):
        cfg_lines.append(f"**Device:** `{config['device']}`")
    if isinstance(config.get("learning_rate"), float):
        cfg_lines.append(f"**Learning Rate:** `{config['learning_rate']}`")
    _add_field(embed, name="Configuration", value="\n".join(cfg_lines), inline=True)


def _add_resources_section(embed: _Embed, config: TrainingConfig) -> None:
    res_lines: list[str] = []
    if isinstance(config.get("cpu_cores"), int):
        res_lines.append(f"**CPU Cores:** `{config['cpu_cores']}`")
    if isinstance(config.get("memory_mb"), int):
        res_lines.append(f"**Memory:** `{config['memory_mb']} MB`")
    if isinstance(config.get("optimal_threads"), int):
        res_lines.append(f"**Threads:** `{config['optimal_threads']}`")
    if isinstance(config.get("optimal_workers"), int):
        res_lines.append(f"**Workers:** `{config['optimal_workers']}`")
    if res_lines:
        _add_field(embed, name="Resources", value="\n".join(res_lines), inline=True)


def _add_augmentations_section(embed: _Embed, config: TrainingConfig) -> None:
    if config.get("augment"):
        aug_lines: list[str] = []
        aug_rotate = config.get("aug_rotate")
        if isinstance(aug_rotate, float) and aug_rotate > 0:
            aug_lines.append(f"**Rotation:** `{aug_rotate}`")
        aug_translate = config.get("aug_translate")
        if isinstance(aug_translate, float) and aug_translate > 0:
            aug_lines.append(f"**Translation:** `{aug_translate * 100:.0f}%`")
        noise_prob = config.get("noise_prob")
        if isinstance(noise_prob, float) and noise_prob > 0:
            aug_lines.append(f"**Noise:** `{noise_prob * 100:.0f}%`")
        dots_prob = config.get("dots_prob")
        if isinstance(dots_prob, float) and dots_prob > 0:
            aug_lines.append(f"**Dots:** `{dots_prob * 100:.0f}%`")
        if aug_lines:
            _add_field(embed, name="Augmentations", value="\n".join(aug_lines), inline=False)
    else:
        _add_field(embed, name="Augmentations", value="*None*", inline=False)


def _add_progress_section(embed: _Embed, config: TrainingConfig, progress: BatchProgress) -> None:
    epoch_pct = ((progress["epoch"] - 1) / max(1, config["total_epochs"])) * 100.0
    batch_pct = (progress["batch"] / max(1, progress["total_batches"])) * 100.0
    epoch_bar = _progress_bar(progress["epoch"] - 1, max(1, config["total_epochs"]))
    batch_bar = _progress_bar(progress["batch"], max(1, progress["total_batches"]))

    prog_text = (
        f"**Epoch {progress['epoch']}/{config['total_epochs']}** ({epoch_pct:.0f}%)\n"
        f"`{epoch_bar}`\n\n"
        f"**Batch {progress['batch']}/{progress['total_batches']}** ({batch_pct:.0f}%)\n"
        f"`{batch_bar}`"
    )
    _add_field(embed, name="Progress", value=prog_text, inline=False)

    batch_metrics = [
        f"**Batch Loss:** `{progress['batch_loss']:.4f}`",
        f"**Batch Accuracy:** `{progress['batch_acc']:.2%}`",
    ]
    _add_field(embed, name="Current Batch", value="\n".join(batch_metrics), inline=True)

    overall_metrics = [
        f"**Average Loss:** `{progress['avg_loss']:.4f}`",
        f"**Speed:** `{progress['samples_per_sec']:.1f} samples/sec`",
    ]
    _add_field(embed, name="Overall", value="\n".join(overall_metrics), inline=True)

    total_process_mb = progress["main_rss_mb"] + progress["workers_rss_mb"]
    mem_pct = f"**Memory:** `{progress['cgroup_pct']:.1f}%`"
    mem_used = f" ({progress['cgroup_usage_mb']}/{progress['cgroup_limit_mb']} MB)"
    proc_line = (
        f"**Process:** `{total_process_mb} MB` "
        f"(main: {progress['main_rss_mb']}, workers: {progress['workers_rss_mb']})"
    )
    mem_details = "\n".join([mem_pct + mem_used, proc_line])
    _add_field(embed, name="Memory", value=mem_details, inline=False)


def _add_completion_summary(embed: _Embed, final_metrics: TrainingMetrics | None) -> None:
    if not isinstance(final_metrics, dict):
        return
    parts: list[str] = []
    if final_metrics.get("final_avg_loss", 0.0) > 0:
        parts.append(f"**Final Avg Loss:** `{final_metrics['final_avg_loss']:.4f}`")
    if final_metrics.get("final_train_loss", 0.0) > 0:
        parts.append(f"**Final Train Loss:** `{final_metrics['final_train_loss']:.4f}`")
    if final_metrics.get("total_time_s", 0.0) > 0:
        mins = int(final_metrics["total_time_s"] // 60)
        secs = int(final_metrics["total_time_s"] % 60)
        parts.append(
            f"**Total Time:** `{mins}m {secs}s`" if mins > 0 else f"**Total Time:** `{secs}s`"
        )
    if final_metrics.get("avg_samples_per_sec", 0.0) > 0:
        parts.append(f"**Avg Speed:** `{final_metrics['avg_samples_per_sec']:.1f} samples/sec`")
    if final_metrics.get("best_epoch", 0) > 0:
        parts.append(f"**Best Epoch:** `{final_metrics['best_epoch']}`")
    if final_metrics.get("peak_memory_mb", 0) > 0:
        parts.append(f"**Peak Memory:** `{final_metrics['peak_memory_mb']} MB`")
    if parts:
        _add_field(embed, name="Training Summary", value="\n".join(parts), inline=False)


def _add_final_performance_and_run_id(
    embed: _Embed, *, final_val_acc: float | None, run_id: str | None
) -> None:
    if isinstance(final_val_acc, float):
        _add_field(
            embed,
            name="Final Performance",
            value=f"**Best Validation Accuracy:** `{final_val_acc:.2%}`",
            inline=False,
        )
    if isinstance(run_id, str) and run_id:
        _add_field(embed, name="Run ID", value=f"`{run_id}`", inline=True)


def _add_failure_section(embed: _Embed, *, error_kind: str | None, error_message: str) -> None:
    if error_kind == "user":
        _add_field(
            embed,
            name="Configuration Issue",
            value=f"```{error_message}```",
            inline=False,
        )
        _add_field(
            embed,
            name="Next Steps",
            value="Please check your configuration and try again.",
            inline=False,
        )
        return

    _add_field(embed, name="System Error", value=f"```{error_message}```", inline=False)
    lower = error_message.lower()
    if ("memory" in lower) or ("oom" in lower):
        next_steps = (
            "Memory issue detected.\n"
            "- Reduce batch size in your training config\n"
            "- Reduce DataLoader workers\n"
            "- Try training with fewer epochs to conserve resources"
        )
    elif ("upload" in lower) or ("artifact" in lower):
        next_steps = (
            "Artifact upload failed.\n"
            "The model trained successfully but could not be saved. "
            "Check worker logs and try again."
        )
    else:
        next_steps = (
            "Please try again. If the issue persists, check worker logs or contact support."
        )
    _add_field(embed, name="Next Steps", value=next_steps, inline=False)


def build_training_embed(
    *,
    request_id: str,
    config: TrainingConfig,
    status: str,
    progress: BatchProgress | None = None,
    final_val_acc: float | None = None,
    final_metrics: TrainingMetrics | None = None,
    run_id: str | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> _Embed:
    """Construct a consistent, ASCII-only training status embed."""

    title, color = _status_title_and_color(status)
    model_id = config["model_id"]
    embed = _create_embed(title=title, description=f"Training **{model_id}**", color=color)

    _add_config_section(embed, config)
    _add_resources_section(embed, config)
    _add_augmentations_section(embed, config)

    if progress is not None:
        _add_progress_section(embed, config, progress)

    if status == "completed":
        _add_completion_summary(embed, final_metrics)
        _add_final_performance_and_run_id(embed, final_val_acc=final_val_acc, run_id=run_id)

    if status in {"failed", "canceled"} and isinstance(error_message, str):
        _add_failure_section(embed, error_kind=error_kind, error_message=error_message)

    job_info = [f"**Queue:** `{config['queue']}`", f"**Status:** `{status}`"]
    _add_field(embed, name="Job Info", value="\n".join(job_info), inline=False)
    _set_footer(embed, text=f"Request ID: {request_id}")
    return embed


__all__ = ["build_training_embed"]
