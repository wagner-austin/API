from __future__ import annotations

from platform_discord.discord_types import Embed as _Embed
from platform_discord.embed_helpers import add_field as _add_field
from platform_discord.embed_helpers import create_embed as _create_embed
from platform_discord.embed_helpers import set_footer as _set_footer

from .types import FinalMetrics, Progress, TrainingConfig


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
    items: list[str] = [
        f"**Family:** `{config['model_family']}`",
        f"**Size:** `{config['model_size']}`",
        f"**Epochs:** `{config['total_epochs']}`",
    ]
    if isinstance(config.get("batch_size"), int):
        items.append(f"**Batch Size:** `{config['batch_size']}`")
    if isinstance(config.get("learning_rate"), float):
        items.append(f"**Learning Rate:** `{config['learning_rate']}`")
    _add_field(embed, name="Configuration", value="\n".join(items), inline=True)


def _add_resources_section(embed: _Embed, config: TrainingConfig) -> None:
    res: list[str] = []
    if isinstance(config.get("cpu_cores"), int):
        res.append(f"**CPU Cores:** `{config['cpu_cores']}`")
    if isinstance(config.get("memory_mb"), int):
        res.append(f"**Memory:** `{config['memory_mb']} MB`")
    if isinstance(config.get("optimal_threads"), int):
        res.append(f"**Threads:** `{config['optimal_threads']}`")
    if isinstance(config.get("optimal_workers"), int):
        res.append(f"**Workers:** `{config['optimal_workers']}`")
    if res:
        _add_field(embed, name="Resources", value="\n".join(res), inline=True)


def _add_progress_section(embed: _Embed, config: TrainingConfig, prog: Progress) -> None:
    epoch_pct = (prog["epoch"] / max(1, config["total_epochs"])) * 100.0
    epoch_bar = _progress_bar(prog["epoch"], max(1, config["total_epochs"]))
    lines = [
        f"**Epoch {prog['epoch']}/{config['total_epochs']}** ({epoch_pct:.1f}%)",
        f"`{epoch_bar}`",
        "",
        f"**Step:** `{prog['step']}` | **Speed:** `{prog['samples_per_sec']:.1f} samples/sec`",
        f"**Train Loss:** `{prog['train_loss']:.4f}` | **Train PPL:** `{prog['train_ppl']:.2f}`",
        f"**Grad Norm:** `{prog['grad_norm']:.4f}`",
    ]
    # Add validation metrics if present (epoch boundaries only)
    val_loss = prog.get("val_loss")
    val_ppl = prog.get("val_ppl")
    if val_loss is not None and val_ppl is not None:
        lines.append(f"**Val Loss:** `{val_loss:.4f}` | **Val PPL:** `{val_ppl:.2f}`")
    _add_field(embed, name="Progress", value="\n".join(lines), inline=False)


def _add_final_section(embed: _Embed, final: FinalMetrics | None) -> None:
    if final is None:
        return
    bits: list[str] = [
        f"**Test Loss:** `{final['test_loss']:.4f}`",
        f"**Test PPL:** `{final['test_ppl']:.2f}`",
        f"**Artifact:** `{final['artifact_path']}`",
    ]
    _add_field(embed, name="Results", value="\n".join(bits), inline=False)


def _add_failure_section(embed: _Embed, *, error_kind: str | None, message: str) -> None:
    if error_kind == "user":
        _add_field(embed, name="Configuration Issue", value=f"```{message}```", inline=False)
        _add_field(
            embed,
            name="Next Steps",
            value="Please check your configuration and try again.",
            inline=False,
        )
        return
    _add_field(embed, name="System Error", value=f"```{message}```", inline=False)


def build_training_embed(
    *,
    request_id: str,
    config: TrainingConfig,
    status: str,
    progress: Progress | None = None,
    final: FinalMetrics | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> _Embed:
    title, color = _status_title_and_color(status)
    embed = _create_embed(
        title=title,
        description=(
            f"Training **{config['model_family']} {config['model_size']}**\n"
            f"Queue: `{config['queue']}`"
        ),
        color=color,
    )

    _add_config_section(embed, config)
    _add_resources_section(embed, config)
    if progress is not None:
        _add_progress_section(embed, config, progress)
    if status == "completed":
        _add_final_section(embed, final)
    if status in {"failed", "canceled"} and isinstance(error_message, str):
        _add_failure_section(embed, error_kind=error_kind, message=error_message)

    _set_footer(embed, text=f"Request ID: {request_id}")
    return embed


__all__ = ["build_training_embed"]
