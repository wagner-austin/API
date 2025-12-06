from __future__ import annotations

from platform_discord.discord_types import Embed as _Embed
from platform_discord.embed_helpers import add_field as _add_field
from platform_discord.embed_helpers import create_embed as _create_embed
from platform_discord.embed_helpers import set_footer as _set_footer

from .types import JobConfig, JobProgress, JobResult


def _status_title_and_color(status: str) -> tuple[str, int]:
    colors: dict[str, int] = {
        "starting": 0x5865F2,
        "processing": 0x5865F2,
        "completed": 0x57F287,
        "failed": 0xED4245,
        "canceled": 0xFAA61A,
    }
    return f"Turkic {status.title()}", colors.get(status, 0x5865F2)


def _progress_bar(pct: int, width: int = 20) -> str:
    p = max(0, min(100, int(pct)))
    filled = int((p / 100.0) * width)
    return "#" * filled + "-" * (width - filled)


def _add_progress_section(embed: _Embed, prog: JobProgress) -> None:
    pct = max(0, min(100, int(prog["progress"])))
    bar = _progress_bar(pct)
    text = f"**Progress:** `{pct}%`\n`{bar}`"
    msg = prog.get("message")
    if isinstance(msg, str) and msg.strip():
        text = text + f"\n\n**Message:** `{msg}`"
    _add_field(embed, name="Status", value=text, inline=False)


def _add_results_section(embed: _Embed, res: JobResult | None) -> None:
    if not isinstance(res, dict):
        return
    _add_field(
        embed,
        name="Result",
        value=f"**Result ID:** `{res['result_id']}`\n**Bytes:** `{res['result_bytes']}`",
        inline=False,
    )


def _add_failure_section(embed: _Embed, *, error_kind: str | None, message: str) -> None:
    if error_kind == "user":
        _add_field(embed, name="Configuration Issue", value=f"```{message}```", inline=False)
        _add_field(
            embed,
            name="Next Steps",
            value="Please review your request parameters and try again.",
            inline=False,
        )
        return
    _add_field(embed, name="System Error", value=f"```{message}```", inline=False)


def build_turkic_embed(
    *,
    job_id: str,
    config: JobConfig,
    status: str,
    progress: JobProgress | None = None,
    result: JobResult | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> _Embed:
    title, color = _status_title_and_color(status)
    embed = _create_embed(
        title=title,
        description="Processing corpus request",
        color=color,
    )

    _add_field(
        embed,
        name="Job Info",
        value=f"**Queue:** `{config['queue']}`\n**Status:** `{status}`",
        inline=False,
    )
    if progress is not None:
        _add_progress_section(embed, progress)
    if status == "completed":
        _add_results_section(embed, result)
    if status in {"failed", "canceled"} and isinstance(error_message, str):
        _add_failure_section(embed, error_kind=error_kind, message=error_message)

    _set_footer(embed, text=f"Job ID: {job_id}")
    return embed


__all__ = ["build_turkic_embed"]
