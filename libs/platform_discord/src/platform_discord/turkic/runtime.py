from __future__ import annotations

from typing import TypedDict

from platform_discord.discord_types import Embed as _Embed

from .embeds import build_turkic_embed
from .types import JobConfig, JobProgress, JobResult


class RequestAction(TypedDict):
    job_id: str
    user_id: int
    embed: _Embed | None


class TurkicRuntime(TypedDict):
    _configs: dict[str, JobConfig]


def new_runtime() -> TurkicRuntime:
    return {"_configs": {}}


def on_started(
    runtime: TurkicRuntime,
    *,
    user_id: int | None,
    job_id: str,
    queue: str,
) -> RequestAction:
    cfg: JobConfig = {"queue": queue}
    runtime["_configs"][job_id] = cfg
    embed = build_turkic_embed(job_id=job_id, config=cfg, status="starting")
    if not isinstance(user_id, int):
        return {"job_id": job_id, "user_id": 0, "embed": None}
    return {"job_id": job_id, "user_id": user_id, "embed": embed}


def on_progress(
    runtime: TurkicRuntime,
    *,
    user_id: int | None,
    job_id: str,
    progress: int,
    message: str | None = None,
) -> RequestAction:
    cfg = runtime["_configs"].get(job_id, {"queue": "turkic"})
    prog: JobProgress = {"progress": int(progress)}
    if isinstance(message, str):
        prog["message"] = message
    embed = build_turkic_embed(job_id=job_id, config=cfg, status="processing", progress=prog)
    if not isinstance(user_id, int):
        return {"job_id": job_id, "user_id": 0, "embed": None}
    return {"job_id": job_id, "user_id": user_id, "embed": embed}


def on_completed(
    runtime: TurkicRuntime,
    *,
    user_id: int | None,
    job_id: str,
    result_id: str,
    result_bytes: int,
) -> RequestAction:
    cfg = runtime["_configs"].get(job_id, {"queue": "turkic"})
    res: JobResult = {"result_id": result_id, "result_bytes": result_bytes}
    embed = build_turkic_embed(job_id=job_id, config=cfg, status="completed", result=res)
    runtime["_configs"].pop(job_id, None)
    if not isinstance(user_id, int):
        return {"job_id": job_id, "user_id": 0, "embed": None}
    return {"job_id": job_id, "user_id": user_id, "embed": embed}


def on_failed(
    runtime: TurkicRuntime,
    *,
    user_id: int | None,
    job_id: str,
    error_kind: str,
    message: str,
    status: str,
) -> RequestAction:
    cfg = runtime["_configs"].get(job_id, {"queue": "turkic"})
    embed = build_turkic_embed(
        job_id=job_id,
        config=cfg,
        status=status,
        error_kind=error_kind,
        error_message=message,
    )
    runtime["_configs"].pop(job_id, None)
    if not isinstance(user_id, int):
        return {"job_id": job_id, "user_id": 0, "embed": None}
    return {"job_id": job_id, "user_id": user_id, "embed": embed}


__all__ = [
    "RequestAction",
    "TurkicRuntime",
    "new_runtime",
    "on_completed",
    "on_failed",
    "on_progress",
    "on_started",
]
