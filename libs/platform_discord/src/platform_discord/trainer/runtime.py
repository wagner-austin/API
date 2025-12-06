"""Trainer runtime for Discord embed generation.

Manages training state and produces Discord embeds from trainer events.
Uses platform_core types directly for DRY.
"""

from __future__ import annotations

from typing import TypedDict

from platform_discord.discord_types import Embed as _Embed

from .embeds import build_training_embed
from .types import FinalMetrics, Progress, TrainingConfig


class RequestAction(TypedDict):
    request_id: str
    user_id: int
    embed: _Embed | None


class TrainerRuntime(TypedDict):
    _configs: dict[str, TrainingConfig]
    _finals: dict[str, FinalMetrics]


def new_runtime() -> TrainerRuntime:
    return {"_configs": {}, "_finals": {}}


def on_config(runtime: TrainerRuntime, event: TrainingConfig) -> RequestAction:
    """Handle a config event at training start.

    Stores the configuration and returns an embed for the starting status.
    """
    request_id = event["job_id"]
    user_id = event["user_id"]
    runtime["_configs"][request_id] = event
    runtime["_finals"].pop(request_id, None)
    embed = build_training_embed(request_id=request_id, config=event, status="starting")
    return {"request_id": request_id, "user_id": user_id, "embed": embed}


def on_progress(runtime: TrainerRuntime, event: Progress) -> RequestAction:
    """Handle a progress metrics event during training.

    Updates progress and returns an embed with current metrics.
    """
    request_id = event["job_id"]
    user_id = event["user_id"]
    cfg = runtime["_configs"].get(request_id)
    if cfg is None:
        # Create minimal config from progress event
        cfg = _minimal_config(request_id, user_id, event["total_epochs"])
        runtime["_configs"][request_id] = cfg
    embed = build_training_embed(
        request_id=request_id,
        config=cfg,
        status="training",
        progress=event,
    )
    return {"request_id": request_id, "user_id": user_id, "embed": embed}


def on_completed(runtime: TrainerRuntime, event: FinalMetrics) -> RequestAction:
    """Handle a completed metrics event at training completion.

    Returns an embed with final metrics and cleans up state.
    """
    request_id = event["job_id"]
    user_id = event["user_id"]
    cfg = runtime["_configs"].get(request_id)
    if cfg is None:
        cfg = _minimal_config(request_id, user_id, 0)
        runtime["_configs"][request_id] = cfg
    runtime["_finals"][request_id] = event
    embed = build_training_embed(request_id=request_id, config=cfg, status="completed", final=event)
    runtime["_configs"].pop(request_id, None)
    runtime["_finals"].pop(request_id, None)
    return {"request_id": request_id, "user_id": user_id, "embed": embed}


def on_failed(
    runtime: TrainerRuntime,
    *,
    user_id: int,
    request_id: str,
    error_kind: str,
    message: str,
    status: str,
) -> RequestAction:
    """Handle a failed job event.

    Returns an embed with error details and cleans up state.
    """
    cfg = runtime["_configs"].get(request_id)
    if cfg is None:
        cfg = _minimal_config(request_id, user_id, 0)
    embed = build_training_embed(
        request_id=request_id,
        config=cfg,
        status=status,
        error_kind=error_kind,
        error_message=message,
    )
    runtime["_configs"].pop(request_id, None)
    runtime["_finals"].pop(request_id, None)
    return {"request_id": request_id, "user_id": user_id, "embed": embed}


def _minimal_config(job_id: str, user_id: int, total_epochs: int) -> TrainingConfig:
    """Create a minimal config for fallback when no config event was received."""
    return {
        "type": "trainer.metrics.config.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_family": "unknown",
        "model_size": "unknown",
        "total_epochs": total_epochs,
        "queue": "training",
    }


__all__ = [
    "RequestAction",
    "TrainerRuntime",
    "new_runtime",
    "on_completed",
    "on_config",
    "on_failed",
    "on_progress",
]
