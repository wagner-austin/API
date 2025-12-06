from __future__ import annotations

from typing import NotRequired, TypedDict


class JobConfig(TypedDict):
    """Immutable job configuration for display in progress messages."""

    queue: str


class JobProgress(TypedDict):
    """Progress metrics for turkic job updates."""

    progress: int
    message: NotRequired[str]


class JobResult(TypedDict):
    """Final job result metadata."""

    result_id: str
    result_bytes: int


__all__ = ["JobConfig", "JobProgress", "JobResult"]
