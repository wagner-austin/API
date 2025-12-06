from __future__ import annotations

from typing import Final, TypedDict


class TurkicLogFields(TypedDict, total=False):
    """Structured log fields emitted by turkic-api services."""

    job_id: str
    language: str
    url: str
    status: int | str
    file_id: str
    redis_url: str
    queue_name: str
    events_channel: str
    has_url: bool
    has_key: bool


LOG_EXTRA_FIELDS: Final[list[str]] = list(TurkicLogFields.__annotations__.keys())


__all__ = ["LOG_EXTRA_FIELDS", "TurkicLogFields"]
