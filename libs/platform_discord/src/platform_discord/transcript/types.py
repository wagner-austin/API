from __future__ import annotations

from typing import NotRequired, TypedDict


class TranscriptInfo(TypedDict):
    url: str
    video_id: str
    chars: NotRequired[int]


__all__ = ["TranscriptInfo"]
