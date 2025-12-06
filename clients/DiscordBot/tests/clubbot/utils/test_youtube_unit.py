from __future__ import annotations

import logging
import urllib.parse as _url

import pytest
from platform_core.errors import AppError

from clubbot.utils.youtube import canonicalize_youtube_url, extract_video_id


def test_extract_video_id_requires_non_empty() -> None:
    with pytest.raises(AppError):
        extract_video_id("  ")


def test_extract_video_id_watch_missing_v_param() -> None:
    with pytest.raises(AppError):
        extract_video_id("https://www.youtube.com/watch?x=y")


def test_extract_video_id_urlsplit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from urllib.parse import SplitResult

    def _boom(url: str) -> SplitResult:
        raise ValueError("bad parse")

    monkeypatch.setattr(_url, "urlsplit", _boom)
    with pytest.raises(AppError):
        extract_video_id("youtube.com/watch?v=dQw4w9WgXcQ")


def test_canonicalize_produces_standard_watch_url() -> None:
    url = canonicalize_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


logger = logging.getLogger(__name__)
