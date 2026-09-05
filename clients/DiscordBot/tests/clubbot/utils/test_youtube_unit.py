from __future__ import annotations

import logging

import pytest
from platform_core.errors import AppError

from clubbot.utils.youtube import canonicalize_youtube_url, extract_video_id


def test_extract_video_id_requires_non_empty() -> None:
    with pytest.raises(AppError):
        extract_video_id("  ")


def test_extract_video_id_watch_missing_v_param() -> None:
    with pytest.raises(AppError):
        extract_video_id("https://www.youtube.com/watch?x=y")


def test_extract_video_id_refuses_a_url_the_stdlib_cannot_parse() -> None:
    """This used to bind a `urlsplit` hook that raised, to reach the branch
    for an unparseable URL. An invalid IPv6 bracket produces the real
    ValueError, so the hook -- and the only reason it existed -- is gone."""
    with pytest.raises(AppError, match="Invalid YouTube URL format"):
        extract_video_id("http://[")


def test_canonicalize_produces_standard_watch_url() -> None:
    url = canonicalize_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


logger = logging.getLogger(__name__)
