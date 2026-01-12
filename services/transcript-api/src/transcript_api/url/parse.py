"""Unified video URL parsing entry point."""

from __future__ import annotations

from platform_core.errors import AppError, TranscriptErrorCode

from .direct import is_direct_url, parse_direct_url
from .types import ParsedURL
from .vimeo import is_vimeo_url, parse_vimeo_url
from .youtube import is_youtube_url, parse_youtube_url


def parse_video_url(url: str) -> ParsedURL:
    """Parse a video URL and return typed result.

    Automatically detects the video source (YouTube, Vimeo, or direct URL)
    and returns the appropriate parsed result type.

    Args:
        url: Video URL to parse.

    Returns:
        Parsed URL with source-specific fields.

    Raises:
        AppError: If URL is empty, invalid, or from unsupported source.
    """
    raw = url.strip()
    if not raw:
        raise AppError(
            TranscriptErrorCode.VIDEO_URL_REQUIRED,
            "Please provide a video URL",
            400,
        )

    # Check each source in order of specificity
    if is_youtube_url(raw):
        return parse_youtube_url(raw)

    if is_vimeo_url(raw):
        return parse_vimeo_url(raw)

    if is_direct_url(raw):
        return parse_direct_url(raw)

    raise AppError(
        TranscriptErrorCode.VIDEO_URL_UNSUPPORTED,
        "URL must be from YouTube, Vimeo, or a direct video file link",
        400,
    )


__all__ = [
    "parse_video_url",
]
