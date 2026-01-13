"""YouTube URL parsing and validation."""

from __future__ import annotations

import re
import urllib.parse as urlparse

from platform_core.errors import AppError, TranscriptErrorCode

from .types import YouTubeParsedURL

_YT_HOSTS: frozenset[str] = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }
)

_VIDEO_ID_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_-]{11}$")


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a YouTube URL.

    Args:
        url: URL string to check.

    Returns:
        True if the URL host matches a known YouTube domain.
    """
    raw = url.strip()
    if not raw:
        return False
    parsed = urlparse.urlsplit(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc.lower()
    return host in _YT_HOSTS


def _extract_watch_id(parsed: urlparse.SplitResult) -> str | None:
    """Extract video ID from YouTube watch URL query params.

    Args:
        parsed: Parsed URL result.

    Returns:
        Video ID if found, None otherwise.
    """
    query = urlparse.parse_qs(parsed.query)
    vals = query.get("v")
    if not vals:
        return None
    first = vals[0]
    return first if isinstance(first, str) and first else None


def parse_youtube_url(url: str) -> YouTubeParsedURL:
    """Parse and validate a YouTube URL.

    Args:
        url: YouTube URL to parse.

    Returns:
        Parsed YouTube URL with video ID and canonical URL.

    Raises:
        AppError: If URL is empty, invalid, not YouTube, or has invalid video ID.
    """
    raw = url.strip()
    if not raw:
        raise AppError(
            TranscriptErrorCode.YOUTUBE_URL_REQUIRED,
            "Please provide a YouTube URL",
            400,
        )

    parsed = urlparse.urlsplit(raw if "://" in raw else f"https://{raw}")

    host = parsed.netloc.lower()
    if host not in _YT_HOSTS:
        raise AppError(
            TranscriptErrorCode.YOUTUBE_URL_UNSUPPORTED,
            "Only YouTube URLs are supported",
            400,
        )

    path = parsed.path.strip("/")
    vid: str | None = None

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if path == "watch":
            vid = _extract_watch_id(parsed)
        else:
            parts = path.split("/")
            if len(parts) >= 2 and parts[0] in {"shorts", "live"}:
                vid = parts[1]
    else:
        # youtu.be short URLs
        parts = path.split("/")
        if parts and parts[0]:
            vid = parts[0]

    if vid is None or not _VIDEO_ID_RE.match(vid):
        raise AppError(
            TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID,
            "Could not extract a valid YouTube video id",
            400,
        )

    canonical = f"https://www.youtube.com/watch?v={vid}"
    result: YouTubeParsedURL = {
        "source": "youtube",
        "video_id": vid,
        "canonical_url": canonical,
    }
    return result


__all__ = [
    "is_youtube_url",
    "parse_youtube_url",
]
