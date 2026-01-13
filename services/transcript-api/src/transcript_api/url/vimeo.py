"""Vimeo URL parsing and validation."""

from __future__ import annotations

import re
import urllib.parse as urlparse

from platform_core.errors import AppError, TranscriptErrorCode

from .types import VimeoParsedURL

_VIMEO_HOSTS: frozenset[str] = frozenset(
    {
        "vimeo.com",
        "www.vimeo.com",
        "player.vimeo.com",
    }
)

_VIMEO_ID_RE: re.Pattern[str] = re.compile(r"^\d+$")


def is_vimeo_url(url: str) -> bool:
    """Check if a URL is a Vimeo URL.

    Args:
        url: URL string to check.

    Returns:
        True if the URL host matches a known Vimeo domain.
    """
    raw = url.strip()
    if not raw:
        return False
    parsed = urlparse.urlsplit(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc.lower()
    return host in _VIMEO_HOSTS


def parse_vimeo_url(url: str) -> VimeoParsedURL:
    """Parse and validate a Vimeo URL.

    Supports formats:
        - https://vimeo.com/123456789
        - https://vimeo.com/123456789?query=params
        - https://player.vimeo.com/video/123456789

    Args:
        url: Vimeo URL to parse.

    Returns:
        Parsed Vimeo URL with video ID and canonical URL.

    Raises:
        AppError: If URL is invalid or video ID cannot be extracted.
    """
    raw = url.strip()
    if not raw:
        raise AppError(
            TranscriptErrorCode.VIMEO_URL_INVALID,
            "Please provide a Vimeo URL",
            400,
        )

    parsed = urlparse.urlsplit(raw if "://" in raw else f"https://{raw}")

    host = parsed.netloc.lower()
    if host not in _VIMEO_HOSTS:
        raise AppError(
            TranscriptErrorCode.VIMEO_URL_INVALID,
            "Invalid Vimeo URL",
            400,
        )

    path = parsed.path.strip("/")
    vid: str | None = None

    if host == "player.vimeo.com":
        # Format: player.vimeo.com/video/123456789
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] == "video":
            vid = parts[1]
    else:
        # Format: vimeo.com/123456789
        parts = path.split("/")
        if parts and parts[0]:
            vid = parts[0]

    if vid is None or not _VIMEO_ID_RE.match(vid):
        raise AppError(
            TranscriptErrorCode.VIMEO_VIDEO_ID_INVALID,
            "Could not extract a valid Vimeo video id",
            400,
        )

    canonical = f"https://vimeo.com/{vid}"
    result: VimeoParsedURL = {
        "source": "vimeo",
        "video_id": vid,
        "canonical_url": canonical,
    }
    return result


__all__ = [
    "is_vimeo_url",
    "parse_vimeo_url",
]
