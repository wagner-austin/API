"""Direct video URL parsing and validation."""

from __future__ import annotations

import hashlib
import urllib.parse as urlparse

from platform_core.errors import AppError, TranscriptErrorCode

from .types import DirectParsedURL

# Supported video/audio file extensions for direct URLs
_VALID_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Video formats
        "mp4",
        "webm",
        "mkv",
        "avi",
        "mov",
        "flv",
        "wmv",
        "m4v",
        "ogv",
        # Audio formats (for audio-only transcription)
        "mp3",
        "wav",
        "flac",
        "aac",
        "ogg",
        "m4a",
        "wma",
    }
)


def _extract_extension(path: str) -> str | None:
    """Extract file extension from URL path.

    Args:
        path: URL path component.

    Returns:
        Lowercase extension without dot, or None if not found.
    """
    if not path:
        return None
    # Remove query params that might be appended
    clean_path = path.split("?")[0]
    if "." not in clean_path:
        return None
    ext = clean_path.rsplit(".", 1)[-1].lower()
    return ext if ext else None


def _generate_video_id(url: str) -> str:
    """Generate a deterministic video ID from URL.

    Args:
        url: The URL to hash.

    Returns:
        MD5 hex digest of the URL (32 characters).
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def is_direct_url(url: str) -> bool:
    """Check if a URL is a direct video/audio file URL.

    Args:
        url: URL string to check.

    Returns:
        True if the URL has a valid video/audio file extension.
    """
    raw = url.strip()
    if not raw:
        return False
    parsed = urlparse.urlsplit(raw if "://" in raw else f"https://{raw}")
    ext = _extract_extension(parsed.path)
    return ext is not None and ext in _VALID_EXTENSIONS


def parse_direct_url(url: str) -> DirectParsedURL:
    """Parse and validate a direct video/audio URL.

    Args:
        url: Direct URL to parse.

    Returns:
        Parsed direct URL with generated video ID and extension.

    Raises:
        AppError: If URL is invalid or has unsupported extension.
    """
    raw = url.strip()
    if not raw:
        raise AppError(
            TranscriptErrorCode.DIRECT_URL_INVALID,
            "Please provide a video URL",
            400,
        )

    parsed = urlparse.urlsplit(raw if "://" in raw else f"https://{raw}")
    ext = _extract_extension(parsed.path)

    if ext is None:
        raise AppError(
            TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID,
            "URL must have a valid video/audio file extension",
            400,
        )

    if ext not in _VALID_EXTENSIONS:
        raise AppError(
            TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID,
            f"Unsupported file extension: .{ext}",
            400,
        )

    # Use the URL as-is for canonical (no normalization for direct URLs)
    canonical = raw
    video_id = _generate_video_id(raw)

    result: DirectParsedURL = {
        "source": "direct",
        "video_id": video_id,
        "canonical_url": canonical,
        "extension": ext,
    }
    return result


__all__ = [
    "is_direct_url",
    "parse_direct_url",
]
