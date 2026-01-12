"""Direct video URL parsing and validation."""

from __future__ import annotations

import hashlib
import re
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

# Patterns that indicate a file download URL (even without visible extension)
# These patterns match CDN/LMS download endpoints
_DOWNLOAD_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/files/\d+/download", re.IGNORECASE),  # Canvas: /files/123/download
    re.compile(r"/download/\d+", re.IGNORECASE),  # Generic: /download/123
    re.compile(r"/media/\d+", re.IGNORECASE),  # Media endpoints: /media/123
    re.compile(r"/attachments/\d+", re.IGNORECASE),  # Attachments: /attachments/123
    re.compile(r"/blob/", re.IGNORECASE),  # Azure/cloud blob storage
    re.compile(r"/objects/", re.IGNORECASE),  # S3-style object storage
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


def _is_download_pattern(path: str) -> bool:
    """Check if URL path matches common download patterns.

    Args:
        path: URL path component.

    Returns:
        True if path matches a known download pattern.
    """
    return any(pattern.search(path) for pattern in _DOWNLOAD_PATH_PATTERNS)


def _has_download_params(query: str) -> bool:
    """Check if URL has query params indicating a file download.

    Args:
        query: URL query string.

    Returns:
        True if query contains download-related parameters.
    """
    if not query:
        return False
    params = urlparse.parse_qs(query)
    # Common download indicators: verifier, token, download flag
    download_keys = {
        "verifier",
        "token",
        "download",
        "download_frd",
        "response-content-disposition",
    }
    return bool(download_keys & set(params.keys()))


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

    Accepts URLs with valid media extensions OR URLs matching common
    download patterns (CDN, LMS, cloud storage).

    Args:
        url: URL string to check.

    Returns:
        True if URL has valid extension or matches download pattern.
    """
    raw = url.strip()
    if not raw:
        return False
    parsed = urlparse.urlsplit(raw if "://" in raw else f"https://{raw}")

    # Check for valid media extension
    ext = _extract_extension(parsed.path)
    if ext is not None and ext in _VALID_EXTENSIONS:
        return True

    # Check for download URL patterns
    if _is_download_pattern(parsed.path):
        return True

    # Check for download query params combined with non-webpage path
    if _has_download_params(parsed.query):
        # Reject obvious webpage extensions
        return ext not in {"html", "htm", "php", "asp", "aspx", "jsp"}

    return False


def parse_direct_url(url: str) -> DirectParsedURL:
    """Parse and validate a direct video/audio URL.

    Handles both extension-based URLs and download pattern URLs.
    For download URLs without visible extension, extension is set to
    empty string (content type determined at download time).

    Args:
        url: Direct URL to parse.

    Returns:
        Parsed direct URL with generated video ID and extension.

    Raises:
        AppError: If URL is invalid or clearly not a media file.
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

    # Determine if this is a valid direct URL and extract extension
    if ext is not None and ext in _VALID_EXTENSIONS:
        # Standard media file URL with extension
        final_ext = ext
    elif _is_download_pattern(parsed.path) or _has_download_params(parsed.query):
        # Download URL - extension unknown until download
        # Reject obvious webpage extensions
        if ext in {"html", "htm", "php", "asp", "aspx", "jsp"}:
            raise AppError(
                TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID,
                f"URL appears to be a webpage, not a media file: .{ext}",
                400,
            )
        final_ext = ""  # Extension determined at download time
    elif ext is None:
        raise AppError(
            TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID,
            "URL must have a valid video/audio file extension or be a download link",
            400,
        )
    else:
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
        "extension": final_ext,
    }
    return result


__all__ = [
    "is_direct_url",
    "parse_direct_url",
]
