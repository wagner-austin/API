"""Turning a YouTube URL into this service's video id, or into its own error.

The PARSING lives in platform_core.youtube_urls, shared with clubbot. What
stays here is the error policy: four traceable codes, one per way a URL can
fail, which is the half a shared raiser would have had to flatten.
"""

from __future__ import annotations

from platform_core.errors import AppError, TranscriptErrorCode
from platform_core.youtube_urls import canonical_watch_url, read_video_id

_REFUSALS: dict[str, tuple[TranscriptErrorCode, str]] = {
    "empty": (TranscriptErrorCode.YOUTUBE_URL_REQUIRED, "Please provide a YouTube URL"),
    "unparseable": (TranscriptErrorCode.YOUTUBE_URL_INVALID, "Invalid YouTube URL format"),
    "not_youtube": (
        TranscriptErrorCode.YOUTUBE_URL_UNSUPPORTED,
        "Only YouTube URLs are supported",
    ),
    "bad_video_id": (
        TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID,
        "Could not extract a valid YouTube video id",
    ),
}
"""Which refusal each parse failure earns.

A mapping rather than a chain of ifs because the outcome's `kind` already
enumerates the failures; restating them as branches would be a second place
for the set to drift from the one that produces it.
"""


def extract_video_id(url: str) -> str:
    """Read the video id from a YouTube URL, or refuse the URL.

    Args:
        url: The URL as the caller supplied it.

    Returns:
        The eleven-character video id.

    Raises:
        AppError: 400, with the code naming which of the four ways the URL
            was wrong: absent, unparseable, not a YouTube host, or carrying
            nothing that looks like a video id.
    """
    outcome = read_video_id(url)
    video_id = outcome["video_id"]
    if video_id is None:
        code, message = _REFUSALS[outcome["kind"]]
        raise AppError(code, message, 400)
    return video_id


def canonicalize_youtube_url(url: str) -> str:
    """Return the canonical watch URL for whatever YouTube URL was given.

    Args:
        url: The URL as the caller supplied it.

    Returns:
        The canonical ``watch?v=`` form.

    Raises:
        AppError: 400, as :func:`extract_video_id`.
    """
    return canonical_watch_url(extract_video_id(url))


def validate_youtube_url(url: str) -> str:
    """Check a YouTube URL and return its canonical form.

    Args:
        url: The URL as the caller supplied it.

    Returns:
        The canonical ``watch?v=`` form.

    Raises:
        AppError: 400, as :func:`extract_video_id`.
    """
    return canonicalize_youtube_url(url)


__all__ = ["canonicalize_youtube_url", "extract_video_id", "validate_youtube_url"]
