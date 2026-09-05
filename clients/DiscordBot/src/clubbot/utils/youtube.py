"""Turning a YouTube URL into a video id, or into a message for the user.

The PARSING lives in platform_core.youtube_urls, shared with transcript-api.
What stays here is this bot's error policy: the same four messages it has
always shown, under the code its handlers expect.

Worth knowing if you touch it: the four failures are DISTINGUISHED by the
shared outcome, so giving each its own ErrorCode is now a one-line change to
the mapping below rather than a rewrite. This module keeps the single
INVALID_INPUT it has always used, because changing what a Discord user sees
is a product decision and not part of removing a duplicate.
"""

from __future__ import annotations

from platform_core.errors import AppError, ErrorCode
from platform_core.youtube_urls import canonical_watch_url, read_video_id

_MESSAGES: dict[str, str] = {
    "empty": "Please provide a YouTube URL",
    "unparseable": "Invalid YouTube URL format",
    "not_youtube": "Only YouTube URLs are supported for /transcript",
    "bad_video_id": "Could not extract a valid YouTube video id",
}
"""What the user is told for each way a URL can fail."""


def extract_video_id(url: str) -> str:
    """Read the video id from a YouTube URL, or refuse it.

    Args:
        url: The URL as the user typed it.

    Returns:
        The eleven-character video id.

    Raises:
        AppError: 400 INVALID_INPUT, with a message naming which of the four
            ways the URL was wrong.
    """
    outcome = read_video_id(url)
    video_id = outcome["video_id"]
    if video_id is None:
        raise AppError(ErrorCode.INVALID_INPUT, _MESSAGES[outcome["kind"]], http_status=400)
    return video_id


def canonicalize_youtube_url(url: str) -> str:
    """Return the canonical watch URL for whatever YouTube URL was given.

    Args:
        url: The URL as the user typed it.

    Returns:
        The canonical ``watch?v=`` form.

    Raises:
        AppError: 400, as :func:`extract_video_id`.
    """
    return canonical_watch_url(extract_video_id(url))


def validate_youtube_url(url: str) -> str:
    """Validate and return canonical URL for a YouTube video.

    Args:
        url: The URL as the user typed it.

    Returns:
        The canonical ``watch?v=`` form.

    Raises:
        AppError: 400, as :func:`extract_video_id`.
    """
    return canonicalize_youtube_url(url)


__all__ = ["canonicalize_youtube_url", "extract_video_id", "validate_youtube_url"]
