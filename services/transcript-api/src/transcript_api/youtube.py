from __future__ import annotations

import re
import urllib.parse as _url

from platform_core.errors import AppError, TranscriptErrorCode

_YT_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _extract_watch_id(parsed: _url.SplitResult) -> str | None:
    q = _url.parse_qs(parsed.query)
    vals = q.get("v")
    if not vals:
        return None
    first = vals[0]
    return first if isinstance(first, str) and first else None


def extract_video_id(url: str) -> str:
    raw = url.strip()
    if not raw:
        raise AppError(
            TranscriptErrorCode.YOUTUBE_URL_REQUIRED, "Please provide a YouTube URL", 400
        )
    try:
        parsed = _url.urlsplit(raw if "://" in raw else f"https://{raw}")
    except ValueError:
        raise AppError(
            TranscriptErrorCode.YOUTUBE_URL_INVALID, "Invalid YouTube URL format", 400
        ) from None

    host = parsed.netloc.lower()
    if host not in _YT_HOSTS:
        raise AppError(
            TranscriptErrorCode.YOUTUBE_URL_UNSUPPORTED, "Only YouTube URLs are supported", 400
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
        parts = path.split("/")
        if parts and parts[0]:
            vid = parts[0]

    is_valid = isinstance(vid, str) and bool(_VIDEO_ID_RE.match(vid))
    if not is_valid:
        raise AppError(
            TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID,
            "Could not extract a valid YouTube video id",
            400,
        )
    return str(vid)


def canonicalize_youtube_url(url: str) -> str:
    vid = extract_video_id(url)
    return f"https://www.youtube.com/watch?v={vid}"


def validate_youtube_url(url: str) -> str:
    return canonicalize_youtube_url(url)
